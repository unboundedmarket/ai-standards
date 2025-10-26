"""
Weight calculation for consensus mechanism based on certification and benchmarking
"""
from typing import Dict, List
import numpy as np


class WeightCalculator:
    """
    Calculates model weights based on certification status and benchmarking results.
    
    As per the research paper:
    W(f_i) = (B(f_i) * C(f_i)) / sum(B(f_j) * C(f_j))
    
    where:
    - B(f_i) is the benchmark score
    - C(f_i) is the certification status (1 if certified, 0 otherwise)
    """
    
    def __init__(
        self,
        require_certification: bool = True,
        min_benchmark_score: float = 0.1,
        normalization_method: str = "softmax"
    ):
        self.require_certification = require_certification
        self.min_benchmark_score = min_benchmark_score
        self.normalization_method = normalization_method
    
    def calculate_weights(
        self,
        model_ids: List[str],
        benchmark_scores: Dict[str, float],
        certification_status: Dict[str, bool]
    ) -> Dict[str, float]:
        """
        Calculate normalized weights for models.
        
        Args:
            model_ids: List of model identifiers
            benchmark_scores: Dictionary mapping model_id to benchmark score (0-1)
            certification_status: Dictionary mapping model_id to certification status
            
        Returns:
            Dictionary mapping model_id to weight (sum to 1.0)
        """
        weights = {}
        
        # Calculate raw weights
        for model_id in model_ids:
            # Get benchmark score
            B_i = benchmark_scores.get(model_id, 0.0)
            
            # Get certification status (1 or 0)
            C_i = 1.0 if certification_status.get(model_id, False) else 0.0
            
            # Filter by certification requirement
            if self.require_certification and C_i == 0:
                weights[model_id] = 0.0
                continue
            
            # Filter by minimum benchmark score
            if B_i < self.min_benchmark_score:
                weights[model_id] = 0.0
                continue
            
            # Calculate weight
            weights[model_id] = B_i * C_i
        
        # Normalize weights
        return self._normalize_weights(weights)
    
    def _normalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Normalize weights to sum to 1.0"""
        total = sum(weights.values())
        
        if total == 0:
            # All models excluded - return uniform weights
            n = len(weights)
            return {k: 1.0 / n for k in weights.keys()}
        
        if self.normalization_method == "softmax":
            # Apply softmax for smoother distribution
            values = np.array(list(weights.values()))
            exp_values = np.exp(values - np.max(values))  # Numerical stability
            softmax_values = exp_values / exp_values.sum()
            
            return {
                model_id: float(softmax_values[i])
                for i, model_id in enumerate(weights.keys())
            }
        else:
            # Simple normalization
            return {
                model_id: weight / total
                for model_id, weight in weights.items()
            }
    
    def apply_byzantine_penalty(
        self,
        weights: Dict[str, float],
        deviation_scores: Dict[str, float],
        lambda_param: float = 1.0,
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        Apply Byzantine fault tolerance penalty.
        
        Models with high deviation from consensus are penalized:
        W(f_i) <- W(f_i) * exp(-lambda * D(f_i))
        
        Args:
            weights: Current model weights
            deviation_scores: Deviation score for each model (0-1)
            lambda_param: Penalty scaling factor
            threshold: Deviation threshold for applying penalty
            
        Returns:
            Updated weights
        """
        penalized_weights = {}
        
        for model_id, weight in weights.items():
            deviation = deviation_scores.get(model_id, 0.0)
            
            if deviation > threshold:
                # Apply exponential penalty
                penalty = np.exp(-lambda_param * deviation)
                penalized_weights[model_id] = weight * penalty
            else:
                penalized_weights[model_id] = weight
        
        # Re-normalize
        return self._normalize_weights(penalized_weights)

