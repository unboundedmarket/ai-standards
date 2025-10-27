"""
Modality-specific aggregators for consensus mechanism
"""
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
import numpy as np
from collections import Counter


class BaseAggregator(ABC):
    """Base class for output aggregators"""
    
    @abstractmethod
    def aggregate(
        self,
        outputs: List[Any],
        weights: Dict[str, float],
        model_ids: List[str]
    ) -> Any:
        """
        Aggregate outputs from multiple models.
        
        Args:
            outputs: List of model outputs
            weights: Dictionary of model weights
            model_ids: List of model IDs corresponding to outputs
            
        Returns:
            Aggregated output
        """
        pass


class DiscreteAggregator(BaseAggregator):
    """
    Aggregator for discrete outputs using weighted majority voting.
    
    As per research paper:
    ŷ = argmax_y sum(W(f_i) * δ(f_i(x), y))
    
    where δ is the Kronecker delta function.
    """
    
    def aggregate(
        self,
        outputs: List[Any],
        weights: Dict[str, float],
        model_ids: List[str]
    ) -> Any:
        """
        Weighted majority voting for discrete outputs.
        
        Args:
            outputs: List of discrete predictions (labels, classes, etc.)
            weights: Model weights
            model_ids: Model identifiers
            
        Returns:
            Consensus prediction
        """
        if not outputs:
            return None
        
        # Calculate weighted votes for each unique output
        vote_weights = {}
        
        for output, model_id in zip(outputs, model_ids):
            weight = weights.get(model_id, 0.0)
            
            if output not in vote_weights:
                vote_weights[output] = 0.0
            
            vote_weights[output] += weight
        
        # Return output with highest weighted vote
        return max(vote_weights.items(), key=lambda x: x[1])[0]
    
    def get_confidence(
        self,
        outputs: List[Any],
        weights: Dict[str, float],
        model_ids: List[str]
    ) -> float:
        """Get confidence score for consensus (normalized vote weight)"""
        if not outputs:
            return 0.0
        
        vote_weights = {}
        total_weight = 0.0
        
        for output, model_id in zip(outputs, model_ids):
            weight = weights.get(model_id, 0.0)
            total_weight += weight
            
            if output not in vote_weights:
                vote_weights[output] = 0.0
            vote_weights[output] += weight
        
        if total_weight == 0:
            return 0.0
        
        max_vote = max(vote_weights.values())
        return max_vote / total_weight


class TextAggregator(BaseAggregator):
    """
    Aggregator for text outputs using distillation-based reranking.
    
    As per research paper:
    ŷ = D({(y_i, W(f_i))})
    
    where D is a distillation model that synthesizes candidate responses.
    """
    
    def __init__(self, aggregator_model: Optional[Any] = None):
        """
        Args:
            aggregator_model: Optional distillation model for synthesis.
                            If None, uses simple reranking.
        """
        self.aggregator_model = aggregator_model
    
    def aggregate(
        self,
        outputs: List[str],
        weights: Dict[str, float],
        model_ids: List[str]
    ) -> str:
        """
        Aggregate text outputs via distillation.
        
        Args:
            outputs: List of text outputs from models
            weights: Model weights
            model_ids: Model identifiers
            
        Returns:
            Synthesized text output
        """
        if not outputs:
            return ""
        
        # Create weighted candidates
        candidates = [
            {"text": output, "weight": weights.get(model_id, 0.0)}
            for output, model_id in zip(outputs, model_ids)
        ]
        
        # Sort by weight
        candidates.sort(key=lambda x: x["weight"], reverse=True)
        
        if self.aggregator_model is not None:
            # Use distillation model to synthesize
            return self._distill_output(candidates)
        else:
            # Simple reranking: return highest weighted output
            return candidates[0]["text"]
    
    def _distill_output(self, candidates: List[Dict[str, Any]]) -> str:
        """
        Use distillation model to synthesize final output.
        
        In a full implementation, this would:
        1. Format candidates with weights
        2. Pass to LLM with instruction to synthesize
        3. Return refined output
        """
        # Placeholder: In real implementation, call aggregator_model
        # For now, implement simple weighted combination
        
        # Create prompt for distillation
        prompt = "Synthesize the following candidate responses into a single, coherent output:\n\n"
        
        for i, cand in enumerate(candidates[:5]):  # Top 5 candidates
            prompt += f"Candidate {i+1} (weight: {cand['weight']:.3f}): {cand['text']}\n\n"
        
        prompt += "Synthesized output:"
        
        # In real implementation: return self.aggregator_model.generate(prompt)
        # For now, return highest weighted
        return candidates[0]["text"]
    
    def rerank(
        self,
        outputs: List[str],
        weights: Dict[str, float],
        model_ids: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Return reranked list of candidates with weights.
        
        Useful for inspection and debugging.
        """
        candidates = [
            {
                "text": output,
                "model_id": model_id,
                "weight": weights.get(model_id, 0.0)
            }
            for output, model_id in zip(outputs, model_ids)
        ]
        
        candidates.sort(key=lambda x: x["weight"], reverse=True)
        return candidates


class VisionAggregator(BaseAggregator):
    """
    Aggregator for vision outputs using latent space fusion.
    
    As per research paper:
    V̂ = Ψ(D_V({(Φ(V_i), W(f_i))}))
    
    where Φ is encoder, D_V is fusion function, Ψ is decoder.
    """
    
    def __init__(
        self,
        encoder: Optional[Any] = None,
        decoder: Optional[Any] = None
    ):
        """
        Args:
            encoder: Vision encoder (e.g., CLIP, VAE encoder)
            decoder: Vision decoder (e.g., VAE decoder, diffusion model)
        """
        self.encoder = encoder
        self.decoder = decoder
    
    def aggregate(
        self,
        outputs: List[Any],  # Images or latent representations
        weights: Dict[str, float],
        model_ids: List[str]
    ) -> Any:
        """
        Aggregate vision outputs via latent space fusion.
        
        Args:
            outputs: List of image outputs (or latent vectors)
            weights: Model weights
            model_ids: Model identifiers
            
        Returns:
            Synthesized image output
        """
        if not outputs:
            return None
        
        if self.encoder is not None and self.decoder is not None:
            # Full latent space aggregation
            return self._latent_fusion(outputs, weights, model_ids)
        else:
            # Simple weighted averaging in pixel space
            return self._pixel_averaging(outputs, weights, model_ids)
    
    def _latent_fusion(
        self,
        outputs: List[Any],
        weights: Dict[str, float],
        model_ids: List[str]
    ) -> Any:
        """Aggregate in latent space"""
        # Encode all outputs to latent space
        latents = [self.encoder(output) for output in outputs]
        
        # Weighted average in latent space
        weighted_latent = np.zeros_like(latents[0])
        
        for latent, model_id in zip(latents, model_ids):
            weight = weights.get(model_id, 0.0)
            weighted_latent += weight * latent
        
        # Decode back to image space
        return self.decoder(weighted_latent)
    
    def _pixel_averaging(
        self,
        outputs: List[Any],
        weights: Dict[str, float],
        model_ids: List[str]
    ) -> Any:
        """Simple weighted averaging in pixel space"""
        # Assume outputs are numpy arrays
        weighted_output = np.zeros_like(outputs[0])
        
        for output, model_id in zip(outputs, model_ids):
            weight = weights.get(model_id, 0.0)
            weighted_output += weight * np.array(output)
        
        return weighted_output


class AudioAggregator(BaseAggregator):
    """
    Aggregator for audio outputs using spectro-temporal fusion.
    
    As per research paper:
    â = ISTFT(D_A({(A_i, W(f_i)) | I_i}))
    
    where A_i are spectrograms, I_i are quality indicators, ISTFT is inverse STFT.
    """
    
    def __init__(self, quality_metrics: Optional[List[str]] = None):
        """
        Args:
            quality_metrics: List of quality metric names to use for ranking
        """
        self.quality_metrics = quality_metrics or ["clarity", "fidelity"]
    
    def aggregate(
        self,
        outputs: List[Any],  # Audio waveforms or spectrograms
        weights: Dict[str, float],
        model_ids: List[str],
        quality_scores: Optional[Dict[str, Dict[str, float]]] = None
    ) -> Any:
        """
        Aggregate audio outputs via spectro-temporal fusion.
        
        Args:
            outputs: List of audio outputs (waveforms or spectrograms)
            weights: Model weights
            model_ids: Model identifiers
            quality_scores: Optional quality indicators for each model
            
        Returns:
            Synthesized audio output
        """
        if not outputs:
            return None
        
        if quality_scores:
            # Use quality indicators to select best output
            return self._quality_based_selection(
                outputs, weights, model_ids, quality_scores
            )
        else:
            # Simple weighted selection
            return self._weighted_selection(outputs, weights, model_ids)
    
    def _quality_based_selection(
        self,
        outputs: List[Any],
        weights: Dict[str, float],
        model_ids: List[str],
        quality_scores: Dict[str, Dict[str, float]]
    ) -> Any:
        """Select output based on quality indicators and weights"""
        scores = []
        
        for output, model_id in zip(outputs, model_ids):
            # Combine weight with quality scores
            weight = weights.get(model_id, 0.0)
            quality = quality_scores.get(model_id, {})
            
            # Average quality metrics
            avg_quality = np.mean([
                quality.get(metric, 0.0) 
                for metric in self.quality_metrics
            ])
            
            # Combined score
            combined_score = weight * avg_quality
            scores.append(combined_score)
        
        # Return output with highest score
        best_idx = np.argmax(scores)
        return outputs[best_idx]
    
    def _weighted_selection(
        self,
        outputs: List[Any],
        weights: Dict[str, float],
        model_ids: List[str]
    ) -> Any:
        """Select output with highest weight"""
        weight_list = [weights.get(model_id, 0.0) for model_id in model_ids]
        best_idx = np.argmax(weight_list)
        return outputs[best_idx]

