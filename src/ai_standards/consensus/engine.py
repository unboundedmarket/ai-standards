"""
Consensus Engine - Main orchestrator for AI consensus mechanism
"""
from typing import List, Dict, Any, Optional
from ai_standards.consensus.weighting import WeightCalculator
from ai_standards.consensus.aggregators import (
    DiscreteAggregator,
    TextAggregator,
    VisionAggregator,
    AudioAggregator,
)
from ai_standards.certification.model_card import OutputModality


class ConsensusEngine:
    """
    Main consensus engine that orchestrates the AI consensus mechanism.
    
    Implements the four-pillar approach:
    1. Retrieves model cards (certification)
    2. Retrieves benchmark results
    3. Calculates model weights
    4. Aggregates outputs based on modality
    """
    
    def __init__(
        self,
        weight_calculator: Optional[WeightCalculator] = None,
        require_certification: bool = True
    ):
        self.weight_calculator = weight_calculator or WeightCalculator(
            require_certification=require_certification
        )
        
        # Initialize aggregators for different modalities
        self.aggregators = {
            OutputModality.DISCRETE: DiscreteAggregator(),
            OutputModality.TEXT: TextAggregator(),
            OutputModality.VISION: VisionAggregator(),
            OutputModality.AUDIO: AudioAggregator(),
        }
    
    def reach_consensus(
        self,
        model_outputs: Dict[str, Any],
        model_metadata: Dict[str, Dict[str, Any]],
        output_modality: OutputModality
    ) -> Dict[str, Any]:
        """
        Reach consensus among multiple models.
        
        Args:
            model_outputs: Dictionary mapping model_id to output
            model_metadata: Dictionary mapping model_id to metadata
                          (must include 'benchmark_score' and 'certified')
            output_modality: Type of output (discrete, text, vision, audio)
            
        Returns:
            Dictionary containing consensus output and metadata
        """
        # Extract model IDs
        model_ids = list(model_outputs.keys())
        
        # Extract benchmark scores and certification status
        benchmark_scores = {
            mid: metadata.get('benchmark_score', 0.0)
            for mid, metadata in model_metadata.items()
        }
        
        certification_status = {
            mid: metadata.get('certified', False)
            for mid, metadata in model_metadata.items()
        }
        
        # Calculate weights
        weights = self.weight_calculator.calculate_weights(
            model_ids=model_ids,
            benchmark_scores=benchmark_scores,
            certification_status=certification_status
        )
        
        # Get appropriate aggregator
        aggregator = self.aggregators.get(output_modality)
        if aggregator is None:
            raise ValueError(f"Unsupported output modality: {output_modality}")
        
        # Aggregate outputs
        outputs = [model_outputs[mid] for mid in model_ids]
        consensus_output = aggregator.aggregate(outputs, weights, model_ids)
        
        return {
            "consensus_output": consensus_output,
            "weights": weights,
            "num_models": len(model_ids),
            "eligible_models": sum(1 for w in weights.values() if w > 0),
            "output_modality": output_modality.value,
        }
    
    def reach_consensus_with_models(
        self,
        models: List[Any],
        model_ids: List[str],
        input_data: Any,
        model_metadata: Dict[str, Dict[str, Any]],
        output_modality: OutputModality
    ) -> Dict[str, Any]:
        """
        Execute models and reach consensus on outputs.
        
        Args:
            models: List of model objects
            model_ids: List of model identifiers
            input_data: Input to pass to all models
            model_metadata: Metadata for each model (benchmark scores, certification)
            output_modality: Type of output
            
        Returns:
            Consensus result
        """
        # Execute all models
        model_outputs = {}
        
        for model, model_id in zip(models, model_ids):
            try:
                if output_modality == OutputModality.DISCRETE:
                    output = model.predict(input_data)
                elif output_modality == OutputModality.TEXT:
                    output = model.generate(input_data)
                else:
                    output = model(input_data)
                
                model_outputs[model_id] = output
            except Exception as e:
                print(f"Error executing model {model_id}: {e}")
        
        # Reach consensus
        return self.reach_consensus(
            model_outputs=model_outputs,
            model_metadata=model_metadata,
            output_modality=output_modality
        )
    
    def set_aggregator(
        self,
        modality: OutputModality,
        aggregator: Any
    ):
        """Set custom aggregator for a modality"""
        self.aggregators[modality] = aggregator
    
    def calculate_deviation_scores(
        self,
        model_outputs: Dict[str, Any],
        consensus_output: Any,
        output_modality: OutputModality
    ) -> Dict[str, float]:
        """
        Calculate deviation scores for Byzantine fault tolerance.
        
        D(f_i) = |f_i(x) - F̄(x)|
        
        Args:
            model_outputs: Dictionary of model outputs
            consensus_output: The consensus output
            output_modality: Type of output
            
        Returns:
            Dictionary mapping model_id to deviation score (0-1)
        """
        deviations = {}
        
        for model_id, output in model_outputs.items():
            if output_modality == OutputModality.DISCRETE:
                # Binary deviation: match or not
                deviations[model_id] = 0.0 if output == consensus_output else 1.0
            
            elif output_modality == OutputModality.TEXT:
                # Text similarity (simplified)
                # In real implementation, use embedding similarity or edit distance
                if output == consensus_output:
                    deviations[model_id] = 0.0
                else:
                    # Simple character-level similarity
                    from difflib import SequenceMatcher
                    similarity = SequenceMatcher(None, output, consensus_output).ratio()
                    deviations[model_id] = 1.0 - similarity
            
            else:
                # For vision/audio, would use appropriate distance metrics
                # Placeholder: random deviation
                deviations[model_id] = 0.0
        
        return deviations

