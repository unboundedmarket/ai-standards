"""
Example 4: Consensus Mechanism

This example demonstrates how to:
1. Set up multiple models
2. Calculate model weights
3. Reach consensus on outputs
4. Apply Byzantine fault tolerance
"""

from ai_standards.consensus import ConsensusEngine, WeightCalculator
from ai_standards.certification.model_card import OutputModality


# Mock models for demonstration
class MockClassifier:
    """Mock classification model"""
    
    def __init__(self, name, bias=0.0):
        self.name = name
        self.bias = bias
    
    def predict(self, input_data):
        """Mock prediction"""
        # Simulate different models with slight biases
        import random
        random.seed(hash(self.name + str(input_data)))
        
        labels = ['positive', 'negative', 'neutral']
        weights = [0.4 + self.bias, 0.3, 0.3 - self.bias]
        return random.choices(labels, weights=weights, k=1)[0]


def main():
    print("=" * 60)
    print("Example 4: Consensus Mechanism")
    print("=" * 60)
    
    # Step 1: Set up multiple models
    print("\n1. Setting up models...")
    models = {
        "model_a": MockClassifier("model_a", bias=0.1),
        "model_b": MockClassifier("model_b", bias=0.0),
        "model_c": MockClassifier("model_c", bias=-0.1),
        "model_d": MockClassifier("model_d", bias=0.05),
    }
    print(f"✓ Created {len(models)} models")
    
    # Step 2: Define model metadata (certification & benchmarking)
    print("\n2. Setting up model metadata...")
    model_metadata = {
        "model_a": {
            "certified": True,
            "benchmark_score": 0.85,
        },
        "model_b": {
            "certified": True,
            "benchmark_score": 0.90,  # Best performing
        },
        "model_c": {
            "certified": True,
            "benchmark_score": 0.75,
        },
        "model_d": {
            "certified": False,  # Not certified
            "benchmark_score": 0.88,
        },
    }
    
    for model_id, metadata in model_metadata.items():
        cert_status = "✓" if metadata["certified"] else "✗"
        print(f"  {cert_status} {model_id}: score={metadata['benchmark_score']:.2f}")
    
    # Step 3: Calculate model weights
    print("\n3. Calculating model weights...")
    weight_calculator = WeightCalculator(require_certification=True)
    
    weights = weight_calculator.calculate_weights(
        model_ids=list(models.keys()),
        benchmark_scores={k: v["benchmark_score"] for k, v in model_metadata.items()},
        certification_status={k: v["certified"] for k, v in model_metadata.items()}
    )
    
    print("  Model weights:")
    for model_id, weight in weights.items():
        print(f"    {model_id}: {weight:.4f}")
    
    # Step 4: Run models and collect outputs
    print("\n4. Running inference on all models...")
    input_data = "This product is amazing!"
    
    model_outputs = {}
    for model_id, model in models.items():
        output = model.predict(input_data)
        model_outputs[model_id] = output
        print(f"  {model_id}: {output}")
    
    # Step 5: Reach consensus
    print("\n5. Reaching consensus...")
    consensus_engine = ConsensusEngine(weight_calculator=weight_calculator)
    
    consensus_result = consensus_engine.reach_consensus(
        model_outputs=model_outputs,
        model_metadata=model_metadata,
        output_modality=OutputModality.DISCRETE
    )
    
    print(f"✓ Consensus reached!")
    print(f"  Consensus output: {consensus_result['consensus_output']}")
    print(f"  Models participated: {consensus_result['eligible_models']}/{consensus_result['num_models']}")
    print(f"  Output modality: {consensus_result['output_modality']}")
    
    # Step 6: Calculate deviation scores (Byzantine fault tolerance)
    print("\n6. Calculating deviation scores...")
    deviation_scores = consensus_engine.calculate_deviation_scores(
        model_outputs=model_outputs,
        consensus_output=consensus_result['consensus_output'],
        output_modality=OutputModality.DISCRETE
    )
    
    print("  Deviation from consensus:")
    for model_id, deviation in deviation_scores.items():
        status = "✓" if deviation == 0.0 else "✗"
        print(f"    {status} {model_id}: {deviation:.2f}")
    
    # Step 7: Apply Byzantine penalty
    print("\n7. Applying Byzantine fault tolerance...")
    updated_weights = weight_calculator.apply_byzantine_penalty(
        weights=weights,
        deviation_scores=deviation_scores,
        lambda_param=2.0,
        threshold=0.5
    )
    
    print("  Updated weights (after penalty):")
    for model_id in weights.keys():
        old_weight = weights[model_id]
        new_weight = updated_weights[model_id]
        change = "↓" if new_weight < old_weight else "="
        print(f"    {model_id}: {old_weight:.4f} → {new_weight:.4f} {change}")
    
    # Step 8: Demonstrate text consensus
    print("\n8. Demonstrating text consensus...")
    
    text_outputs = {
        "model_a": "The product quality is excellent and exceeded expectations.",
        "model_b": "This product has outstanding quality and performance.",
        "model_c": "Great product with excellent build quality.",
    }
    
    from ai_standards.consensus import TextAggregator
    text_aggregator = TextAggregator()
    
    consensus_text = text_aggregator.aggregate(
        outputs=list(text_outputs.values()),
        weights={k: v for k, v in weights.items() if k in text_outputs},
        model_ids=list(text_outputs.keys())
    )
    
    print("  Individual outputs:")
    for model_id, output in text_outputs.items():
        weight = weights.get(model_id, 0.0)
        print(f"    {model_id} (w={weight:.2f}): {output}")
    
    print(f"\n  Consensus (reranked): {consensus_text}")
    
    print("\n" + "=" * 60)
    print("Consensus example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

