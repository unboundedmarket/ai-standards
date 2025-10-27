"""
Example 2: Model Benchmarking

This example demonstrates how to:
1. Register benchmark tasks
2. Run benchmarks on a model
3. Store results on-chain
4. View leaderboards
"""

from ai_standards.benchmarking import BenchmarkHarness
from ai_standards.benchmarking.tasks import register_default_tasks
import shutil


# Mock model for demonstration
class MockSentimentModel:
    """Simple mock model for demonstration"""
    
    def predict(self, input_data):
        """Mock prediction"""
        # In real implementation, this would call actual model
        text = input_data.get('text', '')
        
        # Simple keyword-based mock
        if any(word in text.lower() for word in ['good', 'great', 'excellent', 'love']):
            return 'positive'
        elif any(word in text.lower() for word in ['bad', 'terrible', 'hate', 'awful']):
            return 'negative'
        else:
            return 'neutral'


def main():
    print("=" * 60)
    print("Example 2: Model Benchmarking")
    print("=" * 60)
    
    # Clean up any previous test data
    print("\nCleaning up previous test data...")
    harness = BenchmarkHarness()
    if harness.results_path.exists():
        shutil.rmtree(harness.results_path)
    harness.results_path.mkdir(parents=True, exist_ok=True)
    print("✓ Test environment prepared")
    
    # Step 1: Register default benchmark tasks
    print("\n1. Registering benchmark tasks...")
    register_default_tasks()
    tasks = harness.task_registry.list_tasks()
    print(f"✓ Registered {len(tasks)} benchmark tasks:")
    for task_name in tasks:
        task = harness.task_registry.get(task_name)
        print(f"  - {task_name} ({task.task_type})")
    
    # Step 2: Create a mock model
    print("\n2. Creating mock model...")
    model = MockSentimentModel()
    model_id = "sentiment_model_v1"
    model_name = "Mock Sentiment Analyzer"
    print(f"✓ Model created: {model_name}")
    
    # Step 3: Prepare test data
    print("\n3. Preparing test data...")
    test_data = [
        {'input': {'text': 'This is great!'}, 'label': 'positive'},
        {'input': {'text': 'This is terrible.'}, 'label': 'negative'},
        {'input': {'text': 'It is okay.'}, 'label': 'neutral'},
        {'input': {'text': 'I love this!'}, 'label': 'positive'},
        {'input': {'text': 'I hate this.'}, 'label': 'negative'},
    ]
    print(f"✓ Prepared {len(test_data)} test samples for evaluation")
    
    # Step 4: Run benchmark (using mock evaluation for demo)
    print("\n4. Running benchmark evaluation...")
    print("  Note: Using mock evaluation for demonstration")
    
    # Create mock benchmark result
    from ai_standards.benchmarking import BenchmarkResult
    
    result = BenchmarkResult(
        model_id=model_id,
        model_name=model_name,
        task_name="sentiment_analysis",
        task_type="classification",
        metrics={
            'accuracy': 0.85,
            'f1': 0.83,
            'precision': 0.86,
            'recall': 0.82,
        },
        benchmark_score=0.84,
        num_samples=len(test_data),
        execution_time=0.123
    )
    
    print("✓ Benchmark completed")
    print(f"  Task: {result.task_name}")
    print(f"  Overall Score: {result.benchmark_score:.4f}")
    print(f"  Metrics:")
    for metric, value in result.metrics.items():
        print(f"    - {metric}: {value:.4f}")
    
    # Step 5: Store result (would store on-chain in production)
    print("\n5. Storing results...")
    print("  Note: In production, this stores on blockchain")
    harness._save_result(result)
    print("✓ Results stored")
    
    # Step 6: Retrieve results
    print("\n6. Retrieving benchmark results...")
    aggregated = harness.load_results(model_id)
    print(f"✓ Retrieved results for {aggregated.model_name}")
    print(f"  Overall Score: {aggregated.overall_score:.4f}")
    print(f"  Number of evaluations stored: {len(aggregated.results)}")
    print(f"  (Each evaluation used {result.num_samples} test samples)")
    
    # Step 7: View leaderboard
    print("\n7. Viewing leaderboard...")
    leaderboard = harness.get_leaderboard(task_name="sentiment_analysis")
    if leaderboard:
        print(f"  Top models for sentiment_analysis:")
        print(f"  ({len(leaderboard)} model evaluation{'s' if len(leaderboard) != 1 else ''} on record)")
        for i, entry in enumerate(leaderboard[:5], 1):
            print(f"  {i}. {entry['model_name']}: {entry['score']:.4f}")
    else:
        print("  No entries yet")
    
    print("\n" + "=" * 60)
    print("Benchmarking example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

