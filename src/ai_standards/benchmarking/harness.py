"""
Benchmark harness for evaluating AI models
"""
from typing import Dict, Any, List, Optional
import time
from pathlib import Path
import json

from ai_standards.benchmarking.result import BenchmarkResult, AggregatedBenchmarkResults
from ai_standards.benchmarking.tasks import BenchmarkTask, TaskRegistry, get_task_registry
from ai_standards.utils.blockchain import BlockchainConnector


class BenchmarkHarness:
    """
    Harness for running benchmark evaluations on AI models.
    
    Responsibilities:
    - Run models on benchmark tasks
    - Calculate performance metrics
    - Store results on-chain
    - Track performance over time
    """
    
    def __init__(
        self,
        task_registry: Optional[TaskRegistry] = None,
        blockchain_connector: Optional[BlockchainConnector] = None,
        results_path: Optional[Path] = None
    ):
        self.task_registry = task_registry or get_task_registry()
        self.blockchain = blockchain_connector or BlockchainConnector(use_mock=True)
        self.results_path = results_path or Path("./benchmark_results")
        self.results_path.mkdir(parents=True, exist_ok=True)
    
    def run_benchmark(
        self,
        model: Any,
        model_id: str,
        model_name: str,
        task_name: str,
        test_data: Optional[List[Dict[str, Any]]] = None,
        store_on_chain: bool = True
    ) -> BenchmarkResult:
        """
        Run a benchmark evaluation on a model.
        
        Args:
            model: Model to evaluate
            model_id: Unique model identifier
            model_name: Human-readable model name
            task_name: Name of benchmark task
            test_data: Optional test data (if None, loads from task)
            store_on_chain: Whether to store results on blockchain
            
        Returns:
            BenchmarkResult
        """
        # Get task
        task = self.task_registry.get(task_name)
        if task is None:
            raise ValueError(f"Task '{task_name}' not found in registry")
        
        # Load test data if not provided
        if test_data is None:
            test_data = task.get_test_data()
        
        # Run evaluation
        start_time = time.time()
        metrics = task.evaluate(model, test_data)
        execution_time = time.time() - start_time
        
        # Calculate overall benchmark score (normalized)
        benchmark_score = self._calculate_benchmark_score(metrics)
        
        # Create result
        result = BenchmarkResult(
            model_id=model_id,
            model_name=model_name,
            task_name=task_name,
            task_type=task.task_type,
            metrics=metrics,
            benchmark_score=benchmark_score,
            num_samples=len(test_data),
            execution_time=execution_time,
            details={
                "task_description": task.description,
            }
        )
        
        # Store on blockchain if requested
        if store_on_chain:
            tx_id = self.blockchain.store_benchmark(result.to_dict())
            result.tx_id = tx_id
        
        # Save locally
        self._save_result(result)
        
        return result
    
    def run_multiple_benchmarks(
        self,
        model: Any,
        model_id: str,
        model_name: str,
        task_names: List[str],
        store_on_chain: bool = True
    ) -> AggregatedBenchmarkResults:
        """
        Run multiple benchmark tasks on a model.
        
        Args:
            model: Model to evaluate
            model_id: Unique model identifier
            model_name: Human-readable model name
            task_names: List of task names to run
            store_on_chain: Whether to store results on blockchain
            
        Returns:
            AggregatedBenchmarkResults
        """
        aggregated = AggregatedBenchmarkResults(
            model_id=model_id,
            model_name=model_name
        )
        
        for task_name in task_names:
            try:
                result = self.run_benchmark(
                    model=model,
                    model_id=model_id,
                    model_name=model_name,
                    task_name=task_name,
                    store_on_chain=store_on_chain
                )
                aggregated.add_result(result)
            except Exception as e:
                print(f"Error running task {task_name}: {e}")
        
        return aggregated
    
    def _calculate_benchmark_score(self, metrics: Dict[str, float]) -> float:
        """
        Calculate overall benchmark score from metrics.
        Normalized to 0-1 range.
        """
        if not metrics:
            return 0.0
        
        # Simple average of all metrics (assuming they're already 0-1)
        # In real implementation, this would be more sophisticated
        valid_metrics = [v for v in metrics.values() if 0 <= v <= 1]
        
        if not valid_metrics:
            # Try to normalize if metrics are not in 0-1 range
            return min(1.0, sum(metrics.values()) / len(metrics) / 100)
        
        return sum(valid_metrics) / len(valid_metrics)
    
    def _save_result(self, result: BenchmarkResult):
        """Save benchmark result to local storage"""
        # Create directory for model if it doesn't exist
        model_dir = self.results_path / result.model_id
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save result
        filename = f"{result.task_name}_{result.timestamp.isoformat()}.json"
        file_path = model_dir / filename
        
        with open(file_path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
    
    def load_results(self, model_id: str) -> AggregatedBenchmarkResults:
        """Load all benchmark results for a model"""
        model_dir = self.results_path / model_id
        
        if not model_dir.exists():
            return AggregatedBenchmarkResults(
                model_id=model_id,
                model_name="Unknown"
            )
        
        results = []
        for file_path in model_dir.glob("*.json"):
            with open(file_path, 'r') as f:
                data = json.load(f)
                results.append(BenchmarkResult.from_dict(data))
        
        aggregated = AggregatedBenchmarkResults(
            model_id=model_id,
            model_name=results[0].model_name if results else "Unknown",
            results=results
        )
        
        # Calculate overall score
        aggregated._recalculate_overall_score()
        
        return aggregated
    
    def get_leaderboard(
        self,
        task_name: Optional[str] = None,
        metric: str = "benchmark_score"
    ) -> List[Dict[str, Any]]:
        """
        Get leaderboard of models by performance.
        
        Args:
            task_name: Optional task name to filter by
            metric: Metric to rank by (default: benchmark_score)
            
        Returns:
            List of model rankings
        """
        all_results = []
        
        for model_dir in self.results_path.iterdir():
            if not model_dir.is_dir():
                continue
            
            aggregated = self.load_results(model_dir.name)
            
            for result in aggregated.results:
                if task_name and result.task_name != task_name:
                    continue
                
                all_results.append({
                    "model_id": result.model_id,
                    "model_name": result.model_name,
                    "task_name": result.task_name,
                    "score": result.benchmark_score if metric == "benchmark_score" else result.metrics.get(metric, 0),
                    "timestamp": result.timestamp,
                })
        
        # Sort by score descending
        all_results.sort(key=lambda x: x["score"], reverse=True)
        
        return all_results

