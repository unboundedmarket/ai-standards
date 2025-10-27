"""
Benchmark result data structures
"""
from typing import Dict, Any, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field


class BenchmarkResult(BaseModel):
    """
    Stores results from benchmarking evaluation.
    Designed to be stored on-chain along with model cards.
    """
    
    model_id: str = Field(..., description="ID of evaluated model")
    model_name: str = Field(..., description="Name of evaluated model")
    
    task_name: str = Field(..., description="Name of benchmark task")
    task_type: str = Field(..., description="Type of task (classification, generation, etc.)")
    
    # Performance metrics
    metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="Performance metrics (accuracy, F1, BLEU, etc.)"
    )
    
    # Benchmark score (normalized 0-1)
    benchmark_score: float = Field(
        ...,
        description="Overall benchmark score (0-1 range)"
    )
    
    # Execution details
    num_samples: int = Field(..., description="Number of test samples")
    execution_time: float = Field(..., description="Total execution time in seconds")
    
    # Metadata
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When benchmark was run"
    )
    
    dataset_name: Optional[str] = Field(None, description="Dataset used for benchmarking")
    dataset_version: Optional[str] = Field(None, description="Dataset version")
    
    # Additional details
    details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional benchmark details"
    )
    
    # Blockchain tracking
    tx_id: Optional[str] = Field(
        None,
        description="Transaction ID if stored on-chain"
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return self.model_dump(mode='json')
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BenchmarkResult':
        """Create from dictionary"""
        return cls(**data)
    
    def get_weighted_score(self, weights: Optional[Dict[str, float]] = None) -> float:
        """
        Calculate weighted score across multiple metrics.
        
        Args:
            weights: Dictionary of metric weights (defaults to equal weighting)
            
        Returns:
            Weighted score
        """
        if not self.metrics:
            return self.benchmark_score
        
        if weights is None:
            # Equal weighting
            return sum(self.metrics.values()) / len(self.metrics)
        
        # Weighted average
        total_weight = sum(weights.get(k, 0) for k in self.metrics.keys())
        if total_weight == 0:
            return self.benchmark_score
        
        weighted_sum = sum(
            self.metrics[k] * weights.get(k, 0) 
            for k in self.metrics.keys()
        )
        return weighted_sum / total_weight


class AggregatedBenchmarkResults(BaseModel):
    """
    Aggregates multiple benchmark results for a single model.
    """
    
    model_id: str
    model_name: str
    
    results: List[BenchmarkResult] = Field(default_factory=list)
    
    overall_score: Optional[float] = None
    
    def add_result(self, result: BenchmarkResult):
        """Add a benchmark result"""
        self.results.append(result)
        self._recalculate_overall_score()
    
    def _recalculate_overall_score(self):
        """Recalculate overall score from all results"""
        if not self.results:
            self.overall_score = 0.0
        else:
            self.overall_score = sum(
                r.benchmark_score for r in self.results
            ) / len(self.results)
    
    def get_results_by_task(self, task_name: str) -> List[BenchmarkResult]:
        """Get all results for a specific task"""
        return [r for r in self.results if r.task_name == task_name]
    
    def get_latest_result(self, task_name: Optional[str] = None) -> Optional[BenchmarkResult]:
        """Get most recent result, optionally filtered by task"""
        filtered = self.results
        if task_name:
            filtered = self.get_results_by_task(task_name)
        
        if not filtered:
            return None
        
        return max(filtered, key=lambda r: r.timestamp)

