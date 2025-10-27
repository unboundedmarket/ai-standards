"""
Benchmark task definitions and registry
"""
from typing import Dict, Any, Callable, List, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class BenchmarkTask(ABC):
    """
    Base class for benchmark tasks.
    """
    
    name: str
    task_type: str  # classification, generation, qa, etc.
    description: str
    
    @abstractmethod
    def evaluate(self, model: Any, test_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Evaluate a model on this task.
        
        Args:
            model: Model to evaluate
            test_data: Test dataset
            
        Returns:
            Dictionary of metrics
        """
        pass
    
    @abstractmethod
    def get_test_data(self) -> List[Dict[str, Any]]:
        """Load test data for this task"""
        pass


class ClassificationTask(BenchmarkTask):
    """Classification benchmark task"""
    
    def __init__(self, name: str, num_classes: int, dataset_name: str):
        super().__init__(
            name=name,
            task_type="classification",
            description=f"Classification task with {num_classes} classes"
        )
        self.num_classes = num_classes
        self.dataset_name = dataset_name
    
    def evaluate(self, model: Any, test_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate classification model"""
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        
        predictions = []
        labels = []
        
        for sample in test_data:
            pred = model.predict(sample['input'])
            predictions.append(pred)
            labels.append(sample['label'])
        
        metrics = {
            'accuracy': accuracy_score(labels, predictions),
            'f1': f1_score(labels, predictions, average='weighted', zero_division=0),
            'precision': precision_score(labels, predictions, average='weighted', zero_division=0),
            'recall': recall_score(labels, predictions, average='weighted', zero_division=0),
        }
        
        return metrics
    
    def get_test_data(self) -> List[Dict[str, Any]]:
        """Load test data - placeholder for actual implementation"""
        # In real implementation, load from datasets library
        return []


class TextGenerationTask(BenchmarkTask):
    """Text generation benchmark task"""
    
    def __init__(self, name: str, dataset_name: str):
        super().__init__(
            name=name,
            task_type="generation",
            description="Text generation task"
        )
        self.dataset_name = dataset_name
    
    def evaluate(self, model: Any, test_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate text generation model"""
        # In real implementation, use BLEU, ROUGE, etc.
        # For now, return placeholder metrics
        
        total_samples = len(test_data)
        outputs = []
        
        for sample in test_data:
            output = model.generate(sample['input'])
            outputs.append(output)
        
        # Placeholder metrics
        metrics = {
            'samples_evaluated': float(total_samples),
            'avg_length': float(sum(len(o.split()) for o in outputs) / len(outputs)),
        }
        
        return metrics
    
    def get_test_data(self) -> List[Dict[str, Any]]:
        """Load test data"""
        return []


class TaskRegistry:
    """
    Registry for benchmark tasks.
    Allows registration and retrieval of benchmark tasks.
    """
    
    def __init__(self):
        self.tasks: Dict[str, BenchmarkTask] = {}
    
    def register(self, task: BenchmarkTask):
        """Register a new benchmark task"""
        self.tasks[task.name] = task
    
    def get(self, name: str) -> Optional[BenchmarkTask]:
        """Get a task by name"""
        return self.tasks.get(name)
    
    def list_tasks(self) -> List[str]:
        """List all registered task names"""
        return list(self.tasks.keys())
    
    def list_by_type(self, task_type: str) -> List[BenchmarkTask]:
        """List all tasks of a specific type"""
        return [t for t in self.tasks.values() if t.task_type == task_type]


# Global task registry
_global_registry = TaskRegistry()


def get_task_registry() -> TaskRegistry:
    """Get the global task registry"""
    return _global_registry


def register_default_tasks():
    """Register some default benchmark tasks"""
    registry = get_task_registry()
    
    # Register common tasks
    registry.register(ClassificationTask(
        name="sentiment_analysis",
        num_classes=3,
        dataset_name="imdb"
    ))
    
    registry.register(ClassificationTask(
        name="toxicity_detection",
        num_classes=2,
        dataset_name="toxic_comments"
    ))
    
    registry.register(TextGenerationTask(
        name="text_completion",
        dataset_name="wikitext"
    ))
    
    registry.register(TextGenerationTask(
        name="question_answering",
        dataset_name="squad"
    ))

