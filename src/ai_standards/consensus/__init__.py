"""
Consensus module for AI model output aggregation
"""

from ai_standards.consensus.engine import ConsensusEngine
from ai_standards.consensus.aggregators import (
    DiscreteAggregator,
    TextAggregator,
    VisionAggregator,
    AudioAggregator,
)
from ai_standards.consensus.weighting import WeightCalculator

__all__ = [
    "ConsensusEngine",
    "DiscreteAggregator",
    "TextAggregator",
    "VisionAggregator",
    "AudioAggregator",
    "WeightCalculator",
]

