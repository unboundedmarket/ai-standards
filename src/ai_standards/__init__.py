"""
AI-Blockchain Integration Standards - Reference Implementation

This package provides reference implementations for the four pillars of
AI-Blockchain integration:

1. API Framework - Standardized interfaces for AI model access
2. Certification - NFT-based model cards and validation
3. Benchmarking - Transparent evaluation and performance tracking
4. Consensus - Multi-model aggregation and decision-making
"""

__version__ = "0.1.0"
__author__ = "Cardano AI Standards Team"

from ai_standards.api import APIFramework
from ai_standards.certification import ModelCard, CertificationManager
from ai_standards.benchmarking import BenchmarkHarness, BenchmarkResult
from ai_standards.consensus import ConsensusEngine

__all__ = [
    "APIFramework",
    "ModelCard",
    "CertificationManager",
    "BenchmarkHarness",
    "BenchmarkResult",
    "ConsensusEngine",
]

