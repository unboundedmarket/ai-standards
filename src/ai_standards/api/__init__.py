"""
API Framework module for AI model access
"""

from ai_standards.api.models import (
    InferenceServiceRequest,
    InferenceRequest,
    InferenceResponse,
)
from ai_standards.api.server import app, APIFramework

__all__ = [
    "app",
    "APIFramework",
    "InferenceServiceRequest",
    "InferenceRequest",
    "InferenceResponse",
]

