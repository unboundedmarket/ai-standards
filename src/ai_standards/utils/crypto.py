"""
Cryptographic utilities for model cards and blockchain integration
"""
import hashlib
import json
from typing import Dict, Any
from datetime import datetime


def hash_model_card(model_card: Dict[str, Any]) -> str:
    """
    Generate cryptographic hash of a model card.
    
    Args:
        model_card: Dictionary containing model card data
        
    Returns:
        Hexadecimal hash string
    """
    # Serialize to canonical JSON format
    canonical_json = json.dumps(model_card, sort_keys=True, separators=(',', ':'))
    
    # Compute SHA-256 hash
    hash_object = hashlib.sha256(canonical_json.encode('utf-8'))
    return hash_object.hexdigest()


def generate_model_id(model_name: str, version: str = "1.0.0") -> str:
    """
    Generate unique model identifier.
    
    Args:
        model_name: Name of the model
        version: Model version
        
    Returns:
        Unique model identifier
    """
    timestamp = datetime.utcnow().isoformat()
    content = f"{model_name}:{version}:{timestamp}"
    hash_object = hashlib.sha256(content.encode('utf-8'))
    return hash_object.hexdigest()[:16]


def verify_hash(data: Dict[str, Any], expected_hash: str) -> bool:
    """
    Verify that data matches expected hash.
    
    Args:
        data: Data to verify
        expected_hash: Expected hash value
        
    Returns:
        True if hash matches, False otherwise
    """
    computed_hash = hash_model_card(data)
    return computed_hash == expected_hash

