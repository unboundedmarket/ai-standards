"""
Model Card implementation following the research specification
"""
from typing import Dict, Any, Optional, List
from datetime import datetime
from pydantic import BaseModel, Field, field_validator
from enum import Enum


class ModelArchitecture(str, Enum):
    """Supported model architectures"""
    TRANSFORMER = "transformer"
    CNN = "cnn"
    RNN = "rnn"
    LSTM = "lstm"
    GAN = "gan"
    DIFFUSION = "diffusion"
    ENSEMBLE = "ensemble"
    OTHER = "other"


class OutputModality(str, Enum):
    """Output modalities for AI models"""
    DISCRETE = "discrete"
    TEXT = "text"
    VISION = "vision"
    AUDIO = "audio"
    MULTIMODAL = "multimodal"


class ModelCard(BaseModel):
    """
    NFT-based Model Card following the certification specification.
    
    Includes both fixed (mandatory) and optional parameters as defined
    in the research paper (Section 4: Certification).
    """
    
    # === Fixed Parameters (Mandatory) ===
    model_id: str = Field(..., description="Unique model identifier")
    model_name: str = Field(..., description="Human-readable model name")
    version: str = Field(default="1.0.0", description="Model version")
    
    usage_instructions: str = Field(
        ..., 
        description="Clear description of how to implement and interact with the model"
    )
    
    associated_costs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Financial or computational costs (e.g., per-query cost, hardware requirements)"
    )
    
    model_size: int = Field(
        ..., 
        description="Number of parameters indicating model scale and complexity"
    )
    
    token_limits: Optional[Dict[str, int]] = Field(
        None,
        description="Token or usage limits (e.g., max_input_tokens, max_output_tokens)"
    )
    
    instruction_tuning: Optional[str] = Field(
        None,
        description="Details on fine-tuning with instruction-following data"
    )
    
    architecture: ModelArchitecture = Field(
        ...,
        description="Underlying model architecture"
    )
    
    output_modality: OutputModality = Field(
        ...,
        description="Primary output modality of the model"
    )
    
    licensing_terms: str = Field(
        ...,
        description="Explicit licensing information (usage, distribution, modification)"
    )
    
    # === Optional Parameters ===
    training_data_sources: Optional[List[str]] = Field(
        None,
        description="Datasets used for model training (provenance and potential biases)"
    )
    
    ethical_considerations: Optional[str] = Field(
        None,
        description="Fairness, bias mitigation, and ethical concerns"
    )
    
    intended_use_cases: Optional[List[str]] = Field(
        None,
        description="Scenarios where the model is expected to perform effectively"
    )
    
    limitations: Optional[List[str]] = Field(
        None,
        description="Known limitations or risks in model application"
    )
    
    # === Metadata ===
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp of model card creation"
    )
    
    updated_at: Optional[datetime] = Field(
        None,
        description="Timestamp of last update"
    )
    
    owner_address: Optional[str] = Field(
        None,
        description="Blockchain wallet address of model owner"
    )
    
    # === Certification Status ===
    certified: bool = Field(
        default=False,
        description="Whether the model has been certified"
    )
    
    certification_metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional certification details"
    )
    
    nft_id: Optional[str] = Field(
        None,
        description="NFT token ID on blockchain (set after minting)"
    )
    
    card_hash: Optional[str] = Field(
        None,
        description="Cryptographic hash of the model card"
    )
    
    @field_validator('model_size')
    @classmethod
    def validate_model_size(cls, v):
        if v <= 0:
            raise ValueError("Model size must be positive")
        return v
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model card to dictionary"""
        return self.model_dump(mode='json')
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelCard':
        """Create model card from dictionary"""
        return cls(**data)
    
    def update_timestamp(self):
        """Update the updated_at timestamp"""
        self.updated_at = datetime.utcnow()
    
    def mark_certified(self, metadata: Optional[Dict[str, Any]] = None):
        """Mark the model as certified"""
        self.certified = True
        self.certification_metadata = metadata or {}
        self.update_timestamp()

