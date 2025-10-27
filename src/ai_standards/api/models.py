"""
API request/response models
"""
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


# ============= Administration Routes =============

class InferenceServiceRequest(BaseModel):
    """Request to register an inference service"""
    endpoint_id: str = Field(..., description="Identifier for the AI model endpoint")
    num_queries: int = Field(..., description="Number of intended queries", gt=0)
    user_wallet_address: str = Field(..., description="Blockchain wallet address of the user")
    is_revisable: bool = Field(
        default=False,
        description="Indicates if the contract can be terminated early"
    )
    is_modifiable: bool = Field(
        default=False,
        description="Allows for changes in contract terms"
    )


class InferenceServiceResponse(BaseModel):
    """Response from registering an inference service"""
    inference_service_id: str = Field(..., description="Unique identifier for the registered service")
    tx_id: Optional[str] = Field(None, description="Blockchain transaction ID")
    status: str = Field(default="active")


class RevokeServiceRequest(BaseModel):
    """Request to revoke an inference service"""
    inference_service_id: str
    user_wallet_address: str


class UpdateServiceRequest(BaseModel):
    """Request to update an inference service"""
    inference_service_id: str
    user_wallet_address: str
    updates: Dict[str, Any] = Field(
        default_factory=dict,
        description="Fields to update (e.g., add_funds, withdraw_funds)"
    )


# ============= Inference Routes =============

class InferenceRequest(BaseModel):
    """Single inference request"""
    endpoint: str = Field(..., description="URI or identifier of the AI model")
    query: Any = Field(..., description="Input data for the AI model")
    inference_service_id: str = Field(..., description="Identifier for the associated smart contract")
    optional_parameters: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional parameters for model-specific customization"
    )


class BatchInferenceRequest(BaseModel):
    """Batch inference request"""
    endpoint: str = Field(..., description="URI or identifier of the AI model")
    query_list: List[Any] = Field(..., description="List of input queries")
    inference_service_id: str = Field(..., description="Identifier for the associated smart contract")
    optional_parameters: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional parameters for model-specific customization"
    )


class InferenceResponse(BaseModel):
    """Response from inference"""
    result: Any = Field(..., description="Model output")
    inference_service_id: str
    query_count: int = Field(..., description="Number of queries used")
    execution_time: float = Field(..., description="Execution time in seconds")
    model_metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Metadata about the model used"
    )


class BatchInferenceResponse(BaseModel):
    """Response from batch inference"""
    results: List[Any] = Field(..., description="List of model outputs")
    inference_service_id: str
    query_count: int = Field(..., description="Total number of queries used")
    execution_time: float = Field(..., description="Total execution time in seconds")
    model_metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Metadata about the model used"
    )


class ConsensusInferenceRequest(BaseModel):
    """Request for consensus-based inference using multiple models"""
    model_ids: List[str] = Field(..., description="List of model IDs to use for consensus")
    query: Any = Field(..., description="Input data")
    output_modality: str = Field(..., description="Output type: discrete, text, vision, audio")
    inference_service_id: str
    optional_parameters: Optional[Dict[str, Any]] = None


class ConsensusInferenceResponse(BaseModel):
    """Response from consensus inference"""
    consensus_output: Any = Field(..., description="Aggregated consensus result")
    model_outputs: Dict[str, Any] = Field(..., description="Individual model outputs")
    weights: Dict[str, float] = Field(..., description="Model weights used in consensus")
    num_models: int
    eligible_models: int = Field(..., description="Number of models that participated")
    inference_service_id: str
    execution_time: float

