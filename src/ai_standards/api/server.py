"""
FastAPI server implementation
"""
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any, Optional
import time
import uuid

from ai_standards.api.models import (
    InferenceServiceRequest,
    InferenceServiceResponse,
    RevokeServiceRequest,
    UpdateServiceRequest,
    InferenceRequest,
    BatchInferenceRequest,
    InferenceResponse,
    BatchInferenceResponse,
    ConsensusInferenceRequest,
    ConsensusInferenceResponse,
)
from ai_standards.certification import CertificationManager
from ai_standards.benchmarking import BenchmarkHarness
from ai_standards.consensus import ConsensusEngine
from ai_standards.certification.model_card import OutputModality
from ai_standards.utils.blockchain import BlockchainConnector


# Initialize FastAPI app
app = FastAPI(
    title="AI-Blockchain Integration API",
    description="Reference implementation of the AI Standards API Framework",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class APIFramework:
    """
    API Framework manager - coordinates components
    """
    
    def __init__(self):
        self.blockchain = BlockchainConnector(use_mock=True)
        self.certification_manager = CertificationManager(blockchain_connector=self.blockchain)
        self.benchmark_harness = BenchmarkHarness(blockchain_connector=self.blockchain)
        self.consensus_engine = ConsensusEngine()
        
        # In-memory storage for inference services (in real app, use database)
        self.inference_services: Dict[str, Dict[str, Any]] = {}
        
        # Mock model registry (in real app, integrate with actual model serving)
        self.model_registry: Dict[str, Any] = {}
    
    def register_model(self, model_id: str, model: Any):
        """Register a model for inference"""
        self.model_registry[model_id] = model
    
    def get_model(self, endpoint: str) -> Optional[Any]:
        """Get model by endpoint/ID"""
        return self.model_registry.get(endpoint)


# Global API framework instance
api_framework = APIFramework()


# ============= Administration Routes =============

@app.post(
    "/api/v1/inference-service/register",
    response_model=InferenceServiceResponse,
    tags=["Administration"],
    summary="Register Inference Service",
    description="Registers a new inference service and creates associated smart contract"
)
async def register_inference_service(request: InferenceServiceRequest):
    """
    Register a new inference service.
    
    This creates a smart contract on the blockchain to manage the inference service,
    tracking usage and payments.
    """
    # Generate unique service ID
    service_id = str(uuid.uuid4())
    
    # Create smart contract record on blockchain
    contract_data = {
        "service_id": service_id,
        "endpoint_id": request.endpoint_id,
        "user_wallet_address": request.user_wallet_address,
        "num_queries": request.num_queries,
        "queries_used": 0,
        "is_revisable": request.is_revisable,
        "is_modifiable": request.is_modifiable,
        "status": "active",
    }
    
    tx_id = api_framework.blockchain.backend.store_record({
        "type": "inference_service",
        "data": contract_data,
    })
    
    # Store in memory
    api_framework.inference_services[service_id] = contract_data
    
    return InferenceServiceResponse(
        inference_service_id=service_id,
        tx_id=tx_id,
        status="active"
    )


@app.post(
    "/api/v1/inference-service/revoke",
    tags=["Administration"],
    summary="Revoke Inference Service"
)
async def revoke_inference_service(request: RevokeServiceRequest):
    """Terminate an inference service prematurely (if permitted)"""
    service = api_framework.inference_services.get(request.inference_service_id)
    
    if not service:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Inference service not found"
        )
    
    if not service.get("is_revisable"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Service is not revisable"
        )
    
    if service["user_wallet_address"] != request.user_wallet_address:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Wallet address mismatch"
        )
    
    service["status"] = "revoked"
    
    return {"status": "revoked", "service_id": request.inference_service_id}


@app.post(
    "/api/v1/inference-service/update",
    tags=["Administration"],
    summary="Update Inference Service"
)
async def update_inference_service(request: UpdateServiceRequest):
    """Modify an existing inference service (e.g., add/withdraw funds)"""
    service = api_framework.inference_services.get(request.inference_service_id)
    
    if not service:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Inference service not found"
        )
    
    if not service.get("is_modifiable"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Service is not modifiable"
        )
    
    if service["user_wallet_address"] != request.user_wallet_address:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Wallet address mismatch"
        )
    
    # Apply updates
    for key, value in request.updates.items():
        if key in ["num_queries", "endpoint_id"]:
            service[key] = value
    
    return {"status": "updated", "service_id": request.inference_service_id}


# ============= Inference Routes =============

@app.post(
    "/api/v1/inference/single",
    response_model=InferenceResponse,
    tags=["Inference"],
    summary="Single Inference",
    description="Execute a single AI inference query"
)
async def single_inference(request: InferenceRequest):
    """Execute a single AI inference query"""
    # Verify service
    service = api_framework.inference_services.get(request.inference_service_id)
    if not service or service["status"] != "active":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or inactive inference service"
        )
    
    if service["queries_used"] >= service["num_queries"]:
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail="Query limit exceeded"
        )
    
    # Get model
    model = api_framework.get_model(request.endpoint)
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model not found: {request.endpoint}"
        )
    
    # Execute inference
    start_time = time.time()
    try:
        if hasattr(model, 'predict'):
            result = model.predict(request.query)
        elif hasattr(model, 'generate'):
            result = model.generate(request.query)
        elif callable(model):
            result = model(request.query)
        else:
            raise ValueError("Model does not have a valid inference method")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}"
        )
    
    execution_time = time.time() - start_time
    
    # Update query count
    service["queries_used"] += 1
    
    return InferenceResponse(
        result=result,
        inference_service_id=request.inference_service_id,
        query_count=service["queries_used"],
        execution_time=execution_time,
        model_metadata={"endpoint": request.endpoint}
    )


@app.post(
    "/api/v1/inference/batch",
    response_model=BatchInferenceResponse,
    tags=["Inference"],
    summary="Batch Inference"
)
async def batch_inference(request: BatchInferenceRequest):
    """Execute multiple inference queries in a batch"""
    service = api_framework.inference_services.get(request.inference_service_id)
    if not service or service["status"] != "active":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or inactive inference service"
        )
    
    num_queries = len(request.query_list)
    if service["queries_used"] + num_queries > service["num_queries"]:
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail="Query limit would be exceeded"
        )
    
    model = api_framework.get_model(request.endpoint)
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model not found: {request.endpoint}"
        )
    
    # Execute batch
    start_time = time.time()
    results = []
    
    for query in request.query_list:
        try:
            if hasattr(model, 'predict'):
                result = model.predict(query)
            elif hasattr(model, 'generate'):
                result = model.generate(query)
            else:
                result = model(query)
            results.append(result)
        except Exception as e:
            results.append({"error": str(e)})
    
    execution_time = time.time() - start_time
    
    # Update query count
    service["queries_used"] += num_queries
    
    return BatchInferenceResponse(
        results=results,
        inference_service_id=request.inference_service_id,
        query_count=service["queries_used"],
        execution_time=execution_time,
        model_metadata={"endpoint": request.endpoint}
    )


@app.post(
    "/api/v1/inference/consensus",
    response_model=ConsensusInferenceResponse,
    tags=["Inference"],
    summary="Consensus Inference",
    description="Execute inference using multiple models and reach consensus"
)
async def consensus_inference(request: ConsensusInferenceRequest):
    """Execute consensus-based inference using multiple models"""
    service = api_framework.inference_services.get(request.inference_service_id)
    if not service or service["status"] != "active":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or inactive inference service"
        )
    
    # Get models and metadata
    models = []
    model_metadata = {}
    
    for model_id in request.model_ids:
        model = api_framework.get_model(model_id)
        if not model:
            continue
        
        models.append(model)
        
        # Get model card and benchmark results
        model_card = api_framework.certification_manager.load_model_card(model_id)
        benchmark_results = api_framework.benchmark_harness.load_results(model_id)
        
        model_metadata[model_id] = {
            "certified": model_card.certified if model_card else False,
            "benchmark_score": benchmark_results.overall_score or 0.5,
        }
    
    if not models:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No valid models found"
        )
    
    # Execute consensus
    start_time = time.time()
    
    try:
        modality = OutputModality(request.output_modality)
        consensus_result = api_framework.consensus_engine.reach_consensus_with_models(
            models=models,
            model_ids=request.model_ids[:len(models)],
            input_data=request.query,
            model_metadata=model_metadata,
            output_modality=modality
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Consensus error: {str(e)}"
        )
    
    execution_time = time.time() - start_time
    
    # Update query count (count as number of models used)
    service["queries_used"] += len(models)
    
    # Get individual outputs from consensus result
    model_outputs = {}
    for model_id in request.model_ids[:len(models)]:
        model_outputs[model_id] = "output"  # Placeholder
    
    return ConsensusInferenceResponse(
        consensus_output=consensus_result["consensus_output"],
        model_outputs=model_outputs,
        weights=consensus_result["weights"],
        num_models=consensus_result["num_models"],
        eligible_models=consensus_result["eligible_models"],
        inference_service_id=request.inference_service_id,
        execution_time=execution_time
    )


@app.get("/health", tags=["System"])
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "version": "0.1.0"}


def main():
    """Entry point for running the API server"""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()

