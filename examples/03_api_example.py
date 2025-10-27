"""
Example 3: API Framework

This example demonstrates how to:
1. Set up the API server
2. Register inference services
3. Make inference requests
4. Use consensus-based inference via API

Note: This example shows the programmatic usage.
For interactive testing, start the server with:
    uvicorn ai_standards.api.server:app --reload
Then visit http://localhost:8000/docs
"""

import requests
import time
from ai_standards.api.server import api_framework


# Mock sentiment model for demonstration
class SimpleSentimentModel:
    """Simple mock model for API demonstration"""
    
    def __init__(self, name, positive_bias=0.0):
        self.name = name
        self.positive_bias = positive_bias
    
    def predict(self, text):
        """Predict sentiment"""
        positive_words = ['good', 'great', 'excellent', 'love', 'best', 'amazing']
        negative_words = ['bad', 'terrible', 'hate', 'worst', 'awful', 'disappointed']
        
        text_lower = text.lower()
        
        if any(word in text_lower for word in positive_words):
            return 'positive'
        elif any(word in text_lower for word in negative_words):
            return 'negative'
        else:
            return 'neutral'


def main():
    print("=" * 70)
    print("Example 3: API Framework")
    print("=" * 70)
    
    # Step 1: Register models with the API framework
    print("\n1. Registering models with API framework...")
    
    # Create mock models
    model_a = SimpleSentimentModel("Model A", positive_bias=0.1)
    model_b = SimpleSentimentModel("Model B", positive_bias=0.0)
    model_c = SimpleSentimentModel("Model C", positive_bias=-0.1)
    
    # Register models
    api_framework.register_model("sentiment_model_a", model_a)
    api_framework.register_model("sentiment_model_b", model_b)
    api_framework.register_model("sentiment_model_c", model_c)
    
    print("✓ Registered 3 models with the API framework")
    print("  - sentiment_model_a")
    print("  - sentiment_model_b")
    print("  - sentiment_model_c")
    
    # Step 2: Demonstrate inference service registration
    print("\n2. Registering an inference service...")
    print("  Note: This would normally be done via API POST request")
    
    service_request = {
        "endpoint_id": "sentiment_model_a",
        "num_queries": 100,
        "user_wallet_address": "addr1_test_wallet_address",
        "is_revisable": True,
        "is_modifiable": True
    }
    
    # Simulate service registration (in real usage, this would be via HTTP POST)
    import uuid
    service_id = str(uuid.uuid4())
    api_framework.inference_services[service_id] = {
        "service_id": service_id,
        "endpoint_id": service_request["endpoint_id"],
        "user_wallet_address": service_request["user_wallet_address"],
        "num_queries": service_request["num_queries"],
        "queries_used": 0,
        "is_revisable": service_request["is_revisable"],
        "is_modifiable": service_request["is_modifiable"],
        "status": "active"
    }
    
    print(f"✓ Inference service registered")
    print(f"  Service ID: {service_id}")
    print(f"  Endpoint: {service_request['endpoint_id']}")
    print(f"  Query Limit: {service_request['num_queries']}")
    
    # Step 3: Demonstrate single inference
    print("\n3. Single Inference Example")
    print("  Testing with text: 'This product is amazing!'")
    
    # Get the model and run inference
    model = api_framework.get_model("sentiment_model_a")
    result = model.predict("This product is amazing!")
    
    print(f"✓ Inference completed")
    print(f"  Result: {result}")
    print(f"  Queries used: 1")
    
    # Step 4: Demonstrate batch inference
    print("\n4. Batch Inference Example")
    
    test_queries = [
        "This is excellent!",
        "I'm very disappointed.",
        "It's okay, nothing special.",
        "Best purchase ever!",
        "Terrible experience."
    ]
    
    print(f"  Processing {len(test_queries)} queries...")
    
    results = []
    for query in test_queries:
        result = model.predict(query)
        results.append(result)
    
    print(f"✓ Batch inference completed")
    print(f"  Results:")
    for i, (query, result) in enumerate(zip(test_queries, results), 1):
        print(f"    {i}. '{query}' → {result}")
    
    # Step 5: Demonstrate consensus inference
    print("\n5. Consensus Inference Example")
    print("  Using multiple models for robust predictions...")
    
    from ai_standards.consensus import ConsensusEngine
    from ai_standards.certification.model_card import OutputModality
    
    # Prepare test query
    test_query = "This product is amazing!"
    print(f"  Query: '{test_query}'")
    
    # Get predictions from all models
    model_outputs = {}
    for model_id in ["sentiment_model_a", "sentiment_model_b", "sentiment_model_c"]:
        m = api_framework.get_model(model_id)
        output = m.predict(test_query)
        model_outputs[model_id] = output
        print(f"    {model_id}: {output}")
    
    # Set up model metadata
    model_metadata = {
        "sentiment_model_a": {"benchmark_score": 0.85, "certified": True},
        "sentiment_model_b": {"benchmark_score": 0.90, "certified": True},
        "sentiment_model_c": {"benchmark_score": 0.80, "certified": True}
    }
    
    # Run consensus
    engine = ConsensusEngine()
    consensus_result = engine.reach_consensus(
        model_outputs=model_outputs,
        model_metadata=model_metadata,
        output_modality=OutputModality.DISCRETE
    )
    
    print(f"\n✓ Consensus reached")
    print(f"  Consensus prediction: {consensus_result['consensus_output']}")
    print(f"  Model weights:")
    for model_id, weight in consensus_result['weights'].items():
        print(f"    {model_id}: {weight:.4f}")
    print(f"  Participating models: {consensus_result['eligible_models']}/{consensus_result['num_models']}")
    
    # Step 6: API Server Information
    print("\n6. API Server Usage")
    print("=" * 70)
    print("\nTo use the REST API interactively:")
    print("\n1. Start the server:")
    print("   uvicorn ai_standards.api.server:app --reload")
    print("\n2. Visit the interactive documentation:")
    print("   http://localhost:8000/docs")
    print("\n3. Available endpoints:")
    print("   POST /api/v1/inference-service/register")
    print("   POST /api/v1/inference-service/revoke")
    print("   POST /api/v1/inference-service/update")
    print("   POST /api/v1/inference/single")
    print("   POST /api/v1/inference/batch")
    print("   POST /api/v1/inference/consensus")
    print("   GET  /health")
    print("\n4. Example API request (using curl):")
    print("""
   curl -X POST "http://localhost:8000/api/v1/inference-service/register" \\
     -H "Content-Type: application/json" \\
     -d '{
       "endpoint_id": "sentiment_model_a",
       "num_queries": 100,
       "user_wallet_address": "addr1_test",
       "is_revisable": true,
       "is_modifiable": true
     }'
    """)
    
    print("\n5. Testing with Python requests:")
    print("""
   import requests
   
   # Register service
   response = requests.post(
       "http://localhost:8000/api/v1/inference-service/register",
       json={
           "endpoint_id": "sentiment_model_a",
           "num_queries": 100,
           "user_wallet_address": "addr1_test",
           "is_revisable": True,
           "is_modifiable": True
       }
   )
   service_id = response.json()["inference_service_id"]
   
   # Run inference
   response = requests.post(
       "http://localhost:8000/api/v1/inference/single",
       json={
           "endpoint": "sentiment_model_a",
           "query": "This is amazing!",
           "inference_service_id": service_id
       }
   )
   result = response.json()
   print(result["result"])
    """)
    
    print("=" * 70)
    print("\nAPI Framework example completed successfully!")
    print("\nKey Features Demonstrated:")
    print("  1. Model registration with API framework")
    print("  2. Inference service management")
    print("  3. Single and batch inference")
    print("  4. Consensus-based multi-model inference")
    print("  5. REST API endpoint documentation")
    print("=" * 70)


if __name__ == "__main__":
    main()

