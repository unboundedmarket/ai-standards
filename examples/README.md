# Examples Directory

This directory contains complete working examples demonstrating the AI-Blockchain Integration Standards toolbox.

## Available Examples

### 1. Certification Example (`01_certification_example.py`)

Demonstrates the complete model certification workflow:
- Creating a model card with all required and optional parameters
- Validating the model card
- Certifying the model
- Minting an NFT on blockchain (mock)
- Retrieving and listing model cards

**Run:**
```bash
python examples/01_certification_example.py
```

**Learn:**
- Model card schema and parameters
- Validation rules
- Certification process
- NFT minting

### 2. Benchmarking Example (`02_benchmarking_example.py`)

Shows how to benchmark AI models:
- Registering benchmark tasks
- Running evaluations
- Storing results on-chain
- Viewing leaderboards

**Run:**
```bash
python examples/02_benchmarking_example.py
```

**Learn:**
- Benchmark task registration
- Performance metrics
- Result storage
- Leaderboard queries

### 3. API Framework Example (`03_api_example.py`)

Demonstrates the REST API framework:
- Registering models with the API
- Creating inference services
- Single and batch inference
- Consensus-based API inference
- Starting and using the API server

**Run:**
```bash
python examples/03_api_example.py
```

**Interactive API Server:**

To use the REST API interactively:
```bash
# Start the server
uvicorn ai_standards.api.server:app --reload

# Visit the interactive documentation
# http://localhost:8000/docs
```

Available endpoints:
- `POST /api/v1/inference-service/register` - Register inference service
- `POST /api/v1/inference-service/revoke` - Revoke service
- `POST /api/v1/inference-service/update` - Update service
- `POST /api/v1/inference/single` - Single inference query
- `POST /api/v1/inference/batch` - Batch inference
- `POST /api/v1/inference/consensus` - Consensus-based inference
- `GET /health` - Health check endpoint

**Learn:**
- API framework setup
- Model registration
- Inference service management
- REST API endpoints
- Interactive API documentation

### 4. Consensus Example (`04_consensus_example.py`)

Demonstrates multi-model consensus mechanisms:
- Setting up multiple models
- Calculating model weights based on certification and benchmarking
- Reaching consensus on discrete outputs
- Text aggregation via reranking
- Byzantine fault tolerance

**Run:**
```bash
python examples/04_consensus_example.py
```

**Learn:**
- Weight calculation
- Discrete aggregation (weighted voting)
- Text aggregation (distillation)
- Deviation scores and penalties

### 5. Full Integration Example (`05_full_integration_example.py`)

Complete end-to-end workflow using all four pillars:
- Pillar 1: API Framework for model access
- Pillar 2: Certification of multiple models with NFTs
- Pillar 3: Benchmarking all models
- Pillar 4: Consensus-based inference

**Run:**
```bash
python examples/05_full_integration_example.py
```

**Learn:**
- Complete workflow
- Integration between pillars
- Real-world usage patterns
- Best practices

## Running Examples

All examples can be run directly:

```bash
# Run individual example
python examples/01_certification_example.py

# Run all examples
for example in examples/*.py; do python "$example"; done
```

## Example Output

Each example provides detailed console output showing:
- Success indicators
- Step-by-step progress
- Key results and metrics
- Summary information

## Customizing Examples

The examples use mock models for demonstration. To use with real models:

1. Replace mock model classes with actual model implementations
2. Ensure models have appropriate inference methods (`predict()`, `generate()`, etc.)
3. Update model metadata (size, architecture, etc.)
4. Configure blockchain connector for your network

## Prerequisites

Ensure you have installed the package:

```bash
cd /path/to/ai-standards
pip install -e .
```

## Next Steps

After running the examples:
1. Read the main README.md for detailed API reference
2. Explore the source code (../src/ai_standards/) to understand implementation
3. Try integrating with your own models
4. Experiment with custom benchmark tasks
5. Test consensus with different modalities

## Support

If you encounter issues:
- Check that all dependencies are installed: `pip install -r requirements.txt`
- Verify package installation: `pip list | grep ai-standards`
- Read error messages carefully
- Open an issue on GitHub with details

## Contributing

Found a bug or have an improvement? We welcome contributions:
1. Fork the repository
2. Create a new example or improve existing ones
3. Submit a pull request

