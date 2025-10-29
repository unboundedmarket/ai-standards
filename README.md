# Integrating AI and Blockchain: Developing AI Standards for Cardano

Project Catalyst: 1200134

Welcome to the AI-Blockchain Integration Standards repository. This project provides both research and reference implementation for integrating AI with blockchain technology, specifically tailored for the Cardano ecosystem.

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Installation](#installation)
4. [Architecture](#architecture)
5. [Four Pillars](#four-pillars)
6. [API Reference](#api-reference)
7. [CLI Tools](#cli-tools)
8. [Examples](#examples)
9. [Use Cases](#use-cases)
10. [Contributing](#contributing)
11. [License](#license)

## Overview

This repository contains:

- **Research Papers** - Theoretical framework and specifications
  - [Research Proposal](reports/1200134_Integrating_AI_and_Blockchain.pdf)
  - [API & Certification Benchmarking](reports/API_Certification_Benchmarking.pdf)
  - [AI Consensus Paper](reports/AI_Consensus.pdf)
- **Reference Implementation** - [MVP toolbox] (`src/`)
- **Examples** - Usage examples and demonstrations (`examples/`)
- **CLI Tools** - Command-line utilities for certification, benchmarking, and consensus

## Quick Start

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/unboundedmarket/ai-standards.git
cd ai-standards

# Install dependencies
python -m venv ai-standards
source ai-standards/bin/activate
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Run Your First Example

```bash
# Run the certification example
python examples/01_certification_example.py

# Run the benchmarking example
python examples/02_benchmarking_example.py

# Run the API framework example
python examples/03_api_example.py

# Run the consensus example
python examples/04_consensus_example.py
```

### Try the Full Integration

```bash
# Run the complete workflow demonstrating all four pillars
python examples/05_full_integration_example.py
```

### Start the API Server

```bash
# Start the FastAPI server
uvicorn ai_standards.api.server:app --reload
```

Visit `http://localhost:8000/docs` for interactive API documentation.

The API server provides the following endpoints:

**Administration**
- `POST /api/v1/inference-service/register` - Register inference service
- `POST /api/v1/inference-service/revoke` - Revoke service
- `POST /api/v1/inference-service/update` - Update service

**Inference**
- `POST /api/v1/inference/single` - Single inference query
- `POST /api/v1/inference/batch` - Batch inference
- `POST /api/v1/inference/consensus` - Consensus-based inference

**System**
- `GET /health` - Health check endpoint

## Architecture

The toolbox is organized into four main modules corresponding to the four pillars:

```
ai-standards/
├── src/ai_standards/
│   ├── api/              # Pillar 1: REST API Framework
│   ├── certification/    # Pillar 2: Model Cards & NFTs
│   ├── benchmarking/     # Pillar 3: Performance Evaluation
│   ├── consensus/        # Pillar 4: Multi-Model Aggregation
│   └── utils/            # Shared utilities (blockchain, crypto)
├── examples/             # Usage examples
└── report/               # Research papers (LaTeX)
```

## Four Pillars

### Pillar 1: API Framework

The API framework provides standardized REST endpoints for AI model access with blockchain integration.

#### Key Features

- Administration Routes: Register, revoke, update inference services
- Inference Routes: Single and batch inference
- Consensus Inference: Multi-model aggregation via API
- Smart Contract Integration: On-chain tracking of usage and payments

#### API Endpoints

**Administration**
- `POST /api/v1/inference-service/register` - Register inference service
- `POST /api/v1/inference-service/revoke` - Revoke service
- `POST /api/v1/inference-service/update` - Update service

**Inference**
- `POST /api/v1/inference/single` - Single inference query
- `POST /api/v1/inference/batch` - Batch inference
- `POST /api/v1/inference/consensus` - Consensus-based inference

#### Usage Example

```python
from ai_standards.api import APIFramework

# Initialize framework
api = APIFramework()

# Register a model
api.register_model("model_id", model_instance)

# Start API server
# Run: uvicorn ai_standards.api.server:app --reload
```

### Pillar 2: Certification

NFT-based model cards that ensure transparency and traceability.

#### Model Card Schema

**Fixed Parameters (Mandatory)**
- `model_name`: Human-readable name
- `model_size`: Number of parameters
- `architecture`: Model architecture type
- `output_modality`: Output type (discrete, text, vision, audio)
- `usage_instructions`: How to use the model
- `licensing_terms`: License information
- `associated_costs`: Financial/computational costs
- `token_limits`: Input/output token restrictions

**Optional Parameters**
- `training_data_sources`: Training datasets
- `ethical_considerations`: Bias and fairness information
- `intended_use_cases`: Recommended applications
- `limitations`: Known restrictions

#### Usage Example

```python
from ai_standards.certification import CertificationManager

manager = CertificationManager()

# Create model card
model_card = manager.create_model_card(
    model_name="Sentiment Analyzer",
    model_size=110_000_000,
    architecture="transformer",
    output_modality="discrete",
    usage_instructions="POST /predict with {'text': '...'}",
    licensing_terms="MIT License"
)

# Validate model card
validation_result = manager.validate_model_card(model_card)

# Certify and mint NFT
result = manager.certify_model(model_card, mint_nft=True)
print(f"NFT ID: {result['nft_id']}")
```

### Pillar 3: Benchmarking

Systematic evaluation of AI models with on-chain result storage.

#### Key Features

- Task Registry: Register custom benchmark tasks
- Performance Metrics: Comprehensive metric tracking
- On-Chain Storage: Immutable benchmark results
- Leaderboards: Compare model performance

#### Usage Example

```python
from ai_standards.benchmarking import BenchmarkHarness

harness = BenchmarkHarness()

# Run benchmark
result = harness.run_benchmark(
    model=my_model,
    model_id="model_123",
    model_name="My Model",
    task_name="sentiment_analysis",
    store_on_chain=True
)

# View results
print(f"Benchmark score: {result.benchmark_score:.4f}")

# Get leaderboard
leaderboard = harness.get_leaderboard(task_name="sentiment_analysis")
```

### Pillar 4: Consensus Mechanisms

Multi-model aggregation with modality-specific strategies.

#### Aggregation Strategies

1. **Discrete Aggregator** - Weighted majority voting
   - For: Classification, discrete predictions
   - Formula: `ŷ = argmax_y Σ W(f_i) * δ(f_i(x), y)`

2. **Text Aggregator** - Distillation-based reranking
   - For: Text generation, QA, chatbots
   - Formula: `ŷ = D({(y_i, W(f_i))})`

3. **Vision Aggregator** - Latent space fusion
   - For: Image/video generation
   - Formula: `V̂ = Ψ(D_V({(Φ(V_i), W(f_i))}))`

4. **Audio Aggregator** - Spectro-temporal fusion
   - For: TTS, music generation
   - Formula: `â = ISTFT(D_A({(A_i, W(f_i)) | I_i}))`

#### Weight Calculation

Weights are calculated based on certification and benchmarking:

```
W(f_i) = (B(f_i) * C(f_i)) / Σ(B(f_j) * C(f_j))
```

where:
- `B(f_i)` = Benchmark score (0-1)
- `C(f_i)` = Certification status (1 if certified, 0 otherwise)

#### Usage Example

More information can be found in [examples/README.md](examples/README.md).

```python
from ai_standards.consensus import ConsensusEngine
from ai_standards.certification.model_card import OutputModality

engine = ConsensusEngine()

# Prepare model outputs and metadata
model_outputs = {
    "model_a": "positive",
    "model_b": "positive",
    "model_c": "neutral"
}

model_metadata = {
    "model_a": {"benchmark_score": 0.85, "certified": True},
    "model_b": {"benchmark_score": 0.90, "certified": True},
    "model_c": {"benchmark_score": 0.75, "certified": False}
}

# Reach consensus
result = engine.reach_consensus(
    model_outputs=model_outputs,
    model_metadata=model_metadata,
    output_modality=OutputModality.DISCRETE
)

print(f"Consensus: {result['consensus_output']}")
print(f"Weights: {result['weights']}")
```

#### Byzantine Fault Tolerance

Apply penalties to models that deviate from consensus:

```python
from ai_standards.consensus import WeightCalculator

# Calculate deviation scores
deviations = engine.calculate_deviation_scores(
    model_outputs=model_outputs,
    consensus_output=result['consensus_output'],
    output_modality=OutputModality.DISCRETE
)

# Apply penalty
calculator = WeightCalculator()
updated_weights = calculator.apply_byzantine_penalty(
    weights=result['weights'],
    deviation_scores=deviations,
    lambda_param=2.0,
    threshold=0.5
)
```

## CLI Tools

### Certification CLI

```bash
# Create model card
ai-standards-certify create \
  --name "My Model" \
  --size 125000000 \
  --architecture transformer \
  --modality text \
  --usage "API usage instructions" \
  --license "MIT" \
  --output model_card.json

# Validate model card
ai-standards-certify validate model_card.json

# Certify model
ai-standards-certify certify model_card.json --mint-nft

# Show model card details
ai-standards-certify show <model_id>

# List all model cards
ai-standards-certify list
```

### Benchmarking CLI

```bash
# List available benchmark tasks
ai-standards-benchmark list-tasks

# Show benchmark results for a model
ai-standards-benchmark show-results <model_id>

# View leaderboard
ai-standards-benchmark leaderboard --task sentiment_analysis --limit 10
```

### API Server

```bash
# Start API server with default settings
ai-standards-api

# Or with custom settings
uvicorn ai_standards.api.server:app --host 0.0.0.0 --port 8000 --reload
```

## Examples

The `examples/` directory contains complete working examples:

1. **`01_certification_example.py`** - Model card creation and certification
2. **`02_benchmarking_example.py`** - Running benchmarks and viewing results
3. **`03_api_example.py`** - REST API framework and server usage
4. **`04_consensus_example.py`** - Multi-model consensus with Byzantine fault tolerance
5. **`05_full_integration_example.py`** - Complete workflow using all four pillars

### Running Examples

```bash
python examples/01_certification_example.py
python examples/02_benchmarking_example.py
python examples/03_api_example.py
python examples/04_consensus_example.py
python examples/05_full_integration_example.py
```

## Blockchain Integration

### Mock Blockchain (Development)

By default, the toolbox uses a mock blockchain for development:

```python
from ai_standards.utils.blockchain import BlockchainConnector

# Mock blockchain (in-memory)
blockchain = BlockchainConnector(use_mock=True)

# Store record
tx_id = blockchain.mint_nft(metadata)
```

### Real Blockchain (Production)

For production deployment with Cardano:

```python
# TODO: Implement Cardano integration
# blockchain = BlockchainConnector(
#     network="mainnet",
#     use_mock=False
# )
```

## Use Cases

- **Model Providers**: Certify and benchmark AI models with blockchain-backed credentials
- **Application Developers**: Access certified models via standardized API with on-chain verification
- **Researchers**: Implement and test consensus mechanisms for multi-model systems
- **DAO Governance**: Use benchmarking results and consensus for decentralized AI decision-making

## Contributing

We welcome contributions. Please follow these guidelines:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License.

## Citation

If you use this work in your research, please cite:

```bibtex
@article{cardano-ai-standards,
  title={Integrating AI and Blockchain: Developing AI Standards for Cardano},
  author={UnboundedMarket Team},
  year={2025},
}
```

## Acknowledgments

This project is funded by Project Catalyst ([1200134](https://milestones.projectcatalyst.io/projects/1200134/milestones)).
We are very grateful for the support from the Cardano community in developing these AI-blockchain integration standards.
