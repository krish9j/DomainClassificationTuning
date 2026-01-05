# Childcare Observation Domain Classification

A machine learning system for automatically classifying childcare observations into standardized developmental domains using fine-tuned large language models (LLMs) on Google Cloud Vertex AI.

## 🎯 Overview

This project provides an end-to-end pipeline for training and deploying AI models that classify childcare observations (text notes and photos) according to a developmental taxonomy. The system supports both **managed Gemini fine-tuning** (Google's multimodal models) and **open-source model training** (Phi-3, Llama) using Parameter-Efficient Fine-Tuning (PEFT) techniques.

### Key Features

- **Multimodal Classification**: Processes both text observations and images
- **Dual Training Pipelines**: 
  - Managed Gemini tuning via Vertex AI (TPU-optimized)
  - Custom OSS model training (Phi-3, Llama) on GPU instances
- **Efficient Fine-Tuning**: Uses LoRA (Low-Rank Adaptation) for cost-effective training
- **Production-Ready**: Includes deployment scripts, evaluation tools, and inference clients
- **Taxonomy-Driven**: Validates against structured developmental domains (Physical, Social-Emotional, Language, Cognitive, Approaches to Learning)

## 🏗️ Architecture

The system follows a modular architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│                    Data Transformation Layer                 │
│  (observation_transformer.py, dataset_builder.py)          │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    Storage Layer (GCS)                      │
│  Training Data │ Model Artifacts │ Observations           │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Compute Layer (Vertex AI)                      │
│  ┌──────────────┐              ┌──────────────┐           │
│  │  Training    │              │   Serving     │           │
│  │  (TPU/GPU)   │─────────────▶│  (Endpoints)  │           │
│  └──────────────┘              └──────────────┘           │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Inference Layer (Client SDK)                   │
│  (inference_client.py)                                      │
└─────────────────────────────────────────────────────────────┘
```

For detailed architecture documentation, see [docs/architecture.md](docs/architecture.md).

## 🛠️ Tech Stack

### Core Technologies
- **Python 3.9+**: Primary development language
- **Google Cloud Platform**: Infrastructure and ML services
  - Vertex AI: Model training and serving
  - Cloud Storage (GCS): Data and artifact storage
  - Cloud IAM: Authentication and authorization

### ML Frameworks & Libraries
- **Vertex AI SDK** (`google-cloud-aiplatform`): Managed ML services
- **JAX**: Framework for Gemini models (TPU-optimized)
- **PyTorch**: Framework for OSS model training (Phi-3, Llama)
- **Transformers** (Hugging Face): Model loading and training utilities
- **PEFT/LoRA**: Parameter-efficient fine-tuning
- **scikit-learn**: Evaluation metrics and analysis

### Data Processing
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **JSONL**: Training data format

### Development Tools
- **pytest**: Testing framework
- **tqdm**: Progress bars
- **python-dotenv**: Environment variable management

## 📋 Prerequisites

- Python 3.9 or higher
- Google Cloud Platform account with billing enabled
- GCP project with Vertex AI API enabled
- `gcloud` CLI installed and authenticated
- Sufficient GCP quotas for:
  - Vertex AI Training jobs
  - Cloud Storage buckets
  - Vertex AI Endpoints (for deployment)

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/krish9j/DomainClassificationTuning.git
cd DomainClassificationTuning
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure GCP

Set your GCP project and region in `config.py` or via environment variables:

```bash
export GCP_PROJECT_ID="your-project-id"
export GCP_REGION="us-central1"
export GCS_BUCKET="your-bucket-name"
```

Or edit `config.py` directly:

```python
gcp_config = GCPConfig(
    project_id="your-project-id",
    region="us-central1",
    bucket_name="your-bucket-name"
)
```

### 4. Initialize GCP Resources

Set up required GCP resources (buckets, service accounts):

```bash
python scripts/gcp_setup.py --setup
```

### 5. Prepare Training Data

Place your raw observation data in `data/raw/`, then transform it:

```bash
# Transform observations to canonical format
python src/observation_transformer.py

# Build training/validation datasets
python src/dataset_builder.py
```

### 6. Start Fine-Tuning

**Option A: Managed Gemini Tuning (Recommended)**
```bash
python scripts/tuning_orchestrator.py --start
```

**Option B: OSS Model Training (Phi-3)**
```bash
python scripts/tuning_orchestrator_oss.py --start
```

### 7. Monitor Training

Check job status:
```bash
python scripts/tuning_orchestrator.py --status
```

### 8. Deploy Model

Once training completes, deploy to an endpoint:

```bash
python scripts/endpoint_deployer.py --model <model_resource_name>
```

### 9. Evaluate Model

Run evaluation on validation set:

```bash
python scripts/evaluation.py --dataset data/training/valid.jsonl --endpoint <endpoint_id>
```

## 📁 Project Structure

```
DomainClassification/
├── client/                    # Inference client SDK
│   ├── __init__.py
│   └── inference_client.py   # Client for calling deployed models
├── data/                      # Data directories
│   ├── raw/                   # Raw observation data
│   ├── processed/             # Processed/canonical observations
│   └── training/              # Training datasets (JSONL)
├── docs/                      # Documentation
│   ├── architecture.md        # System architecture details
│   ├── cost_estimates.md      # Cost analysis
│   ├── evaluation_guide.md    # Evaluation methodology
│   ├── runbook_operations.md  # Operations guide
│   ├── runbook_training.md    # Training guide
│   └── runbook_training_oss.md # OSS training guide
├── scripts/                   # Orchestration scripts
│   ├── endpoint_deployer.py   # Model deployment
│   ├── evaluation.py          # Model evaluation
│   ├── gcp_setup.py           # GCP resource setup
│   ├── tuning_orchestrator.py # Gemini tuning orchestration
│   └── tuning_orchestrator_oss.py # OSS tuning orchestration
├── src/                       # Core source code
│   ├── dataset_builder.py     # JSONL dataset generation
│   ├── observation_transformer.py # Data transformation
│   ├── taxonomy_parser.py     # Taxonomy parsing and validation
│   └── train_phi.py           # Phi-3 training script (runs in container)
├── config.py                  # Configuration management
├── learningdomains.json       # Developmental taxonomy definition
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🔧 Configuration

All configuration is centralized in `config.py`. Key settings include:

- **GCP Configuration**: Project ID, region, bucket names
- **Model Configuration**: Base model, hyperparameters, endpoint settings
- **Data Configuration**: Train/validation splits, batch sizes
- **Prompt Templates**: System instructions and classification prompts

See `config.py` for detailed configuration options.

## 📊 Data Format

### Input: Canonical Observations

Observations are transformed into a canonical format with:
- Observation ID
- Notes (text description)
- Photo URIs (GCS paths)
- Classification labels (domain, attribute, progression)

### Output: Training JSONL

Each line in `train.jsonl` / `valid.jsonl` contains:

**Text-only format:**
```json
{
  "input_text": "Analyze the following childcare observation...",
  "output_text": "{\"domain_key\": \"physical_development\", ...}"
}
```

**Multimodal format:**
```json
{
  "contents": [
    {
      "role": "user",
      "parts": [
        {"fileData": {"fileUri": "gs://bucket/image.jpg"}},
        {"text": "Analyze the following observation..."}
      ]
    },
    {
      "role": "model",
      "parts": [{"text": "{\"domain_key\": \"physical_development\", ...}"}]
    }
  ]
}
```

## 🎓 Training Methodology

### Parameter-Efficient Fine-Tuning (PEFT)

The system uses **LoRA (Low-Rank Adaptation)** instead of full fine-tuning:

- **Efficiency**: Only ~0.1-1% of model parameters are trainable
- **Performance**: Achieves comparable accuracy to full fine-tuning
- **Cost**: Significantly reduces training time and compute costs
- **Preservation**: Maintains base model's general knowledge

### Hyperparameters

Default settings (configurable in `config.py`):
- **Epochs**: 4
- **Adapter Size**: 4 (LoRA rank)
- **Learning Rate Multiplier**: 1.0
- **Batch Size**: Varies by model type

## 📈 Evaluation Metrics

The evaluation script calculates:
- **Exact Match Accuracy**: Percentage of perfect (domain, attribute, progression) matches
- **Per-Domain Accuracy**: Accuracy broken down by developmental domain
- **Per-Attribute Accuracy**: Accuracy for each attribute
- **Per-Progression Accuracy**: Accuracy for each progression level
- **Confusion Matrices**: Detailed error analysis
- **JSON Validity Rate**: Percentage of valid JSON responses

## 🔐 Authentication

The system uses Google Cloud Application Default Credentials (ADC):

```bash
gcloud auth application-default login
```

Or set service account key:
```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"
```

## 💰 Cost Considerations

Training costs vary based on:
- **Model Type**: Gemini (TPU) vs. OSS (GPU)
- **Dataset Size**: Number of training examples
- **Training Duration**: Epochs and convergence time
- **Region**: Different regions have different pricing

See [docs/cost_estimates.md](docs/cost_estimates.md) for detailed cost analysis.

## 📚 Documentation

- [Architecture Guide](docs/architecture.md): Detailed system design and patterns
- [Training Runbook](docs/runbook_training.md): Step-by-step training instructions
- [OSS Training Guide](docs/runbook_training_oss.md): Open-source model training
- [Operations Runbook](docs/runbook_operations.md): Production operations guide
- [Evaluation Guide](docs/evaluation_guide.md): Model evaluation methodology

## 🧪 Testing

Run tests with pytest:

```bash
pytest tests/
```

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📝 License

[Add your license here]

## 🆘 Support

For issues and questions:
- Open an issue on GitHub
- Check the documentation in `docs/`
- Review the runbooks for common operations

## 🙏 Acknowledgments

- Google Cloud Vertex AI team for managed ML infrastructure
- Hugging Face for Transformers and PEFT libraries
- Microsoft for Phi-3 model

---

**Note**: This project requires a Google Cloud Platform account with billing enabled. Ensure you understand the costs associated with Vertex AI training and serving before running large-scale jobs.

