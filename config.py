"""
Configuration for Childcare Observation Domain Classification Pipeline.

This module contains all project-wide settings including GCP configuration,
model parameters, and file paths. Update the placeholder values before running.
"""

import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

# =============================================================================
# PROJECT PATHS
# =============================================================================

# Base project directory
PROJECT_ROOT = Path(__file__).parent.resolve()

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
TRAINING_DATA_DIR = DATA_DIR / "training"

# Source files
TAXONOMY_FILE = PROJECT_ROOT / "learningdomains.json"

# Output files
TRAIN_JSONL = TRAINING_DATA_DIR / "train.jsonl"
VALID_JSONL = TRAINING_DATA_DIR / "valid.jsonl"
TAXONOMY_NORMALIZED = PROCESSED_DATA_DIR / "taxonomy_normalized.json"


# =============================================================================
# GCP CONFIGURATION
# =============================================================================

@dataclass
class GCPConfig:
    """Google Cloud Platform configuration."""
    
    # REQUIRED: Update these values before running
    project_id: str = os.getenv("GCP_PROJECT_ID", "your-gcp-project-id")
    region: str = os.getenv("GCP_REGION", "us-central1")
    
    # GCS bucket for training data and model artifacts
    bucket_name: str = os.getenv("GCS_BUCKET", "childcare-observation-tuning")
    
    # GCS paths within the bucket
    training_data_gcs_path: str = "training-data"
    model_artifacts_gcs_path: str = "model-artifacts"
    observations_gcs_path: str = "observations"
    
    @property
    def bucket_uri(self) -> str:
        return f"gs://{self.bucket_name}"
    
    @property
    def training_data_uri(self) -> str:
        return f"{self.bucket_uri}/{self.training_data_gcs_path}"
    
    @property
    def model_artifacts_uri(self) -> str:
        return f"{self.bucket_uri}/{self.model_artifacts_gcs_path}"


# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

@dataclass
class ModelConfig:
    """Vertex AI model and tuning configuration."""
    
    # Base model for fine-tuning
    # Options: "gemini-1.5-flash-002" (multimodal), "gemini-1.5-pro-002"
    base_model: str = "gemini-1.5-flash-002"
    
    # Whether to use multimodal (image + text) or text-only
    multimodal: bool = True
    
    # Fine-tuning hyperparameters
    epochs: int = 4
    adapter_size: int = 4  # LoRA adapter size (1, 4, 8, 16)
    learning_rate_multiplier: float = 1.0
    
    # Tuning job settings
    tuned_model_display_name: str = "childcare-observation-classifier"
    
    # Endpoint settings
    endpoint_display_name: str = "childcare-classifier-endpoint"
    min_replica_count: int = 1
    max_replica_count: int = 1


# =============================================================================
# DATA CONFIGURATION
# =============================================================================

@dataclass
class DataConfig:
    """Data processing and dataset configuration."""
    
    # Train/validation split ratio
    train_split: float = 0.8
    validation_split: float = 0.2
    
    # Random seed for reproducibility
    random_seed: int = 42
    
    # Minimum samples per class for stratification
    min_samples_per_class: int = 2
    
    # Whether to stratify by domain, attribute, or progression
    stratify_by: str = "domain"  # Options: "domain", "attribute", "progression"
    
    # Maximum prompt length (characters)
    max_prompt_length: int = 8000
    
    # Batch size for processing
    batch_size: int = 100


# =============================================================================
# PROMPT TEMPLATES
# =============================================================================

# System instruction for the classification task
SYSTEM_INSTRUCTION = """You are a childcare development classifier specialized in early childhood education. 
Your task is to analyze observation notes (and optionally photos) of children's activities and classify them 
according to a standardized developmental taxonomy.

You must respond with a valid JSON object containing the classification."""

# Prompt template for classification
CLASSIFICATION_PROMPT_TEMPLATE = """Analyze the following childcare observation and classify it according to the developmental taxonomy.

## Observation Notes:
{notes}

## Task:
Classify this observation into the appropriate developmental domain, attribute, and progression level.

## Response Format:
Respond with a JSON object containing these exact fields:
- domain_key: The normalized key for the domain (e.g., "physical_development")
- domain_name: The full human-readable domain name
- attribute_name: The specific attribute within the domain
- progression_title: One of "Emerging", "Developing", "Progressing", "Advancing", or "Refining"
- progression_description: The specific behavioral description from the taxonomy

## JSON Response:"""

# Prompt template with image reference (for multimodal)
CLASSIFICATION_PROMPT_TEMPLATE_MULTIMODAL = """Analyze the following childcare observation photo and notes, then classify according to the developmental taxonomy.

## Observation Notes:
{notes}

## Photo:
[Image attached showing the child's activity]

## Task:
Classify this observation into the appropriate developmental domain, attribute, and progression level.

## Response Format:
Respond with a JSON object containing these exact fields:
- domain_key: The normalized key for the domain (e.g., "physical_development")
- domain_name: The full human-readable domain name
- attribute_name: The specific attribute within the domain
- progression_title: One of "Emerging", "Developing", "Progressing", "Advancing", or "Refining"
- progression_description: The specific behavioral description from the taxonomy

## JSON Response:"""


# =============================================================================
# DOMAIN KEY MAPPING
# =============================================================================

# Maps domain IDs to normalized keys for consistent output
DOMAIN_ID_TO_KEY = {
    1: "physical_development",
    2: "social_emotional_development",
    3: "language_literacy",
    4: "cognitive_development",
    5: "approaches_to_learning",
}

# Valid progression levels in order
PROGRESSION_LEVELS = [
    "Emerging",
    "Developing",
    "Progressing",
    "Advancing",
    "Refining",
]


# =============================================================================
# EVALUATION CONFIGURATION
# =============================================================================

@dataclass
class EvaluationConfig:
    """Configuration for model evaluation."""
    
    # Target accuracy thresholds
    min_exact_match_accuracy: float = 0.70
    target_exact_match_accuracy: float = 0.80
    
    # Report output
    evaluation_report_path: Path = field(
        default_factory=lambda: PROJECT_ROOT / "evaluation_report.json"
    )


# =============================================================================
# CREATE DEFAULT INSTANCES
# =============================================================================

gcp_config = GCPConfig()
model_config = ModelConfig()
data_config = DataConfig()
evaluation_config = EvaluationConfig()


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def ensure_directories():
    """Create all required directories if they don't exist."""
    for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, TRAINING_DATA_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def validate_config() -> list[str]:
    """Validate configuration and return list of issues."""
    issues = []
    
    if gcp_config.project_id == "your-gcp-project-id":
        issues.append("GCP_PROJECT_ID not set - update config.py or set environment variable")
    
    if not TAXONOMY_FILE.exists():
        issues.append(f"Taxonomy file not found: {TAXONOMY_FILE}")
    
    if data_config.train_split + data_config.validation_split != 1.0:
        issues.append("Train and validation splits must sum to 1.0")
    
    return issues


if __name__ == "__main__":
    # Validate and print configuration
    print("=" * 60)
    print("Childcare Observation Classification - Configuration")
    print("=" * 60)
    print(f"\nProject Root: {PROJECT_ROOT}")
    print(f"Taxonomy File: {TAXONOMY_FILE}")
    print(f"\nGCP Project: {gcp_config.project_id}")
    print(f"GCP Region: {gcp_config.region}")
    print(f"GCS Bucket: {gcp_config.bucket_name}")
    print(f"\nBase Model: {model_config.base_model}")
    print(f"Multimodal: {model_config.multimodal}")
    print(f"\nTrain Split: {data_config.train_split}")
    print(f"Validation Split: {data_config.validation_split}")
    
    issues = validate_config()
    if issues:
        print("\n⚠️  Configuration Issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\n✅ Configuration valid")
