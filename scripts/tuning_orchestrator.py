"""
Tuning Orchestrator for Childcare Observation Classification.

This script orchestrates Vertex AI supervised fine-tuning jobs for the
childcare observation classification model.

Features:
- Creates and monitors fine-tuning jobs
- Supports both text-only and multimodal tuning
- Handles job resumption and status tracking
- Stores tuned model resource names for deployment

Usage:
    python scripts/tuning_orchestrator.py --start       # Start new tuning job
    python scripts/tuning_orchestrator.py --status      # Check job status
    python scripts/tuning_orchestrator.py --list        # List all tuning jobs
"""

import argparse
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Optional
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    gcp_config,
    model_config,
    data_config,
    TRAIN_JSONL,
    VALID_JSONL,
    PROJECT_ROOT,
    SYSTEM_INSTRUCTION,
)

# Vertex AI imports (only when running)
try:
    from google.cloud import aiplatform
    from vertexai.tuning import sft as supervised_fine_tuning
    import vertexai
    VERTEX_AI_AVAILABLE = True
except ImportError:
    VERTEX_AI_AVAILABLE = False
    print("⚠️  Vertex AI SDK not installed. Run: pip install google-cloud-aiplatform")


# =============================================================================
# CONSTANTS
# =============================================================================

TUNING_JOBS_FILE = PROJECT_ROOT / "tuning_jobs.json"


# =============================================================================
# JOB TRACKING
# =============================================================================

def load_tuning_jobs() -> dict:
    """Load the tuning jobs tracking file."""
    if TUNING_JOBS_FILE.exists():
        with open(TUNING_JOBS_FILE, 'r') as f:
            return json.load(f)
    return {"jobs": []}


def save_tuning_jobs(data: dict):
    """Save the tuning jobs tracking file."""
    with open(TUNING_JOBS_FILE, 'w') as f:
        json.dump(data, f, indent=2)


def record_tuning_job(
    job_name: str,
    job_resource_name: str,
    base_model: str,
    train_data_uri: str,
    config: dict
):
    """Record a new tuning job in the tracking file."""
    data = load_tuning_jobs()
    
    job_record = {
        "job_name": job_name,
        "resource_name": job_resource_name,
        "base_model": base_model,
        "train_data_uri": train_data_uri,
        "config": config,
        "started_at": datetime.now().isoformat(),
        "status": "RUNNING",
        "tuned_model_name": None,
        "completed_at": None
    }
    
    data["jobs"].append(job_record)
    save_tuning_jobs(data)
    
    return job_record


def update_job_status(resource_name: str, status: str, tuned_model_name: str = None):
    """Update the status of a tuning job."""
    data = load_tuning_jobs()
    
    for job in data["jobs"]:
        if job["resource_name"] == resource_name:
            job["status"] = status
            if tuned_model_name:
                job["tuned_model_name"] = tuned_model_name
            if status in ["SUCCEEDED", "FAILED", "CANCELLED"]:
                job["completed_at"] = datetime.now().isoformat()
            break
    
    save_tuning_jobs(data)


# =============================================================================
# TUNING FUNCTIONS
# =============================================================================

def initialize_vertex_ai():
    """Initialize Vertex AI SDK."""
    if not VERTEX_AI_AVAILABLE:
        print("❌ Vertex AI SDK not available")
        sys.exit(1)
    
    vertexai.init(
        project=gcp_config.project_id,
        location=gcp_config.region
    )
    print(f"✅ Initialized Vertex AI")
    print(f"   Project: {gcp_config.project_id}")
    print(f"   Region: {gcp_config.region}")


def upload_training_data() -> tuple[str, Optional[str]]:
    """
    Upload training data to GCS and return URIs.
    
    Returns:
        Tuple of (train_uri, valid_uri)
    """
    from google.cloud import storage
    
    client = storage.Client(project=gcp_config.project_id)
    bucket = client.bucket(gcp_config.bucket_name)
    
    # Generate timestamped paths
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    train_blob_name = f"{gcp_config.training_data_gcs_path}/{timestamp}/train.jsonl"
    valid_blob_name = f"{gcp_config.training_data_gcs_path}/{timestamp}/valid.jsonl"
    
    # Upload training data
    if not TRAIN_JSONL.exists():
        raise FileNotFoundError(f"Training data not found: {TRAIN_JSONL}")
    
    print(f"\n📤 Uploading training data...")
    train_blob = bucket.blob(train_blob_name)
    train_blob.upload_from_filename(str(TRAIN_JSONL))
    train_uri = f"gs://{gcp_config.bucket_name}/{train_blob_name}"
    print(f"   ✅ Uploaded: {train_uri}")
    
    # Upload validation data if exists
    valid_uri = None
    if VALID_JSONL.exists() and VALID_JSONL.stat().st_size > 0:
        valid_blob = bucket.blob(valid_blob_name)
        valid_blob.upload_from_filename(str(VALID_JSONL))
        valid_uri = f"gs://{gcp_config.bucket_name}/{valid_blob_name}"
        print(f"   ✅ Uploaded: {valid_uri}")
    
    return train_uri, valid_uri


def start_tuning_job(
    train_uri: str,
    valid_uri: Optional[str] = None,
    job_name: Optional[str] = None
) -> str:
    """
    Start a supervised fine-tuning job on Vertex AI.
    
    Args:
        train_uri: GCS URI to training data
        valid_uri: Optional GCS URI to validation data
        job_name: Optional custom job name
        
    Returns:
        The tuned model resource name (or job resource name if running)
    """
    if not job_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        job_name = f"childcare-classifier-{timestamp}"
    
    print(f"\n🚀 Starting fine-tuning job: {job_name}")
    print(f"   Base model: {model_config.base_model}")
    print(f"   Training data: {train_uri}")
    if valid_uri:
        print(f"   Validation data: {valid_uri}")
    
    # Tuning configuration
    tuning_config = {
        "epochs": model_config.epochs,
        "adapter_size": model_config.adapter_size,
        "learning_rate_multiplier": model_config.learning_rate_multiplier,
    }
    print(f"   Config: {tuning_config}")
    
    # Create the tuning job
    sft_tuning_job = supervised_fine_tuning.train(
        source_model=model_config.base_model,
        train_dataset=train_uri,
        validation_dataset=valid_uri,
        epochs=model_config.epochs,
        adapter_size=model_config.adapter_size,
        learning_rate_multiplier=model_config.learning_rate_multiplier,
        tuned_model_display_name=model_config.tuned_model_display_name,
    )
    
    # Record the job
    job_resource_name = sft_tuning_job.resource_name
    record_tuning_job(
        job_name=job_name,
        job_resource_name=job_resource_name,
        base_model=model_config.base_model,
        train_data_uri=train_uri,
        config=tuning_config
    )
    
    print(f"\n✅ Tuning job started!")
    print(f"   Job resource: {job_resource_name}")
    print(f"\n   Monitor progress with:")
    print(f"   python scripts/tuning_orchestrator.py --status")
    
    return job_resource_name


def check_job_status(resource_name: Optional[str] = None) -> dict:
    """
    Check the status of a tuning job.
    
    Args:
        resource_name: Specific job to check (default: latest)
        
    Returns:
        Job status dictionary
    """
    data = load_tuning_jobs()
    
    if not data["jobs"]:
        print("No tuning jobs found.")
        return {}
    
    # Find the job
    if resource_name:
        job_record = next(
            (j for j in data["jobs"] if j["resource_name"] == resource_name),
            None
        )
    else:
        job_record = data["jobs"][-1]  # Latest job
    
    if not job_record:
        print(f"Job not found: {resource_name}")
        return {}
    
    print(f"\n📊 Tuning Job Status")
    print(f"   Job: {job_record['job_name']}")
    print(f"   Resource: {job_record['resource_name']}")
    print(f"   Started: {job_record['started_at']}")
    
    # Get current status from Vertex AI
    try:
        sft_tuning_job = supervised_fine_tuning.SupervisedTuningJob(
            job_record['resource_name']
        )
        
        # Get state
        job_state = str(sft_tuning_job._job.state)
        print(f"   Status: {job_state}")
        
        # Update local record
        if "SUCCEEDED" in job_state:
            tuned_model = sft_tuning_job.tuned_model_name
            update_job_status(
                job_record['resource_name'],
                "SUCCEEDED",
                tuned_model
            )
            print(f"\n✅ Tuning complete!")
            print(f"   Tuned model: {tuned_model}")
            print(f"\n   Deploy with:")
            print(f"   python scripts/endpoint_deployer.py --deploy --model {tuned_model}")
            
        elif "FAILED" in job_state:
            update_job_status(job_record['resource_name'], "FAILED")
            print(f"\n❌ Tuning failed")
            
        elif "CANCELLED" in job_state:
            update_job_status(job_record['resource_name'], "CANCELLED")
            print(f"\n⚠️ Tuning cancelled")
            
        else:
            # Still running
            print(f"\n⏳ Tuning in progress...")
            print(f"   This can take 1-3 hours depending on dataset size")
        
        return {
            "job_name": job_record['job_name'],
            "resource_name": job_record['resource_name'],
            "state": job_state,
            "tuned_model": getattr(sft_tuning_job, 'tuned_model_name', None)
        }
        
    except Exception as e:
        print(f"\n⚠️ Could not get live status: {e}")
        print(f"   Last known status: {job_record['status']}")
        return job_record


def list_tuning_jobs():
    """List all recorded tuning jobs."""
    data = load_tuning_jobs()
    
    if not data["jobs"]:
        print("No tuning jobs found.")
        return
    
    print(f"\n📋 Tuning Jobs ({len(data['jobs'])} total)")
    print("-" * 80)
    
    for job in data["jobs"]:
        status_emoji = {
            "RUNNING": "⏳",
            "SUCCEEDED": "✅",
            "FAILED": "❌",
            "CANCELLED": "⚠️"
        }.get(job["status"], "❓")
        
        print(f"\n{status_emoji} {job['job_name']}")
        print(f"   Status: {job['status']}")
        print(f"   Started: {job['started_at']}")
        if job['completed_at']:
            print(f"   Completed: {job['completed_at']}")
        if job['tuned_model_name']:
            print(f"   Tuned Model: {job['tuned_model_name']}")


def interactive_tuning():
    """Interactive mode for starting tuning with user prompts."""
    print("\n🔧 Interactive Tuning Setup")
    print("-" * 40)
    
    # Check training data
    if not TRAIN_JSONL.exists():
        print(f"\n❌ Training data not found: {TRAIN_JSONL}")
        print("   Run the data pipeline first to generate training data.")
        return
    
    # Count training examples
    with open(TRAIN_JSONL, 'r') as f:
        train_count = sum(1 for line in f if line.strip())
    
    valid_count = 0
    if VALID_JSONL.exists():
        with open(VALID_JSONL, 'r') as f:
            valid_count = sum(1 for line in f if line.strip())
    
    print(f"\n📊 Training Data:")
    print(f"   Training examples: {train_count}")
    print(f"   Validation examples: {valid_count}")
    
    if train_count < 10:
        print(f"\n⚠️ Warning: Only {train_count} training examples.")
        print("   Recommended minimum is 100+ for good results.")
    
    # Confirm configuration
    print(f"\n⚙️ Tuning Configuration:")
    print(f"   Base model: {model_config.base_model}")
    print(f"   Epochs: {model_config.epochs}")
    print(f"   Adapter size: {model_config.adapter_size}")
    print(f"   Learning rate multiplier: {model_config.learning_rate_multiplier}")
    
    confirm = input("\nStart tuning with this configuration? [y/N]: ")
    if confirm.lower() != 'y':
        print("Tuning cancelled.")
        return
    
    # Upload data and start tuning
    train_uri, valid_uri = upload_training_data()
    start_tuning_job(train_uri, valid_uri)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Vertex AI Tuning Orchestrator for Childcare Classification"
    )
    parser.add_argument(
        "--start",
        action="store_true",
        help="Start a new tuning job"
    )
    parser.add_argument(
        "--status",
        action="store_true", 
        help="Check status of latest tuning job"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all tuning jobs"
    )
    parser.add_argument(
        "--job",
        type=str,
        help="Specific job resource name for status check"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive mode with prompts"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Vertex AI Tuning Orchestrator")
    print("=" * 60)
    
    # Check configuration
    if gcp_config.project_id == "your-gcp-project-id":
        print("\n❌ GCP Project ID not configured")
        print("   Update config.py or set GCP_PROJECT_ID environment variable")
        sys.exit(1)
    
    if args.start or args.status:
        if not VERTEX_AI_AVAILABLE:
            print("\n❌ Vertex AI SDK required")
            print("   Run: pip install google-cloud-aiplatform")
            sys.exit(1)
        initialize_vertex_ai()
    
    if args.start:
        if args.interactive:
            interactive_tuning()
        else:
            train_uri, valid_uri = upload_training_data()
            start_tuning_job(train_uri, valid_uri)
    
    elif args.status:
        check_job_status(args.job)
    
    elif args.list:
        list_tuning_jobs()
    
    elif args.interactive:
        if not VERTEX_AI_AVAILABLE:
            print("\n❌ Vertex AI SDK required")
            sys.exit(1)
        initialize_vertex_ai()
        interactive_tuning()
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
