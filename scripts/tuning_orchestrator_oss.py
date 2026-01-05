"""
OSS Tuning Orchestrator for Childcare Classification.

This script submits a Custom Job to Vertex AI to fine-tune an open-source model
(e.g., Microsoft Phi-3 Vision) using a custom training script (`src/train_phi.py`).

Configuration:
- Uses a pre-built PyTorch GPU container.
- Requests L4 GPUs (g2-standard-4) or A100s (a2-highgpu-1g).
- Mounts GCS bucket for output.

Usage:
    python scripts/tuning_orchestrator_oss.py \
        --model_id "microsoft/Phi-3-vision-128k-instruct" \
        --accelerator_type "NVIDIA_L4"
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import gcp_config

try:
    from google.cloud import aiplatform
except ImportError:
    print("❌ Vertex AI SDK required: pip install google-cloud-aiplatform")
    sys.exit(1)

# Training Image (PyTorch 2.1 + Python 3.10 + CUDA)
TRAINING_IMAGE_URI = "us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-1.py310:latest"

def main():
    parser = argparse.ArgumentParser(description="Submit OSS Tuning Job to Vertex AI")
    parser.add_argument("--model_id", type=str, default="microsoft/Phi-3-vision-128k-instruct")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--accelerator_type", type=str, default="NVIDIA_L4", 
                        choices=["NVIDIA_L4", "NVIDIA_TESLA_A100", "NVIDIA_TESLA_T4"])
    parser.add_argument("--machine_type", type=str, default=None, help="Override machine type")
    parser.add_argument("--staging_bucket", type=str, default=None)
    args = parser.parse_args()

    # Initialize Vertex AI
    aiplatform.init(
        project=gcp_config.project_id,
        location=gcp_config.region,
        staging_bucket=args.staging_bucket or f"gs://{gcp_config.bucket_name}/staging"
    )

    # Determine Machine Type based on Accelerator
    # L4 -> g2-standard-4 (1 GPU)
    # A100 -> a2-highgpu-1g (1 GPU)
    # T4 -> n1-standard-4 (1 GPU)
    if not args.machine_type:
        if args.accelerator_type == "NVIDIA_L4":
            machine_type = "g2-standard-4"
        elif args.accelerator_type == "NVIDIA_TESLA_A100":
            machine_type = "a2-highgpu-1g"
        elif args.accelerator_type == "NVIDIA_TESLA_T4":
            machine_type = "n1-standard-4"
        else:
            machine_type = "n1-standard-4"
    else:
        machine_type = args.machine_type

    print(f"🚀 Submitting Tuning Job for {args.model_id}")
    print(f"   Accelerator: {args.accelerator_type}")
    print(f"   Machine: {machine_type}")

    # Define the job
    # We pass the training script from src/train_phi.py
    # We need to ensure we install dependencies at runtime since we use a generic image
    
    # Path to our training script
    script_path = "src/train_phi.py" 
    
    # Dependencies to install in the container
    requirements = [
        "transformers>=4.40.0",
        "peft>=0.10.0",
        "bitsandbytes>=0.43.0",
        "accelerate>=0.29.0",
        "datasets>=2.19.0",
        "trl>=0.8.0",
        "Pillow",
        "google-cloud-storage",
        "scipy" # often needed for flash attn fallback or other parts
    ]

    job_name = f"oss-tuning-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    # Verify script exists
    if not Path(script_path).exists():
        print(f"❌ Custom training script not found at {script_path}")
        sys.exit(1)
        
    # We need to construct the training data URI
    # Assuming the data pipeline has run and provided train.jsonl
    # We use the most recent or fixed path. For robustness, let's look for it or assert it.
    
    train_data_uri = f"gs://{gcp_config.bucket_name}/{gcp_config.training_data_gcs_path}/train.jsonl"
    # Note: In a real scenario, we might want to check if this exists or use the timestamped one.
    # We'll use the "latest" logic or require user to upload first effectively.
    # Ideally, we pass the exact URI. For "Easy Button" usage, we'll try to find the latest.
    
    # Just pass the base path and let the script find it or fail? 
    # Let's try to be smart and verify.
    # Actually, simpler is better: Just assume the user has run the gcp_setup and data pipeline
    # which uploads to a known location, or we upload it right here?
    # `tuning_orchestrator.py` (Gemini) does upload explicitly. Let's do that here too.
    
    from google.cloud import storage
    def upload_dataset_if_needed():
        local_path = Path("data/training/train.jsonl")
        if not local_path.exists():
            print(f"❌ Local dataset not found: {local_path}")
            sys.exit(1)
        
        client = storage.Client(project=gcp_config.project_id)
        bucket = client.bucket(gcp_config.bucket_name)
        blob = bucket.blob(f"{gcp_config.training_data_gcs_path}/latest/train.jsonl")
        blob.upload_from_filename(str(local_path))
        return f"gs://{gcp_config.bucket_name}/{gcp_config.training_data_gcs_path}/latest/train.jsonl"

    print("📤 Uploading latest dataset...")
    try:
        data_uri = upload_dataset_if_needed()
    except Exception as e:
        print(f"⚠️ Warning: utilizing hardcoded URI due to upload error: {e}")
        data_uri = f"gs://{gcp_config.bucket_name}/{gcp_config.training_data_gcs_path}/latest/train.jsonl"

    print(f"   Data URI: {data_uri}")

    job = aiplatform.CustomJob.from_local_script(
        display_name=job_name,
        script_path=script_path,
        container_uri=TRAINING_IMAGE_URI,
        requirements=requirements,
        args=[
            f"--model_id={args.model_id}",
            f"--train_data_uri={data_uri}",
            "--output_dir=/gcs/" + gcp_config.bucket_name + "/oss_models/" + job_name,
            f"--epochs={args.epochs}"
        ],
        replica_count=1,
        machine_type=machine_type,
        accelerator_type=args.accelerator_type,
        accelerator_count=1,
    )

    print("🚀 submitting job...")
    job.run(sync=False) # Run asynchronously
    
    print(f"\n✅ Job submitted: {job.resource_name}")
    print(f"   You can monitor it in the Vertex AI Console.")
    print(f"   Outputs will be saved to: gs://{gcp_config.bucket_name}/oss_models/{job_name}")

if __name__ == "__main__":
    main()
