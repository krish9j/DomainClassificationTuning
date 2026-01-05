# OSS Model Training Runbook (Phi-3 Vision)

This guide details how to fine-tune open-source models (like Microsoft Phi-3 Vision or Llama 3) on Vertex AI using Custom Jobs.

## Prerequisites

1.  **GCP Quota**: Ensure your project has quota for **L4 GPUs** (`NVIDIA_L4`) or **A100 GPUs** in your selected region.
    *   *Check Console*: IAM & Admin -> Quotas -> Filter by "NVIDIA L4" or "Restricted image training GPUs".

2.  **Docker Auth**: Your local environment generally doesn't need Docker, but the job runs in a pre-built Vertex AI container.

## Step 1: Prepare Data

Follow the standard data preparation steps from the main runbook. The OSS pipeline uses the **same** `train.jsonl` format. The training script automatically converts it to the model-specific format.

1.  Transform and Build:
    ```bash
    python src/observation_transformer.py
    python src/dataset_builder.py
    ```

## Step 2: Launch Training

Run the OSS orchestrator script. This submits a job to Vertex AI that provisions a GPU machine and runs your training logic.

```bash
python scripts/tuning_orchestrator_oss.py \
    --model_id "microsoft/Phi-3-vision-128k-instruct" \
    --accelerator_type "NVIDIA_L4" \
    --epochs 3
```

### Configuration Options
*   `--model_id`: Any Hugging Face model ID (must be accessible publicly or via token).
*   `--accelerator_type`:
    *   `NVIDIA_L4` (Cheapest, good for <10B models).
    *   `NVIDIA_TESLA_A100` (Fastest, required for >13B models).
*   `--machine_type`: Optional override (e.g., `g2-standard-4`).

## Step 3: Monitor Job

1.  **Console**: Go to [Vertex AI Training](https://console.cloud.google.com/vertex-ai/training/custom-jobs).
2.  **Logs**: Click on the job -> View Logs. You will see the training progress bar from the `transformers` library.

## Step 4: Access Artifacts

Once the job succeeds:
1.  Navigate to your GCS bucket: `gs://<your-bucket>/oss_models/oss-tuning-<timestamp>/`.
2.  You will find:
    *   `adapter_model.bin` (The LoRA weights)
    *   `adapter_config.json`
    *   `special_tokens_map.json` (Tokenizer files)

## Step 5: Inference (Local or Cloud)

To run inference with the fine-tuned model, you need to load the base model and apply the adapter.

**Example Python Snippet:**
```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base_model_id = "microsoft/Phi-3-vision-128k-instruct"
adapter_path = "gs://your-bucket/oss_models/..."

# Load Base
model = AutoModelForCausalLM.from_pretrained(base_model_id, trust_remote_code=True)

# Load Adapter
model = PeftModel.from_pretrained(model, adapter_path)

# Run Inference...
```
