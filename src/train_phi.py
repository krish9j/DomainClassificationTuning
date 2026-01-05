"""
Training Script for Phi-3 Vision on Vertex AI.

This script runs INSIDE the Vertex AI training container. It:
1. Downloads data from GCS.
2. Loads the Phi-3 Vision model (quantized).
3. Fine-tunes using LoRA.
4. Saves the adapter back to GCS.
"""

import os
import argparse
import json
import torch
from pathlib import Path
from google.cloud import storage
from PIL import Image
from io import BytesIO
from transformers import (
    AutoModelForCausalLM, 
    AutoProcessor, 
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import Dataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="microsoft/Phi-3-vision-128k-instruct")
    parser.add_argument("--train_data_uri", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="/gcs/output") # Fuse mounted or local
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    return parser.parse_args()

def download_gcs_file(uri: str, local_path: str):
    """Downloads a file from GCS."""
    client = storage.Client()
    bucket_name = uri.split("/")[2]
    blob_name = "/".join(uri.split("/")[3:])
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(local_path)

def load_image_from_gcs(uri: str) -> Image.Image:
    """Loads an image directly from GCS."""
    if not uri.startswith("gs://"):
        return None # Handle local files or None
    
    client = storage.Client()
    bucket_name = uri.split("/")[2]
    blob_name = "/".join(uri.split("/")[3:])
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    data = blob.download_as_bytes()
    return Image.open(BytesIO(data)).convert("RGB")

def prepare_dataset(data_path: str, processor):
    """
    Reads the Gemini-formatted JSONL and converts to Phi-3 training format.
    Expects JSONL with: {"input_text": "...", "target_text": "...", "image_gcs_uri": "..."}
    """
    records = []
    with open(data_path, 'r') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    # Convert to HF Dataset
    def format_item(item):
        # Construct chat template for Phi-3
        # user: <|user|>\n<|image_1|>\n{prompt}<|end|>\n
        # assist: <|assistant|>\n{response}<|end|>
        
        prompt = item.get("input_text", "")
        # Clean up prompt if it contains markdown headers from our template
        if "## Observation Notes:" in prompt:
             # Basic cleanup to just get the core text if needed, or keep as is
             pass

        image_uri = item.get("image_gcs_uri")
        
        messages = [
            {"role": "user", "content": f"<|image_1|>\n{prompt}"},
            {"role": "assistant", "content": item["target_text"]}
        ]
        
        # Load image
        image = None
        if image_uri:
            try:
                image = load_image_from_gcs(image_uri)
            except Exception as e:
                print(f"Failed to load image {image_uri}: {e}")
        
        # Note: In a real script we might preload images or load lazily. 
        # For simplicity here we assume dataset fits in memory or we'd map it.
        # But SFTTrainer expects text. For VLM, we usually need a custom data collator
        # or pre-process inputs.
        
        # Using processor to format inputs directly
        inputs = processor(text=prompt, images=image, return_tensors="pt")
        
        return {
            "text": processor.tokenizer.apply_chat_template(messages, tokenize=False),
            "image": image
        }

    dataset = Dataset.from_list([format_item(r) for r in records])
    return dataset

def main():
    args = parse_args()
    
    # 1. Download Data
    local_data_path = "train.jsonl"
    download_gcs_file(args.train_data_uri, local_data_path)
    
    # 2. Load Model & Processor
    print(f"Loading model: {args.model_id}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, 
        quantization_config=bnb_config, 
        trust_remote_code=True,
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    
    # 3. Apply LoRA
    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], # Phi-3 target modules
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # 4. Prepare Data
    # Note: For Vision Language Models, standard SFTTrainer needs careful setup with collators.
    # This is a simplified placeholder. In production, use processor as part of data collator.
    dataset = prepare_dataset(local_data_path, processor)
    
    def collate_fn(examples):
        texts = [ex["text"] for ex in examples]
        images = [ex["image"] for ex in examples]
        # Filter out None images if any
        images = [img for img in images if img is not None]
        
        if not images: images = None
        
        batch = processor(text=texts, images=images, return_tensors="pt", padding=True)
        
        # Create labels (same as input_ids but masked is typical, SFTTrainer handles this usually)
        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        batch["labels"] = labels
        
        return batch

    # 5. Training Arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.learning_rate,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
    )
    
    # 6. Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        dataset_text_field="text", # Used if no collator, but we use custom
        peft_config=peft_config,
        args=training_args,
        data_collator=collate_fn,
        packing=False
    )
    
    # 7. Train
    print("Starting training...")
    trainer.train()
    
    # 8. Save
    print("Saving model...")
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
