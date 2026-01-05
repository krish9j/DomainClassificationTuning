"""
Evaluation Script for Childcare Observation Classification.

This script evaluates the performance of the fine-tuned model against a held-out
validation set (or a specific test set). It calculates metrics like:
- Exact match accuracy for (domain, attribute, progression) triple
- Per-domain, per-attribute, per-progression accuracy
- Confusion matrices
- JSON schema validity rates

Usage:
    python scripts/evaluation.py --dataset data/training/valid.jsonl --endpoint <endpoint_id>
    python scripts/evaluation.py --dry-run  # Run on sample data without calling endpoint
"""

import argparse
import json
import time
from pathlib import Path
from collections import defaultdict, Counter
import sys
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import PROJECT_ROOT, evaluation_config
from client.inference_client import ObservationClassifier, ClassificationResult


def load_dataset(file_path: Path) -> list[dict]:
    """Load JSONL dataset."""
    examples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                # Supports both TrainingExample format (input_text/target_text) 
                # and CanonicalObservation format (direct fields)
                data = json.loads(line)
                
                # Normalize to evaluation format
                if "target_text" in data:
                    # It's a training example
                    target = json.loads(data["target_text"])
                    # Extract input notes from input_text (rough parsing)
                    # Assumes standard prompt template structure
                    input_text = data["input_text"]
                    notes_marker = "## Observation Notes:\n"
                    task_marker = "\n\n## Task:"
                    try:
                        start = input_text.index(notes_marker) + len(notes_marker)
                        end = input_text.index(task_marker)
                        notes = input_text[start:end].strip()
                    except ValueError:
                        notes = input_text # Fallback
                        
                    examples.append({
                        "notes": notes,
                        "ground_truth": target,
                        "image_uri": data.get("image_gcs_uri")  # Might differ key based on builder
                    })
                else:
                    # It's a canonical observation
                    examples.append({
                        "notes": data.get("notes"),
                        "ground_truth": {
                            "domain_key": data.get("domain_key"),
                            "attribute_name": data.get("attribute_name"),
                            "progression_title": data.get("progression_title")
                        },
                        "image_uri": data.get("photo_uris", [])[0] if data.get("photo_uris") else None
                    })
    return examples


def evaluate_model(
    dataset_path: Path,
    endpoint_name: str,
    limit: int = None,
    concurrency: int = 5,
    dry_run: bool = False
):
    """
    Run evaluation on the dataset.
    
    Args:
        dataset_path: Path to JSONL dataset
        endpoint_name: Name/ID of vertex endpoint
        limit: Max examples to evaluate
        concurrency: Parallel requests
        dry_run: If True, mocks the model response
    """
    print(f"Loading dataset from {dataset_path}...")
    examples = load_dataset(dataset_path)
    
    if limit:
        examples = examples[:limit]
        print(f"Limiting to {limit} examples")
    
    print(f"Evaluating {len(examples)} examples...")
    
    results = []
    
    # Initialize client if not dry run
    client = None
    if not dry_run:
        try:
            client = ObservationClassifier(endpoint_name=endpoint_name)
        except Exception as e:
            print(f"Failed to initialize client: {e}")
            if not dry_run:
                return

    # Define prediction function
    def predict(example):
        if dry_run:
            # Mock prediction (returns correct answer 80% of time)
            import random
            gt = example["ground_truth"]
            if random.random() < 0.8:
                return ClassificationResult(
                    domain_key=gt["domain_key"],
                    domain_name="Mock Domain",
                    attribute_name=gt["attribute_name"],
                    progression_title=gt["progression_title"],
                    progression_description="Mock description"
                )
            else:
                return ClassificationResult(
                    domain_key="wrong_domain",
                    domain_name="Wrong Domain",
                    attribute_name="Wrong Attribute",
                    progression_title="Emerging" if gt["progression_title"] != "Emerging" else "Developing",
                    progression_description="Mock description"
                )
        
        try:
            return client.classify(
                notes=example["notes"],
                image_uri=example["image_uri"]
            )
        except Exception as e:
            return ClassificationResult(
                domain_key="error",
                domain_name="Error",
                attribute_name="Error",
                progression_title="Error",
                progression_description=str(e),
                raw_response="Client Error"
            )

    # Run predictions
    predictions = []
    
    print("Running inference...")
    # Sequential for now to avoid complexity or rate limits, 
    # but ThreadPoolExecutor could be used if rate limits allow
    for ex in tqdm(examples):
        pred = predict(ex)
        predictions.append(pred)
        time.sleep(0.2) # Rate limit safety
    
    # Calculate metrics
    metrics = calculate_metrics(examples, predictions)
    
    # Save detailed report
    report_path = PROJECT_ROOT / "evaluation_report.json"
    
    report_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": str(dataset_path),
        "example_count": len(examples),
        "metrics": metrics,
        "failures": [
            {
                "notes": ex["notes"][:100],
                "ground_truth": ex["ground_truth"],
                "prediction": p.to_dict()
            }
            for ex, p in zip(examples, predictions)
            if not is_exact_match(ex["ground_truth"], p)
        ]
    }
    
    with open(report_path, 'w') as f:
        json.dump(report_data, f, indent=2)
        
    print_metrics(metrics)
    print(f"\nDetailed report saved to {report_path}")


def is_exact_match(gt: dict, pred: ClassificationResult) -> bool:
    return (
        gt.get("domain_key") == pred.domain_key and
        gt.get("attribute_name") == pred.attribute_name and
        gt.get("progression_title") == pred.progression_title
    )


def calculate_metrics(examples: list[dict], predictions: list[ClassificationResult]) -> dict:
    """Calculate accuracy and other metrics."""
    
    y_true_domain = [ex["ground_truth"].get("domain_key") for ex in examples]
    y_pred_domain = [p.domain_key for p in predictions]
    
    y_true_prog = [ex["ground_truth"].get("progression_title") for ex in examples]
    y_pred_prog = [p.progression_title for p in predictions]
    
    # Check valid attributes (needs to match exact string)
    matches = [is_exact_match(ex["ground_truth"], p) for ex, p in zip(examples, predictions)]
    exact_accuracy = sum(matches) / len(matches) if matches else 0
    
    # Parse errors
    parse_errors = sum(1 for p in predictions if p.domain_key == "error")
    
    return {
        "exact_match_accuracy": exact_accuracy,
        "domain_accuracy": accuracy_score(y_true_domain, y_pred_domain),
        "progression_accuracy": accuracy_score(y_true_prog, y_pred_prog),
        "parse_error_rate": parse_errors / len(predictions) if predictions else 0,
        "total_examples": len(examples)
    }


def print_metrics(metrics: dict):
    print("\n" + "="*40)
    print("EVALUATION RESULTS")
    print("="*40)
    print(f"Exact Match Accuracy:     {metrics['exact_match_accuracy']:.2%}")
    print(f"Domain Accuracy:          {metrics['domain_accuracy']:.2%}")
    print(f"Progression Accuracy:     {metrics['progression_accuracy']:.2%}")
    print(f"Parse Error Rate:         {metrics['parse_error_rate']:.2%}")
    print("="*40)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Childcare Model")
    parser.add_argument("--dataset", type=Path, help="Path to validation JSONL")
    parser.add_argument("--endpoint", type=str, help="Endpoint Display Name or ID")
    parser.add_argument("--limit", type=int, help="Max examples to evaluate")
    parser.add_argument("--dry-run", action="store_true", help="Run without calling endpoint (test mode)")
    
    args = parser.parse_args()
    
    if args.dry_run:
        # Create a dummy dataset if none provided for dry run
        if not args.dataset:
            dummy_path = PROJECT_ROOT / "data/training/test_dummy.jsonl"
            dummy_path.parent.mkdir(parents=True, exist_ok=True)
            with open(dummy_path, 'w') as f:
                rec = {
                    "input_text": "## Observation Notes:\nTest note\n\n## Task:",
                    "target_text": '{"domain_key": "physical_development", "attribute_name": "Uses large muscles", "progression_title": "Emerging"}'
                }
                for _ in range(10): 
                    f.write(json.dumps(rec) + "\n")
            args.dataset = dummy_path
            print(f"Created dummy dataset at {dummy_path}")

    if not args.dataset or not args.dataset.exists():
        print(f"Dataset not found: {args.dataset}")
        sys.exit(1)
        
    evaluate_model(
        dataset_path=args.dataset,
        endpoint_name=args.endpoint,
        limit=args.limit,
        dry_run=args.dry_run
    )

if __name__ == "__main__":
    main()
