# Childcare Classification Operations Runbook

This guide covers ongoing operations, troubleshooting, and maintenance of the deployed model.

## Model Usage

### Using the Client Library
Internal services should use the provided Python client:

```python
from client.inference_client import ObservationClassifier

# Initialize (uses config.py settings by default)
classifier = ObservationClassifier()

# Classify
result = classifier.classify(
    notes="Child was stacking blocks...",
    image_uri="gs://bucket/photo.jpg"
)

print(f"Domain: {result.domain_name}")
print(f"Progression: {result.progression_title}")
```

### Error Handling
The client handles common errors:
- **No prediction**: Retries or returns error object.
- **Parsing failure**: Returns `ClassificationResult` with `domain_key="error"`. Applications should check for this.

## Monitoring & Maintenance

### Checking Endpoint Status
View currently deployed models and traffic:
```bash
python scripts/endpoint_deployer.py --status
```

### Updating the Model
To deploy a new version of the model:
1.  Train the new model (see `runbook_training.md`).
2.  Deploy to the **same** endpoint to enable traffic splitting or seamless rollout:
    ```bash
    python scripts/endpoint_deployer.py --deploy --model projects/.../models/...
    ```
    *Note: The script currently defaults to 100% traffic to the new model.*

### Removing Old Models
To undeploy models to save costs:
```bash
python scripts/endpoint_deployer.py --undeploy
```

## Troubleshooting

### Issue: "Quota Exceeded" during Tuning
*   **Cause**: specific region might be out of accelerator capacity (TPU/GPU).
*   **Solution**: Change `region` in `config.py` (e.g., to `us-west1` or `europe-west4`) and re-run.

### Issue: Low Accuracy
*   **Check Data**: Ensure `train.jsonl` has cleaned notes and accurate labels.
*   **Check Splits**: Verify `valid.jsonl` isn't leaking into training.
*   **Hyperparameters**: Try increasing `epochs` in `config.py`.

### Issue: Endpoint Timeout
*   **Cause**: Model cold start or complex input.
*   **Solution**: The client has built-in retries. For high-traffic, increase `min_replica_count` in `config.py`.

## Cost Management
For detailed cost estimates of fine-tuning and inference, see [Cost Estimates](cost_estimates.md).
