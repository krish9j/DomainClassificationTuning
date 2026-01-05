
## Cost Estimation

When running this pipeline on GCP, costs are incurred in three areas. These estimates are approximate and vary by region and usage.

### 1. Fine-Tuning (One-Time Cost)
Vertex AI charges for the training job based on node hours or token count.
*   **Gemini 1.5 Flash Tuning**: Typically priced by token count of the training dataset.
*   **Estimate**: For a dataset of ~500-1000 examples (text-only), the cost is usually **$5 - $20 USD** per run. Multimodal datasets will be higher due to image token processing.

### 2. Inference & Serving (Ongoing Cost)
This is usually the largest component.
*   **Endpoint Hosting**: Deployed tuned models typically incur an **hourly** hosting charge for the active endpoint, regardless of traffic.
*   **Machine Type**: The deployment uses a standard machine node (e.g., `n1-standard-4`).
*   **Estimate**: An `n1-standard-4` node is approximately **$0.20 - $0.30 per hour** (~$150-$200/month) if left running 24/7.
*   **Recommendation**: Undeploy the model when not in use (e.g., development/testing/weekends) using `python scripts/endpoint_deployer.py --undeploy`.

### 3. Storage
*   **GCS Bucket**: Storing images and JSONL files.
*   **Estimate**: Negligible (<$1/month) for datasets under 10GB.
