# Childcare Classification Model Training Runbook

This guide details how to prepare data, train the model, and deploy it using the provided scripts.

## Prerequisites

1.  **Environment Setup**
    Ensure you have the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

2.  **GCP Authentication**
    Authenticate with Google Cloud:
    ```bash
    gcloud auth login
    gcloud config set project <YOUR_PROJECT_ID>
    ```

3.  **Infrastructure Initialization**
    If running for the first time, set up the GCP resources (buckets, service accounts):
    ```bash
    python scripts/gcp_setup.py --setup
    ```

## Step 1: Data Preparation

1.  **Place Data**
    Put your historical observation export or raw data files in `data/raw/`.

2.  **Transform Data**
    Convert raw data into the canonical format. Uses `src/observation_transformer.py`.
    ```bash
    python src/observation_transformer.py
    ```
    *Note: You may need to modify the script to point to your specific input file.*

3.  **Build Dataset**
    Generate `train.jsonl` and `valid.jsonl` for Vertex AI.
    ```bash
    python src/dataset_builder.py
    ```
    This creates files in `data/training/`.

## Step 2: Fine-Tuning

1.  **Start Tuning Job**
    Run the orchestrator to upload data and start the Vertex AI tuning job.
    ```bash
    python scripts/tuning_orchestrator.py --start
    ```
    
    Or use interactive mode to verify configuration:
    ```bash
    python scripts/tuning_orchestrator.py --interactive
    ```

2.  **Monitor Progress**
    Check the status of the job:
    ```bash
    python scripts/tuning_orchestrator.py --status
    ```
    *Tuning typically takes 1-3 hours depending on dataset size.*

## Step 3: Deployment

1.  **Find Tuned Model**
    Once tuning succeeds, the model resource name is saved in `tuning_jobs.json`.

2.  **Deploy to Endpoint**
    Deploy the model to a prediction endpoint:
    ```bash
    python scripts/endpoint_deployer.py --deploy
    ```
    This process takes about 15-20 minutes.

3.  **Verify Deployment**
    Check endpoint status:
    ```bash
    python scripts/endpoint_deployer.py --status
    ```

## Step 4: Verification

1.  **Run Evaluation**
    Run the evaluation script against the validation set:
    ```bash
    python scripts/evaluation.py --dataset data/training/valid.jsonl --endpoint "childcare-classifier-endpoint"
    ```

2.  **Check Report**
    Review `evaluation_report.json` for detailed metrics and failure cases.
