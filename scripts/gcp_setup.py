"""
GCP Infrastructure Setup Script for Childcare Observation Classification.

This script automates the creation of GCP resources needed for the fine-tuning pipeline:
- GCS bucket for training data and model artifacts
- Service accounts with appropriate IAM roles
- Vertex AI API enablement

Usage:
    python scripts/gcp_setup.py --setup          # Create all resources
    python scripts/gcp_setup.py --validate       # Validate existing setup
    python scripts/gcp_setup.py --cleanup        # Remove created resources
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import gcp_config, model_config


# =============================================================================
# CONSTANTS
# =============================================================================

# APIs to enable
REQUIRED_APIS = [
    "aiplatform.googleapis.com",
    "storage.googleapis.com",
    "iam.googleapis.com",
]

# Service accounts to create
SERVICE_ACCOUNTS = {
    "data-prep": {
        "name": "childcare-data-prep",
        "display_name": "Childcare Data Preparation",
        "roles": [
            "roles/storage.objectAdmin",  # Read/write to GCS
        ]
    },
    "tuning": {
        "name": "childcare-tuning",
        "display_name": "Childcare Model Tuning",
        "roles": [
            "roles/aiplatform.user",  # Create and run tuning jobs
            "roles/storage.objectViewer",  # Read training data
        ]
    },
    "inference": {
        "name": "childcare-inference",
        "display_name": "Childcare Model Inference",
        "roles": [
            "roles/aiplatform.user",  # Call endpoints
        ]
    },
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def run_gcloud(args: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run a gcloud command and return the result."""
    cmd = ["gcloud"] + args
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"  ❌ Error: {result.stderr}")
    return result


def run_gsutil(args: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run a gsutil command and return the result."""
    cmd = ["gsutil"] + args
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"  ❌ Error: {result.stderr}")
    return result


def check_gcloud_auth() -> bool:
    """Check if gcloud is authenticated."""
    result = run_gcloud(["auth", "list", "--filter=status:ACTIVE", "--format=value(account)"], check=False)
    if result.returncode != 0 or not result.stdout.strip():
        print("❌ Not authenticated with gcloud. Run: gcloud auth login")
        return False
    print(f"✅ Authenticated as: {result.stdout.strip()}")
    return True


def check_project() -> bool:
    """Check if project is set correctly."""
    result = run_gcloud(["config", "get-value", "project"], check=False)
    current_project = result.stdout.strip()
    
    if current_project != gcp_config.project_id:
        print(f"⚠️  Current project: {current_project}")
        print(f"   Expected project: {gcp_config.project_id}")
        print(f"   Run: gcloud config set project {gcp_config.project_id}")
        return False
    
    print(f"✅ Project: {current_project}")
    return True


# =============================================================================
# SETUP FUNCTIONS
# =============================================================================

def enable_apis():
    """Enable required GCP APIs."""
    print("\n📡 Enabling required APIs...")
    
    for api in REQUIRED_APIS:
        print(f"\n  Enabling {api}...")
        result = run_gcloud([
            "services", "enable", api,
            f"--project={gcp_config.project_id}"
        ])
        if result.returncode == 0:
            print(f"  ✅ {api} enabled")
        else:
            print(f"  ⚠️  {api} may already be enabled or there was an error")


def create_bucket():
    """Create GCS bucket for training data and artifacts."""
    print(f"\n🪣 Creating GCS bucket: {gcp_config.bucket_name}...")
    
    # Check if bucket exists
    result = run_gsutil(["ls", f"gs://{gcp_config.bucket_name}"], check=False)
    if result.returncode == 0:
        print(f"  ✅ Bucket already exists: gs://{gcp_config.bucket_name}")
        return
    
    # Create bucket with uniform access
    result = run_gsutil([
        "mb",
        "-p", gcp_config.project_id,
        "-l", gcp_config.region,
        "-b", "on",  # Uniform bucket-level access
        f"gs://{gcp_config.bucket_name}"
    ])
    
    if result.returncode == 0:
        print(f"  ✅ Bucket created: gs://{gcp_config.bucket_name}")
        
        # Create folder structure
        for folder in ["training-data", "model-artifacts", "observations"]:
            run_gsutil(["cp", "-", f"gs://{gcp_config.bucket_name}/{folder}/.keep"], check=False)
    else:
        print(f"  ❌ Failed to create bucket")


def create_service_accounts():
    """Create service accounts with appropriate roles."""
    print("\n👤 Creating service accounts...")
    
    for sa_type, sa_config in SERVICE_ACCOUNTS.items():
        sa_email = f"{sa_config['name']}@{gcp_config.project_id}.iam.gserviceaccount.com"
        
        print(f"\n  Creating {sa_config['display_name']}...")
        
        # Check if SA exists
        result = run_gcloud([
            "iam", "service-accounts", "describe", sa_email,
            f"--project={gcp_config.project_id}"
        ], check=False)
        
        if result.returncode == 0:
            print(f"  ✅ Service account already exists: {sa_email}")
        else:
            # Create SA
            result = run_gcloud([
                "iam", "service-accounts", "create", sa_config['name'],
                f"--display-name={sa_config['display_name']}",
                f"--project={gcp_config.project_id}"
            ])
            
            if result.returncode == 0:
                print(f"  ✅ Created: {sa_email}")
            else:
                continue
        
        # Assign roles
        for role in sa_config['roles']:
            print(f"    Assigning role: {role}")
            run_gcloud([
                "projects", "add-iam-policy-binding", gcp_config.project_id,
                f"--member=serviceAccount:{sa_email}",
                f"--role={role}",
                "--condition=None"
            ], check=False)


def setup_vertex_ai():
    """Initialize Vertex AI settings."""
    print("\n🤖 Configuring Vertex AI...")
    
    # Set default region for Vertex AI
    run_gcloud([
        "config", "set", "ai/region", gcp_config.region
    ])
    
    print(f"  ✅ Vertex AI region set to: {gcp_config.region}")
    
    # Verify Vertex AI access
    result = run_gcloud([
        "ai", "endpoints", "list",
        f"--region={gcp_config.region}",
        f"--project={gcp_config.project_id}",
        "--format=value(name)"
    ], check=False)
    
    if result.returncode == 0:
        print("  ✅ Vertex AI access verified")
    else:
        print("  ⚠️  Could not verify Vertex AI access. This may be normal if no endpoints exist yet.")


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_setup() -> bool:
    """Validate that all resources are properly set up."""
    print("\n🔍 Validating GCP Setup...")
    all_valid = True
    
    # Check bucket
    print("\n  Checking GCS bucket...")
    result = run_gsutil(["ls", f"gs://{gcp_config.bucket_name}"], check=False)
    if result.returncode == 0:
        print(f"  ✅ Bucket exists: gs://{gcp_config.bucket_name}")
    else:
        print(f"  ❌ Bucket not found: gs://{gcp_config.bucket_name}")
        all_valid = False
    
    # Check APIs
    print("\n  Checking APIs...")
    for api in REQUIRED_APIS:
        result = run_gcloud([
            "services", "list",
            "--enabled",
            f"--filter=config.name:{api}",
            "--format=value(config.name)",
            f"--project={gcp_config.project_id}"
        ], check=False)
        
        if api in result.stdout:
            print(f"  ✅ {api} enabled")
        else:
            print(f"  ❌ {api} not enabled")
            all_valid = False
    
    # Check service accounts
    print("\n  Checking service accounts...")
    for sa_type, sa_config in SERVICE_ACCOUNTS.items():
        sa_email = f"{sa_config['name']}@{gcp_config.project_id}.iam.gserviceaccount.com"
        result = run_gcloud([
            "iam", "service-accounts", "describe", sa_email,
            f"--project={gcp_config.project_id}"
        ], check=False)
        
        if result.returncode == 0:
            print(f"  ✅ {sa_config['display_name']}: {sa_email}")
        else:
            print(f"  ❌ {sa_config['display_name']} not found")
            all_valid = False
    
    return all_valid


# =============================================================================
# CLEANUP FUNCTIONS
# =============================================================================

def cleanup():
    """Remove all created resources (use with caution!)."""
    print("\n⚠️  This will delete all created resources!")
    confirm = input("Type 'DELETE' to confirm: ")
    
    if confirm != "DELETE":
        print("Cleanup cancelled.")
        return
    
    print("\n🧹 Cleaning up resources...")
    
    # Delete bucket
    print(f"\n  Deleting bucket: gs://{gcp_config.bucket_name}")
    run_gsutil(["rm", "-r", f"gs://{gcp_config.bucket_name}"], check=False)
    
    # Delete service accounts
    for sa_type, sa_config in SERVICE_ACCOUNTS.items():
        sa_email = f"{sa_config['name']}@{gcp_config.project_id}.iam.gserviceaccount.com"
        print(f"\n  Deleting service account: {sa_email}")
        run_gcloud([
            "iam", "service-accounts", "delete", sa_email,
            f"--project={gcp_config.project_id}",
            "--quiet"
        ], check=False)
    
    print("\n✅ Cleanup complete")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="GCP Infrastructure Setup for Childcare Observation Classification"
    )
    parser.add_argument(
        "--setup", 
        action="store_true", 
        help="Create all GCP resources"
    )
    parser.add_argument(
        "--validate", 
        action="store_true", 
        help="Validate existing setup"
    )
    parser.add_argument(
        "--cleanup", 
        action="store_true", 
        help="Remove created resources (use with caution!)"
    )
    
    args = parser.parse_args()
    
    # Print configuration
    print("=" * 60)
    print("GCP Setup for Childcare Observation Classification")
    print("=" * 60)
    print(f"\nProject ID: {gcp_config.project_id}")
    print(f"Region: {gcp_config.region}")
    print(f"Bucket: {gcp_config.bucket_name}")
    
    # Check prerequisites
    if not check_gcloud_auth():
        sys.exit(1)
    
    if not check_project():
        if gcp_config.project_id == "your-gcp-project-id":
            print("\n❌ Please update GCP_PROJECT_ID in config.py or set environment variable")
            sys.exit(1)
    
    # Execute requested action
    if args.setup:
        enable_apis()
        create_bucket()
        create_service_accounts()
        setup_vertex_ai()
        print("\n" + "=" * 60)
        print("✅ GCP Setup Complete!")
        print("=" * 60)
        
    elif args.validate:
        if validate_setup():
            print("\n✅ All resources are properly configured")
        else:
            print("\n❌ Some resources are missing or misconfigured")
            sys.exit(1)
            
    elif args.cleanup:
        cleanup()
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
