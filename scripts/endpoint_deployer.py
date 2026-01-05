"""
Endpoint Deployer for Childcare Observation Classification.

This script handles deployment of tuned models to Vertex AI endpoints
for serving predictions.

Features:
- Creates and configures endpoints
- Deploys tuned models with traffic allocation
- Sets up IAM permissions
- Manages endpoint lifecycle

Usage:
    python scripts/endpoint_deployer.py --deploy --model <model_resource>
    python scripts/endpoint_deployer.py --status
    python scripts/endpoint_deployer.py --undeploy
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

from config import gcp_config, model_config, PROJECT_ROOT

# Vertex AI imports
try:
    from google.cloud import aiplatform
    import vertexai
    VERTEX_AI_AVAILABLE = True
except ImportError:
    VERTEX_AI_AVAILABLE = False


# =============================================================================
# CONSTANTS
# =============================================================================

ENDPOINTS_FILE = PROJECT_ROOT / "endpoints.json"


# =============================================================================
# ENDPOINT TRACKING
# =============================================================================

def load_endpoints() -> dict:
    """Load the endpoints tracking file."""
    if ENDPOINTS_FILE.exists():
        with open(ENDPOINTS_FILE, 'r') as f:
            return json.load(f)
    return {"endpoints": []}


def save_endpoints(data: dict):
    """Save the endpoints tracking file."""
    with open(ENDPOINTS_FILE, 'w') as f:
        json.dump(data, f, indent=2)


def record_endpoint(
    endpoint_name: str,
    endpoint_resource_name: str,
    model_resource_name: str,
    deployed_model_id: str
):
    """Record a new endpoint deployment."""
    data = load_endpoints()
    
    endpoint_record = {
        "endpoint_name": endpoint_name,
        "resource_name": endpoint_resource_name,
        "model_resource_name": model_resource_name,
        "deployed_model_id": deployed_model_id,
        "deployed_at": datetime.now().isoformat(),
        "status": "DEPLOYED"
    }
    
    data["endpoints"].append(endpoint_record)
    save_endpoints(data)
    
    return endpoint_record


def get_latest_endpoint() -> Optional[dict]:
    """Get the latest deployed endpoint."""
    data = load_endpoints()
    active = [e for e in data["endpoints"] if e["status"] == "DEPLOYED"]
    return active[-1] if active else None


# =============================================================================
# DEPLOYMENT FUNCTIONS
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
    aiplatform.init(
        project=gcp_config.project_id,
        location=gcp_config.region
    )
    print(f"✅ Initialized Vertex AI")
    print(f"   Project: {gcp_config.project_id}")
    print(f"   Region: {gcp_config.region}")


def get_or_create_endpoint(endpoint_name: str = None) -> aiplatform.Endpoint:
    """
    Get existing endpoint or create a new one.
    
    Args:
        endpoint_name: Optional custom endpoint name
        
    Returns:
        Vertex AI Endpoint object
    """
    endpoint_name = endpoint_name or model_config.endpoint_display_name
    
    # Check for existing endpoint with same name
    endpoints = aiplatform.Endpoint.list(
        filter=f'display_name="{endpoint_name}"'
    )
    
    if endpoints:
        print(f"   Using existing endpoint: {endpoint_name}")
        return endpoints[0]
    
    # Create new endpoint
    print(f"   Creating new endpoint: {endpoint_name}")
    endpoint = aiplatform.Endpoint.create(
        display_name=endpoint_name,
        description="Childcare observation domain classification endpoint"
    )
    
    print(f"   ✅ Endpoint created: {endpoint.resource_name}")
    return endpoint


def deploy_model(
    model_resource_name: str,
    endpoint: Optional[aiplatform.Endpoint] = None,
    traffic_percentage: int = 100
) -> tuple[aiplatform.Endpoint, str]:
    """
    Deploy a tuned model to an endpoint.
    
    Args:
        model_resource_name: The tuned model resource name
        endpoint: Optional existing endpoint to deploy to
        traffic_percentage: Percentage of traffic to route to this model
        
    Returns:
        Tuple of (endpoint, deployed_model_id)
    """
    print(f"\n🚀 Deploying model to endpoint...")
    print(f"   Model: {model_resource_name}")
    
    # Get or create endpoint
    if endpoint is None:
        endpoint = get_or_create_endpoint()
    
    # Get the model
    model = aiplatform.Model(model_resource_name)
    
    # Deploy the model
    print(f"\n   Deploying model (this may take 10-20 minutes)...")
    
    deployed_model = endpoint.deploy(
        model=model,
        deployed_model_display_name=model_config.tuned_model_display_name,
        traffic_percentage=traffic_percentage,
        machine_type="n1-standard-4",  # Adjust based on needs
        min_replica_count=model_config.min_replica_count,
        max_replica_count=model_config.max_replica_count,
    )
    
    # Get deployed model ID
    deployed_model_id = None
    for dm in endpoint.list_models():
        if dm.display_name == model_config.tuned_model_display_name:
            deployed_model_id = dm.id
            break
    
    print(f"\n✅ Model deployed successfully!")
    print(f"   Endpoint: {endpoint.resource_name}")
    if deployed_model_id:
        print(f"   Deployed model ID: {deployed_model_id}")
    
    # Record the deployment
    record_endpoint(
        endpoint_name=endpoint.display_name,
        endpoint_resource_name=endpoint.resource_name,
        model_resource_name=model_resource_name,
        deployed_model_id=deployed_model_id or "unknown"
    )
    
    return endpoint, deployed_model_id


def undeploy_model(endpoint_resource_name: str = None, deployed_model_id: str = None):
    """
    Undeploy a model from an endpoint.
    
    Args:
        endpoint_resource_name: Endpoint to undeploy from (default: latest)
        deployed_model_id: Specific model to undeploy (default: all)
    """
    # Get endpoint info
    if endpoint_resource_name is None:
        latest = get_latest_endpoint()
        if not latest:
            print("No deployed endpoints found.")
            return
        endpoint_resource_name = latest["resource_name"]
        deployed_model_id = deployed_model_id or latest.get("deployed_model_id")
    
    print(f"\n🔄 Undeploying model...")
    print(f"   Endpoint: {endpoint_resource_name}")
    
    endpoint = aiplatform.Endpoint(endpoint_resource_name)
    
    # Undeploy
    if deployed_model_id:
        print(f"   Model ID: {deployed_model_id}")
        endpoint.undeploy(deployed_model_id=deployed_model_id)
    else:
        # Undeploy all models
        endpoint.undeploy_all()
    
    print(f"✅ Model undeployed")
    
    # Update record
    data = load_endpoints()
    for ep in data["endpoints"]:
        if ep["resource_name"] == endpoint_resource_name:
            ep["status"] = "UNDEPLOYED"
            ep["undeployed_at"] = datetime.now().isoformat()
    save_endpoints(data)


def delete_endpoint(endpoint_resource_name: str = None):
    """
    Delete an endpoint entirely.
    
    Args:
        endpoint_resource_name: Endpoint to delete (default: latest)
    """
    if endpoint_resource_name is None:
        latest = get_latest_endpoint()
        if not latest:
            print("No endpoints found.")
            return
        endpoint_resource_name = latest["resource_name"]
    
    print(f"\n🗑️ Deleting endpoint: {endpoint_resource_name}")
    
    confirm = input("Are you sure? This cannot be undone. [y/N]: ")
    if confirm.lower() != 'y':
        print("Cancelled.")
        return
    
    endpoint = aiplatform.Endpoint(endpoint_resource_name)
    
    # Undeploy all models first
    try:
        endpoint.undeploy_all()
    except Exception as e:
        print(f"   Warning: {e}")
    
    # Delete endpoint
    endpoint.delete()
    
    print(f"✅ Endpoint deleted")
    
    # Update record
    data = load_endpoints()
    for ep in data["endpoints"]:
        if ep["resource_name"] == endpoint_resource_name:
            ep["status"] = "DELETED"
            ep["deleted_at"] = datetime.now().isoformat()
    save_endpoints(data)


def check_endpoint_status(endpoint_resource_name: str = None):
    """
    Check the status of an endpoint.
    
    Args:
        endpoint_resource_name: Endpoint to check (default: latest)
    """
    if endpoint_resource_name is None:
        latest = get_latest_endpoint()
        if not latest:
            print("No deployed endpoints found.")
            return
        endpoint_resource_name = latest["resource_name"]
    
    print(f"\n📊 Endpoint Status")
    print(f"   Resource: {endpoint_resource_name}")
    
    try:
        endpoint = aiplatform.Endpoint(endpoint_resource_name)
        
        print(f"   Display Name: {endpoint.display_name}")
        print(f"   Description: {endpoint.description or 'N/A'}")
        
        # List deployed models
        deployed_models = endpoint.list_models()
        print(f"\n   Deployed Models ({len(deployed_models)}):")
        
        for dm in deployed_models:
            print(f"      - {dm.display_name}")
            print(f"        ID: {dm.id}")
            print(f"        Model: {dm.model}")
        
        if not deployed_models:
            print(f"      No models deployed")
        
        # Print prediction endpoint
        print(f"\n   Prediction endpoint:")
        print(f"   {endpoint.resource_name}")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")


def list_endpoints():
    """List all recorded endpoints."""
    data = load_endpoints()
    
    if not data["endpoints"]:
        print("No endpoints recorded.")
        return
    
    print(f"\n📋 Endpoints ({len(data['endpoints'])} total)")
    print("-" * 80)
    
    for ep in data["endpoints"]:
        status_emoji = {
            "DEPLOYED": "✅",
            "UNDEPLOYED": "⚠️",
            "DELETED": "🗑️"
        }.get(ep["status"], "❓")
        
        print(f"\n{status_emoji} {ep['endpoint_name']}")
        print(f"   Status: {ep['status']}")
        print(f"   Resource: {ep['resource_name']}")
        print(f"   Model: {ep['model_resource_name']}")
        print(f"   Deployed: {ep['deployed_at']}")


def get_model_from_tuning_jobs() -> Optional[str]:
    """Get the latest successful tuned model from tuning jobs."""
    tuning_jobs_file = PROJECT_ROOT / "tuning_jobs.json"
    
    if not tuning_jobs_file.exists():
        return None
    
    with open(tuning_jobs_file, 'r') as f:
        data = json.load(f)
    
    # Find latest successful job with a tuned model
    for job in reversed(data.get("jobs", [])):
        if job.get("status") == "SUCCEEDED" and job.get("tuned_model_name"):
            return job["tuned_model_name"]
    
    return None


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Vertex AI Endpoint Deployer for Childcare Classification"
    )
    parser.add_argument(
        "--deploy",
        action="store_true",
        help="Deploy a model to an endpoint"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model resource name to deploy (default: latest from tuning)"
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Check endpoint status"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all endpoints"
    )
    parser.add_argument(
        "--undeploy",
        action="store_true",
        help="Undeploy model from endpoint"
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete endpoint entirely"
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        help="Specific endpoint resource name"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Vertex AI Endpoint Deployer")
    print("=" * 60)
    
    # Check configuration
    if gcp_config.project_id == "your-gcp-project-id":
        print("\n❌ GCP Project ID not configured")
        print("   Update config.py or set GCP_PROJECT_ID environment variable")
        sys.exit(1)
    
    if any([args.deploy, args.status, args.undeploy, args.delete]):
        if not VERTEX_AI_AVAILABLE:
            print("\n❌ Vertex AI SDK required")
            print("   Run: pip install google-cloud-aiplatform")
            sys.exit(1)
        initialize_vertex_ai()
    
    if args.deploy:
        # Get model to deploy
        model_resource = args.model
        
        if not model_resource:
            model_resource = get_model_from_tuning_jobs()
            if model_resource:
                print(f"\n   Using model from tuning: {model_resource}")
            else:
                print("\n❌ No model specified and no tuned model found")
                print("   Run tuning first or specify --model")
                sys.exit(1)
        
        deploy_model(model_resource)
        
        print("\n" + "=" * 60)
        print("Deployment complete!")
        print("\nTest the endpoint with:")
        print("   python client/inference_client.py --test")
        print("=" * 60)
    
    elif args.status:
        check_endpoint_status(args.endpoint)
    
    elif args.list:
        list_endpoints()
    
    elif args.undeploy:
        undeploy_model(args.endpoint)
    
    elif args.delete:
        delete_endpoint(args.endpoint)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
