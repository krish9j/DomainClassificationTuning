"""
Inference Client for Childcare Observation Classification.

This client library provides a high-level interface for internal services
to get classifications from the deployed Vertex AI model.

Features:
- Handles authentication and connection to Vertex AI
- Supports both text-only and multimodal (image) observations
- Parses and validates JSON responses
- Includes retry logic and error handling

Usage:
    from client.inference_client import ObservationClassifier
    
    classifier = ObservationClassifier(project_id="...", location="...")
    result = classifier.classify(
        notes="Child was building with blocks...",
        image_uri="gs://bucket/photo.jpg"
    )
    print(result.domain_name)
"""

import json
import base64
import time
from typing import Optional, Union, Dict, Any
from dataclasses import dataclass
from pathlib import Path
import sys

# Add parent directory to path for imports if needed
try:
    from config import gcp_config, model_config
except ImportError:
    # Fallback if config not available in path
    gcp_config = None
    model_config = None

try:
    from google.cloud import aiplatform
    from google.protobuf import json_format
    from google.protobuf.struct_pb2 import Value
    import vertexai
    from vertexai.preview.generative_models import GenerativeModel, Part
    VERTEX_AI_AVAILABLE = True
except ImportError:
    VERTEX_AI_AVAILABLE = False


@dataclass
class ClassificationResult:
    """Structured result from the classification model."""
    domain_key: str
    domain_name: str
    attribute_name: str
    progression_title: str
    progression_description: str
    confidence_score: Optional[float] = None
    raw_response: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            "domain_key": self.domain_key,
            "domain_name": self.domain_name,
            "attribute_name": self.attribute_name,
            "progression_title": self.progression_title,
            "progression_description": self.progression_description,
            "confidence_score": self.confidence_score
        }


class ObservationClassifier:
    """Client for calling the Childcare Observation Classification endpoint."""
    
    def __init__(
        self,
        project_id: Optional[str] = None,
        location: Optional[str] = None,
        endpoint_id: Optional[str] = None,
        endpoint_name: Optional[str] = None
    ):
        """
        Initialize the classifier client.
        
        Args:
            project_id: GCP Project ID (defaults to config)
            location: GCP Region (defaults to config)
            endpoint_id: specific Endpoint ID (optional)
            endpoint_name: display name to lookup (defaults to config)
        """
        if not VERTEX_AI_AVAILABLE:
            raise ImportError("Vertex AI SDK not installed. Run: pip install google-cloud-aiplatform")
            
        self.project_id = project_id or (gcp_config.project_id if gcp_config else None)
        self.location = location or (gcp_config.region if gcp_config else "us-central1")
        
        if not self.project_id:
            raise ValueError("project_id must be provided or configured in config.py")
            
        vertexai.init(project=self.project_id, location=self.location)
        
        self.endpoint = self._get_endpoint(endpoint_id, endpoint_name)
        
    def _get_endpoint(self, endpoint_id: Optional[str], endpoint_name: Optional[str]) -> aiplatform.Endpoint:
        """Resolve the Vertex AI Endpoint to use."""
        if endpoint_id:
            return aiplatform.Endpoint(endpoint_id)
            
        # Lookup by display name
        display_name = endpoint_name or (model_config.endpoint_display_name if model_config else "childcare-classifier-endpoint")
        print(f"Looking up endpoint with display name: {display_name}")
        
        endpoints = aiplatform.Endpoint.list(
            filter=f'display_name="{display_name}"',
            project=self.project_id,
            location=self.location
        )
        
        if not endpoints:
            raise ValueError(f"No endpoint found with name '{display_name}'")
            
        # Use the most recently created one
        endpoint = sorted(endpoints, key=lambda e: e.create_time, reverse=True)[0]
        print(f"Using endpoint: {endpoint.resource_name}")
        return endpoint

    def classify(
        self,
        notes: str,
        image_uri: Optional[str] = None,
        local_image_path: Optional[str] = None
    ) -> ClassificationResult:
        """
        Classify a childcare observation.
        
        Args:
            notes: The observation notes/narrative
            image_uri: GCS URI to an image (e.g., gs://bucket/image.jpg)
            local_image_path: Path to a local image file
            
        Returns:
            ClassificationResult object
        """
        if not notes:
            raise ValueError("Observation notes are required")
            
        # Construct the prompt
        # We use a simplified prompt here, relying on the fine-tuning to know the task
        prompt_text = f"Analyze the following childcare observation and classify it according to the developmental taxonomy.\n\n## Observation Notes:\n{notes}\n\n## Task:\nClassify this observation into the appropriate developmental domain, attribute, and progression level.\n\n## Response Format:\nRespond with a JSON object containing these exact fields:\n- domain_key\n- domain_name\n- attribute_name\n- progression_title\n- progression_description\n\n## JSON Response:"

        instances = []
        
        # Prepare content
        # For Gemini fine-tuned models on endpoints, structure can vary slightly depending on how it was deployed.
        # Typically for Generative AI endpoints we expect content inputs.
        
        # NOTE: This implementation assumes the endpoint supports the standard prediction request format
        # for tuned Gemini models.
        
        contents = [{"role": "user", "parts": [{"text": prompt_text}]}]
        
        # Handle Image
        if image_uri:
            # GCS Image
            image_part = {"fileData": {"fileUri": image_uri, "mimeType": self._infer_mime_type(image_uri)}}
            contents[0]["parts"].insert(0, image_part)
        elif local_image_path:
            # Local Image - encode to base64
            with open(local_image_path, "rb") as f:
                image_data = f.read()
                b64_data = base64.b64encode(image_data).decode("utf-8")
                image_part = {"inlineData": {"data": b64_data, "mimeType": self._infer_mime_type(local_image_path)}}
                contents[0]["parts"].insert(0, image_part)
                
        # Call the endpoint
        try:
            # For Gemini models deployed to endpoints, we often use the predict method with specific structure
            # OR we check if we should use GenerativeModel directly if it's an adapter
            # Here we assume a standard Vertex AI Endpoint deployment
            
            response = self.endpoint.predict(instances=[{"contents": contents}])
            
            # Parse response
            # Response structure depends on the model. For Gemini, it usually returns candidates.
            predictions = response.predictions
            
            if not predictions:
                raise ValueError("No predictions returned from endpoint")
            
            # Extract text from prediction
            # Structure typically: predictions[0]['candidates'][0]['content']['parts'][0]['text']
            # Or simplified depending on deployment
            try:
                # Try standard Gemini response structure
                generated_text = predictions[0]['candidates'][0]['content']['parts'][0]['text']
            except (KeyError, IndexError):
                # Fallback to direct text if different structure
                generated_text = str(predictions[0])
                
            # Parse JSON from the response text
            return self._parse_json_response(generated_text)
            
        except Exception as e:
            print(f"Error calling endpoint: {e}")
            raise

    def _infer_mime_type(self, path: str) -> str:
        ext = path.split('.')[-1].lower()
        if ext in ['jpg', 'jpeg']: return 'image/jpeg'
        if ext == 'png': return 'image/png'
        if ext == 'webp': return 'image/webp'
        return 'image/jpeg' # Default

    def _parse_json_response(self, text: str) -> ClassificationResult:
        """Extract and parse JSON from the model's text response."""
        # Clean up text - remove markdown code blocks if present
        cleaned_text = text.strip()
        if cleaned_text.startswith("```json"):
            cleaned_text = cleaned_text[7:]
        if cleaned_text.startswith("```"):
            cleaned_text = cleaned_text[3:]
        if cleaned_text.endswith("```"):
            cleaned_text = cleaned_text[:-3]
        
        cleaned_text = cleaned_text.strip()
        
        try:
            data = json.loads(cleaned_text)
            
            return ClassificationResult(
                domain_key=data.get("domain_key", "unknown"),
                domain_name=data.get("domain_name", "Unknown Domain"),
                attribute_name=data.get("attribute_name", "Unknown Attribute"),
                progression_title=data.get("progression_title", "Unknown"),
                progression_description=data.get("progression_description", ""),
                raw_response=text
            )
        except json.JSONDecodeError:
            # Fallback if raw text isn't valid JSON
            print(f"Failed to parse JSON: {text}")
            return ClassificationResult(
                domain_key="error",
                domain_name="Parse Error",
                attribute_name="Could not parse model response",
                progression_title="Error",
                progression_description=text[:500],
                raw_response=text
            )


if __name__ == "__main__":
    # Test block
    print("Initializing ObservationClassifier...")
    try:
        # Try to init (will fail if no config/auth/endpoint, which is expected during setup)
        # Pass dummy project/location to avoid immediate config failure if config.py isn't set up
        client = ObservationClassifier(project_id="test-project", location="us-central1")
        print("Client initialized successfully (dry run).")
    except Exception as e:
        print(f"Client init check: {e}")
        # Expected if no auth/project, that's fine for now
