"""
Observation Transformer for Childcare Classification Training.

This module transforms historical observation exports into a canonical format
ready for JSONL training data generation. Since historical data is already
accurately classified according to the taxonomy, this focuses on:
- Validating observation structure and required fields
- Enriching with human-readable names from taxonomy
- Outputting standardized observation records

Usage:
    from observation_transformer import ObservationTransformer
    
    transformer = ObservationTransformer(taxonomy)
    canonical_observations = transformer.transform(raw_observations)
"""

import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Iterator, Any
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.taxonomy_parser import load_taxonomy, NormalizedTaxonomy


@dataclass
class CanonicalObservation:
    """
    Standardized observation record for training data generation.
    
    This represents a single observation with all required fields
    for generating training prompts and labels.
    """
    # Unique identifier
    observation_id: str
    
    # Observation content
    notes: str
    photo_uris: list[str]  # GCS URIs to photos
    
    # Classification (from taxonomy)
    domain_id: int
    domain_key: str
    domain_name: str
    attribute_id: int
    attribute_name: str
    progression_title: str  # "Emerging", "Developing", etc.
    progression_description: str
    
    # Optional metadata
    child_age_months: Optional[int] = None
    observation_date: Optional[str] = None
    educator_id: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    def get_classification_json(self) -> dict:
        """Get the classification as a JSON-serializable dict (for training labels)."""
        return {
            "domain_key": self.domain_key,
            "domain_name": self.domain_name,
            "attribute_name": self.attribute_name,
            "progression_title": self.progression_title,
            "progression_description": self.progression_description
        }


@dataclass
class TransformResult:
    """Result of transforming observation data."""
    successful: list[CanonicalObservation]
    failed: list[tuple[dict, str]]  # (original_record, error_message)
    
    @property
    def success_count(self) -> int:
        return len(self.successful)
    
    @property
    def failure_count(self) -> int:
        return len(self.failed)
    
    @property
    def total_count(self) -> int:
        return self.success_count + self.failure_count
    
    def get_summary(self) -> dict:
        return {
            "total": self.total_count,
            "successful": self.success_count,
            "failed": self.failure_count,
            "success_rate": self.success_count / self.total_count if self.total_count > 0 else 0
        }


class ObservationTransformer:
    """
    Transforms raw observation exports into canonical format.
    
    Supports multiple input formats and field mappings.
    """
    
    def __init__(
        self, 
        taxonomy: Optional[NormalizedTaxonomy] = None,
        field_mapping: Optional[dict[str, str]] = None
    ):
        """
        Initialize the transformer.
        
        Args:
            taxonomy: Pre-loaded taxonomy (loads default if not provided)
            field_mapping: Optional mapping from source field names to canonical names
        """
        self.taxonomy = taxonomy or load_taxonomy()
        
        # Default field mapping (can be overridden)
        self.field_mapping = field_mapping or {
            # Source field -> Canonical field
            "id": "observation_id",
            "observationId": "observation_id",
            "observation_id": "observation_id",
            
            "notes": "notes",
            "observationNotes": "notes",
            "narrative": "notes",
            "text": "notes",
            
            "photoUri": "photo_uris",
            "photoUris": "photo_uris",
            "photo_uri": "photo_uris",
            "photo_uris": "photo_uris",
            "images": "photo_uris",
            
            "domainId": "domain_id",
            "domain_id": "domain_id",
            
            "attributeId": "attribute_id",
            "attribute_id": "attribute_id",
            
            "progressionTitle": "progression_title",
            "progression_title": "progression_title",
            "progressionLevel": "progression_title",
            
            "progressionDescription": "progression_description",
            "progression_description": "progression_description",
            
            "childAgeMonths": "child_age_months",
            "child_age_months": "child_age_months",
            "age_months": "child_age_months",
            
            "observationDate": "observation_date",
            "observation_date": "observation_date",
            "date": "observation_date",
            
            "educatorId": "educator_id",
            "educator_id": "educator_id",
        }
    
    def transform(
        self, 
        observations: list[dict],
        require_photos: bool = False
    ) -> TransformResult:
        """
        Transform a list of raw observation records.
        
        Args:
            observations: List of raw observation dictionaries
            require_photos: If True, observations without photos are considered failed
            
        Returns:
            TransformResult with successful and failed observations
        """
        successful = []
        failed = []
        
        for raw_obs in observations:
            try:
                canonical = self._transform_single(raw_obs, require_photos)
                successful.append(canonical)
            except Exception as e:
                failed.append((raw_obs, str(e)))
        
        return TransformResult(successful=successful, failed=failed)
    
    def transform_file(
        self, 
        file_path: Path, 
        require_photos: bool = False
    ) -> TransformResult:
        """
        Transform observations from a JSON or JSONL file.
        
        Args:
            file_path: Path to input file (.json or .jsonl)
            require_photos: If True, observations without photos are considered failed
            
        Returns:
            TransformResult with successful and failed observations
        """
        observations = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            if file_path.suffix == '.jsonl':
                # JSONL format: one JSON object per line
                for line in f:
                    line = line.strip()
                    if line:
                        observations.append(json.loads(line))
            else:
                # Regular JSON: expect a list or object with 'observations' key
                data = json.load(f)
                if isinstance(data, list):
                    observations = data
                elif isinstance(data, dict) and 'observations' in data:
                    observations = data['observations']
                else:
                    raise ValueError("Expected a list or object with 'observations' key")
        
        return self.transform(observations, require_photos)
    
    def _transform_single(
        self, 
        raw: dict, 
        require_photos: bool
    ) -> CanonicalObservation:
        """Transform a single observation record."""
        
        # Map fields from source to canonical names
        mapped = self._map_fields(raw)
        
        # Validate required fields
        self._validate_required_fields(mapped)
        
        # Get taxonomy data
        domain_id = int(mapped['domain_id'])
        attribute_id = int(mapped['attribute_id'])
        
        domain = self.taxonomy.get_domain(domain_id)
        if not domain:
            raise ValueError(f"Invalid domain_id: {domain_id}")
        
        attribute = self.taxonomy.get_attribute(attribute_id)
        if not attribute:
            raise ValueError(f"Invalid attribute_id: {attribute_id}")
        
        # Verify attribute belongs to domain
        attr_domain = self.taxonomy.get_attribute_domain(attribute_id)
        if not attr_domain or attr_domain.id != domain_id:
            raise ValueError(
                f"Attribute {attribute_id} does not belong to domain {domain_id}"
            )
        
        # Get progression details
        progression_title = mapped.get('progression_title', '')
        if progression_title not in ['Emerging', 'Developing', 'Progressing', 'Advancing', 'Refining']:
            raise ValueError(f"Invalid progression_title: {progression_title}")
        
        # Get progression description (use provided or lookup from taxonomy)
        progression_description = mapped.get('progression_description', '')
        if not progression_description:
            prog = attribute.get_first_progression_for_level(progression_title)
            if prog:
                progression_description = prog.description
        
        # Handle photo URIs
        photo_uris = self._normalize_photo_uris(mapped.get('photo_uris', []))
        if require_photos and not photo_uris:
            raise ValueError("Observation has no photos but require_photos=True")
        
        # Build canonical observation
        return CanonicalObservation(
            observation_id=str(mapped['observation_id']),
            notes=str(mapped['notes']).strip(),
            photo_uris=photo_uris,
            domain_id=domain_id,
            domain_key=domain.key,
            domain_name=domain.name,
            attribute_id=attribute_id,
            attribute_name=attribute.name,
            progression_title=progression_title,
            progression_description=progression_description,
            child_age_months=self._safe_int(mapped.get('child_age_months')),
            observation_date=mapped.get('observation_date'),
            educator_id=mapped.get('educator_id')
        )
    
    def _map_fields(self, raw: dict) -> dict:
        """Map source fields to canonical field names."""
        mapped = {}
        
        for source_field, value in raw.items():
            canonical_field = self.field_mapping.get(source_field)
            if canonical_field:
                # Handle potential conflicts by preferring non-None values
                if canonical_field not in mapped or mapped[canonical_field] is None:
                    mapped[canonical_field] = value
            else:
                # Keep unmapped fields as-is (for extensibility)
                mapped[source_field] = value
        
        return mapped
    
    def _validate_required_fields(self, mapped: dict):
        """Validate that all required fields are present."""
        required = ['observation_id', 'notes', 'domain_id', 'attribute_id', 'progression_title']
        
        missing = [f for f in required if f not in mapped or mapped[f] is None]
        if missing:
            raise ValueError(f"Missing required fields: {missing}")
        
        # Validate notes is not empty
        notes = str(mapped['notes']).strip()
        if not notes:
            raise ValueError("Notes field is empty")
    
    def _normalize_photo_uris(self, photo_uris: Any) -> list[str]:
        """Normalize photo URIs to a list of strings."""
        if photo_uris is None:
            return []
        
        if isinstance(photo_uris, str):
            # Single URI as string
            return [photo_uris] if photo_uris.strip() else []
        
        if isinstance(photo_uris, list):
            return [str(uri) for uri in photo_uris if uri]
        
        return []
    
    def _safe_int(self, value: Any) -> Optional[int]:
        """Safely convert a value to int."""
        if value is None:
            return None
        try:
            return int(value)
        except (ValueError, TypeError):
            return None


def save_canonical_observations(
    observations: list[CanonicalObservation],
    output_path: Path,
    format: str = "jsonl"
):
    """
    Save canonical observations to a file.
    
    Args:
        observations: List of canonical observations
        output_path: Where to save
        format: "jsonl" or "json"
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        if format == "jsonl":
            for obs in observations:
                f.write(json.dumps(obs.to_dict(), ensure_ascii=False) + "\n")
        else:
            f.write(json.dumps(
                [obs.to_dict() for obs in observations],
                indent=2,
                ensure_ascii=False
            ))


if __name__ == "__main__":
    # Demo with sample data
    print("=" * 60)
    print("Observation Transformer Demo")
    print("=" * 60)
    
    # Sample observation data (mimicking what historical export might look like)
    sample_observations = [
        {
            "id": "obs_001",
            "notes": "Emma was climbing on the playground structure today. She showed excellent balance as she navigated the climbing wall, using both hands and feet to grip the holds. She paused at the top with a big smile before sliding down.",
            "photoUri": "gs://bucket/photos/obs_001.jpg",
            "domainId": 1,
            "attributeId": 2,
            "progressionTitle": "Progressing",
            "childAgeMonths": 42,
            "observationDate": "2024-03-15"
        },
        {
            "id": "obs_002", 
            "notes": "During circle time, Jackson shared a story about his weekend trip to the zoo. He spoke clearly and took turns listening when other children asked questions about the animals he saw.",
            "domainId": 3,
            "attributeId": 15,
            "progressionTitle": "Advancing"
        },
        {
            "id": "obs_003",
            "notes": "Lily worked on a puzzle with her friend Maya. When they both wanted the same piece, Lily suggested they take turns. She waited patiently for Maya to try first.",
            "domainId": 2,
            "attributeId": 7,
            "progressionTitle": "Progressing"
        }
    ]
    
    # Transform
    transformer = ObservationTransformer()
    result = transformer.transform(sample_observations)
    
    print(f"\n📊 Transform Results:")
    print(f"   Total: {result.total_count}")
    print(f"   Successful: {result.success_count}")
    print(f"   Failed: {result.failure_count}")
    
    if result.failed:
        print(f"\n❌ Failed observations:")
        for raw, error in result.failed:
            print(f"   {raw.get('id', 'unknown')}: {error}")
    
    print(f"\n✅ Successful transformations:")
    for obs in result.successful:
        print(f"\n   Observation: {obs.observation_id}")
        print(f"   Domain: {obs.domain_key} ({obs.domain_name})")
        print(f"   Attribute: {obs.attribute_name}")
        print(f"   Progression: {obs.progression_title}")
        print(f"   Notes preview: {obs.notes[:100]}...")
    
    # Save to file
    from config import PROCESSED_DATA_DIR
    output_file = PROCESSED_DATA_DIR / "sample_canonical.jsonl"
    save_canonical_observations(result.successful, output_file)
    print(f"\n💾 Saved to: {output_file}")
