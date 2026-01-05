"""
Dataset Builder for Childcare Observation Classification Training.

This module generates JSONL training and validation datasets from canonical
observations, formatted for Vertex AI supervised fine-tuning.

Supports both text-only and multimodal (image + text) formats.

Usage:
    from dataset_builder import DatasetBuilder
    
    builder = DatasetBuilder()
    builder.build_from_observations(observations, output_dir)
"""

import json
import random
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from collections import defaultdict
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    data_config,
    model_config,
    CLASSIFICATION_PROMPT_TEMPLATE,
    CLASSIFICATION_PROMPT_TEMPLATE_MULTIMODAL,
    SYSTEM_INSTRUCTION,
    TRAIN_JSONL,
    VALID_JSONL,
)
from src.observation_transformer import CanonicalObservation


@dataclass
class TrainingExample:
    """
    A single training example for fine-tuning.
    
    This represents one prompt-response pair for the model to learn from.
    """
    input_text: str
    target_text: str
    image_gcs_uri: Optional[str] = None  # For multimodal
    
    def to_text_only_dict(self) -> dict:
        """Convert to text-only JSONL format."""
        return {
            "input_text": self.input_text,
            "output_text": self.target_text
        }
    
    def to_multimodal_dict(self) -> dict:
        """Convert to multimodal JSONL format for Gemini."""
        example = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": self.input_text}
                    ]
                },
                {
                    "role": "model", 
                    "parts": [
                        {"text": self.target_text}
                    ]
                }
            ]
        }
        
        # Add image if available
        if self.image_gcs_uri:
            example["contents"][0]["parts"].insert(0, {
                "fileData": {
                    "fileUri": self.image_gcs_uri,
                    "mimeType": self._get_mime_type(self.image_gcs_uri)
                }
            })
        
        return example
    
    def _get_mime_type(self, uri: str) -> str:
        """Determine MIME type from file extension."""
        ext = uri.lower().split('.')[-1]
        mime_types = {
            'jpg': 'image/jpeg',
            'jpeg': 'image/jpeg',
            'png': 'image/png',
            'gif': 'image/gif',
            'webp': 'image/webp',
        }
        return mime_types.get(ext, 'image/jpeg')


@dataclass
class DatasetSplit:
    """Container for train/validation split results."""
    train: list[TrainingExample]
    validation: list[TrainingExample]
    
    def get_statistics(self) -> dict:
        """Get statistics about the split."""
        return {
            "train_count": len(self.train),
            "validation_count": len(self.validation),
            "total": len(self.train) + len(self.validation),
            "train_ratio": len(self.train) / (len(self.train) + len(self.validation)) if self.train else 0
        }


class DatasetBuilder:
    """
    Builds training and validation datasets from canonical observations.
    
    Handles:
    - Prompt generation with taxonomy context
    - Train/validation splitting with stratification
    - Output in Vertex AI compatible JSONL format
    """
    
    def __init__(
        self,
        multimodal: bool = None,
        train_split: float = None,
        random_seed: int = None
    ):
        """
        Initialize the dataset builder.
        
        Args:
            multimodal: Whether to generate multimodal examples (default from config)
            train_split: Ratio for training data (default from config)
            random_seed: Seed for reproducible splits (default from config)
        """
        self.multimodal = multimodal if multimodal is not None else model_config.multimodal
        self.train_split = train_split or data_config.train_split
        self.random_seed = random_seed or data_config.random_seed
        
        # Use appropriate prompt template
        if self.multimodal:
            self.prompt_template = CLASSIFICATION_PROMPT_TEMPLATE_MULTIMODAL
        else:
            self.prompt_template = CLASSIFICATION_PROMPT_TEMPLATE
    
    def build_from_observations(
        self,
        observations: list[CanonicalObservation],
        output_dir: Optional[Path] = None,
        stratify_by: str = None
    ) -> DatasetSplit:
        """
        Build training and validation datasets from observations.
        
        Args:
            observations: List of canonical observations
            output_dir: Where to save JSONL files (optional)
            stratify_by: Field to stratify by ("domain", "attribute", or "progression")
            
        Returns:
            DatasetSplit with train and validation examples
        """
        stratify_by = stratify_by or data_config.stratify_by
        
        # Generate training examples
        examples = [self._observation_to_example(obs) for obs in observations]
        
        # Split with stratification
        split = self._stratified_split(
            examples=list(zip(observations, examples)),
            stratify_by=stratify_by
        )
        
        # Save if output directory specified
        if output_dir:
            self._save_datasets(split, output_dir)
        
        return split
    
    def build_from_file(
        self,
        input_file: Path,
        output_dir: Optional[Path] = None
    ) -> DatasetSplit:
        """
        Build datasets from a JSONL file of canonical observations.
        
        Args:
            input_file: Path to canonical observations JSONL
            output_dir: Where to save output files
            
        Returns:
            DatasetSplit with train and validation examples
        """
        observations = []
        
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    obs = CanonicalObservation(**data)
                    observations.append(obs)
        
        return self.build_from_observations(observations, output_dir)
    
    def _observation_to_example(self, obs: CanonicalObservation) -> TrainingExample:
        """Convert a canonical observation to a training example."""
        
        # Generate input prompt
        input_text = self.prompt_template.format(notes=obs.notes)
        
        # Generate target JSON
        target = obs.get_classification_json()
        target_text = json.dumps(target, ensure_ascii=False)
        
        # Get image URI if multimodal and available
        image_uri = None
        if self.multimodal and obs.photo_uris:
            image_uri = obs.photo_uris[0]  # Use first photo
        
        return TrainingExample(
            input_text=input_text,
            target_text=target_text,
            image_gcs_uri=image_uri
        )
    
    def _stratified_split(
        self,
        examples: list[tuple[CanonicalObservation, TrainingExample]],
        stratify_by: str
    ) -> DatasetSplit:
        """
        Perform stratified train/validation split.
        
        Ensures each stratum has proportional representation in both sets.
        """
        random.seed(self.random_seed)
        
        # Group by stratification key
        groups = defaultdict(list)
        for obs, example in examples:
            if stratify_by == "domain":
                key = obs.domain_key
            elif stratify_by == "attribute":
                key = f"{obs.domain_key}_{obs.attribute_id}"
            elif stratify_by == "progression":
                key = obs.progression_title
            else:
                key = "all"  # No stratification
            
            groups[key].append(example)
        
        train = []
        validation = []
        
        # Split each group proportionally
        for key, group_examples in groups.items():
            random.shuffle(group_examples)
            
            split_idx = max(1, int(len(group_examples) * self.train_split))
            
            # Ensure at least 1 in validation if possible
            if len(group_examples) > 1:
                split_idx = min(split_idx, len(group_examples) - 1)
            
            train.extend(group_examples[:split_idx])
            validation.extend(group_examples[split_idx:])
        
        # Shuffle final datasets
        random.shuffle(train)
        random.shuffle(validation)
        
        return DatasetSplit(train=train, validation=validation)
    
    def _save_datasets(self, split: DatasetSplit, output_dir: Path):
        """Save train and validation datasets to JSONL files."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        train_path = output_dir / "train.jsonl"
        valid_path = output_dir / "valid.jsonl"
        
        # Determine format based on multimodal setting
        if self.multimodal:
            self._save_multimodal_jsonl(split.train, train_path)
            self._save_multimodal_jsonl(split.validation, valid_path)
        else:
            self._save_text_jsonl(split.train, train_path)
            self._save_text_jsonl(split.validation, valid_path)
        
        print(f"💾 Saved training data: {train_path} ({len(split.train)} examples)")
        print(f"💾 Saved validation data: {valid_path} ({len(split.validation)} examples)")
    
    def _save_text_jsonl(self, examples: list[TrainingExample], path: Path):
        """Save examples in text-only JSONL format."""
        with open(path, 'w', encoding='utf-8') as f:
            for example in examples:
                f.write(json.dumps(example.to_text_only_dict(), ensure_ascii=False) + "\n")
    
    def _save_multimodal_jsonl(self, examples: list[TrainingExample], path: Path):
        """Save examples in multimodal JSONL format for Gemini."""
        with open(path, 'w', encoding='utf-8') as f:
            for example in examples:
                f.write(json.dumps(example.to_multimodal_dict(), ensure_ascii=False) + "\n")


def analyze_dataset_distribution(
    observations: list[CanonicalObservation]
) -> dict:
    """
    Analyze the distribution of observations across taxonomy categories.
    
    Useful for understanding class balance before training.
    """
    stats = {
        "total": len(observations),
        "by_domain": defaultdict(int),
        "by_progression": defaultdict(int),
        "by_attribute": defaultdict(int),
        "by_domain_progression": defaultdict(lambda: defaultdict(int)),
        "has_photos": 0,
        "no_photos": 0,
    }
    
    for obs in observations:
        stats["by_domain"][obs.domain_key] += 1
        stats["by_progression"][obs.progression_title] += 1
        stats["by_attribute"][obs.attribute_name[:50]] += 1
        stats["by_domain_progression"][obs.domain_key][obs.progression_title] += 1
        
        if obs.photo_uris:
            stats["has_photos"] += 1
        else:
            stats["no_photos"] += 1
    
    # Convert defaultdicts to regular dicts for JSON serialization
    stats["by_domain"] = dict(stats["by_domain"])
    stats["by_progression"] = dict(stats["by_progression"])
    stats["by_attribute"] = dict(stats["by_attribute"])
    stats["by_domain_progression"] = {
        k: dict(v) for k, v in stats["by_domain_progression"].items()
    }
    
    return stats


if __name__ == "__main__":
    # Demo with sample data
    print("=" * 60)
    print("Dataset Builder Demo")
    print("=" * 60)
    
    # Create sample canonical observations
    sample_observations = [
        CanonicalObservation(
            observation_id="obs_001",
            notes="Emma was climbing on the playground structure today. She showed excellent balance as she navigated the climbing wall.",
            photo_uris=["gs://bucket/photos/obs_001.jpg"],
            domain_id=1,
            domain_key="physical_development",
            domain_name="My Growing Body (Physical Development)",
            attribute_id=2,
            attribute_name="Uses large muscles to explore & interact with the environment",
            progression_title="Progressing",
            progression_description="Develops an increasing awareness of body & space",
        ),
        CanonicalObservation(
            observation_id="obs_002",
            notes="During circle time, Jackson shared a story about his weekend trip. He spoke clearly and took turns listening.",
            photo_uris=[],
            domain_id=3,
            domain_key="language_literacy",
            domain_name="How I Communicate (Language & Literacy)",
            attribute_id=15,
            attribute_name="Understands & uses social & conversational rules",
            progression_title="Advancing",
            progression_description="Maintains multi-turn conversations",
        ),
        CanonicalObservation(
            observation_id="obs_003",
            notes="Lily worked on a puzzle with Maya. When they both wanted the same piece, Lily suggested they take turns.",
            photo_uris=[],
            domain_id=2,
            domain_key="social_emotional_development",
            domain_name="My Feelings & Relationships (Social & Emotional Development)",
            attribute_id=7,
            attribute_name="Develops positive relationships with other children",
            progression_title="Progressing",
            progression_description="Joins in play with other children",
        ),
        CanonicalObservation(
            observation_id="obs_004",
            notes="Today Alex discovered that his tower of blocks kept falling. He tried different arrangements before finding one that was stable.",
            photo_uris=["gs://bucket/photos/obs_004.jpg"],
            domain_id=4,
            domain_key="cognitive_development",
            domain_name="My Growing Brain (Cognitive Development)",
            attribute_id=24,
            attribute_name="Uses problem-solving strategies",
            progression_title="Developing",
            progression_description="Tries different strategies",
        ),
    ]
    
    # Analyze distribution
    print("\n📊 Dataset Distribution:")
    stats = analyze_dataset_distribution(sample_observations)
    print(f"   Total observations: {stats['total']}")
    print(f"   With photos: {stats['has_photos']}")
    print(f"   Without photos: {stats['no_photos']}")
    print(f"\n   By domain:")
    for domain, count in stats['by_domain'].items():
        print(f"      {domain}: {count}")
    print(f"\n   By progression:")
    for prog, count in stats['by_progression'].items():
        print(f"      {prog}: {count}")
    
    # Build datasets (text-only mode for demo)
    print("\n" + "=" * 60)
    print("Building Text-Only Dataset")
    print("=" * 60)
    
    builder = DatasetBuilder(multimodal=False)
    
    from config import TRAINING_DATA_DIR
    split = builder.build_from_observations(
        sample_observations,
        output_dir=TRAINING_DATA_DIR
    )
    
    print(f"\n📊 Split Statistics:")
    stats = split.get_statistics()
    print(f"   Train: {stats['train_count']}")
    print(f"   Validation: {stats['validation_count']}")
    print(f"   Train ratio: {stats['train_ratio']:.2%}")
    
    # Show sample examples
    if split.train:
        print(f"\n📝 Sample Training Example:")
        example = split.train[0]
        print(f"   Input (truncated): {example.input_text[:200]}...")
        print(f"   Target: {example.target_text}")
    
    print(f"\n✅ Dataset building complete!")
