"""
Taxonomy Parser and Normalizer for Childcare Observation Classification.

This module parses the learningdomains.json taxonomy file and creates
normalized lookup structures for efficient label generation and validation.
"""

import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import TAXONOMY_FILE, DOMAIN_ID_TO_KEY, PROGRESSION_LEVELS


@dataclass
class ProgressionPoint:
    """Represents a single progression point within an attribute."""
    id: int
    progression_points_id: int  # Level ID (1-5)
    title: str  # "Emerging", "Developing", etc.
    description: str
    sort_order: int
    description_order: str
    is_hidden: bool = False


@dataclass
class Attribute:
    """Represents an attribute within a domain."""
    id: int
    number: int
    name: str
    category_information: str
    sort_order: int
    progression_points: list[ProgressionPoint] = field(default_factory=list)
    
    def get_progressions_by_level(self, level: str) -> list[ProgressionPoint]:
        """Get all progression points for a specific level."""
        return [p for p in self.progression_points if p.title == level and not p.is_hidden]
    
    def get_first_progression_for_level(self, level: str) -> Optional[ProgressionPoint]:
        """Get the first (primary) progression point for a level."""
        progressions = self.get_progressions_by_level(level)
        return progressions[0] if progressions else None


@dataclass
class Domain:
    """Represents a developmental domain."""
    id: int
    name: str
    category_name: str
    category_title: str
    sort_order: int
    key: str = ""  # Normalized key like "physical_development"
    attributes: list[Attribute] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.key:
            self.key = DOMAIN_ID_TO_KEY.get(self.id, f"domain_{self.id}")
    
    def get_attribute_by_id(self, attr_id: int) -> Optional[Attribute]:
        """Get an attribute by its ID."""
        for attr in self.attributes:
            if attr.id == attr_id:
                return attr
        return None
    
    def get_attribute_by_name(self, name: str) -> Optional[Attribute]:
        """Get an attribute by its name (case-insensitive partial match)."""
        name_lower = name.lower()
        for attr in self.attributes:
            if name_lower in attr.name.lower():
                return attr
        return None


@dataclass
class NormalizedTaxonomy:
    """
    Complete normalized taxonomy with efficient lookup structures.
    
    Provides multiple ways to look up domains, attributes, and progressions
    for both training data generation and inference validation.
    """
    domains: list[Domain] = field(default_factory=list)
    
    # Lookup dictionaries (populated by build_lookups)
    domain_by_id: dict[int, Domain] = field(default_factory=dict)
    domain_by_key: dict[str, Domain] = field(default_factory=dict)
    attribute_by_id: dict[int, Attribute] = field(default_factory=dict)
    progression_by_id: dict[int, ProgressionPoint] = field(default_factory=dict)
    
    # Flat list of all valid (domain, attribute, progression) combinations
    valid_combinations: list[tuple[str, str, str]] = field(default_factory=list)
    
    def build_lookups(self):
        """Build all lookup dictionaries after domains are loaded."""
        self.domain_by_id = {}
        self.domain_by_key = {}
        self.attribute_by_id = {}
        self.progression_by_id = {}
        self.valid_combinations = []
        
        for domain in self.domains:
            self.domain_by_id[domain.id] = domain
            self.domain_by_key[domain.key] = domain
            
            for attr in domain.attributes:
                self.attribute_by_id[attr.id] = attr
                
                for prog in attr.progression_points:
                    self.progression_by_id[prog.id] = prog
                    
                    if not prog.is_hidden:
                        self.valid_combinations.append((
                            domain.key,
                            attr.name,
                            prog.title
                        ))
    
    def get_domain(self, identifier: int | str) -> Optional[Domain]:
        """Get domain by ID (int) or key (str)."""
        if isinstance(identifier, int):
            return self.domain_by_id.get(identifier)
        return self.domain_by_key.get(identifier)
    
    def get_attribute(self, attr_id: int) -> Optional[Attribute]:
        """Get attribute by ID."""
        return self.attribute_by_id.get(attr_id)
    
    def get_attribute_domain(self, attr_id: int) -> Optional[Domain]:
        """Get the domain that contains a specific attribute."""
        for domain in self.domains:
            if any(a.id == attr_id for a in domain.attributes):
                return domain
        return None
    
    def get_progression(self, prog_id: int) -> Optional[ProgressionPoint]:
        """Get progression point by ID."""
        return self.progression_by_id.get(prog_id)
    
    def validate_classification(
        self, 
        domain_key: str, 
        attribute_name: str, 
        progression_title: str
    ) -> tuple[bool, list[str]]:
        """
        Validate a classification against the taxonomy.
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check domain
        domain = self.domain_by_key.get(domain_key)
        if not domain:
            errors.append(f"Invalid domain_key: '{domain_key}'. Valid keys: {list(self.domain_by_key.keys())}")
            return False, errors
        
        # Check attribute
        attr = domain.get_attribute_by_name(attribute_name)
        if not attr:
            errors.append(f"Invalid attribute_name: '{attribute_name}' for domain '{domain_key}'")
            return False, errors
        
        # Check progression
        if progression_title not in PROGRESSION_LEVELS:
            errors.append(f"Invalid progression_title: '{progression_title}'. Valid levels: {PROGRESSION_LEVELS}")
            return False, errors
        
        return True, []
    
    def get_classification_json(
        self,
        domain_id: int,
        attribute_id: int,
        progression_title: str,
        progression_description: Optional[str] = None
    ) -> dict:
        """
        Generate a classification JSON object from IDs.
        
        This is used for creating training target labels from historical data.
        """
        domain = self.domain_by_id.get(domain_id)
        attr = self.attribute_by_id.get(attribute_id)
        
        if not domain or not attr:
            raise ValueError(f"Invalid domain_id={domain_id} or attribute_id={attribute_id}")
        
        # If no description provided, try to find the first matching one
        if not progression_description:
            prog = attr.get_first_progression_for_level(progression_title)
            if prog:
                progression_description = prog.description
            else:
                progression_description = ""
        
        return {
            "domain_key": domain.key,
            "domain_name": domain.name,
            "attribute_name": attr.name,
            "progression_title": progression_title,
            "progression_description": progression_description
        }
    
    def get_statistics(self) -> dict:
        """Get summary statistics about the taxonomy."""
        total_attributes = sum(len(d.attributes) for d in self.domains)
        total_progressions = sum(
            len(a.progression_points) 
            for d in self.domains 
            for a in d.attributes
        )
        
        progressions_by_level = {level: 0 for level in PROGRESSION_LEVELS}
        for domain in self.domains:
            for attr in domain.attributes:
                for prog in attr.progression_points:
                    if prog.title in progressions_by_level:
                        progressions_by_level[prog.title] += 1
        
        return {
            "total_domains": len(self.domains),
            "total_attributes": total_attributes,
            "total_progression_points": total_progressions,
            "progressions_by_level": progressions_by_level,
            "domains": [
                {
                    "id": d.id,
                    "key": d.key,
                    "name": d.name,
                    "attribute_count": len(d.attributes)
                }
                for d in self.domains
            ]
        }


class TaxonomyParser:
    """
    Parser for the learningdomains.json taxonomy file.
    
    Usage:
        parser = TaxonomyParser()
        taxonomy = parser.parse()
        
        # Or load from a specific file:
        taxonomy = parser.parse("/path/to/taxonomy.json")
    """
    
    def __init__(self, taxonomy_file: Optional[Path] = None):
        self.taxonomy_file = taxonomy_file or TAXONOMY_FILE
    
    def parse(self, file_path: Optional[Path] = None) -> NormalizedTaxonomy:
        """
        Parse the taxonomy JSON file and return normalized structures.
        
        Args:
            file_path: Optional override for taxonomy file path
            
        Returns:
            NormalizedTaxonomy with all lookup structures built
        """
        path = file_path or self.taxonomy_file
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        taxonomy = NormalizedTaxonomy()
        
        for domain_data in data.get("domains", []):
            domain = self._parse_domain(domain_data)
            taxonomy.domains.append(domain)
        
        # Sort domains by sort_order
        taxonomy.domains.sort(key=lambda d: d.sort_order)
        
        # Build lookup dictionaries
        taxonomy.build_lookups()
        
        return taxonomy
    
    def _parse_domain(self, data: dict) -> Domain:
        """Parse a domain object from JSON data."""
        domain = Domain(
            id=data["id"],
            name=data["name"],
            category_name=data.get("categoryName", ""),
            category_title=data.get("categoryTitle", ""),
            sort_order=data.get("sortOrder", 0)
        )
        
        for attr_data in data.get("attributes", []):
            attr = self._parse_attribute(attr_data)
            domain.attributes.append(attr)
        
        # Sort attributes by sort_order
        domain.attributes.sort(key=lambda a: a.sort_order)
        
        return domain
    
    def _parse_attribute(self, data: dict) -> Attribute:
        """Parse an attribute object from JSON data."""
        attr = Attribute(
            id=data["id"],
            number=data.get("number", 0),
            name=data["name"],
            category_information=data.get("categoryInformation", ""),
            sort_order=data.get("sortOrder", 0)
        )
        
        for prog_data in data.get("progressionPoints", []):
            prog = self._parse_progression(prog_data)
            attr.progression_points.append(prog)
        
        # Sort progressions by sort_order
        attr.progression_points.sort(key=lambda p: p.sort_order)
        
        return attr
    
    def _parse_progression(self, data: dict) -> ProgressionPoint:
        """Parse a progression point object from JSON data."""
        return ProgressionPoint(
            id=data["id"],
            progression_points_id=data.get("progressionPointsId", 0),
            title=data.get("progressionPointsTitle", ""),
            description=data.get("description", ""),
            sort_order=data.get("sortOrder", 0),
            description_order=str(data.get("descriptionOrder", "")),
            is_hidden=bool(data.get("isProgressionHidden", 0))
        )
    
    def save_normalized(
        self, 
        taxonomy: NormalizedTaxonomy, 
        output_path: Optional[Path] = None
    ) -> Path:
        """
        Save the normalized taxonomy to a JSON file for quick loading.
        
        Args:
            taxonomy: The parsed taxonomy
            output_path: Where to save (defaults to processed data dir)
            
        Returns:
            Path to the saved file
        """
        from config import TAXONOMY_NORMALIZED
        
        output_path = output_path or TAXONOMY_NORMALIZED
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to serializable format
        serializable = {
            "domains": [
                {
                    "id": d.id,
                    "key": d.key,
                    "name": d.name,
                    "category_name": d.category_name,
                    "category_title": d.category_title,
                    "sort_order": d.sort_order,
                    "attributes": [
                        {
                            "id": a.id,
                            "number": a.number,
                            "name": a.name,
                            "category_information": a.category_information,
                            "sort_order": a.sort_order,
                            "progression_points": [
                                {
                                    "id": p.id,
                                    "progression_points_id": p.progression_points_id,
                                    "title": p.title,
                                    "description": p.description,
                                    "sort_order": p.sort_order,
                                    "description_order": p.description_order,
                                    "is_hidden": p.is_hidden
                                }
                                for p in a.progression_points
                            ]
                        }
                        for a in d.attributes
                    ]
                }
                for d in taxonomy.domains
            ],
            "statistics": taxonomy.get_statistics()
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)
        
        return output_path


def load_taxonomy(file_path: Optional[Path] = None) -> NormalizedTaxonomy:
    """
    Convenience function to load and parse the taxonomy.
    
    Usage:
        from taxonomy_parser import load_taxonomy
        taxonomy = load_taxonomy()
    """
    parser = TaxonomyParser(file_path)
    return parser.parse()


if __name__ == "__main__":
    # Parse taxonomy and print statistics
    print("=" * 60)
    print("Parsing Taxonomy: learningdomains.json")
    print("=" * 60)
    
    parser = TaxonomyParser()
    taxonomy = parser.parse()
    
    stats = taxonomy.get_statistics()
    
    print(f"\n📊 Taxonomy Statistics:")
    print(f"   Total Domains: {stats['total_domains']}")
    print(f"   Total Attributes: {stats['total_attributes']}")
    print(f"   Total Progression Points: {stats['total_progression_points']}")
    
    print(f"\n📈 Progressions by Level:")
    for level, count in stats['progressions_by_level'].items():
        print(f"   {level}: {count}")
    
    print(f"\n🏷️  Domains:")
    for domain_info in stats['domains']:
        print(f"   [{domain_info['id']}] {domain_info['key']}")
        print(f"       Name: {domain_info['name']}")
        print(f"       Attributes: {domain_info['attribute_count']}")
    
    # Test classification lookup
    print("\n" + "=" * 60)
    print("Testing Classification Lookup")
    print("=" * 60)
    
    # Example: Get classification for domain 1, attribute 1, "Emerging"
    try:
        classification = taxonomy.get_classification_json(
            domain_id=1,
            attribute_id=1,
            progression_title="Emerging"
        )
        print(f"\n✅ Sample Classification (Domain 1, Attribute 1, Emerging):")
        print(json.dumps(classification, indent=2))
    except Exception as e:
        print(f"\n❌ Error: {e}")
    
    # Validate a classification
    is_valid, errors = taxonomy.validate_classification(
        domain_key="physical_development",
        attribute_name="Uses large muscles",
        progression_title="Developing"
    )
    print(f"\n✅ Validation test passed: {is_valid}")
    
    # Save normalized taxonomy
    output_path = parser.save_normalized(taxonomy)
    print(f"\n💾 Saved normalized taxonomy to: {output_path}")
    
    print(f"\n✅ Taxonomy parsing complete!")
