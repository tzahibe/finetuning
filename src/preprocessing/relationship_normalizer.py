from typing import Optional

from src.datasets.schema import RelationshipType

RELATIONSHIP_MAP: dict[str, RelationshipType] = {
    "adjacent": RelationshipType.ADJACENT,
    "door": RelationshipType.DOOR_CONNECTION,
    "window": RelationshipType.WINDOW_CONNECTION,
    "direct_access": RelationshipType.DIRECT_ACCESS,
    "separated": RelationshipType.SEPARATED,
    "near": RelationshipType.NEAR,
}


def normalize_relationship(raw_via: str) -> Optional[RelationshipType]:
    """Returns the normalized RelationshipType, or None if raw_via is not in the controlled vocabulary."""
    return RELATIONSHIP_MAP.get(raw_via.strip().lower())
