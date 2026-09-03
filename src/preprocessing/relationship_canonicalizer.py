from src.datasets.schema import RoomType
from src.preprocessing.boomi_parser import ParsedRelationship


def canonicalize_relationships(relationships: list[ParsedRelationship]) -> list[ParsedRelationship]:
    """Orders a_type/b_type deterministically within each relationship type and drops
    exact duplicate (a_type, b_type, relationship) triples, so `BEDROOM<->BATHROOM`
    and `BATHROOM<->BEDROOM` never appear as two separate facts. Never merges across
    different relationship types (ADJACENT/DOOR_CONNECTION/WINDOW_CONNECTION stay
    distinct) - a pair legitimately connected by both a door and adjacency keeps both.
    """
    seen: set[tuple[str, str, str]] = set()
    result: list[ParsedRelationship] = []
    for rel in relationships:
        a, b = sorted([rel.a_type.value, rel.b_type.value])
        key = (a, b, rel.relationship.value)
        if key in seen:
            continue
        seen.add(key)
        result.append(ParsedRelationship(a_type=RoomType(a), b_type=RoomType(b), relationship=rel.relationship))
    return result
