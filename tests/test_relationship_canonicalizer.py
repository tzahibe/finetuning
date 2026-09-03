from src.datasets.schema import RelationshipType, RoomType
from src.preprocessing.boomi_parser import ParsedRelationship
from src.preprocessing.relationship_canonicalizer import canonicalize_relationships


def test_reversed_pair_collapses_to_one_fact():
    rels = [
        ParsedRelationship(a_type=RoomType.BEDROOM, b_type=RoomType.BATHROOM, relationship=RelationshipType.ADJACENT),
        ParsedRelationship(a_type=RoomType.BATHROOM, b_type=RoomType.BEDROOM, relationship=RelationshipType.ADJACENT),
    ]
    result = canonicalize_relationships(rels)
    assert len(result) == 1


def test_different_relationship_types_for_same_pair_are_kept_distinct():
    rels = [
        ParsedRelationship(a_type=RoomType.BEDROOM, b_type=RoomType.BATHROOM, relationship=RelationshipType.ADJACENT),
        ParsedRelationship(
            a_type=RoomType.BEDROOM, b_type=RoomType.BATHROOM, relationship=RelationshipType.DOOR_CONNECTION
        ),
    ]
    result = canonicalize_relationships(rels)
    assert len(result) == 2
    assert {r.relationship for r in result} == {RelationshipType.ADJACENT, RelationshipType.DOOR_CONNECTION}


def test_ordering_is_deterministic():
    rels = [
        ParsedRelationship(a_type=RoomType.KITCHEN, b_type=RoomType.LIVING, relationship=RelationshipType.ADJACENT),
    ]
    result = canonicalize_relationships(rels)
    assert (result[0].a_type.value, result[0].b_type.value) == tuple(
        sorted([RoomType.KITCHEN.value, RoomType.LIVING.value])
    )


def test_same_type_pair_is_left_as_is():
    rels = [
        ParsedRelationship(a_type=RoomType.BATHROOM, b_type=RoomType.BATHROOM, relationship=RelationshipType.ADJACENT),
    ]
    result = canonicalize_relationships(rels)
    assert len(result) == 1
    assert result[0].a_type == result[0].b_type == RoomType.BATHROOM
