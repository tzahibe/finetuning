import pytest
from pydantic import ValidationError

from src.datasets.schema import (
    ArchitectTrainingExample,
    ArchitecturalSpec,
    Brief,
    Constraint,
    ConstraintType,
    Priority,
    Relationship,
    RelationshipType,
    Room,
    RoomProgram,
    RoomType,
    Zone,
    ZoneType,
)


def _valid_spec() -> ArchitecturalSpec:
    return ArchitecturalSpec(
        program=[RoomProgram(type=RoomType.BEDROOM, count=2, target_area_m2=12.0)],
        zones=[Zone(type=ZoneType.PRIVATE, room_types=[RoomType.BEDROOM])],
        rooms=[Room(type=RoomType.BEDROOM, zone=ZoneType.PRIVATE, target_area_m2=12.0)],
        relationships=[
            Relationship(a_type=RoomType.BEDROOM, b_type=RoomType.BATHROOM, relationship=RelationshipType.ADJACENT)
        ],
    )


def test_valid_training_example():
    example = ArchitectTrainingExample(
        example_id="1",
        brief=Brief(bedrooms=2, bathrooms=1),
        constraints=[
            Constraint(
                id="c1", type=ConstraintType.ROOM_COUNT, target="BEDROOM", value=2, priority=Priority.HARD, source="test"
            )
        ],
        target_spec=_valid_spec(),
    )
    assert example.brief.bedrooms == 2


def test_missing_required_field_raises():
    with pytest.raises(ValidationError):
        ArchitectTrainingExample(brief=Brief(), target_spec=_valid_spec())  # missing example_id


def test_invalid_room_type_rejected():
    with pytest.raises(ValidationError):
        RoomProgram(type="SWIMMING_POOL", count=1, target_area_m2=10.0)


def test_negative_area_rejected():
    with pytest.raises(ValidationError):
        RoomProgram(type=RoomType.BEDROOM, count=1, target_area_m2=-5.0)


def test_invalid_relationship_rejected():
    with pytest.raises(ValidationError):
        Relationship(a_type=RoomType.BEDROOM, b_type=RoomType.BATHROOM, relationship="TOUCHING")


def test_invalid_constraint_type_rejected():
    with pytest.raises(ValidationError):
        Constraint(id="c1", type="MAGIC", priority=Priority.HARD, source="test")


def test_invalid_priority_rejected():
    with pytest.raises(ValidationError):
        Constraint(id="c1", type=ConstraintType.ROOM_COUNT, priority="URGENT", source="test")
