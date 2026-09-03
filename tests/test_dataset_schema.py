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
    RoomProgram,
    RoomType,
    Site,
    SourceType,
    Zone,
    ZoneType,
)


def _valid_spec() -> ArchitecturalSpec:
    return ArchitecturalSpec(
        program=[RoomProgram(type=RoomType.BEDROOM, count=2, area_per_room_m2=12.0, zone=ZoneType.PRIVATE)],
        zones=[Zone(type=ZoneType.PRIVATE, room_types=[RoomType.BEDROOM])],
        relationships=[
            Relationship(a_type=RoomType.BEDROOM, b_type=RoomType.BATHROOM, relationship=RelationshipType.ADJACENT)
        ],
    )


def _valid_site() -> Site:
    return Site(width_m=10.0, length_m=12.0, area_m2=120.0)


def test_valid_training_example():
    example = ArchitectTrainingExample(
        example_id="1",
        brief=Brief(bedrooms=2, bathrooms=1),
        site=_valid_site(),
        constraints=[
            Constraint(
                id="c1",
                type=ConstraintType.ROOM_COUNT,
                target="BEDROOM",
                value=2,
                priority=Priority.HARD,
                source_type=SourceType.USER_REQUIREMENT,
                source="test",
            )
        ],
        target_spec=_valid_spec(),
    )
    assert example.brief.bedrooms == 2


def test_missing_required_field_raises():
    with pytest.raises(ValidationError):
        ArchitectTrainingExample(brief=Brief(), site=_valid_site(), target_spec=_valid_spec())  # missing example_id


def test_missing_site_raises():
    with pytest.raises(ValidationError):
        ArchitectTrainingExample(example_id="1", brief=Brief(), target_spec=_valid_spec())  # missing site


def test_invalid_room_type_rejected():
    with pytest.raises(ValidationError):
        RoomProgram(type="SWIMMING_POOL", count=1, area_per_room_m2=10.0, zone=ZoneType.PRIVATE)


def test_negative_area_rejected():
    with pytest.raises(ValidationError):
        RoomProgram(type=RoomType.BEDROOM, count=1, area_per_room_m2=-5.0, zone=ZoneType.PRIVATE)


def test_invalid_relationship_rejected():
    with pytest.raises(ValidationError):
        Relationship(a_type=RoomType.BEDROOM, b_type=RoomType.BATHROOM, relationship="TOUCHING")


def test_relationship_defaults_to_observed_geometry():
    rel = Relationship(a_type=RoomType.BEDROOM, b_type=RoomType.BATHROOM, relationship=RelationshipType.ADJACENT)
    assert rel.source_type == SourceType.OBSERVED_GEOMETRY


def test_invalid_constraint_type_rejected():
    with pytest.raises(ValidationError):
        Constraint(id="c1", type="MAGIC", priority=Priority.HARD, source_type=SourceType.USER_REQUIREMENT, source="test")


def test_invalid_priority_rejected():
    with pytest.raises(ValidationError):
        Constraint(
            id="c1",
            type=ConstraintType.ROOM_COUNT,
            priority="URGENT",
            source_type=SourceType.USER_REQUIREMENT,
            source="test",
        )


def test_constraint_requires_source_type():
    with pytest.raises(ValidationError):
        Constraint(id="c1", type=ConstraintType.ROOM_COUNT, priority=Priority.HARD, source="test")


def test_site_area_not_recomputed_from_dimensions():
    # Non-rectangular sites are common in BOOMI (verified: width*length overstates
    # area_m2 in ~98% of sampled plans) - Site must accept area independent of w*l.
    site = Site(width_m=10.0, length_m=10.0, area_m2=64.25)
    assert site.area_m2 == 64.25
