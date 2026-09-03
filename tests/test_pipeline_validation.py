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
    Site,
    SourceType,
    Zone,
    ZoneType,
)
from src.datasets.schema import RoomType
from src.validation.dataset_qa import (
    anti_leakage_ratio,
    assert_no_split_leakage,
    find_constraint_violations,
    find_contradictions,
)


def _example(constraints: list[Constraint]) -> ArchitectTrainingExample:
    spec = ArchitecturalSpec(
        program=[RoomProgram(type=RoomType.BEDROOM, count=2, area_per_room_m2=12.0, zone=ZoneType.PRIVATE)],
        zones=[Zone(type=ZoneType.PRIVATE, room_types=[RoomType.BEDROOM])],
        relationships=[
            Relationship(a_type=RoomType.BEDROOM, b_type=RoomType.KITCHEN, relationship=RelationshipType.ADJACENT)
        ],
        metadata={"total_area_m2": 60.0},
    )
    return ArchitectTrainingExample(
        example_id="1",
        brief=Brief(bedrooms=2),
        site=Site(width_m=10.0, length_m=10.0, area_m2=90.0),
        constraints=constraints,
        target_spec=spec,
    )


def test_constraint_grounded_in_target_has_no_violations():
    c = Constraint(
        id="c1",
        type=ConstraintType.ROOM_COUNT,
        target="BEDROOM",
        value=2,
        priority=Priority.HARD,
        source_type=SourceType.USER_REQUIREMENT,
        source="test",
    )
    assert find_constraint_violations(_example([c])) == []


def test_constraint_not_grounded_in_target_is_flagged():
    c = Constraint(
        id="c1",
        type=ConstraintType.ROOM_COUNT,
        target="BEDROOM",
        value=99,
        priority=Priority.HARD,
        source_type=SourceType.USER_REQUIREMENT,
        source="test",
    )
    violations = find_constraint_violations(_example([c]))
    assert len(violations) == 1


def test_adjacency_constraint_without_matching_relationship_is_flagged():
    c = Constraint(
        id="c1",
        type=ConstraintType.ADJACENCY,
        target="BEDROOM<->BALCONY",
        priority=Priority.SOFT,
        source_type=SourceType.OBSERVED_GEOMETRY,
        source="test",
    )
    violations = find_constraint_violations(_example([c]))
    assert len(violations) == 1


def test_conflicting_room_count_is_a_contradiction():
    c1 = Constraint(
        id="c1", type=ConstraintType.ROOM_COUNT, target="BEDROOM", value=2, priority=Priority.HARD,
        source_type=SourceType.USER_REQUIREMENT, source="test",
    )
    c2 = Constraint(
        id="c2", type=ConstraintType.ROOM_COUNT, target="BEDROOM", value=3, priority=Priority.HARD,
        source_type=SourceType.USER_REQUIREMENT, source="test",
    )
    assert len(find_contradictions([c1, c2])) == 1


def test_min_area_over_max_area_is_a_contradiction():
    c1 = Constraint(
        id="c1", type=ConstraintType.MIN_AREA, target="BEDROOM", value=20.0, priority=Priority.HARD,
        source_type=SourceType.USER_REQUIREMENT, source="test",
    )
    c2 = Constraint(
        id="c2", type=ConstraintType.MAX_AREA, target="BEDROOM", value=15.0, priority=Priority.HARD,
        source_type=SourceType.USER_REQUIREMENT, source="test",
    )
    assert len(find_contradictions([c1, c2])) == 1


def test_no_contradictions_in_consistent_constraints():
    c1 = Constraint(
        id="c1", type=ConstraintType.ROOM_COUNT, target="BEDROOM", value=2, priority=Priority.HARD,
        source_type=SourceType.USER_REQUIREMENT, source="test",
    )
    c2 = Constraint(
        id="c2", type=ConstraintType.MIN_AREA, target="BEDROOM", value=10.0, priority=Priority.HARD,
        source_type=SourceType.USER_REQUIREMENT, source="test",
    )
    assert find_contradictions([c1, c2]) == []


def test_anti_leakage_ratio():
    assert anti_leakage_ratio(0, 10) == 0.0
    assert anti_leakage_ratio(5, 10) == 0.5
    assert anti_leakage_ratio(10, 10) == 1.0
    assert anti_leakage_ratio(0, 0) == 0.0


def test_split_leakage_detected():
    import pytest

    with pytest.raises(AssertionError):
        assert_no_split_leakage({"train": {1, 2, 3}, "validation": {3, 4}})


def test_no_split_leakage_when_disjoint():
    assert_no_split_leakage({"train": {1, 2, 3}, "validation": {4, 5}, "test": {6, 7}})
