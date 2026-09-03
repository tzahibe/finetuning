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
from src.datasets.sft_format import to_sft_pair


def _example() -> ArchitectTrainingExample:
    spec = ArchitecturalSpec(
        program=[RoomProgram(type=RoomType.BEDROOM, count=2, area_per_room_m2=12.0, zone=ZoneType.PRIVATE)],
        zones=[Zone(type=ZoneType.PRIVATE, room_types=[RoomType.BEDROOM])],
        relationships=[
            Relationship(a_type=RoomType.BEDROOM, b_type=RoomType.KITCHEN, relationship=RelationshipType.ADJACENT)
        ],
        metadata={"source": "BOOMI", "source_plan_id": 123, "bhk_label": "2BHK", "total_area_m2": 24.0},
    )
    return ArchitectTrainingExample(
        example_id="123-C1",
        brief=Brief(bedrooms=2, target_area_m2=24.0),
        site=Site(width_m=10.0, length_m=10.0, area_m2=90.0),
        constraints=[
            Constraint(
                id="room_count:BEDROOM",
                type=ConstraintType.ROOM_COUNT,
                target="BEDROOM",
                value=2,
                priority=Priority.HARD,
                source_type=SourceType.USER_REQUIREMENT,
                source="boomi:123",
            )
        ],
        target_spec=spec,
        metadata={"source_plan_id": 123, "caption": "a two bedroom flat", "pipeline_version": "0.3.0"},
    )


def test_target_excludes_dataset_metadata():
    pair = to_sft_pair(_example())
    assert "metadata" not in pair["target"]
    assert set(pair["target"].keys()) == {"program", "zones", "relationships", "circulation"}


def test_input_constraints_exclude_lineage_fields():
    pair = to_sft_pair(_example())
    c = pair["input"]["constraints"][0]
    assert "id" not in c
    assert "source" not in c
    assert c["type"] == "ROOM_COUNT"
    assert c["target"] == "BEDROOM"


def test_input_contains_brief_and_site():
    pair = to_sft_pair(_example())
    assert pair["input"]["brief"]["bedrooms"] == 2
    assert pair["input"]["site"]["area_m2"] == 90.0
