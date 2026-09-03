import json
from dataclasses import dataclass, field

from src.datasets.schema import RelationshipType, RoomType
from src.preprocessing.relationship_normalizer import normalize_relationship
from src.preprocessing.room_normalizer import normalize_room_type


@dataclass
class ParsedRoomProgram:
    type: RoomType
    count: int
    target_area_m2: float


@dataclass
class ParsedRelationship:
    a_type: RoomType
    b_type: RoomType
    relationship: RelationshipType


@dataclass
class ParsedBoomiRecord:
    plan_id: int
    level: str
    caption: str
    bhk_label: str
    total_area_m2: float
    plot_width_m: float
    plot_length_m: float
    room_program: list[ParsedRoomProgram]
    relationships: list[ParsedRelationship]
    unknown_room_types: set[str] = field(default_factory=set)
    unknown_relationship_types: set[str] = field(default_factory=set)


def parse_boomi_record(row: dict) -> ParsedBoomiRecord:
    spec = json.loads(row["spec"])

    unknown_room_types: set[str] = set()
    unknown_relationship_types: set[str] = set()

    room_program = []
    for rp in spec.get("room_program", []):
        room_type = normalize_room_type(rp["type"])
        if room_type is None:
            unknown_room_types.add(rp["type"])
            continue
        room_program.append(
            ParsedRoomProgram(type=room_type, count=rp["count"], target_area_m2=rp["target_area_m2"])
        )

    relationships = []
    for adj in spec.get("adjacency", []):
        a_type = normalize_room_type(adj["a_type"])
        b_type = normalize_room_type(adj["b_type"])
        relationship = normalize_relationship(adj["via"])
        if a_type is None:
            unknown_room_types.add(adj["a_type"])
        if b_type is None:
            unknown_room_types.add(adj["b_type"])
        if relationship is None:
            unknown_relationship_types.add(adj["via"])
        if a_type is None or b_type is None or relationship is None:
            continue
        relationships.append(ParsedRelationship(a_type=a_type, b_type=b_type, relationship=relationship))

    plot = spec.get("plot", {})

    return ParsedBoomiRecord(
        plan_id=row["plan_id"],
        level=row["level"],
        caption=row["caption"],
        bhk_label=spec.get("bhk_label", ""),
        total_area_m2=spec["total_area_m2"],
        plot_width_m=plot["width_mm"] / 1000.0,
        plot_length_m=plot["depth_mm"] / 1000.0,
        room_program=room_program,
        relationships=relationships,
        unknown_room_types=unknown_room_types,
        unknown_relationship_types=unknown_relationship_types,
    )
