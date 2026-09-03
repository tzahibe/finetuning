import json
from dataclasses import dataclass, field

from src.datasets.schema import RelationshipType, RoomType
from src.preprocessing.relationship_normalizer import normalize_relationship
from src.preprocessing.room_normalizer import normalize_room_type

# Float precision policy (SKILL.md audit Task 28): site dimensions get 2 decimals,
# total/building areas get 1 decimal, per-room-type areas get 1 decimal.
SITE_DIM_PRECISION = 2
AREA_PRECISION = 1


@dataclass
class ParsedRoomProgram:
    type: RoomType
    count: int
    area_per_room_m2: float


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
    # Built/program area: verified equal to sum(count * area_per_room_m2) across the
    # dataset (mean residual ~0). This is NOT the same as the site/plot area below,
    # and is NOT what BOOMI captions' stated area figure refers to (see site_area_m2).
    total_area_m2: float
    # Site area, taken directly from BOOMI's plot.area_m2 - verified this is what the
    # caption's stated area actually matches (not total_area_m2), and verified that
    # site_width_m * site_length_m does NOT reliably reproduce site_area_m2 (site is
    # not a simple rectangle in ~98% of sampled plans), so area_m2 must never be
    # recomputed from the width/length below.
    site_width_m: float
    site_length_m: float
    site_area_m2: float
    room_program: list[ParsedRoomProgram]
    relationships: list[ParsedRelationship]
    # abs(total_area_m2 - sum(count * area_per_room_m2)); should be ~0, kept for QA.
    area_consistency_diff_m2: float
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
            ParsedRoomProgram(
                type=room_type,
                count=rp["count"],
                area_per_room_m2=round(rp["target_area_m2"], AREA_PRECISION),
            )
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
    total_area_m2 = spec["total_area_m2"]
    program_area_sum = sum(rp["count"] * rp["target_area_m2"] for rp in spec.get("room_program", []))

    return ParsedBoomiRecord(
        plan_id=row["plan_id"],
        level=row["level"],
        caption=row["caption"],
        bhk_label=spec.get("bhk_label", ""),
        total_area_m2=round(total_area_m2, AREA_PRECISION),
        site_width_m=round(plot["width_mm"] / 1000.0, SITE_DIM_PRECISION),
        site_length_m=round(plot["depth_mm"] / 1000.0, SITE_DIM_PRECISION),
        site_area_m2=round(plot["area_m2"], SITE_DIM_PRECISION),
        room_program=room_program,
        relationships=relationships,
        area_consistency_diff_m2=round(abs(total_area_m2 - program_area_sum), 3),
        unknown_room_types=unknown_room_types,
        unknown_relationship_types=unknown_relationship_types,
    )
