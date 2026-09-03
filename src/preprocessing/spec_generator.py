from src.datasets.schema import ArchitecturalSpec, Relationship, RoomProgram, SourceType, Zone, ZoneType
from src.preprocessing.boomi_parser import ParsedBoomiRecord
from src.preprocessing.zone_classifier import classify_zone


def generate_spec(record: ParsedBoomiRecord) -> ArchitecturalSpec:
    # NOTE: BOOMI has no room-instance identifiers anywhere (verified across the raw
    # spec's full key set) - room_program is aggregated by type only. So `program` is
    # the finest granularity available; there is no separate, honest "rooms" list of
    # individual instances to build (see SKILL.md audit Task 13/14/21).
    program = [
        RoomProgram(type=rp.type, count=rp.count, area_per_room_m2=rp.area_per_room_m2, zone=classify_zone(rp.type))
        for rp in record.room_program
    ]

    zones: dict[ZoneType, set] = {}
    for rp in record.room_program:
        zone = classify_zone(rp.type)
        zones.setdefault(zone, set()).add(rp.type)
    zone_list = [
        Zone(type=zone_type, room_types=sorted(types, key=lambda t: t.value)) for zone_type, types in zones.items()
    ]

    # All BOOMI adjacency/door/window edges are observed facts about one solved plan,
    # not universal requirements (see SKILL.md audit Task 15/34) - tagged accordingly.
    relationships = [
        Relationship(
            a_type=rel.a_type,
            b_type=rel.b_type,
            relationship=rel.relationship,
            source_type=SourceType.OBSERVED_GEOMETRY,
        )
        for rel in record.relationships
    ]

    circulation = sorted(
        {rp.type for rp in record.room_program if classify_zone(rp.type) == ZoneType.CIRCULATION},
        key=lambda t: t.value,
    )

    return ArchitecturalSpec(
        program=program,
        zones=zone_list,
        relationships=relationships,
        circulation=circulation,
        metadata={
            "source": "BOOMI",
            "source_plan_id": record.plan_id,
            "source_level": record.level,
            "bhk_label": record.bhk_label,
            "total_area_m2": record.total_area_m2,
            "area_consistency_diff_m2": record.area_consistency_diff_m2,
        },
    )
