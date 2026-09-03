from src.datasets.schema import ArchitecturalSpec, Relationship, Room, RoomProgram, Zone, ZoneType
from src.preprocessing.boomi_parser import ParsedBoomiRecord
from src.preprocessing.zone_classifier import classify_zone


def generate_spec(record: ParsedBoomiRecord) -> ArchitecturalSpec:
    program = [
        RoomProgram(type=rp.type, count=rp.count, target_area_m2=rp.target_area_m2) for rp in record.room_program
    ]

    rooms = [
        Room(type=rp.type, zone=classify_zone(rp.type), target_area_m2=rp.target_area_m2)
        for rp in record.room_program
    ]

    zones: dict[ZoneType, set] = {}
    for rp in record.room_program:
        zone = classify_zone(rp.type)
        zones.setdefault(zone, set()).add(rp.type)
    zone_list = [Zone(type=zone_type, room_types=sorted(types, key=lambda t: t.value)) for zone_type, types in zones.items()]

    relationships = [
        Relationship(a_type=rel.a_type, b_type=rel.b_type, relationship=rel.relationship)
        for rel in record.relationships
    ]

    circulation = sorted(
        {rp.type for rp in record.room_program if classify_zone(rp.type) == ZoneType.CIRCULATION},
        key=lambda t: t.value,
    )

    return ArchitecturalSpec(
        program=program,
        zones=zone_list,
        rooms=rooms,
        relationships=relationships,
        circulation=circulation,
        metadata={
            "source": "BOOMI",
            "plan_id": record.plan_id,
            "bhk_label": record.bhk_label,
            "total_area_m2": record.total_area_m2,
        },
    )
