from src.datasets.schema import Brief, RoomType
from src.preprocessing.boomi_parser import ParsedBoomiRecord


def generate_brief(record: ParsedBoomiRecord) -> Brief:
    bedrooms = sum(
        rp.count for rp in record.room_program if rp.type in (RoomType.BEDROOM, RoomType.MASTER_BEDROOM)
    )
    bathrooms = sum(rp.count for rp in record.room_program if rp.type in (RoomType.BATHROOM, RoomType.WC))
    balconies = sum(rp.count for rp in record.room_program if rp.type == RoomType.BALCONY)

    return Brief(
        # Every BOOMI record is a dwelling unit (has a bhk_label) - safe to infer, not invented.
        building_type="residential",
        # This is the built/program area (sum of room areas), NOT the site area - see
        # ParsedBoomiRecord.total_area_m2 for the verification behind this distinction.
        target_area_m2=record.total_area_m2,
        bedrooms=bedrooms,
        bathrooms=bathrooms,
        balconies=balconies,
        # floors: not present anywhere in BOOMI - left unset rather than assuming 1.
    )
