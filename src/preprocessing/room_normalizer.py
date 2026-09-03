from typing import Optional

from src.datasets.schema import RoomType

ROOM_TYPE_MAP: dict[str, RoomType] = {
    "BED ROOM": RoomType.BEDROOM,
    "BEDROOM": RoomType.BEDROOM,
    "MASTER BED ROOM": RoomType.MASTER_BEDROOM,
    "MASTER ROOM": RoomType.MASTER_BEDROOM,
    "TOILET": RoomType.BATHROOM,
    "BATH": RoomType.BATHROOM,
    "BATHROOM": RoomType.BATHROOM,
    "WASHROOM": RoomType.BATHROOM,
    "WC": RoomType.WC,
    "LIVING": RoomType.LIVING,
    "LIVING ROOM": RoomType.LIVING,
    "DRAWING ROOM": RoomType.LIVING,
    "KITCHEN": RoomType.KITCHEN,
    "DINING": RoomType.DINING,
    "DINING ROOM": RoomType.DINING,
    "BALCONY": RoomType.BALCONY,
    "CORRIDOR": RoomType.CORRIDOR,
    "ENTRANCE": RoomType.ENTRANCE,
    "STORE": RoomType.STORAGE,
    "STORAGE": RoomType.STORAGE,
    "UTILITY": RoomType.UTILITY,
    "STAIR": RoomType.STAIRCASE,
    "STAIRCASE": RoomType.STAIRCASE,
    "PARKING": RoomType.PARKING,
}


def normalize_room_type(raw_type: str) -> Optional[RoomType]:
    """Returns the normalized RoomType, or None if raw_type is not in the controlled vocabulary."""
    return ROOM_TYPE_MAP.get(raw_type.strip().upper())
