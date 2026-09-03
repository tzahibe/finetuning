from src.datasets.schema import RoomType, ZoneType

ROOM_ZONE_MAP: dict[RoomType, ZoneType] = {
    RoomType.LIVING: ZoneType.PUBLIC,
    RoomType.DINING: ZoneType.PUBLIC,
    RoomType.BEDROOM: ZoneType.PRIVATE,
    RoomType.MASTER_BEDROOM: ZoneType.PRIVATE,
    RoomType.KITCHEN: ZoneType.SERVICE,
    RoomType.BATHROOM: ZoneType.SERVICE,
    RoomType.WC: ZoneType.SERVICE,
    RoomType.UTILITY: ZoneType.SERVICE,
    RoomType.STORAGE: ZoneType.SERVICE,
    RoomType.CORRIDOR: ZoneType.CIRCULATION,
    RoomType.ENTRANCE: ZoneType.CIRCULATION,
    RoomType.STAIRCASE: ZoneType.CIRCULATION,
    RoomType.BALCONY: ZoneType.OUTDOOR,
    RoomType.PARKING: ZoneType.OUTDOOR,
}


def classify_zone(room_type: RoomType) -> ZoneType:
    return ROOM_ZONE_MAP[room_type]
