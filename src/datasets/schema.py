from enum import Enum
from typing import Optional, Union

from pydantic import BaseModel, Field


class RoomType(str, Enum):
    BEDROOM = "BEDROOM"
    MASTER_BEDROOM = "MASTER_BEDROOM"
    BATHROOM = "BATHROOM"
    WC = "WC"
    LIVING = "LIVING"
    KITCHEN = "KITCHEN"
    DINING = "DINING"
    BALCONY = "BALCONY"
    CORRIDOR = "CORRIDOR"
    ENTRANCE = "ENTRANCE"
    STORAGE = "STORAGE"
    UTILITY = "UTILITY"
    STAIRCASE = "STAIRCASE"
    PARKING = "PARKING"


class RelationshipType(str, Enum):
    ADJACENT = "ADJACENT"
    DIRECT_ACCESS = "DIRECT_ACCESS"
    DOOR_CONNECTION = "DOOR_CONNECTION"
    WINDOW_CONNECTION = "WINDOW_CONNECTION"
    SEPARATED = "SEPARATED"
    NEAR = "NEAR"


class ZoneType(str, Enum):
    PUBLIC = "PUBLIC"
    PRIVATE = "PRIVATE"
    SERVICE = "SERVICE"
    CIRCULATION = "CIRCULATION"
    OUTDOOR = "OUTDOOR"


class ConstraintType(str, Enum):
    REQUIRED_ROOM = "REQUIRED_ROOM"
    FORBIDDEN_ROOM = "FORBIDDEN_ROOM"
    ROOM_COUNT = "ROOM_COUNT"
    MIN_AREA = "MIN_AREA"
    MAX_AREA = "MAX_AREA"
    MIN_WIDTH = "MIN_WIDTH"
    MAX_WIDTH = "MAX_WIDTH"
    ADJACENCY = "ADJACENCY"
    DIRECT_ACCESS = "DIRECT_ACCESS"
    SEPARATION = "SEPARATION"
    TOTAL_AREA = "TOTAL_AREA"
    SITE_BOUNDARY = "SITE_BOUNDARY"


class Priority(str, Enum):
    HARD = "HARD"
    SOFT = "SOFT"


class Brief(BaseModel):
    building_type: Optional[str] = None
    built_area_m2: Optional[float] = Field(default=None, gt=0)
    floors: Optional[int] = Field(default=None, gt=0)
    bedrooms: Optional[int] = Field(default=None, ge=0)
    bathrooms: Optional[int] = Field(default=None, ge=0)
    balconies: Optional[int] = Field(default=None, ge=0)
    plot_width_m: Optional[float] = Field(default=None, gt=0)
    plot_length_m: Optional[float] = Field(default=None, gt=0)
    preferences: list[str] = Field(default_factory=list)


class Constraint(BaseModel):
    id: str
    type: ConstraintType
    target: Optional[str] = None
    value: Optional[Union[float, str]] = None
    unit: Optional[str] = None
    priority: Priority
    source: str


class RoomProgram(BaseModel):
    type: RoomType
    count: int = Field(gt=0)
    target_area_m2: float = Field(gt=0)


class Relationship(BaseModel):
    a_type: RoomType
    b_type: RoomType
    relationship: RelationshipType


class Room(BaseModel):
    type: RoomType
    zone: ZoneType
    target_area_m2: float = Field(gt=0)


class Zone(BaseModel):
    type: ZoneType
    room_types: list[RoomType]


class ArchitecturalSpec(BaseModel):
    program: list[RoomProgram]
    zones: list[Zone]
    rooms: list[Room]
    relationships: list[Relationship]
    circulation: list[RoomType] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


class ArchitectTrainingExample(BaseModel):
    example_id: str
    brief: Brief
    constraints: list[Constraint] = Field(default_factory=list)
    target_spec: ArchitecturalSpec
    metadata: dict = Field(default_factory=dict)
