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


class SourceType(str, Enum):
    """Why a constraint or relationship exists. See .claude/skills/tasks/SKILL.md audit, Task 16/67."""

    USER_REQUIREMENT = "USER_REQUIREMENT"
    REGULATION = "REGULATION"
    SITE_CONDITION = "SITE_CONDITION"
    ARCHITECTURAL_PREFERENCE = "ARCHITECTURAL_PREFERENCE"
    OBSERVED_GEOMETRY = "OBSERVED_GEOMETRY"
    DATASET_AUGMENTATION = "DATASET_AUGMENTATION"


class Site(BaseModel):
    """A simple rectangular-bounding-box site representation (V1 scope).

    width_m/length_m are bounding dimensions, not necessarily implying
    area_m2 = width_m * length_m - verified against BOOMI that the plot is
    frequently non-rectangular (width*depth overstates area_m2 in ~98% of
    sampled plans). area_m2 always comes directly from the source, never
    recomputed from width*length.
    """

    width_m: float = Field(gt=0)
    length_m: float = Field(gt=0)
    area_m2: float = Field(gt=0)


class Brief(BaseModel):
    building_type: Optional[str] = None
    target_area_m2: Optional[float] = Field(default=None, gt=0)
    floors: Optional[int] = Field(default=None, gt=0)
    bedrooms: Optional[int] = Field(default=None, ge=0)
    bathrooms: Optional[int] = Field(default=None, ge=0)
    balconies: Optional[int] = Field(default=None, ge=0)
    preferences: list[str] = Field(default_factory=list)


class Constraint(BaseModel):
    id: str
    type: ConstraintType
    target: Optional[str] = None
    value: Optional[Union[float, str]] = None
    unit: Optional[str] = None
    priority: Priority
    source_type: SourceType
    source: str


class RoomProgram(BaseModel):
    type: RoomType
    count: int = Field(gt=0)
    # Area of ONE room of this type in the source plan (verified: count * area_per_room_m2
    # sums to total_area_m2 almost exactly across the dataset) - not a total, not an average.
    area_per_room_m2: float = Field(gt=0)
    zone: ZoneType


class Relationship(BaseModel):
    a_type: RoomType
    b_type: RoomType
    relationship: RelationshipType
    # All BOOMI-derived relationships are observed facts about one solved plan,
    # not universal architectural rules - see SKILL.md audit Task 15/34.
    source_type: SourceType = SourceType.OBSERVED_GEOMETRY


class Zone(BaseModel):
    type: ZoneType
    room_types: list[RoomType]


class ArchitecturalSpec(BaseModel):
    program: list[RoomProgram]
    zones: list[Zone]
    relationships: list[Relationship]
    circulation: list[RoomType] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


class ArchitectTrainingExample(BaseModel):
    example_id: str
    brief: Brief
    site: Site
    constraints: list[Constraint] = Field(default_factory=list)
    target_spec: ArchitecturalSpec
    metadata: dict = Field(default_factory=dict)
