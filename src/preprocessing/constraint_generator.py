import random
from enum import Enum

from src.datasets.schema import Constraint, ConstraintType, Priority, RelationshipType
from src.preprocessing.boomi_parser import ParsedBoomiRecord


class DifficultyLevel(str, Enum):
    C0 = "C0"
    C1 = "C1"
    C2 = "C2"
    C3 = "C3"


_SAMPLE_SIZES = {
    DifficultyLevel.C0: (0, 0),
    DifficultyLevel.C1: (1, 2),
    DifficultyLevel.C2: (3, 5),
    DifficultyLevel.C3: (6, 8),
}


def build_constraint_pool(record: ParsedBoomiRecord) -> list[Constraint]:
    """Builds every constraint that the source BOOMI plan actually satisfies."""
    source = f"boomi:{record.plan_id}"
    pool: list[Constraint] = []

    for rp in record.room_program:
        pool.append(
            Constraint(
                id=f"required_room:{rp.type.value}",
                type=ConstraintType.REQUIRED_ROOM,
                target=rp.type.value,
                priority=Priority.HARD,
                source=source,
            )
        )
        pool.append(
            Constraint(
                id=f"room_count:{rp.type.value}",
                type=ConstraintType.ROOM_COUNT,
                target=rp.type.value,
                value=rp.count,
                priority=Priority.HARD,
                source=source,
            )
        )

    pool.append(
        Constraint(
            id="total_area",
            type=ConstraintType.TOTAL_AREA,
            value=record.total_area_m2,
            unit="m2",
            priority=Priority.HARD,
            source=source,
        )
    )
    pool.append(
        Constraint(
            id="site_boundary:width",
            type=ConstraintType.SITE_BOUNDARY,
            target="width",
            value=record.plot_width_m,
            unit="m",
            priority=Priority.HARD,
            source=source,
        )
    )
    pool.append(
        Constraint(
            id="site_boundary:length",
            type=ConstraintType.SITE_BOUNDARY,
            target="length",
            value=record.plot_length_m,
            unit="m",
            priority=Priority.HARD,
            source=source,
        )
    )

    for i, rel in enumerate(record.relationships):
        if rel.relationship == RelationshipType.ADJACENT:
            pool.append(
                Constraint(
                    id=f"adjacency:{i}",
                    type=ConstraintType.ADJACENCY,
                    target=f"{rel.a_type.value}<->{rel.b_type.value}",
                    priority=Priority.HARD,
                    source=source,
                )
            )
        elif rel.relationship == RelationshipType.DOOR_CONNECTION:
            pool.append(
                Constraint(
                    id=f"direct_access:{i}",
                    type=ConstraintType.DIRECT_ACCESS,
                    target=f"{rel.a_type.value}<->{rel.b_type.value}",
                    priority=Priority.HARD,
                    source=source,
                )
            )

    return pool


def sample_constraints(pool: list[Constraint], difficulty: DifficultyLevel, rng: random.Random) -> list[Constraint]:
    """Samples a subset of the constraint pool for one training variant.

    Caps below the full pool size (when the pool is large enough to allow it) so a
    variant never exposes every fact the parser extracted about the target plan.
    """
    lo, hi = _SAMPLE_SIZES[difficulty]
    cap = min(hi, len(pool))
    if difficulty != DifficultyLevel.C0 and cap >= len(pool) and len(pool) > lo:
        cap -= 1
    size = rng.randint(lo, cap) if cap > lo else cap
    return rng.sample(pool, size) if size else []
