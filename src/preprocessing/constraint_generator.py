import random
from enum import Enum

from src.datasets.schema import Constraint, ConstraintType, Priority, RelationshipType, SourceType
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
    """Builds every constraint candidate the source BOOMI plan actually satisfies.

    Priority/source_type follow SKILL.md audit Task 15-17: room counts mirror facts
    the Brief itself states (USER_REQUIREMENT, HARD). Relationships and total area are
    a single observed solution, not a universal rule the model must reproduce exactly
    - kept SOFT / OBSERVED_GEOMETRY (or USER_REQUIREMENT for total area, treated as an
    approximate target by consumers, not exact equality - see Task 27).

    Site facts (width/length/area) are NOT generated here - Site is always provided as
    a first-class top-level field on the training example (Task 25), so a SITE_BOUNDARY
    constraint here would just duplicate it.

    `window` adjacency edges are not mapped to a constraint: their exact architectural
    meaning could not be determined from the source data alone (Task 20), so they are
    only preserved as raw observed relationships in the target spec, not promoted into
    an input-facing constraint.
    """
    source = f"boomi:{record.plan_id}"
    pool: list[Constraint] = []

    for rp in record.room_program:
        pool.append(
            Constraint(
                id=f"required_room:{rp.type.value}",
                type=ConstraintType.REQUIRED_ROOM,
                target=rp.type.value,
                priority=Priority.HARD,
                source_type=SourceType.USER_REQUIREMENT,
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
                source_type=SourceType.USER_REQUIREMENT,
                source=source,
            )
        )

    pool.append(
        Constraint(
            id="total_area",
            type=ConstraintType.TOTAL_AREA,
            value=record.total_area_m2,
            unit="m2",
            # Approximate target, not exact floating-point equality - see Task 27.
            priority=Priority.SOFT,
            source_type=SourceType.USER_REQUIREMENT,
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
                    priority=Priority.SOFT,
                    source_type=SourceType.OBSERVED_GEOMETRY,
                    source=source,
                )
            )
        elif rel.relationship == RelationshipType.DOOR_CONNECTION:
            pool.append(
                Constraint(
                    id=f"direct_access:{i}",
                    type=ConstraintType.DIRECT_ACCESS,
                    target=f"{rel.a_type.value}<->{rel.b_type.value}",
                    priority=Priority.SOFT,
                    source_type=SourceType.OBSERVED_GEOMETRY,
                    source=source,
                )
            )

    return pool


def sample_constraints(pool: list[Constraint], difficulty: DifficultyLevel, rng: random.Random) -> list[Constraint]:
    """Samples a subset of the constraint pool for one training variant.

    Caps below the full pool size (when the pool is large enough to allow it) so a
    variant never exposes every fact the parser extracted about the target plan.

    Stratifies by constraint type (Task 5: C0-C3 must vary in constraint-type
    diversity, not just count) - round-robins across REQUIRED_ROOM/ROOM_COUNT/
    TOTAL_AREA/ADJACENCY/DIRECT_ACCESS in a per-call shuffled order, rather than a
    flat random sample that could land on e.g. eight ROOM_COUNT facts and nothing
    else.
    """
    lo, hi = _SAMPLE_SIZES[difficulty]
    cap = min(hi, len(pool))
    if difficulty != DifficultyLevel.C0 and cap >= len(pool) and len(pool) > lo:
        cap -= 1
    size = rng.randint(lo, cap) if cap > lo else cap
    if size == 0:
        return []

    by_type: dict[ConstraintType, list[Constraint]] = {}
    for c in pool:
        by_type.setdefault(c.type, []).append(c)
    for group in by_type.values():
        rng.shuffle(group)
    type_order = list(by_type.keys())
    rng.shuffle(type_order)

    selected: list[Constraint] = []
    pointers = {t: 0 for t in type_order}
    while len(selected) < size:
        progressed = False
        for t in type_order:
            if len(selected) >= size:
                break
            p = pointers[t]
            if p < len(by_type[t]):
                selected.append(by_type[t][p])
                pointers[t] = p + 1
                progressed = True
        if not progressed:
            break
    return selected
