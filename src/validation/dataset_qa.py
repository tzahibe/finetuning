from src.datasets.schema import ArchitectTrainingExample, Constraint, ConstraintType, Priority, RoomType, SourceType

# Area tolerance policy (docs/DATA_CONTRACT.md "Area tolerance"). Evidence: over ALL
# 14,891 unique BOOMI plans, abs(total_area_m2 - sum(count*area_per_room_m2)):
# mean=0.96m2 median=0.78 p95=2.45 p99=3.50 max=7.47 (m2); as a % of total_area_m2:
# mean=0.81% median=0.69% p95=1.95% p99=2.53% max=3.97%. The relative error is small
# and bounded across the entire dataset (never >4%) - the signature of source-side
# floating point rounding, not data loss (which would show large/erratic/one-sided
# errors). A plan passes if EITHER tolerance holds, so small-total-area plans (where
# a tiny absolute rounding could look large in %) are still covered by the absolute
# bound, and large plans (where a few m2 is negligible) are covered by the relative
# bound - the observed max of both (7.47m2 / 3.97%) sits safely under these.
AREA_TOLERANCE_ABS_M2 = 5.0
AREA_TOLERANCE_REL_PCT = 0.05


def area_within_tolerance(value_a: float, value_b: float) -> bool:
    abs_diff = abs(value_a - value_b)
    if abs_diff <= AREA_TOLERANCE_ABS_M2:
        return True
    denom = max(abs(value_a), abs(value_b))
    return denom > 0 and abs_diff / denom <= AREA_TOLERANCE_REL_PCT


def find_constraint_violations(example: ArchitectTrainingExample) -> list[str]:
    """Returns a list of human-readable violation descriptions for constraints that
    are NOT actually satisfied by example.target_spec (Task 44: source-target
    consistency check). Empty list means every sampled constraint is grounded in the
    target this example was built from.
    """
    spec = example.target_spec
    room_counts = {rp.type.value: rp.count for rp in spec.program}
    room_types_present = set(room_counts.keys())
    rel_pairs = {(r.a_type.value, r.b_type.value, r.relationship.value) for r in spec.relationships}

    violations = []
    for c in example.constraints:
        if c.type == ConstraintType.REQUIRED_ROOM:
            if c.target not in room_types_present:
                violations.append(f"{c.id}: REQUIRED_ROOM {c.target} not present in target_spec.program")
        elif c.type == ConstraintType.ROOM_COUNT:
            if room_counts.get(c.target) != c.value:
                violations.append(
                    f"{c.id}: ROOM_COUNT {c.target}={c.value} does not match target count {room_counts.get(c.target)}"
                )
        elif c.type == ConstraintType.TOTAL_AREA:
            if abs(float(c.value) - spec.metadata.get("total_area_m2", float("nan"))) > 1e-6:
                violations.append(f"{c.id}: TOTAL_AREA {c.value} does not match target total_area_m2")
        elif c.type in (ConstraintType.ADJACENCY, ConstraintType.DIRECT_ACCESS):
            a, b = c.target.split("<->")
            want_rel = "ADJACENT" if c.type == ConstraintType.ADJACENCY else "DOOR_CONNECTION"
            if (a, b, want_rel) not in rel_pairs and (b, a, want_rel) not in rel_pairs:
                violations.append(f"{c.id}: {c.type.value} {c.target} has no matching relationship in target_spec")
    return violations


def classify_constraint_violations(example: ArchitectTrainingExample) -> tuple[list[str], list[str]]:
    """Splits find_constraint_violations() output into (hard_violations, soft_violations)
    by looking up each violated constraint's declared priority - HARD violations are
    correctness bugs (a stated requirement the target doesn't satisfy); SOFT ones
    (e.g. an ADJACENCY/DIRECT_ACCESS constraint not matching) are reported separately
    since a SOFT/OBSERVED_GEOMETRY fact not holding is a softer signal.
    """
    by_id = {c.id: c for c in example.constraints}
    hard, soft = [], []
    for v in find_constraint_violations(example):
        constraint_id = v.split(":", 1)[0]
        c = by_id.get(constraint_id)
        (hard if c and c.priority == Priority.HARD else soft).append(v)
    return hard, soft


def find_contradictions(constraints: list[Constraint]) -> list[str]:
    """Detects internally contradictory constraints within one example (Task 45):
    conflicting ROOM_COUNT values for the same target, or MIN_AREA > MAX_AREA for the
    same target. Returns human-readable descriptions; empty list means no contradiction.
    """
    contradictions = []

    room_counts: dict[str, set] = {}
    min_area: dict[str, float] = {}
    max_area: dict[str, float] = {}

    for c in constraints:
        if c.type == ConstraintType.ROOM_COUNT:
            room_counts.setdefault(c.target, set()).add(c.value)
        elif c.type == ConstraintType.MIN_AREA:
            min_area[c.target] = float(c.value)
        elif c.type == ConstraintType.MAX_AREA:
            max_area[c.target] = float(c.value)

    for target, values in room_counts.items():
        if len(values) > 1:
            contradictions.append(f"ROOM_COUNT conflict for {target}: {sorted(values)}")

    for target in set(min_area) & set(max_area):
        if min_area[target] > max_area[target]:
            contradictions.append(f"MIN_AREA {min_area[target]} > MAX_AREA {max_area[target]} for {target}")

    return contradictions


def find_duplicate_relationships(example: ArchitectTrainingExample) -> list[str]:
    """Flags (a_type, b_type, relationship) triples that appear more than once after
    canonicalization (Task 7) - should be empty if relationship_canonicalizer ran.
    """
    seen: dict[tuple, int] = {}
    for r in example.target_spec.relationships:
        a, b = sorted([r.a_type.value, r.b_type.value])
        key = (a, b, r.relationship.value)
        seen[key] = seen.get(key, 0) + 1
    return [f"duplicate relationship {k}: appears {v} times" for k, v in seen.items() if v > 1]


def find_invariant_violations(example: ArchitectTrainingExample) -> list[str]:
    """Task 2's explicit per-example invariant checks, beyond schema-level (pydantic)
    validation and the constraint/contradiction checks above.
    """
    violations = []
    spec = example.target_spec
    brief = example.brief

    program_area_sum = sum(rp.count * rp.area_per_room_m2 for rp in spec.program)
    if brief.target_area_m2 is not None and not area_within_tolerance(program_area_sum, brief.target_area_m2):
        violations.append(
            f"sum(program.count*area_per_room_m2)={program_area_sum:.2f} outside tolerance of "
            f"brief.target_area_m2={brief.target_area_m2} (>{AREA_TOLERANCE_ABS_M2}m2 and >{AREA_TOLERANCE_REL_PCT:.0%})"
        )

    spec_total = spec.metadata.get("total_area_m2")
    if brief.target_area_m2 is not None and spec_total is not None:
        if abs(spec_total - brief.target_area_m2) > 1e-6:
            violations.append(f"target_spec.metadata.total_area_m2={spec_total} != brief.target_area_m2={brief.target_area_m2}")

    if example.site.area_m2 <= 0:
        violations.append(f"site.area_m2={example.site.area_m2} is not positive")

    room_counts = {rp.type.value: rp.count for rp in spec.program}
    for field_name, room_types in (
        ("bedrooms", (RoomType.BEDROOM.value, RoomType.MASTER_BEDROOM.value)),
        ("bathrooms", (RoomType.BATHROOM.value, RoomType.WC.value)),
        ("balconies", (RoomType.BALCONY.value,)),
    ):
        brief_value = getattr(brief, field_name)
        spec_value = sum(room_counts.get(t, 0) for t in room_types)
        if brief_value is not None and brief_value != spec_value:
            violations.append(f"brief.{field_name}={brief_value} != target_spec program count {spec_value}")

    for c in example.constraints:
        if c.source_type == SourceType.OBSERVED_GEOMETRY and c.priority == Priority.HARD:
            violations.append(f"{c.id}: HARD + OBSERVED_GEOMETRY is not allowed (observed facts must be SOFT)")

    violations.extend(find_constraint_violations(example))
    violations.extend(find_duplicate_relationships(example))

    return violations


def constraint_coverage_ratio(n_sampled_constraints: int, pool_size: int) -> float:
    """Fraction of the full observed-fact pool exposed as input constraints (Task 4/37):
    facts-from-target-that-appear-in-input / total-facts-in-target. 0.0 = no facts
    exposed (Brief only). 1.0 = every extractable fact exposed - an example teaching
    "copy the input" rather than "design a plan".
    """
    if pool_size == 0:
        return 0.0
    return n_sampled_constraints / pool_size


def assert_no_split_leakage(plan_id_sets: dict[str, set[int]]) -> None:
    """Raises AssertionError if any two named plan_id sets (e.g. {"train": ..., "validation": ...})
    share a plan_id (Task 53/54). No return value - call it for its side effect.
    """
    names = list(plan_id_sets)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            overlap = plan_id_sets[a] & plan_id_sets[b]
            assert not overlap, f"plan_id leakage between {a} and {b}: {sorted(overlap)[:10]}"
