from src.datasets.schema import ArchitectTrainingExample, Constraint, ConstraintType


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


def anti_leakage_ratio(n_sampled_constraints: int, pool_size: int) -> float:
    """Fraction of the full observed-fact pool exposed as input constraints (Task 37).
    0.0 = no facts exposed (Brief only). 1.0 = every extractable fact exposed. High
    values mean the example teaches "copy the input" rather than "design a plan".
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
