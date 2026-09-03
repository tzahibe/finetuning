from src.datasets.build_full_dataset import DIFFICULTY_LEVELS, _variant_rng
from src.datasets.build_dataset import build_example


def _fake_row(plan_id: int) -> dict:
    spec = (
        '{"adjacency":[{"a_type":"BED ROOM","b_type":"BATH","via":"door"},'
        '{"a_type":"BED ROOM","b_type":"KITCHEN","via":"adjacent"}],'
        '"bhk_label":"1BHK","plot":{"area_m2":75.888,"depth_mm":8367.5,"width_mm":10994.6},'
        '"room_program":[{"count":1,"target_area_m2":16.0,"type":"BED ROOM"},'
        '{"count":1,"target_area_m2":3.0,"type":"BATH"},{"count":1,"target_area_m2":8.0,"type":"KITCHEN"}],'
        '"total_area_m2":27.0}'
    )
    return {"plan_id": plan_id, "level": "L3", "caption": "test", "spec": spec}


def test_same_plan_and_difficulty_always_produces_identical_constraints():
    """Task 2: regenerating {plan_id}-{difficulty} must always produce exactly the
    same constraints, independent of iteration order elsewhere.
    """
    row = _fake_row(5397)
    difficulty = DIFFICULTY_LEVELS[2]  # C2

    example_a, _ = build_example(row, difficulty, _variant_rng(5397, difficulty))
    example_b, _ = build_example(row, difficulty, _variant_rng(5397, difficulty))

    ids_a = [c.id for c in example_a.constraints]
    ids_b = [c.id for c in example_b.constraints]
    assert ids_a == ids_b
    assert len(example_a.constraints) > 0  # C2 always samples something


def test_different_difficulty_same_plan_uses_different_seed():
    row = _fake_row(5397)
    rng_c0 = _variant_rng(5397, DIFFICULTY_LEVELS[0])
    rng_c3 = _variant_rng(5397, DIFFICULTY_LEVELS[3])
    assert rng_c0.getstate() != rng_c3.getstate()


def test_different_plan_same_difficulty_uses_different_seed():
    difficulty = DIFFICULTY_LEVELS[1]
    rng_a = _variant_rng(111, difficulty)
    rng_b = _variant_rng(222, difficulty)
    assert rng_a.getstate() != rng_b.getstate()
