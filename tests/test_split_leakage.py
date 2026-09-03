import pytest
from datasets import load_dataset

from src.validation.dataset_qa import assert_no_split_leakage


@pytest.mark.slow
def test_boomi_splits_have_no_plan_id_overlap():
    """Mandatory per SKILL.md audit Task 53/54: the build must fail if any two
    splits share a source plan_id, since all L0-L3 (and future C0-C3) variants of
    a plan must stay in the same split.
    """
    ds = load_dataset("BDivyesh/boomi-stage-a-text-spec")
    plan_id_sets = {split: set(ds[split]["plan_id"]) for split in ds}
    assert_no_split_leakage(plan_id_sets)
