import json

import pytest
from datasets import load_dataset

from src.datasets.build_dataset import build_dataset
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


@pytest.mark.slow
def test_generated_examples_from_different_splits_do_not_share_plan_ids(tmp_path):
    """End-to-end version of the same rule (Task 6): build a small example set from
    two different BOOMI splits and confirm OUR generator's own output never mixes
    plan_ids across them, not just the raw source data.
    """
    train_path = tmp_path / "train.jsonl"
    val_path = tmp_path / "validation.jsonl"
    build_dataset(n_samples=5, output_path=str(train_path), split="train")
    build_dataset(n_samples=5, output_path=str(val_path), split="validation")

    def _plan_ids(path):
        with open(path) as f:
            return {json.loads(line)["metadata"]["source_plan_id"] for line in f}

    assert_no_split_leakage({"train": _plan_ids(train_path), "validation": _plan_ids(val_path)})
