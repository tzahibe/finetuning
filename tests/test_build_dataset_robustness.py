import random

import pytest
from datasets import load_dataset
from pydantic import ValidationError

from src.datasets.build_dataset import DIFFICULTY_LEVELS, build_example


@pytest.mark.slow
def test_known_bad_plan_raises_a_catchable_validation_error():
    """plan_id 15861 has degenerate site dims in BOOMI itself (width_mm=depth_mm=8.0
    -> area_m2=0.0), a confirmed isolated source defect (1/14891 unique plans,
    0.007%) - not a pipeline bug. build_example() must fail on it with a
    ValidationError (which build_dataset() catches and rejects with a clear reason),
    not silently produce a zero-area Site.
    """
    ds = load_dataset("BDivyesh/boomi-stage-a-text-spec", split="train")
    row = next(r for r in ds if r["plan_id"] == 15861 and r["level"] == "L3")

    with pytest.raises(ValidationError):
        build_example(row, DIFFICULTY_LEVELS[0], random.Random(0))
