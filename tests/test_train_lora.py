import pytest
from transformers import AutoTokenizer

from src.training.train_lora import MODEL_ID, _collate, build_training_example


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_ID)


def _row():
    return {
        "input": {
            "brief": {"building_type": "residential", "target_area_m2": 60.0, "bedrooms": 2, "bathrooms": 1, "balconies": 0, "preferences": []},
            "site": {"width_m": 10.0, "length_m": 8.0, "area_m2": 80.0},
            "constraints": [],
        },
        "target": {
            "program": [{"type": "BEDROOM", "count": 2, "area_per_room_m2": 15.0, "zone": "PRIVATE"}],
            "zones": [{"type": "PRIVATE", "room_types": ["BEDROOM"]}],
            "relationships": [],
            "circulation": [],
        },
    }


@pytest.mark.slow
def test_labels_mask_the_prompt_not_the_completion(tokenizer):
    example = build_training_example(tokenizer, _row())
    n_masked = sum(1 for l in example["labels"] if l == -100)
    n_unmasked = sum(1 for l in example["labels"] if l != -100)
    assert n_masked > 0
    assert n_unmasked > 0
    assert n_masked + n_unmasked == len(example["input_ids"])


@pytest.mark.slow
def test_unmasked_labels_decode_to_target_json(tokenizer):
    example = build_training_example(tokenizer, _row())
    completion_ids = [l for l in example["labels"] if l != -100]
    decoded = tokenizer.decode(completion_ids, skip_special_tokens=True)
    assert '"BEDROOM"' in decoded
    assert decoded.strip().startswith("{")


@pytest.mark.slow
def test_truncation_respects_max_length(tokenizer):
    example = build_training_example(tokenizer, _row(), max_length=20)
    assert len(example["input_ids"]) == 20
    assert len(example["labels"]) == 20


def test_collate_pads_to_longest_in_batch():
    batch = [
        {"input_ids": [1, 2, 3], "labels": [-100, -100, 5], "attention_mask": [1, 1, 1]},
        {"input_ids": [1, 2], "labels": [-100, 7], "attention_mask": [1, 1]},
    ]
    out = _collate(batch, pad_token_id=0)
    assert out["input_ids"].shape == (2, 3)
    assert out["input_ids"][1].tolist() == [1, 2, 0]
    assert out["labels"][1].tolist() == [-100, 7, -100]
    assert out["attention_mask"][1].tolist() == [1, 1, 0]
