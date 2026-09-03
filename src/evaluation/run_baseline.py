import json
import random
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.evaluation.metrics import (
    area_error,
    check_schema_validity,
    extract_json,
    hard_constraint_satisfaction,
    relationship_accuracy,
    room_count_accuracy,
)
from src.evaluation.prompts import build_prompt

MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"
TEST_PATH = "data/training/architect_test.jsonl"
SAMPLE_SEED = 42
MAX_NEW_TOKENS = 1024


def load_fixed_sample(n: int, path: str = TEST_PATH, seed: int = SAMPLE_SEED) -> list[dict]:
    """Deterministic sample: fixed seed, fixed source file, fixed indexing - the same
    n always returns the same examples.
    """
    with open(path) as f:
        rows = [json.loads(line) for line in f]
    rng = random.Random(seed)
    indices = list(range(len(rows)))
    rng.shuffle(indices)
    chosen = sorted(indices[:n])  # sort for stable file order
    return [{"index": i, **rows[i]} for i in chosen]


def load_model():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16).to(device)
    model.eval()
    return tokenizer, model, device


def generate(tokenizer, model, device, messages: list[dict]) -> str:
    prompt_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        output_ids = model.generate(
            prompt_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,  # deterministic, greedy decoding
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output_ids[0][prompt_ids.shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def evaluate_example(raw_output: str, example: dict) -> dict:
    target = example["target"]
    input_ = example["input"]

    parsed, parse_error = extract_json(raw_output)
    result = {
        "index": example["index"],
        "json_valid": parsed is not None,
        "json_error": parse_error,
        "schema_valid": False,
        "schema_error": None,
        "room_count_accuracy": None,
        "hard_constraint_satisfaction": None,
        "relationship_accuracy": None,
        "area_error": None,
    }
    if parsed is None:
        return result

    schema_valid, schema_error = check_schema_validity(parsed)
    result["schema_valid"] = schema_valid
    result["schema_error"] = schema_error

    predicted_program = parsed.get("program") if isinstance(parsed.get("program"), list) else []
    predicted_relationships = parsed.get("relationships") if isinstance(parsed.get("relationships"), list) else []

    result["room_count_accuracy"] = round(room_count_accuracy(predicted_program, target["program"]), 3)
    result["hard_constraint_satisfaction"] = hard_constraint_satisfaction(input_["constraints"], predicted_program)
    result["relationship_accuracy"] = relationship_accuracy(predicted_relationships, target["relationships"])
    if input_["brief"].get("target_area_m2") is not None:
        result["area_error"] = area_error(predicted_program, input_["brief"]["target_area_m2"])

    return result


def run_baseline(n: int, output_dir: str = "data/evaluation") -> dict:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    raw_path = Path(output_dir) / "baseline_qwen2.5-coder-7b_raw.jsonl"
    results_path = Path(output_dir) / "baseline_qwen2.5-coder-7b_results.jsonl"

    examples = load_fixed_sample(n)
    tokenizer, model, device = load_model()
    print(f"Model loaded on {device}. Running {len(examples)} examples...")

    all_results = []
    with raw_path.open("w") as rf, results_path.open("w") as ef:
        for example in examples:
            t0 = time.time()
            messages = build_prompt(example["input"])
            raw_output = generate(tokenizer, model, device, messages)
            elapsed = round(time.time() - t0, 1)

            rf.write(json.dumps({"index": example["index"], "example_id": example["example_id"], "raw_output": raw_output, "seconds": elapsed}) + "\n")

            result = evaluate_example(raw_output, example)
            ef.write(json.dumps(result) + "\n")
            all_results.append(result)
            print(f"  [{example['index']}] {example['example_id']} json_valid={result['json_valid']} schema_valid={result['schema_valid']} ({elapsed}s)")

    return _summarize(all_results, n)


def _summarize(results: list[dict], n: int) -> dict:
    json_valid_n = sum(1 for r in results if r["json_valid"])
    schema_valid_n = sum(1 for r in results if r["schema_valid"])

    room_acc = [r["room_count_accuracy"] for r in results if r["room_count_accuracy"] is not None]
    hard_sat = [r["hard_constraint_satisfaction"] for r in results if r["hard_constraint_satisfaction"] is not None]
    rel_f1 = [r["relationship_accuracy"]["f1"] for r in results if r["relationship_accuracy"] and r["relationship_accuracy"]["f1"] is not None]
    area_abs_err = [r["area_error"]["abs_error_m2"] for r in results if r["area_error"]]
    area_rel_err = [r["area_error"]["rel_error_pct"] for r in results if r["area_error"] and r["area_error"]["rel_error_pct"] is not None]

    def avg(xs):
        return round(sum(xs) / len(xs), 3) if xs else None

    return {
        "model": MODEL_ID,
        "sample_size": n,
        "sample_seed": SAMPLE_SEED,
        "source_file": TEST_PATH,
        "metrics": {
            "json_validity_rate": round(json_valid_n / n, 3),
            "schema_validity_rate": round(schema_valid_n / n, 3),
            "room_count_accuracy_mean": avg(room_acc),
            "hard_constraint_satisfaction_mean": avg(hard_sat),
            "relationship_f1_mean": avg(rel_f1),
            "area_abs_error_m2_mean": avg(area_abs_err),
            "area_rel_error_pct_mean": avg(area_rel_err),
        },
        "n_json_valid": json_valid_n,
        "n_schema_valid": schema_valid_n,
    }


if __name__ == "__main__":
    import sys

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    summary = run_baseline(n)
    Path("data/reports/baseline_qwen2.5-coder-7b_report.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
