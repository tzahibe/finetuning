import json
import random
import subprocess
from pathlib import Path

from datasets import load_dataset

from src.datasets.build_dataset import DATASET_SOURCE, PIPELINE_VERSION, build_example
from src.datasets.sft_format import to_sft_pair
from src.preprocessing.constraint_generator import DifficultyLevel
from src.validation.dataset_qa import assert_no_split_leakage

# Deterministic 80/10/10 split by source_plan_id, computed ONCE over the pooled set of
# every unique plan_id across BOOMI's own train+validation+test (verified disjoint),
# BEFORE any C0-C3 variant is generated. A fresh split, not a reuse of BOOMI's own
# ~88/6/6 split boundaries.
SPLIT_RATIOS = {"train": 0.8, "validation": 0.1, "test": 0.1}
SPLIT_SEED = 20260904
GENERATION_SEED = 42
# Explicit, fixed order - iterated directly, never chosen by rng.randrange().
DIFFICULTY_LEVELS = list(DifficultyLevel)  # [C0, C1, C2, C3]


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _canonical_spec_key(spec_json: str) -> str:
    """Canonicalizes a raw BOOMI spec JSON string for equality comparison, independent
    of key ordering/whitespace.
    """
    return json.dumps(json.loads(spec_json), sort_keys=True)


def _load_canonical_rows() -> tuple[list[dict], list[int]]:
    """Groups every BOOMI row (all splits, all levels) by plan_id, verifies L0-L3
    share the same spec, and selects ONE canonical row per plan (preferring L3) to
    drive generation. L0/L1/L2/L3 must never determine training-example
    multiplicity - only C0/C1/C2/C3 does. Returns (canonical_rows, plan_ids_with_spec_mismatch).
    """
    ds = load_dataset(DATASET_SOURCE)
    by_plan: dict[int, list[dict]] = {}
    for split in ds:
        for row in ds[split]:
            by_plan.setdefault(row["plan_id"], []).append(row)

    level_priority = {"L3": 0, "L2": 1, "L1": 2, "L0": 3}
    canonical_rows = []
    spec_mismatches = []
    for plan_id, rows in by_plan.items():
        spec_keys = {_canonical_spec_key(r["spec"]) for r in rows}
        if len(spec_keys) > 1:
            spec_mismatches.append(plan_id)
            continue  # cannot pick an authoritative spec - reject the whole plan
        rows_sorted = sorted(rows, key=lambda r: level_priority.get(r["level"], 99))
        canonical_rows.append(rows_sorted[0])  # prefer L3

    return canonical_rows, spec_mismatches


def _assign_splits(plan_ids: list[int]) -> dict[int, str]:
    rng = random.Random(SPLIT_SEED)
    ids = sorted(plan_ids)
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(n * SPLIT_RATIOS["train"])
    n_val = int(n * SPLIT_RATIOS["validation"])
    assignment = {}
    for pid in ids[:n_train]:
        assignment[pid] = "train"
    for pid in ids[n_train : n_train + n_val]:
        assignment[pid] = "validation"
    for pid in ids[n_train + n_val :]:
        assignment[pid] = "test"
    return assignment


def _variant_rng(plan_id: int, difficulty: DifficultyLevel) -> random.Random:
    """Deterministic seed derived from GENERATION_SEED + source_plan_id + difficulty
    (Task 2) - regenerating a single (plan_id, difficulty) always reproduces the same
    constraints regardless of iteration order elsewhere.
    """
    return random.Random(f"{GENERATION_SEED}:{plan_id}:{difficulty.value}")


def build_full_dataset(
    processed_dir: str = "data/processed",
    training_dir: str = "data/training",
    report_path: str = "data/reports/architect_full_report.json",
) -> dict:
    """Generates the complete BOOMI-derived Architect AI dataset: exactly ONE
    canonical source spec per plan_id, and exactly FOUR variants per accepted plan
    (plan_id-C0, -C1, -C2, -C3), split 80/10/10 by plan_id (all 4 variants of a plan
    always in the same split). Uses build_example() UNCHANGED from the validated
    pipeline - no schema/normalization/constraint semantics touched here.
    """
    canonical_rows, spec_mismatches = _load_canonical_rows()
    plan_ids = sorted(r["plan_id"] for r in canonical_rows)
    split_of = _assign_splits(plan_ids)

    rows_by_split: dict[str, list[dict]] = {"train": [], "validation": [], "test": []}
    for row in canonical_rows:
        rows_by_split[split_of[row["plan_id"]]].append(row)

    Path(processed_dir).mkdir(parents=True, exist_ok=True)
    Path(training_dir).mkdir(parents=True, exist_ok=True)

    split_summaries = {}
    all_rejected = []
    per_split_plan_ids: dict[str, set[int]] = {}
    global_difficulty_counts = {d.value: 0 for d in DIFFICULTY_LEVELS}
    global_example_ids: list[str] = []

    for split_name, rows in rows_by_split.items():
        full_path = Path(processed_dir) / f"architect_full_{split_name}.jsonl"
        train_path = Path(training_dir) / f"architect_{split_name}.jsonl"

        accepted_plan_ids: set[int] = set()
        accepted_examples = 0

        with full_path.open("w") as ff, train_path.open("w") as tf:
            for row in rows:
                plan_id = row["plan_id"]
                variants: list[tuple] = []  # (example, qa_info)
                reject_reason = None

                for difficulty in DIFFICULTY_LEVELS:  # explicit C0,C1,C2,C3 - never random
                    rng = _variant_rng(plan_id, difficulty)
                    try:
                        example, qa_info = build_example(row, difficulty, rng)
                    except Exception as e:
                        reject_reason = {"reasons": ["processing_error"], "error": f"{type(e).__name__}: {e}", "difficulty": difficulty.value}
                        break

                    reasons = []
                    if qa_info["unknown_room_types"] or qa_info["unknown_relationship_types"]:
                        reasons.append("unknown_vocabulary")
                    if qa_info["invariant_violations"]:
                        reasons.append("invariant_violation")
                    if qa_info["contradictions"]:
                        reasons.append("contradiction")
                    if reasons:
                        reject_reason = {"reasons": reasons, "difficulty": difficulty.value}
                        break

                    variants.append((example, qa_info))

                # Atomic per-plan acceptance: a plan contributes ALL 4 variants or NONE.
                if reject_reason is not None or len(variants) != 4:
                    all_rejected.append({"plan_id": plan_id, "split": split_name, **(reject_reason or {"reasons": ["incomplete_variants"]})})
                    continue

                for example, qa_info in variants:
                    ff.write(example.model_dump_json() + "\n")
                    tf.write(json.dumps(to_sft_pair(example)) + "\n")
                    global_difficulty_counts[qa_info["difficulty"]] += 1
                    global_example_ids.append(example.example_id)
                accepted_examples += 4
                accepted_plan_ids.add(plan_id)

        split_summaries[split_name] = {"examples": accepted_examples, "unique_source_plan_ids": len(accepted_plan_ids)}
        per_split_plan_ids[split_name] = accepted_plan_ids

    assert_no_split_leakage(per_split_plan_ids)

    total_accepted_plans = sum(s["unique_source_plan_ids"] for s in split_summaries.values())
    total_examples = sum(s["examples"] for s in split_summaries.values())

    # ---- Task 6: mandatory assertions - the process MUST fail if any is false ----
    assert total_examples == len(set(global_example_ids)), (
        f"TOTAL_EXAMPLES ({total_examples}) != UNIQUE_EXAMPLE_IDS ({len(set(global_example_ids))})"
    )
    for d in DIFFICULTY_LEVELS:
        assert global_difficulty_counts[d.value] == total_accepted_plans, (
            f"{d.value}_count ({global_difficulty_counts[d.value]}) != ACCEPTED_SOURCE_PLAN_IDS ({total_accepted_plans})"
        )

    report = {
        "dataset_version": PIPELINE_VERSION,
        "source_dataset": DATASET_SOURCE,
        "git_commit": _git_commit(),
        "split_seed": SPLIT_SEED,
        "generation_seed": GENERATION_SEED,
        "split_ratios": SPLIT_RATIOS,
        "source_unique_plan_ids": len(plan_ids) + len(spec_mismatches),
        "spec_mismatch_plan_ids": spec_mismatches,
        "rejected_source_plan_ids": len(spec_mismatches) + len(all_rejected),
        "accepted_source_plan_ids": total_accepted_plans,
        "splits": split_summaries,
        "total_examples": total_examples,
        "unique_example_ids": len(set(global_example_ids)),
        "difficulty_counts": global_difficulty_counts,
        "source_plan_split_leakage": 0,
        "rejected_detail": all_rejected[:100],
    }

    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return report


if __name__ == "__main__":
    report = build_full_dataset()
    print(json.dumps({k: v for k, v in report.items() if k != "rejected_detail"}, indent=2))
