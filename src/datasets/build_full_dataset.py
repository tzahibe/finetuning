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
# BEFORE any per-level/per-difficulty variant is generated (Task 2/3 of the full-
# generation brief). This is intentionally a fresh split, not a reuse of BOOMI's own
# split boundaries (which are ~88/6/6, not 80/10/10).
SPLIT_RATIOS = {"train": 0.8, "validation": 0.1, "test": 0.1}
SPLIT_SEED = 20260904
GENERATION_SEED = 42
DIFFICULTY_LEVELS = list(DifficultyLevel)


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _load_all_rows() -> list[dict]:
    ds = load_dataset(DATASET_SOURCE)
    rows = []
    for split in ds:
        rows.extend(ds[split])
    return rows


def _assign_splits(plan_ids: list[int]) -> dict[int, str]:
    rng = random.Random(SPLIT_SEED)
    ids = sorted(plan_ids)  # deterministic base order before shuffling
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


def build_full_dataset(
    processed_dir: str = "data/processed",
    training_dir: str = "data/training",
    report_path: str = "data/reports/architect_full_report.json",
) -> dict:
    """Generates the complete BOOMI-derived Architect AI dataset: every (plan_id,
    level) pair, split 80/10/10 by plan_id (all levels of a plan stay together, per
    Task 2). Uses build_example() UNCHANGED from the validated 1000-sample pipeline -
    no schema/normalization/constraint semantics are touched here.

    Writes, per split:
    - {processed_dir}/architect_full_{split}.jsonl: full ArchitectTrainingExample
      records (source_plan_id, caption, pipeline_version, ... - QA/debug traceability)
    - {training_dir}/architect_{split}.jsonl: clean {"input":..., "target":...} pairs
      via to_sft_pair() - what actually gets tokenized, no dataset metadata
    """
    all_rows = _load_all_rows()
    plan_ids = sorted({r["plan_id"] for r in all_rows})
    split_of = _assign_splits(plan_ids)

    rows_by_split: dict[str, list[dict]] = {"train": [], "validation": [], "test": []}
    for r in all_rows:
        rows_by_split[split_of[r["plan_id"]]].append(r)

    rng = random.Random(GENERATION_SEED)

    Path(processed_dir).mkdir(parents=True, exist_ok=True)
    Path(training_dir).mkdir(parents=True, exist_ok=True)

    split_summaries = {}
    all_rejected = []

    for split_name, rows in rows_by_split.items():
        full_path = Path(processed_dir) / f"architect_full_{split_name}.jsonl"
        train_path = Path(training_dir) / f"architect_{split_name}.jsonl"

        accepted = 0
        accepted_plan_ids: set[int] = set()
        with full_path.open("w") as ff, train_path.open("w") as tf:
            for row in rows:
                difficulty = DIFFICULTY_LEVELS[rng.randrange(len(DIFFICULTY_LEVELS))]
                try:
                    example, qa_info = build_example(row, difficulty, rng)
                except Exception as e:
                    all_rejected.append(
                        {
                            "plan_id": row["plan_id"],
                            "level": row["level"],
                            "split": split_name,
                            "reasons": ["processing_error"],
                            "error": f"{type(e).__name__}: {e}",
                        }
                    )
                    continue

                reasons = []
                if qa_info["unknown_room_types"] or qa_info["unknown_relationship_types"]:
                    reasons.append("unknown_vocabulary")
                if qa_info["invariant_violations"]:
                    reasons.append("invariant_violation")
                if qa_info["contradictions"]:
                    reasons.append("contradiction")

                if reasons:
                    all_rejected.append({"plan_id": qa_info["plan_id"], "split": split_name, "reasons": reasons, **qa_info})
                    continue

                ff.write(example.model_dump_json() + "\n")
                tf.write(json.dumps(to_sft_pair(example)) + "\n")
                accepted += 1
                accepted_plan_ids.add(row["plan_id"])

        split_summaries[split_name] = {
            "examples": accepted,
            "unique_source_plan_ids": len(accepted_plan_ids),
            "_plan_ids": accepted_plan_ids,  # used for leakage check below, stripped before saving
        }

    assert_no_split_leakage({name: s["_plan_ids"] for name, s in split_summaries.items()})
    leakage_checked_ok = True  # assert_no_split_leakage raises on failure, so reaching here means OK

    report = {
        "dataset_version": PIPELINE_VERSION,
        "source_dataset": DATASET_SOURCE,
        "git_commit": _git_commit(),
        "split_seed": SPLIT_SEED,
        "generation_seed": GENERATION_SEED,
        "split_ratios": SPLIT_RATIOS,
        "splits": {name: {"examples": s["examples"], "unique_source_plan_ids": s["unique_source_plan_ids"]} for name, s in split_summaries.items()},
        "total_examples": sum(s["examples"] for s in split_summaries.values()),
        "total_unique_source_plan_ids": sum(s["unique_source_plan_ids"] for s in split_summaries.values()),
        "total_rejected": len(all_rejected),
        "source_plan_split_leakage": 0 if leakage_checked_ok else "FAILED",
        "rejected_detail": all_rejected[:100],
    }

    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return report


if __name__ == "__main__":
    report = build_full_dataset()
    print(json.dumps({k: v for k, v in report.items() if k != "rejected_detail"}, indent=2))
