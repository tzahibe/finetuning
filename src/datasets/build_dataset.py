import json
import random
import subprocess
from pathlib import Path

from datasets import load_dataset

from src.datasets.schema import ArchitectTrainingExample
from src.preprocessing.boomi_parser import parse_boomi_record
from src.preprocessing.brief_generator import generate_brief
from src.preprocessing.constraint_generator import DifficultyLevel, build_constraint_pool, sample_constraints
from src.preprocessing.site_generator import generate_site
from src.preprocessing.spec_generator import generate_spec
from src.validation.dataset_qa import anti_leakage_ratio, find_constraint_violations, find_contradictions

DIFFICULTY_LEVELS = list(DifficultyLevel)
PIPELINE_VERSION = "0.2.0"  # bump when normalization/schema/constraint logic changes
DATASET_SOURCE = "BDivyesh/boomi-stage-a-text-spec"


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def build_example(row: dict, difficulty: DifficultyLevel, rng: random.Random) -> tuple[ArchitectTrainingExample, dict]:
    """Returns (example, qa_info). qa_info always describes the example even if it's later rejected."""
    record = parse_boomi_record(row)
    brief = generate_brief(record)
    site = generate_site(record)
    target_spec = generate_spec(record)
    pool = build_constraint_pool(record)
    constraints = sample_constraints(pool, difficulty, rng)

    example = ArchitectTrainingExample(
        example_id=f"{record.plan_id}-{difficulty.value}",
        brief=brief,
        site=site,
        constraints=constraints,
        target_spec=target_spec,
        metadata={
            "source": "BOOMI",
            "source_dataset": DATASET_SOURCE,
            "source_plan_id": record.plan_id,
            "source_level": record.level,
            "caption": record.caption,
            "difficulty": difficulty.value,
            "pipeline_version": PIPELINE_VERSION,
            "unknown_room_types": sorted(record.unknown_room_types),
            "unknown_relationship_types": sorted(record.unknown_relationship_types),
            "area_consistency_diff_m2": record.area_consistency_diff_m2,
        },
    )

    violations = find_constraint_violations(example)
    contradictions = find_contradictions(constraints)
    qa_info = {
        "plan_id": record.plan_id,
        "difficulty": difficulty.value,
        "pool_size": len(pool),
        "n_constraints": len(constraints),
        "anti_leakage_ratio": round(anti_leakage_ratio(len(constraints), len(pool)), 3),
        "constraint_violations": violations,
        "contradictions": contradictions,
        "unknown_room_types": sorted(record.unknown_room_types),
        "unknown_relationship_types": sorted(record.unknown_relationship_types),
        "area_consistency_diff_m2": record.area_consistency_diff_m2,
    }
    return example, qa_info


def build_dataset(
    n_samples: int,
    output_path: str,
    report_path: str | None = None,
    split: str = "train",
    seed: int = 42,
    level: str = "L3",
) -> dict:
    """Builds up to n_samples ArchitectTrainingExamples from distinct BOOMI plans, writes
    them as JSONL, and returns/writes a generation report (Task 56). Examples with
    constraint violations, internal contradictions, or unknown vocabulary are rejected
    (not silently dropped - logged in the report, Task 57) rather than written out.
    """
    ds = load_dataset(DATASET_SOURCE, split=split)
    ds = ds.filter(lambda r: r["level"] == level)

    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    accepted = 0
    rejected: list[dict] = []
    qa_records: list[dict] = []

    with out_path.open("w") as f:
        for i in indices:
            if accepted >= n_samples:
                break
            difficulty = DIFFICULTY_LEVELS[rng.randrange(len(DIFFICULTY_LEVELS))]
            example, qa_info = build_example(ds[i], difficulty, rng)
            qa_records.append(qa_info)

            # area_consistency_diff_m2 is monitored (Task 11) but NOT a rejection
            # reason: verified over 500 random plans it's source-inherent rounding
            # noise (mean ~0, range -5..+4.6 m2) rather than a real defect, and
            # Task 7 explicitly warns against auto-rejecting outliers without
            # investigation.
            reasons = []
            if qa_info["unknown_room_types"] or qa_info["unknown_relationship_types"]:
                reasons.append("unknown_vocabulary")
            if qa_info["constraint_violations"]:
                reasons.append("constraint_violation")
            if qa_info["contradictions"]:
                reasons.append("contradiction")

            if reasons:
                rejected.append({"plan_id": qa_info["plan_id"], "reasons": reasons, **qa_info})
                continue

            f.write(example.model_dump_json() + "\n")
            accepted += 1

    report = {
        "dataset_version": PIPELINE_VERSION,
        "source_dataset": DATASET_SOURCE,
        "source_split": split,
        "source_level": level,
        "git_commit": _git_commit(),
        "requested_samples": n_samples,
        "accepted_examples": accepted,
        "rejected_examples": len(rejected),
        "rejection_reasons": _count_reasons(rejected),
        "anti_leakage_ratio": {
            "mean": round(sum(r["anti_leakage_ratio"] for r in qa_records) / len(qa_records), 3)
            if qa_records
            else 0.0,
            "max": max((r["anti_leakage_ratio"] for r in qa_records), default=0.0),
            "n_above_0.8": sum(1 for r in qa_records if r["anti_leakage_ratio"] > 0.8),
        },
        "area_consistency_diff_m2": {
            "mean": round(sum(r["area_consistency_diff_m2"] for r in qa_records) / len(qa_records), 3)
            if qa_records
            else 0.0,
            "max": max((r["area_consistency_diff_m2"] for r in qa_records), default=0.0),
        },
        "rejected_detail": rejected[:50],
    }

    if report_path:
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

    return report


def _count_reasons(rejected: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in rejected:
        for reason in r["reasons"]:
            counts[reason] = counts.get(reason, 0) + 1
    return counts


if __name__ == "__main__":
    report = build_dataset(
        n_samples=100,
        output_path="data/processed/architect_v0_100.jsonl",
        report_path="data/reports/architect_v0_100_report.json",
    )
    print(json.dumps(report, indent=2))
