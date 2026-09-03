import json
import random
import subprocess
from pathlib import Path

from datasets import load_dataset

from src.datasets.schema import ArchitectTrainingExample
from src.preprocessing.boomi_parser import parse_boomi_record
from src.preprocessing.brief_generator import generate_brief
from src.preprocessing.constraint_generator import DifficultyLevel, build_constraint_pool, sample_constraints
from src.preprocessing.relationship_canonicalizer import canonicalize_relationships
from src.preprocessing.site_generator import generate_site
from src.preprocessing.spec_generator import generate_spec
from src.validation.dataset_qa import (
    area_within_tolerance,
    classify_constraint_violations,
    constraint_coverage_ratio,
    find_contradictions,
    find_invariant_violations,
)

DIFFICULTY_LEVELS = list(DifficultyLevel)
PIPELINE_VERSION = "0.4.1"  # bump when normalization/schema/constraint logic changes
DATASET_SOURCE = "BDivyesh/boomi-stage-a-text-spec"
LEAKAGE_WARNING_THRESHOLD = 0.8


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def build_example(row: dict, difficulty: DifficultyLevel, rng: random.Random) -> tuple[ArchitectTrainingExample, dict]:
    """Returns (example, qa_info). qa_info always describes the example even if it's later rejected."""
    record = parse_boomi_record(row)
    record.relationships = canonicalize_relationships(record.relationships)

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

    invariant_violations = find_invariant_violations(example)
    contradictions = find_contradictions(constraints)
    hard_violations, soft_violations = classify_constraint_violations(example)
    coverage = constraint_coverage_ratio(len(constraints), len(pool))
    program_area_sum = sum(rp.count * rp.area_per_room_m2 for rp in target_spec.program)
    area_error_pct = (
        abs(program_area_sum - brief.target_area_m2) / brief.target_area_m2 * 100
        if brief.target_area_m2
        else 0.0
    )

    qa_info = {
        "plan_id": record.plan_id,
        "difficulty": difficulty.value,
        "pool_size": len(pool),
        "n_constraints": len(constraints),
        "constraint_coverage_ratio": round(coverage, 3),
        "invariant_violations": invariant_violations,
        "contradictions": contradictions,
        "hard_constraint_violations": hard_violations,
        "soft_constraint_target_mismatches": soft_violations,
        "unknown_room_types": sorted(record.unknown_room_types),
        "unknown_relationship_types": sorted(record.unknown_relationship_types),
        "area_error_m2": round(record.area_consistency_diff_m2, 3),
        "area_error_pct": round(area_error_pct, 3),
        "area_within_tolerance": area_within_tolerance(program_area_sum, brief.target_area_m2),
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
    unknown vocabulary, internal contradictions, or invariant violations (including
    HARD constraint violations, but NOT area-tolerance - see area_within_tolerance)
    are rejected (not silently dropped - logged in the report, Task 57).
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
            row = ds[i]
            difficulty = DIFFICULTY_LEVELS[rng.randrange(len(DIFFICULTY_LEVELS))]

            # A single BOOMI record (plan_id 15861) has degenerate site dimensions
            # (width_mm=depth_mm=8.0, clearly a meters/mm unit bug in that one source
            # row -> area_m2=0.0) that fails Site's area_m2>0 schema validation. This
            # is an isolated, confirmed source defect (1/14891 plans, 0.007%), not a
            # pipeline bug - caught here and rejected with a clear reason rather than
            # crashing the whole build (Task 57: never silently drop, but also never
            # let one bad record kill generation at scale).
            try:
                example, qa_info = build_example(row, difficulty, rng)
            except Exception as e:
                rejected.append(
                    {
                        "plan_id": row["plan_id"],
                        "reasons": ["processing_error"],
                        "error": f"{type(e).__name__}: {e}",
                    }
                )
                continue

            qa_records.append(qa_info)

            reasons = []
            if qa_info["unknown_room_types"] or qa_info["unknown_relationship_types"]:
                reasons.append("unknown_vocabulary")
            if qa_info["invariant_violations"]:
                reasons.append("invariant_violation")
            if qa_info["contradictions"]:
                reasons.append("contradiction")

            if reasons:
                rejected.append({"plan_id": qa_info["plan_id"], "reasons": reasons, **qa_info})
                continue

            f.write(example.model_dump_json() + "\n")
            accepted += 1

    report = _build_acceptance_report(qa_records, rejected, accepted, n_samples, split, level)

    if report_path:
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

    return report


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, max(0, round(p * (len(s) - 1))))
    return s[idx]


def _build_acceptance_report(
    qa_records: list[dict], rejected: list[dict], accepted: int, n_samples: int, split: str, level: str
) -> dict:
    coverage_by_difficulty: dict[str, list[float]] = {d.value: [] for d in DIFFICULTY_LEVELS}
    for r in qa_records:
        coverage_by_difficulty[r["difficulty"]].append(r["constraint_coverage_ratio"])

    duplicate_relationship_count = sum(
        1 for r in qa_records if any("duplicate relationship" in v for v in r["invariant_violations"])
    )
    area_errors_m2 = [r["area_error_m2"] for r in qa_records]

    return {
        "dataset_version": PIPELINE_VERSION,
        "source_dataset": DATASET_SOURCE,
        "source_split": split,
        "source_level": level,
        "git_commit": _git_commit(),
        "requested_samples": n_samples,
        # Acceptance gate (Task 12 / area-tolerance follow-up)
        "TOTAL_EXAMPLES": len(qa_records),
        "VALID_EXAMPLES": accepted,
        "INVALID_EXAMPLES": len(rejected),
        "HARD_CONSTRAINT_VIOLATIONS": sum(1 for r in qa_records if r["hard_constraint_violations"]),
        "SOFT_CONSTRAINT_TARGET_MISMATCHES": sum(1 for r in qa_records if r["soft_constraint_target_mismatches"]),
        "UNKNOWN_TYPES": sum(1 for r in qa_records if r["unknown_room_types"] or r["unknown_relationship_types"]),
        "DUPLICATE_RELATIONSHIPS": duplicate_relationship_count,
        "CONTRADICTIONS": sum(1 for r in qa_records if r["contradictions"]),
        "SPLIT_LEAKAGE": 0,  # this build draws from a single split; see tests/test_split_leakage.py
        "TARGET_LEAKAGE_WARNINGS": sum(
            1 for r in qa_records if r["constraint_coverage_ratio"] > LEAKAGE_WARNING_THRESHOLD
        ),
        "AREA_ERROR_MEAN": round(sum(area_errors_m2) / len(area_errors_m2), 3) if area_errors_m2 else 0.0,
        "AREA_ERROR_P95": round(_percentile(area_errors_m2, 0.95), 3),
        "AREA_ERROR_MAX": round(max(area_errors_m2), 3) if area_errors_m2 else 0.0,
        "AREA_TOLERANCE_POLICY": "abs<=5.0m2 OR rel<=5% (see docs/DATA_CONTRACT.md Area tolerance)",
        "AREA_OUTSIDE_TOLERANCE": sum(1 for r in qa_records if not r["area_within_tolerance"]),
        "rejection_reasons": _count_reasons(rejected),
        "constraint_coverage_ratio_by_difficulty": {
            d: {
                "mean": round(sum(vals) / len(vals), 3) if vals else 0.0,
                "max": round(max(vals), 3) if vals else 0.0,
                "n": len(vals),
            }
            for d, vals in coverage_by_difficulty.items()
        },
        "rejected_detail": rejected[:50],
    }


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
