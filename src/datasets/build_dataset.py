import random
from pathlib import Path

from datasets import load_dataset

from src.datasets.schema import ArchitectTrainingExample
from src.preprocessing.boomi_parser import parse_boomi_record
from src.preprocessing.brief_generator import generate_brief
from src.preprocessing.constraint_generator import DifficultyLevel, build_constraint_pool, sample_constraints
from src.preprocessing.spec_generator import generate_spec

DIFFICULTY_LEVELS = list(DifficultyLevel)


def build_example(row: dict, difficulty: DifficultyLevel, rng: random.Random) -> ArchitectTrainingExample:
    record = parse_boomi_record(row)
    brief = generate_brief(record)
    target_spec = generate_spec(record)
    pool = build_constraint_pool(record)
    constraints = sample_constraints(pool, difficulty, rng)

    return ArchitectTrainingExample(
        example_id=f"{record.plan_id}-{difficulty.value}",
        brief=brief,
        constraints=constraints,
        target_spec=target_spec,
        metadata={
            "plan_id": record.plan_id,
            "level": record.level,
            "caption": record.caption,
            "difficulty": difficulty.value,
            "unknown_room_types": sorted(record.unknown_room_types),
            "unknown_relationship_types": sorted(record.unknown_relationship_types),
        },
    )


def build_dataset(
    n_samples: int,
    output_path: str,
    split: str = "train",
    seed: int = 42,
    level: str = "L3",
) -> None:
    """Builds n_samples ArchitectTrainingExamples from distinct BOOMI plans and writes them as JSONL."""
    ds = load_dataset("BDivyesh/boomi-stage-a-text-spec", split=split)
    ds = ds.filter(lambda r: r["level"] == level)

    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    chosen = indices[:n_samples]

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for i in chosen:
            difficulty = DIFFICULTY_LEVELS[rng.randrange(len(DIFFICULTY_LEVELS))]
            example = build_example(ds[i], difficulty, rng)
            f.write(example.model_dump_json() + "\n")


if __name__ == "__main__":
    build_dataset(n_samples=100, output_path="data/processed/architect_v0_100.jsonl")
