import json
import re

from pydantic import ValidationError

from src.datasets.schema import ArchitecturalSpec


def extract_json(raw_text: str) -> tuple[dict | None, str | None]:
    """Best-effort extraction of a JSON object from raw model output (strips markdown
    code fences, takes the first balanced {...} block). Returns (parsed_dict, error).
    """
    text = raw_text.strip()
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()

    start = text.find("{")
    if start == -1:
        return None, "no '{' found in output"

    depth = 0
    end = None
    for i, ch in enumerate(text[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end is None:
        return None, "unbalanced braces"

    candidate = text[start:end]
    try:
        return json.loads(candidate), None
    except json.JSONDecodeError as e:
        return None, f"JSONDecodeError: {e}"


def check_schema_validity(parsed: dict) -> tuple[bool, str | None]:
    try:
        ArchitecturalSpec.model_validate(parsed)
        return True, None
    except ValidationError as e:
        return False, str(e)


def room_count_accuracy(predicted_program: list[dict], target_program: list[dict]) -> float:
    """Fraction of target room types whose count the prediction matches exactly."""
    target_counts = {rp["type"]: rp["count"] for rp in target_program}
    pred_counts = {rp.get("type"): rp.get("count") for rp in predicted_program if isinstance(rp, dict)}
    if not target_counts:
        return 1.0
    correct = sum(1 for t, c in target_counts.items() if pred_counts.get(t) == c)
    return correct / len(target_counts)


def hard_constraint_satisfaction(constraints: list[dict], predicted_program: list[dict]) -> float | None:
    """Fraction of HARD constraints (REQUIRED_ROOM/ROOM_COUNT) from the input that the
    prediction satisfies. Returns None if there were no HARD constraints (C0 examples).
    """
    hard = [c for c in constraints if c.get("priority") == "HARD"]
    if not hard:
        return None
    pred_counts = {rp.get("type"): rp.get("count") for rp in predicted_program if isinstance(rp, dict)}
    pred_types = set(pred_counts.keys())
    satisfied = 0
    for c in hard:
        if c["type"] == "REQUIRED_ROOM":
            if c["target"] in pred_types:
                satisfied += 1
        elif c["type"] == "ROOM_COUNT":
            if pred_counts.get(c["target"]) == c["value"]:
                satisfied += 1
        else:
            satisfied += 1  # unrecognized HARD type - don't penalize
    return satisfied / len(hard)


def _canonical_rel_set(relationships: list[dict]) -> set[tuple[str, str, str]]:
    result = set()
    for r in relationships:
        if not isinstance(r, dict) or "a_type" not in r or "b_type" not in r or "relationship" not in r:
            continue
        a, b = sorted([r["a_type"], r["b_type"]])
        result.add((a, b, r["relationship"]))
    return result


def relationship_accuracy(predicted_relationships: list[dict], target_relationships: list[dict]) -> dict:
    pred_set = _canonical_rel_set(predicted_relationships)
    target_set = _canonical_rel_set(target_relationships)
    if not target_set:
        return {"precision": None, "recall": None, "f1": None}
    tp = len(pred_set & target_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(target_set)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3)}


def area_error(predicted_program: list[dict], brief_target_area_m2: float) -> dict | None:
    try:
        predicted_total = sum(rp["count"] * rp["area_per_room_m2"] for rp in predicted_program)
    except (KeyError, TypeError):
        return None
    abs_err = abs(predicted_total - brief_target_area_m2)
    rel_err = abs_err / brief_target_area_m2 if brief_target_area_m2 else None
    return {
        "predicted_total_m2": round(predicted_total, 2),
        "target_total_m2": round(brief_target_area_m2, 2),
        "abs_error_m2": round(abs_err, 2),
        "rel_error_pct": round(rel_err * 100, 2) if rel_err is not None else None,
    }
