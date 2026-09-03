from src.datasets.schema import ArchitectTrainingExample

# Float precision policy (docs/DATA_CONTRACT.md) - applied HERE, at serialization
# time, not during parsing/normalization. Internal storage (data/processed/*.jsonl)
# keeps full source precision for traceability; only the SFT-facing view is rounded.
SITE_DIM_PRECISION = 2
AREA_PRECISION = 1


def to_sft_pair(example: ArchitectTrainingExample) -> dict:
    """Strips an ArchitectTrainingExample down to exactly what should be tokenized for
    training (Task 10), with the documented rounding policy applied. Internal storage
    keeps full-precision traceability (source_plan_id, caption, pipeline_version,
    unrounded areas, ...) for QA/debugging; none of that belongs in the model's input
    or target - see docs/DATA_CONTRACT.md.

    input:  brief, site, constraints (stripped of dataset-lineage id/source fields)
    target: target_spec's architectural content only (program/zones/relationships/
            circulation) - NOT target_spec.metadata (source_plan_id, bhk_label, ...)
    """
    brief = example.brief.model_dump(exclude_none=True)
    if "target_area_m2" in brief:
        brief["target_area_m2"] = round(brief["target_area_m2"], AREA_PRECISION)

    site = example.site.model_dump()
    for key in ("width_m", "length_m", "area_m2"):
        site[key] = round(site[key], SITE_DIM_PRECISION)

    constraints = []
    for c in example.constraints:
        d = c.model_dump(include={"type", "target", "value", "unit", "priority", "source_type"}, exclude_none=True)
        if d.get("unit") == "m2" and isinstance(d.get("value"), (int, float)):
            d["value"] = round(d["value"], AREA_PRECISION)
        constraints.append(d)

    target = example.target_spec.model_dump(include={"program", "zones", "relationships", "circulation"})
    for rp in target["program"]:
        rp["area_per_room_m2"] = round(rp["area_per_room_m2"], AREA_PRECISION)

    return {"input": {"brief": brief, "site": site, "constraints": constraints}, "target": target}
