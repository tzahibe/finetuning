from src.datasets.schema import ArchitectTrainingExample


def to_sft_pair(example: ArchitectTrainingExample) -> dict:
    """Strips an ArchitectTrainingExample down to exactly what should be tokenized for
    training (Task 10). Internal storage keeps full traceability (source_plan_id,
    caption, pipeline_version, ...) for QA/debugging; none of that belongs in the
    model's input or target - see docs/DATA_CONTRACT.md.

    input:  brief, site, constraints (stripped of dataset-lineage id/source fields)
    target: target_spec's architectural content only (program/zones/relationships/
            circulation) - NOT target_spec.metadata (source_plan_id, bhk_label, ...)
    """
    input_ = {
        "brief": example.brief.model_dump(exclude_none=True),
        "site": example.site.model_dump(),
        "constraints": [
            c.model_dump(include={"type", "target", "value", "unit", "priority", "source_type"}, exclude_none=True)
            for c in example.constraints
        ],
    }
    target = example.target_spec.model_dump(include={"program", "zones", "relationships", "circulation"})
    return {"input": input_, "target": target}
