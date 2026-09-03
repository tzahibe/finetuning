# Architect AI — Data Contract

Binding definitions for every field produced by `src/preprocessing` /
`src/datasets/build_dataset.py`. All claims below were verified directly
against the raw `BDivyesh/boomi-stage-a-text-spec` dataset (not assumed) —
see `data/reports/boomi_area_consistency.json` for the underlying numbers.
Do not rename or reinterpret these fields without re-running that
verification.

## Area fields

| Field | Meaning | Source | Notes |
|---|---|---|---|
| `brief.target_area_m2` | Built/program area (sum of all room areas) | `spec.total_area_m2` | Verified: `spec.total_area_m2 == sum(room_program[i].count * room_program[i].target_area_m2)` almost exactly (mean residual ~0 over 500 sampled plans) |
| `site.area_m2` | Site/plot area | `spec.plot.area_m2` | This — **not** `total_area_m2` — is what the BOOMI caption's stated area number matches (stdev 0.3 m² vs `plot.area_m2`, vs 37.7 m² against `total_area_m2`, n=242 parsed captions) |
| `site.width_m`, `site.length_m` | Bounding dimensions of the site | `spec.plot.width_mm/1000`, `depth_mm/1000` | **Not** used to compute `site.area_m2`. Verified `width_m * length_m != area_m2` for ~98% of sampled plans (site is not a simple rectangle) |
| `program[].area_per_room_m2` | Area of **one** room of that type in this plan | `spec.room_program[].target_area_m2` | Uniform across all instances of the type in a plan — BOOMI gives only one number per type, not per instance. Not a total, not an average of varying values (renamed from BOOMI's own `target_area_m2` to avoid that ambiguity) |
| `program[].count` | Number of rooms of that type | `spec.room_program[].count` | |

## Room instances and relationships

BOOMI has **no room-instance identifiers anywhere** — verified by scanning
every key across all 67,564 rows (every split): top-level spec keys are only
`{adjacency, plot, total_area_m2, bhk_label, room_program}`; `room_program`
entries only have `{type, count, target_area_m2}`; `adjacency` entries only
have `{a_type, b_type, via}`. No `room_id`/`index`/`node_id`/`space_id`.

Consequence: **relationships in `target_spec.relationships` are TYPE-LEVEL
only** (`a_type <-> b_type`), never instance-level (`BEDROOM_1 <->
BATHROOM_1`). This is a hard limitation of the source data, not a
preprocessing shortcut. Do not fabricate room IDs or instance-level
relationships for BOOMI-derived examples.

## Relationship types

Kept distinct, never merged (BOOMI's `via` values map 1:1):

- `ADJACENT` (BOOMI `adjacent`) — verified statistically independent of `door` (~30% co-occurrence)
- `DOOR_CONNECTION` (BOOMI `door`) — does **not** imply `ADJACENT` (~35% co-occurrence)
- `WINDOW_CONNECTION` (BOOMI `window`) — exact semantics undetermined from data alone; kept as a raw observed fact only, never promoted to a constraint

Relationships are canonicalized within each type (a_type/b_type ordered
consistently, exact duplicate triples removed) — see
`src/preprocessing/relationship_canonicalizer.py`. Canonicalization never
merges across `relationship` types.

## Constraint provenance

- `ROOM_COUNT` / `REQUIRED_ROOM`: `priority=HARD`, `source_type=USER_REQUIREMENT` — these mirror facts the Brief itself states.
- `TOTAL_AREA`: `priority=SOFT`, `source_type=USER_REQUIREMENT` — treated as an approximate target by consumers, not exact floating-point equality.
- `ADJACENCY` / `DIRECT_ACCESS`: `priority=SOFT`, `source_type=OBSERVED_GEOMETRY` — one observed solution's geometry, never a universal requirement. **`HARD` + `OBSERVED_GEOMETRY` is an invalid combination and is checked for.**
- No `SITE_BOUNDARY` constraints are generated — `site` is always given as a first-class top-level field, so a constraint would just duplicate it.

## What the model actually sees (SFT serialization)

Internal storage (`data/processed/*.jsonl`) keeps full traceability
(`metadata.source_plan_id`, `source_level`, `caption`, `pipeline_version`,
etc.) for debugging and QA. **This is not what gets tokenized for
training.** `src/datasets/sft_format.py::to_sft_pair()` strips it down to:

- `input`: `brief`, `site`, `constraints` (each constraint stripped to `type`/`target`/`value`/`unit`/`priority`/`source_type` — no dataset-lineage `id`/`source` string)
- `target`: `target_spec.program`, `zones`, `relationships`, `circulation` only — **no** `target_spec.metadata` (which carries `source_plan_id`, `bhk_label`, etc.)

The model should never need to reproduce a BOOMI plan ID or a BHK label to
solve the task.

## Difficulty levels (C0-C3)

Difficulty controls both **how many** and **how varied** the sampled
constraints are — not just a random constraint count:

| Level | Constraint count | Intent |
|---|---|---|
| C0 | 0 | Brief + Site only |
| C1 | 1-2 | A couple of simple facts |
| C2 | 3-5 | A moderate, type-diverse set |
| C3 | 6-8 | A richer, type-diverse set — but still capped below the full extractable fact pool (never a near-complete copy of the target) |

`sample_constraints()` stratifies by constraint type (`REQUIRED_ROOM`,
`ROOM_COUNT`, `TOTAL_AREA`, `ADJACENCY`, `DIRECT_ACCESS`) so a C2/C3 example
draws from multiple types rather than, say, eight `ROOM_COUNT` facts and
nothing else. See `data/reports/architect_v0_100_report.json`'s
`constraint_coverage_ratio_by_difficulty` for the measured leakage
distribution per level.

## Source traceability

Every generated example keeps, in its top-level `metadata` (not in
`input`/`target`): `source` ("BOOMI"), `source_dataset` (HF dataset id),
`source_plan_id`, `source_level`, `pipeline_version`, `caption` (for human
QA only). The raw spec itself is not embedded (would bloat every example);
it is deterministically reconstructable via
`load_dataset(source_dataset, split=...).filter(lambda r: r["plan_id"] ==
source_plan_id and r["level"] == source_level)`. Raw BOOMI data under
`data/raw/` (once populated) is never modified in place — all
transformations write to `data/processed/`.
