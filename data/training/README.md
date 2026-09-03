---
license: apache-2.0
language:
- en
task_categories:
- text-generation
tags:
- architecture
- floor-plan
- structured-output
- json
pretty_name: Architect AI Dataset
dataset_info:
  features:
  - name: input
    struct:
    - name: brief
      struct:
      - name: building_type
        dtype: string
      - name: target_area_m2
        dtype: float64
      - name: bedrooms
        dtype: int64
      - name: bathrooms
        dtype: int64
      - name: balconies
        dtype: int64
      - name: preferences
        sequence: string
    - name: site
      struct:
      - name: width_m
        dtype: float64
      - name: length_m
        dtype: float64
      - name: area_m2
        dtype: float64
    - name: constraints
      sequence:
      - name: type
        dtype: string
      - name: target
        dtype: string
      - name: value
        dtype: string
      - name: unit
        dtype: string
      - name: priority
        dtype: string
      - name: source_type
        dtype: string
  - name: target
    struct:
    - name: program
      sequence:
      - name: type
        dtype: string
      - name: count
        dtype: int64
      - name: area_per_room_m2
        dtype: float64
      - name: zone
        dtype: string
    - name: zones
      sequence:
      - name: type
        dtype: string
      - name: room_types
        sequence: string
    - name: relationships
      sequence:
      - name: a_type
        dtype: string
      - name: b_type
        dtype: string
      - name: relationship
        dtype: string
      - name: source_type
        dtype: string
    - name: circulation
      sequence: string
  splits:
  - name: train
    num_examples: 54044
  - name: validation
    num_examples: 6756
  - name: test
    num_examples: 6760
---

# Architect AI Dataset

Trains an LLM to act as a residential architectural planner: given a **Brief**
(what the user wants), **Site** (the physical plot), and a set of
**Constraints** (explicit requirements, at varying levels of detail), produce
an **Architectural SPEC** — a room program, zoning, and functional
relationships. Geometry (coordinates, wall placement, doors) is intentionally
out of scope for this model; that belongs to a separate geometry solver.

## Source and provenance

Derived from [`BDivyesh/boomi-stage-a-text-spec`](https://huggingface.co/datasets/BDivyesh/boomi-stage-a-text-spec)
(Apache-2.0) via a from-scratch preprocessing pipeline (`src/` in the
[Architect AI repository](.)). See `docs/DATA_CONTRACT.md` for the exact,
verified semantics of every field (what `target_area_m2` vs `site.area_m2`
mean, why relationships are type-level only, the area-tolerance policy, etc.)
— every claim there was checked directly against the raw source data, not
assumed.

**Generation contract**: one BOOMI source plan → exactly four variants
(`{plan_id}-C0`, `-C1`, `-C2`, `-C3`), one per constraint-richness level.
BOOMI's 4 caption-detail levels (L0-L3) share an identical underlying spec
per plan (verified) and are collapsed to one canonical source row (L3
preferred) — they do **not** create additional training examples.

## Splits

Deterministic 80/10/10 split **by source plan**, computed once over all
unique BOOMI plans before any variant was generated, so all 4
difficulty-variants of a plan always land in the same split. Verified zero
`source_plan_id` overlap between splits.

| split | examples | unique source plans |
|---|---|---|
| train | 54,044 | 13,511 |
| validation | 6,756 | 1,689 |
| test | 6,760 | 1,690 |

One plan (id 15861) was excluded entirely — a confirmed BOOMI source defect
(degenerate site dimensions, `plot.area_m2 = 0`), not a processing error.

## Difficulty levels (C0-C3)

Controls how many, and how varied, the input `constraints` are — not
architectural quality:

| level | constraints | intent |
|---|---|---|
| C0 | 0 | Brief + Site only |
| C1 | 1-2 | a couple of simple facts |
| C2 | 3-5 | a moderate, type-diverse set |
| C3 | 6-8 | a richer, type-diverse set — capped below the full extractable fact set |

Constraint sampling is deterministic per `(source_plan_id, difficulty)` (seeded
from `GENERATION_SEED:plan_id:difficulty`), so regenerating one variant always
reproduces identical constraints.

**Target-leakage measured, not assumed**: even the single highest-coverage
example never exposes more than ~59% of the extractable facts about its
target (C0 mean 0.22 → C3 mean 0.38, max 0.59) — per-room areas and zone
assignments are **never** exposed in the input at any difficulty, only room
types/counts, some relationships, and total area.

## Fields

- `input.brief`: what the user wants (building type, target built area,
  bedroom/bathroom/balcony counts, free-text preferences).
- `input.site`: the physical plot (`width_m`/`length_m` are bounding
  dimensions only — BOOMI sites are frequently non-rectangular, so `area_m2`
  is authoritative and never derived from width×length).
- `input.constraints`: a sampled subset of facts the target actually
  satisfies, each tagged `priority` (`HARD`/`SOFT`) and `source_type`
  (`USER_REQUIREMENT` for room counts/total area, `OBSERVED_GEOMETRY` for
  relationships — an observed layout choice in one solution, not a universal
  rule the model must reproduce).
- `target.program`: room types, counts, area per room, and zone.
- `target.zones`: which room types belong to each zone
  (PUBLIC/PRIVATE/SERVICE/CIRCULATION/OUTDOOR).
- `target.relationships`: type-level adjacency/door/window connections
  (BOOMI has no room-instance IDs, so relationships cannot be instance-level
  — a documented source limitation, not a shortcut).
- `target.circulation`: room types serving circulation (e.g. staircases).

## Known limitations (V1 scope)

- Single-floor residential only; BOOMI carries no floor count, so a small
  number of examples (108, ~0.16%) have `built_area > site_area`, plausible
  for undeclared multi-floor plans — kept, not removed, and worth revisiting
  before claiming multi-floor support.
- Rare room types: `PARKING` (1,336 examples, ~2%), `STAIRCASE` (2,716,
  ~4%) — usable but under-represented; no rebalancing has been applied.
- No Israeli (or any other) regulatory constraints are encoded — those
  belong to a separate, runtime-retrieved regulation engine by design, never
  baked into this dataset or model weights.

## Not yet done

No base-model baseline evaluation, no fine-tuning, not yet pushed to the Hub.
