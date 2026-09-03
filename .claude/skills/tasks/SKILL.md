# Architect AI — Technical Tasks

## Project Goal

Build an AI architectural planning system capable of converting:

```text
User Requirements
        +
Regulatory Constraints
        ↓
Architectural AI
        ↓
Architectural SPEC
        ↓
Geometry Solver
        ↓
Validated Floor Plan
        ↓
SVG / DXF / CAD
```

The Fine-Tuned model learns **how to design**.

The Regulation Engine provides **what is currently allowed / required**.

The Solver determines **how the design can be geometrically realized**.

The Validator determines **whether the final result is actually valid**.

---

# Milestone 1 — Project Setup

## Task 1.1 — Create Repository Structure

Create:

```text
architect-ai/
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── training/
│   └── reports/
│
├── notebooks/
│   └── 01_boomi_exploration.ipynb
│
├── src/
│   ├── datasets/
│   ├── preprocessing/
│   ├── models/
│   ├── training/
│   ├── evaluation/
│   ├── regulations/
│   ├── solver/
│   ├── validation/
│   └── rendering/
│
├── tests/
│
├── docs/
│
├── pyproject.toml
├── .env
├── .gitignore
└── README.md
```

## Task 1.2 — Install Core Dependencies

Initial dependencies:

```text
datasets
transformers
peft
trl
accelerate
torch
pydantic
pandas
numpy
shapely
ortools
matplotlib
pytest
```

---

# Milestone 2 — BOOMI Dataset Exploration

## Task 2.1 — Load BOOMI

Create:

```text
notebooks/01_boomi_exploration.ipynb
```

Tasks:

- Load BOOMI from Hugging Face
- Inspect train split
- Inspect validation split
- Inspect test split
- Display real samples
- Inspect `caption`
- Parse `spec`
- Inspect L0 / L1 / L2 / L3 examples
- Verify available fields

---

## Task 2.2 — Dataset Statistics

Calculate:

- Total number of examples
- Total unique plans
- Examples per detail level
- Unique room types
- Room count distribution
- Bedroom distribution
- Bathroom distribution
- Built-area distribution
- Plot-size distribution
- BHK distribution
- Relationship types
- Adjacency distribution

Save results:

```text
data/reports/boomi_statistics.json
```

---

## Task 2.3 — Data Quality Analysis

Detect:

- Missing fields
- Invalid JSON
- Unknown room types
- Missing areas
- Zero areas
- Negative areas
- Impossible dimensions
- Duplicate records
- Duplicate plans
- Invalid relationships
- Extreme outliers

Generate:

```text
data/reports/boomi_quality_report.json
```

---

# Milestone 3 — Architectural Vocabulary

## Task 3.1 — Normalize Room Types

Create:

```text
src/preprocessing/room_normalizer.py
```

Create controlled vocabulary.

Example source values:

```text
BED ROOM
BEDROOM
MASTER BED ROOM
MASTER ROOM
TOILET
BATH
WASHROOM
LIVING ROOM
DRAWING ROOM
```

Normalize to:

```text
BEDROOM
MASTER_BEDROOM
BATHROOM
WC
LIVING
KITCHEN
DINING
BALCONY
CORRIDOR
ENTRANCE
STORAGE
UTILITY
```

Do not silently discard unknown room types.

Store them for analysis.

---

## Task 3.2 — Normalize Relationships

Create:

```text
src/preprocessing/relationship_normalizer.py
```

Controlled relationships:

```text
ADJACENT
DIRECT_ACCESS
DOOR_CONNECTION
WINDOW_CONNECTION
SEPARATED
NEAR
```

---

## Task 3.3 — Define Architectural Zones

Create:

```text
src/preprocessing/zone_classifier.py
```

Initial zones:

```text
PUBLIC
PRIVATE
SERVICE
CIRCULATION
OUTDOOR
```

Example mapping:

```text
LIVING         -> PUBLIC
DINING         -> PUBLIC

BEDROOM        -> PRIVATE
MASTER_BEDROOM -> PRIVATE

KITCHEN        -> SERVICE
BATHROOM       -> SERVICE
WC             -> SERVICE
UTILITY        -> SERVICE

CORRIDOR       -> CIRCULATION
ENTRANCE       -> CIRCULATION

BALCONY        -> OUTDOOR
```

---

# Milestone 4 — Architect Dataset Schema

## Task 4.1 — Create Pydantic Models

Create:

```text
src/datasets/schema.py
```

Models:

```text
ArchitectTrainingExample
Brief
Constraint
Room
RoomProgram
Relationship
Zone
ArchitecturalSpec
```

---

## Task 4.2 — Brief Schema

Initial fields:

```text
building_type
built_area_m2
floors
bedrooms
bathrooms
balconies
plot_width
plot_length
preferences
```

The Brief represents:

```text
WHAT THE USER WANTS
```

It must not contain regulatory rules.

---

## Task 4.3 — Constraint Schema

Support:

```text
REQUIRED_ROOM
FORBIDDEN_ROOM
ROOM_COUNT
MIN_AREA
MAX_AREA
MIN_WIDTH
MAX_WIDTH
ADJACENCY
DIRECT_ACCESS
SEPARATION
TOTAL_AREA
SITE_BOUNDARY
```

Each constraint should support:

```text
id
type
target
value
unit
priority
source
```

Priority:

```text
HARD
SOFT
```

---

## Task 4.4 — Architectural SPEC Schema

Initial structure:

```text
ArchitecturalSpec
│
├── program
├── zones
├── rooms
├── relationships
├── circulation
└── metadata
```

The SPEC should remain independent from the renderer.

---

## Task 4.5 — Schema Validation

Create:

```text
tests/test_dataset_schema.py
```

Test:

- Valid samples
- Missing fields
- Invalid room types
- Invalid constraints
- Negative areas
- Invalid relationships
- Invalid priorities

---

# Milestone 5 — BOOMI Parser

## Task 5.1 — Parse BOOMI Records

Create:

```text
src/preprocessing/boomi_parser.py
```

Responsibilities:

```text
BOOMI Record
     ↓
Parse JSON
     ↓
Normalize values
     ↓
Internal representation
```

---

## Task 5.2 — Generate Brief

Create:

```text
src/preprocessing/brief_generator.py
```

Extract where available:

```text
building_type
built_area_m2
bedrooms
bathrooms
balconies
plot dimensions
BHK
room counts
```

Never invent unavailable information.

---

## Task 5.3 — Generate Architectural SPEC

Create:

```text
src/preprocessing/spec_generator.py
```

Convert BOOMI target into our normalized:

```text
ArchitecturalSpec
```

Preserve:

- Room program
- Areas
- Relationships
- Adjacency
- Plot information
- Source metadata

---

# Milestone 6 — Constraint Generator

## Task 6.1 — Extract Constraints From Existing Plans

Create:

```text
src/preprocessing/constraint_generator.py
```

Only generate constraints that the source solution actually satisfies.

Examples:

```text
BEDROOM count = 4

BALCONY required

KITCHEN adjacent LIVING

BEDROOM direct access BATHROOM
```

---

## Task 6.2 — Constraint Sampling

Do not expose every known fact in every training sample.

Generate different subsets.

Example:

```text
PLAN #123

Variant A
---------
Brief only


Variant B
---------
Brief
+
BALCONY required


Variant C
---------
Brief
+
KITCHEN adjacent LIVING


Variant D
---------
Brief
+
BALCONY required
+
KITCHEN adjacent LIVING
```

All variants may point to the same valid solution.

---

## Task 6.3 — Constraint Difficulty Levels

Create levels:

```text
C0 = no explicit constraints
C1 = 1-2 constraints
C2 = 3-5 constraints
C3 = complex constraint combination
```

This allows evaluation of constraint-following behavior.

---

## Task 6.4 — Prevent Information Leakage

Do not provide constraints that completely reveal the target plan.

Avoid inputs containing:

- Exact coordinates
- Exact complete room geometry
- Complete adjacency graph
- Exact dimensions of every room

The model must still perform architectural reasoning.

---

# Milestone 7 — Dataset Builder

## Task 7.1 — Build Architect Dataset V0

Create:

```text
src/datasets/build_dataset.py
```

Pipeline:

```text
BOOMI
  ↓
Parser
  ↓
Normalizer
  ↓
Brief Generator
  ↓
Constraint Generator
  ↓
Architectural SPEC
  ↓
Training Example
```

---

## Task 7.2 — Generate First 100 Samples

Generate only:

```text
100 samples
```

Output:

```text
data/processed/architect_v0_100.jsonl
```

Manually review before continuing.

---

## Task 7.3 — Manual Dataset QA

Check:

- Does the Brief describe the plan?
- Are constraints actually satisfied?
- Is the target reasonable?
- Are room types correct?
- Are areas reasonable?
- Is adjacency correct?
- Is information leaking from target to input?

Document problems.

---

## Task 7.4 — Generate 1,000 Samples

After fixing issues:

```text
data/processed/architect_v0_1000.jsonl
```

Run automated QA.

---

## Task 7.5 — Build Full Dataset

Only after QA passes:

```text
data/training/architect_train.jsonl
data/training/architect_validation.jsonl
data/training/architect_test.jsonl
```

IMPORTANT:

Split by:

```text
source_plan_id
```

before generating variants.

Never allow variants of the same plan across train/test.

---

# Milestone 8 — Baseline Evaluation

## Task 8.1 — Select Base Model

Evaluate candidate models before Fine-Tuning.

Initial experiment:

```text
Qwen family
```

Compare model sizes based on available hardware.

---

## Task 8.2 — Create Baseline Prompt

Create:

```text
src/models/prompts.py
```

Prompt format:

```text
SYSTEM:
You are an architectural planning model.

USER:

Brief:
{brief}

Hard Constraints:
{hard_constraints}

Soft Constraints:
{soft_constraints}

Return ArchitecturalSpec JSON.
```

---

## Task 8.3 — Run Zero-Shot Baseline

Run test set without examples.

Measure performance.

---

## Task 8.4 — Run Few-Shot Baseline

Provide several high-quality examples.

Measure again.

This establishes whether Fine-Tuning actually improves performance.

---

# Milestone 9 — Evaluation Framework

## Task 9.1 — JSON Validity

Metric:

```text
valid_json_rate
```

---

## Task 9.2 — Schema Validity

Metric:

```text
schema_validation_rate
```

Validate output using Pydantic.

---

## Task 9.3 — Room Program Accuracy

Check:

- Requested rooms exist
- Correct counts
- No unexpected mandatory-room loss

Metric:

```text
room_program_accuracy
```

---

## Task 9.4 — Constraint Satisfaction

Metric:

```text
hard_constraint_satisfaction_rate
```

This is one of the most important metrics.

---

## Task 9.5 — Adjacency Accuracy

Measure whether requested relationships exist.

Metric:

```text
adjacency_accuracy
```

---

## Task 9.6 — Area Error

Calculate:

```text
requested_area
vs
generated_area
```

Metrics:

```text
mean_area_error
percentage_area_error
```

---

## Task 9.7 — Solver Success Rate

Measure:

```text
How many generated SPECs can be converted
into valid geometry?
```

Metric:

```text
solver_success_rate
```

This is a critical production metric.

---

# Milestone 10 — Fine-Tuning V1

## Task 10.1 — Prepare SFT Dataset

Convert examples into model training format.

Conceptually:

```text
Brief
+
Constraints
        ↓
Architectural SPEC
```

---

## Task 10.2 — LoRA / QLoRA Configuration

Create:

```text
src/training/train_lora.py
```

Configure:

- Base model
- LoRA rank
- LoRA alpha
- Target modules
- Learning rate
- Batch size
- Gradient accumulation
- Epochs
- Checkpointing

Do not choose final hyperparameters before baseline experiments.

---

## Task 10.3 — Small Training Run

First run:

```text
1K–5K examples
```

Goal:

Validate the pipeline, not achieve final quality.

---

## Task 10.4 — Evaluate Checkpoint

Compare:

```text
Base Model
vs
Fine-Tuned Model
```

using exactly the same test set.

---

## Task 10.5 — Full Training

Only if V1 improves metrics.

Train using the cleaned full dataset.

Save:

```text
models/architect-v1/
```

---

# Milestone 11 — Geometry Solver

## Task 11.1 — Define Solver Input

Solver consumes:

```text
ArchitecturalSpec
```

not natural language.

---

## Task 11.2 — CP-SAT Prototype

Create:

```text
src/solver/cp_sat_solver.py
```

Initial variables:

```text
room.x
room.y
room.width
room.height
```

---

## Task 11.3 — Non-Overlap Constraints

Ensure rooms cannot overlap.

---

## Task 11.4 — Boundary Constraints

Ensure all rooms remain inside building boundary.

---

## Task 11.5 — Area Constraints

Ensure generated room areas satisfy target ranges.

---

## Task 11.6 — Adjacency Constraints

Support:

```text
KITCHEN adjacent LIVING
BEDROOM accessible from circulation
```

---

## Task 11.7 — Door Connections

Generate valid room connections.

---

## Task 11.8 — Solver Objective Function

Optimize soft objectives such as:

```text
minimize corridor area

minimize wasted space

maximize preferred adjacency

maintain reasonable proportions
```

---

# Milestone 12 — Floor Plan Renderer

## Task 12.1 — Geometry Data Model

Create:

```text
src/rendering/geometry.py
```

Represent:

- polygons
- walls
- doors
- windows
- labels
- dimensions

---

## Task 12.2 — SVG Renderer

Create:

```text
src/rendering/svg_renderer.py
```

Render:

```text
Geometry
   ↓
SVG
```

First version needs:

- Walls
- Room labels
- Doors
- Windows
- Dimensions

---

## Task 12.3 — DXF Export

Later add:

```text
src/rendering/dxf_renderer.py
```

Output:

```text
DXF
```

for CAD interoperability.

---

# Milestone 13 — Validation Engine

## Task 13.1 — Architecture Validator

Create:

```text
src/validation/architectural_validator.py
```

Check:

- Missing rooms
- Invalid room counts
- Inaccessible rooms
- Broken adjacency
- Area problems
- Impossible circulation

---

## Task 13.2 — Geometry Validator

Create:

```text
src/validation/geometry_validator.py
```

Check:

- overlaps
- boundary violations
- invalid polygons
- impossible doors
- disconnected spaces

---

## Task 13.3 — Validation Report

Return structured errors:

```json
{
  "valid": false,
  "violations": [
    {
      "code": "ROOM_NOT_ACCESSIBLE",
      "room": "BEDROOM_3",
      "severity": "ERROR"
    }
  ]
}
```

---

# Milestone 14 — Revision Loop

## Task 14.1 — Feed Violations Back to Architect

Pipeline:

```text
Architect
    ↓
SPEC
    ↓
Solver
    ↓
Validator
    ↓
Violations
    ↓
Architect
    ↓
Revised SPEC
```

---

## Task 14.2 — Limit Iterations

Example:

```text
MAX_REVISIONS = 3
```

Never create an uncontrolled Agent loop.

---

## Task 14.3 — Track Revision Success

Metrics:

```text
first_pass_success_rate

revision_success_rate

average_revision_count
```

---

# Milestone 15 — Israeli Regulation Knowledge Base

NOTE:

This milestone is separate from model Fine-Tuning.

## Task 15.1 — Identify Official Sources

Collect official Israeli sources for:

- Planning and Building regulations
- Residential requirements
- Protected spaces / MAMAD
- Accessibility
- Fire safety
- Sanitation
- Parking
- Building types
- Relevant local planning rules

Store source metadata.

---

## Task 15.2 — Regulation Schema

Create:

```text
src/regulations/schema.py
```

Example structure:

```json
{
  "id": "REG-001",

  "jurisdiction": "Israel",

  "building_type": "residential",

  "conditions": {},

  "constraint": {},

  "source": {
    "document": "...",
    "section": "...",
    "effective_date": "..."
  },

  "status": "active"
}
```

---

## Task 15.3 — Regulation Extraction Pipeline

Pipeline:

```text
Official Documents
       ↓
Parser
       ↓
Chunking
       ↓
Structured Rule Extraction
       ↓
Human / QA Validation
       ↓
Regulation DB
```

---

## Task 15.4 — Regulation Retrieval

Input:

```text
Project Context
```

Output:

```text
Applicable Regulations
```

---

## Task 15.5 — Convert Regulations to Constraints

Example:

```text
Regulation
     ↓
Machine-readable constraint
```

Result becomes:

```text
HARD constraints
```

sent to the Architect/Solver.

---

# Milestone 16 — Regulatory Validator

## Task 16.1 — Deterministic Rule Checks

Where possible, regulations should become executable checks.

Example concept:

```text
room width >= required minimum
```

Do not ask an LLM to decide mathematical compliance when deterministic validation is possible.

---

## Task 16.2 — Source Traceability

Every regulatory violation must return:

```text
rule_id
source_document
section
constraint
actual_value
required_value
```

This is essential for a professional product.

---

# Milestone 17 — Architect Critic

## Task 17.1 — Generate Bad Plans

Create controlled violations:

- Remove important adjacency
- Add excessive corridor
- Isolate a room
- Create poor zoning
- Create unreasonable room proportions

---

## Task 17.2 — Critic Dataset

Training format:

```text
Brief
+
Constraints
+
Proposed Plan

        ↓

Architectural Critique
```

---

## Task 17.3 — Critic Output Schema

Example:

```json
{
  "score": 72,

  "valid": true,

  "problems": [
    {
      "type": "POOR_CIRCULATION",
      "severity": "MEDIUM",
      "description": "..."
    }
  ],

  "suggestions": [...]
}
```

---

# Milestone 18 — End-to-End API

## Task 18.1 — FastAPI Service

Create:

```text
src/api/
```

Endpoints:

```text
POST /projects/brief

POST /projects/design

POST /projects/validate

POST /projects/render
```

---

## Task 18.2 — Design Endpoint

Flow:

```text
Request
   ↓
Brief Parser
   ↓
Regulation Retrieval
   ↓
Architect Model
   ↓
Solver
   ↓
Validator
   ↓
Renderer
   ↓
Response
```

---

# Milestone 19 — Observability

## Task 19.1 — Structured Logging

Track:

```text
request_id
model
model_version
dataset_version
prompt_version
solver_version
regulation_version
latency
token_usage
revision_count
```

---

## Task 19.2 — Store Evaluation Data

Store anonymized test/evaluation cases for improving future versions.

Do not automatically turn production user data into training data.

---

# Milestone 20 — V1 Acceptance Criteria

V1 scope:

```text
RESIDENTIAL
+
SINGLE FLOOR
+
SIMPLE RECTANGULAR / POLYGON BOUNDARY
```

V1 must support:

- User Brief
- Multiple bedrooms
- Bathrooms
- Kitchen
- Living area
- Balcony where requested
- Architectural zoning
- Room relationships
- Constraints
- Architectural SPEC
- Geometry generation
- SVG floor plan
- Validation

V1 should NOT initially attempt:

- Schools
- Restaurants
- Hospitals
- Towers
- Complex mixed-use buildings
- Full permit submission
- Multi-floor optimization

Those become later milestones.

---

# Final Architecture

```text
                    USER
                      |
                      v
               +-------------+
               | Brief Parser|
               +-------------+
                      |
                      v
              Structured Brief
                      |
                      |
        +-------------+-------------+
        |                           |
        v                           v
+----------------+          +----------------+
| Regulation RAG |          | Project / Site |
+----------------+          +----------------+
        |                           |
        +-------------+-------------+
                      |
                      v
             Brief + Constraints
                      |
                      v
            +-------------------+
            | Architect Model   |
            | Fine-Tuned LLM    |
            +-------------------+
                      |
                      v
            Architectural SPEC
                      |
                      v
            +-------------------+
            | Geometry Solver   |
            | CP-SAT            |
            +-------------------+
                      |
                      v
                 Geometry
                      |
                      v
            +-------------------+
            | Validators        |
            +-------------------+
                      |
                +-----+-----+
                |           |
              PASS         FAIL
                |           |
                |           v
                |     Revision Feedback
                |           |
                |           +----> Architect
                |
                v
            +-------------------+
            | Renderer          |
            +-------------------+
                |
          +-----+-----+
          |           |
         SVG         DXF
```

---

# Recommended Development Order

Do not build everything simultaneously.

Implement in this order:

1. BOOMI exploration
2. Dataset schema
3. BOOMI parser
4. Room / relationship normalization
5. Brief generator
6. Constraint generator
7. Generate 100 samples
8. Manual dataset QA
9. Generate 1,000 samples
10. Baseline model evaluation
11. LoRA Fine-Tuning V1
12. Fine-Tuned model evaluation
13. CP-SAT geometry prototype
14. SVG renderer
15. Architecture validator
16. Revision loop
17. Israeli Regulation DB
18. Regulation retrieval
19. Regulation → Constraint conversion
20. Regulatory validator
21. FastAPI
22. End-to-end application
23. Architect Critic
24. DXF / CAD export

---

# First Development Sprint

Do ONLY these tasks first:

- [ ] Create repository structure
- [ ] Install dependencies
- [ ] Load BOOMI
- [ ] Inspect 20 real examples
- [ ] List all room types
- [ ] List all relationship types
- [ ] Analyze L0-L3
- [ ] Define Pydantic schema
- [ ] Implement BOOMI parser
- [ ] Implement room normalization
- [ ] Implement brief generator
- [ ] Implement basic constraint generator
- [ ] Generate 100 training examples
- [ ] Manually review the 100 examples
- [ ] Fix dataset problems

STOP HERE.

Do not Fine-Tune until these 100 examples look correct.

The quality of these examples is more important than starting training quickly.