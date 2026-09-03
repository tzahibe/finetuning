# Architect AI

AI architectural planning system that converts user requirements and regulatory
constraints into a validated floor plan.

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

See [.claude/tasks/SKILL.md](.claude/tasks/SKILL.md) for the full technical task
breakdown and development roadmap.

## Project layout

```text
data/           raw / processed / training data, generated reports
notebooks/      exploratory notebooks
src/
  datasets/     dataset schema and builder
  preprocessing/ BOOMI parsing, normalization, brief/constraint generation
  models/       prompts, model selection
  training/     LoRA/QLoRA fine-tuning
  evaluation/   metrics and baseline comparisons
  regulations/  Israeli regulation knowledge base
  solver/       CP-SAT geometry solver
  validation/   architectural and geometry validators
  rendering/    SVG / DXF renderers
tests/          test suite
docs/           documentation
```

## Setup

```bash
pip install -e .
```
