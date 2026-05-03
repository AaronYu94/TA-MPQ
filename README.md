# TA-MPQ: Task-Aware Mixed-Precision Quantization

TA-MPQ is a course project on **task-aware mixed-precision post-training
quantization** for large language models. The current project studies whether a
model can fit inside the same raw weight footprint as **uniform INT4** while
recovering more task accuracy by reallocating precision across model
components.

The core idea is simple: uniform INT4 gives every linear component the same
precision, but not every component matters equally for a target workload. TA-MPQ
profiles benchmark sensitivity, ranks model groups by sensitivity per true
component size, and searches over exact-budget mixed policies using INT2, INT4,
and INT8.

## Current Project Snapshot

- **Model family:** Qwen3.5-9B
- **Main budget target:** exact raw linear-weight footprint of uniform INT4
- **Bit choices:** INT2, INT4, INT8
- **Main method:** Structured Exact-Budget Coarse-to-Fine Mixed-Precision Search
- **Current policy scope:** one code-oriented policy and one math-oriented policy
- **Primary artifacts:** final report in `Final-Report/`, poster in `poster/`

The current mainline is **not** the older evolutionary-search route and is
**not** the older 27B-to-9B uniform-INT8-budget story. Those materials remain in
the repository as historical context, mostly under `docs/archive/`.

## Research Question

The project asks:

> Under the same raw weight budget as uniform INT4, can a task-aware
> mixed-precision policy improve benchmark accuracy and generation behavior by
> spending INT8 precision on sensitive components and compensating with INT2 on
> less valuable components?

This is a memory-allocation problem, not a larger-budget comparison. Mixed
policies are only meaningful because every candidate is constrained to the same
INT4-sized raw weight footprint.

## Method Overview

The active algorithm is **Structured Exact-Budget Coarse-to-Fine
Mixed-Precision Search**.

1. Select a target workload family and calibration/evaluation split.
2. Run a sensitivity profiler over linear model groups.
3. Rank groups by sensitivity normalized by true component size.
4. Build an exact-budget frontier of INT2/INT4/INT8 policies.
5. Divide the frontier into 8 coarse sectors and evaluate one representative per
   sector.
6. Select the best sector, breaking ties toward higher INT4 fraction.
7. Evaluate 4 refined policies inside the winning sector.
8. Instantiate the selected policy as a real post-training quantized artifact and
   evaluate against BF16, uniform INT8, and uniform INT4 baselines.

The search is deliberately structured. Instead of mutating arbitrary bit
assignments, it searches an ordered policy frontier where moving along the
frontier gradually trades INT8 and INT2 usage for more uniform INT4 behavior.

## Why Not Evolutionary Search?

The original project direction explored evolutionary search. In practice,
crossover was a poor fit for exact-budget mixed precision: two valid parent
policies often produced an over-budget child, and repairing that child rewrote so
many assignments that the child no longer preserved useful inherited structure.

The current method replaces that unstable permutation-style search with a
deterministic, interpretable, budget-constrained frontier.

## Current Results Summary

The most complete machine-readable summary is:

- `poster/current_results_summary.yaml`

Headline findings from the current experiments:

- Mixed exact-budget policies outperform uniform INT4 on HumanEval,
  HumanEval+, BigCodeBench-Hard, CodeMMLU, and MATH-500.
- Mixed matches uniform INT4 on MMLU-coding accuracy but uses fewer completion
  tokens.
- Mixed does **not** consistently beat uniform INT8 or BF16 across benchmarks.
- MATH-500 is sensitive to the generation token wall; both 4096-token and
  20000-token settings are reported.
- Current evidence supports **workload-family-aware policies**, not one
  universal mixed policy for all tasks.

Current selected policies:

| Workload family | Policy id | INT2 | INT4 | INT8 | Used for |
| --- | --- | ---: | ---: | ---: | --- |
| Code | `gpf_refine_00_i0112` | 57 | 80 | 112 | MMLU-coding, CodeMMLU, HumanEval, HumanEval+, BigCodeBench-Hard |
| Math | `gpf_refine_02_i0110` | 75 | 64 | 110 | MATH-500 at 4096 and 20000 max-new-token settings |

The project uses `llmcompressor.oneshot` to instantiate quantized artifacts. This
is sufficient for evaluating accuracy and token behavior, but it is not an
optimized INT4/INT8 serving stack. For that reason, raw latency is treated more
cautiously than accuracy and token usage.

## Quick Navigation

If you only want to understand the finished project:

1. Read `Final-Report/report.pdf` for the full write-up.
2. Open `poster/poster.pdf` for the visual summary.
3. Inspect `poster/current_results_summary.yaml` for the exact reported numbers.
4. Check `src/ta_mpq/quant_search/` for the current exact-budget policy search
   implementation.
5. Use `docs/archive/` only when you need historical context.

## Repository Layout

```text
TA-MPQ/
├── Final-Report/              # Current final course report source and PDF
├── poster/                    # Current poster source, PDF, and result summary
├── src/ta_mpq/                # Main Python package
├── scripts/                   # Small workflow scripts for profiling/search
├── configs/                   # Experiment contracts and configuration files
├── artifacts/                 # Policy artifacts, sensitivity outputs, registries
├── outputs/                   # Evaluation outputs and historical experiment logs
├── docs/                      # Notes, active context, and archived materials
├── tests/                     # Unit tests
├── report/                    # Older report copy kept for context
├── MATH500 result/            # Legacy local MATH-500 JSON exports
├── pyproject.toml             # Python package metadata
└── README.md                  # This file
```

### `Final-Report/`

This is the current final report directory. The main files are:

- `Final-Report/report.tex`
- `Final-Report/report.pdf`
- `Final-Report/results_integrated_draft.tex`
- `Final-Report/figures/`

The report follows the course-required MLSys-style structure: introduction,
problem, related work, overview, method, evaluation, limitations, and
conclusion. `Final-Report/mlsys2025style 4/` contains the MLSys style reference
files used while formatting the report.

To rebuild the report from source:

```bash
cd Final-Report
pdflatex report.tex
```

Depending on the local TeX installation, a second `pdflatex report.tex` pass may
be useful for references and cross-references.

### `poster/`

This contains the current poster and poster-side result summaries:

- `poster/poster.html`
- `poster/poster.pdf`
- `poster/poster.md`
- `poster/current_results_summary.yaml`
- `poster/current_results_summary_table.html`
- `poster/render_poster.py`

Use `current_results_summary.yaml` as the canonical compact summary for the
poster numbers. It includes accuracy, token usage, token caps, benchmark
protocols, and source paths where available.

The poster is authored as HTML and exported to PDF. The current committed PDF is
`poster/poster.pdf`; `poster/render_poster.py` is a helper used during rendering.

### `src/ta_mpq/`

This is the main package implementation.

Important files and modules:

- `src/ta_mpq/quant_search/`: exact-budget frontier construction, policy I/O,
  policy hashing, budget accounting, and sensitivity-aware policy builders.
- `src/ta_mpq/sensitivity.py`: sensitivity profiling utilities.
- `src/ta_mpq/quantization.py`: PTQ artifact creation and quantization helpers.
- `src/ta_mpq/policy_export.py`: policy export helpers.
- `src/ta_mpq/baseline.py`: benchmark and baseline execution helpers.
- `src/ta_mpq/tasks/`: task-specific benchmark utilities for MATH-500,
  MMLU-coding, CodeMMLU, and GSM8K.
- `src/ta_mpq/modal_feasibility_app.py`: Modal-based experiment orchestration
  and older large experiment entrypoints.

Some modules still contain legacy routes from earlier project iterations. The
current report/poster route is the exact-budget INT2/INT4/INT8 search described
above.

### `scripts/`

This folder contains smaller command-line workflow scripts:

- `scripts/profile_sensitivity.py`
- `scripts/build_threshold_policies.py`
- `scripts/evaluate_policy_frontier.py`
- `scripts/select_final_policy.py`

These scripts correspond to the current pipeline stages: profile sensitivity,
build candidate policies, evaluate the frontier, and select a final policy.

### `artifacts/`

This folder stores generated but important intermediate artifacts:

- `artifacts/sensitivity/`: saved sensitivity profiles.
- `artifacts/group_registry/`: model-group registries used for component-level
  policy assignment.
- `artifacts/policies/`: saved policy artifacts.
- `artifacts/results/`: selected result collections from the current method.
- `artifacts/configs/`: generated search grids and policy configuration files.

These are not all polished final outputs, but they are useful for tracing how a
policy was produced.

### `outputs/`

This folder contains benchmark and experiment outputs. It includes both current
results and older exploratory runs:

- `outputs/evaluations/`: many MATH/MMLU/GSM8K/simple-evals JSON outputs.
- `outputs/evalplus/`: HumanEval/HumanEval+ EvalPlus outputs.
- `outputs/bigcodebench/`: BigCodeBench-Hard generation and grading outputs.
- `outputs/search/`: search traces.
- `outputs/baselines/`: baseline model outputs.
- `outputs/archive/`, `outputs/closed_loop/`, `outputs/surrogate/`,
  `outputs/ablations/`: historical or diagnostic experiment routes.

Because this project evolved quickly, not every file in `outputs/` is part of
the final claim. Prefer `poster/current_results_summary.yaml` when deciding which
numbers are currently reportable.

### `docs/`

This folder stores readable context and archived materials:

- `docs/ACTIVE_MAINLINE.md`: current route notes from earlier cleanup.
- `docs/coding_task_prompts.md`: prompt protocols used by coding benchmarks.
- `docs/mlsys_paper_draft/`: older MLSys-format draft materials.
- `docs/archive/`: older notes, old reports, old posters, old HTML summaries,
  and course proposal material.

Archived files are kept for traceability. They should not be treated as the
current project specification unless explicitly referenced by the report or
poster.

### `report/` and `MATH500 result/`

These are legacy local folders that remain for context:

- `report/` is an older report copy.
- `MATH500 result/` contains local MATH-500 JSON exports used during poster and
  report preparation.

The current report is in `Final-Report/`; the current poster is in `poster/`.

## Setup

The package metadata is in `pyproject.toml`. A minimal local setup is:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

The declared package dependencies are intentionally light:

- `modal`
- `PyYAML`

Full experiment execution may require additional ML/runtime dependencies such as
PyTorch, Transformers, quantization backends, benchmark harnesses, and access to
the relevant model checkpoints. Many heavy experiments were run remotely.

For lightweight sanity checks after code changes:

```bash
pytest
```

## Typical Workflow

The high-level workflow is:

1. Build or load a model group registry.
2. Profile task sensitivity on a calibration split.
3. Build exact-budget INT2/INT4/INT8 candidate policies.
4. Evaluate coarse frontier representatives.
5. Evaluate refined candidates inside the best sector.
6. Export the selected policy.
7. Instantiate a real quantized model with PTQ.
8. Evaluate against BF16, uniform INT8, and uniform INT4.
9. Summarize reportable numbers in `poster/current_results_summary.yaml`.

The scripts under `scripts/` map to the middle of this workflow. Larger remote
runs and older orchestration code live under `src/ta_mpq/modal_feasibility_app.py`
and related modules.

## Benchmarks and Prompting

Current report/poster benchmarks include:

| Benchmark | Workload family | Prompt/evaluator | Max new tokens in current summary |
| --- | --- | --- | ---: |
| MMLU-coding | Code | `simple_evals` | 4096 |
| CodeMMLU | Code | `simple_evals` | 4096 |
| HumanEval / HumanEval+ | Code | official EvalPlus | 768 |
| BigCodeBench-Hard | Code | official BigCodeBench | 1280 |
| MATH-500 | Math | `simple_evals_nonthinking` | 4096 and 20000 |

All final reported runs use greedy decoding. The coding policy was searched on a
coding-oriented workload and transferred unchanged to multiple code benchmarks.
The math policy was searched for MATH-500 and evaluated under two token-wall
settings.

## Claim Boundary

Supported by the current results:

- Exact-budget mixed precision can outperform uniform INT4 on several code and
  math benchmarks.
- Token usage often improves relative to uniform INT4, suggesting that preserving
  sensitive components may keep generations more on track.
- Workload-family-specific policies are a better supported claim than a single
  universal policy.

Not supported yet:

- A consistent win over uniform INT8.
- A universal policy that works best for all task families.
- A strong latency claim independent of serving backend.
- A claim that the method is fully optimized for token usage rather than
  selected primarily by accuracy.

## Notes for Future Work

Promising next steps:

- Run larger held-out benchmark slices and repeated seeds.
- Test larger model families.
- Compare layer-level and component-level granularity.
- Optimize directly for token usage, not only accuracy.
- Combine TA-MPQ allocation with GPTQ/AWQ-style weight refinement to test whether
  allocation gains stack with quantization-error minimization.
- Evaluate on an optimized INT4/INT8 serving stack before making latency claims.
