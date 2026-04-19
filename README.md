# TA-MPQ

This repository now serves a single active route:

- source model: `Qwen/Qwen3.5-27B`
- primary task: `MATH-500`
- reference budget: matched **uniform-int8 linear-weight** footprint
- active search space: `{4, 8, 16}`
- active method: **hierarchical coarse-to-fine structured search**

The project is no longer centered on the older:

- surrogate-guided evolutionary search
- uniform8-aware closed-loop retraining loop
- no-surrogate local-search branch
- `27B -> 9B budget` narrative

Those branches are still present in the repository history, but they are no longer the mainline. Historical workflow notes have been moved under [docs/archive/legacy-mainline](/Users/aaronyu/Desktop/TA-MPQ/docs/archive/legacy-mainline).

## Active Goal

The active research question is:

> Under a matched uniform-int8 linear-weight budget, can a structured `{4,8,16}` mixed-precision policy beat uniform int8 on `MATH-500` while remaining exportable and runnable through the current PTQ pipeline?

This means the mainline is now about:

1. choosing where to spend high-precision budget
2. structuring search rather than relying on free-form evolutionary mutation/crossover
3. validating saved PTQ artifacts, not just proxy scores

## Current Mainline Documents

If you are continuing this project, start from these files:

- mainline overview: [ACTIVE_MAINLINE.md](/Users/aaronyu/Desktop/TA-MPQ/ACTIVE_MAINLINE.md)
- current status / limitations: [status_quo.md](/Users/aaronyu/Desktop/TA-MPQ/status_quo.md)
- BF16 / high-precision budget framing: [BF16_BUDGET_PROBLEM_STATEMENT.md](/Users/aaronyu/Desktop/TA-MPQ/BF16_BUDGET_PROBLEM_STATEMENT.md)
- why the old evolutionary framing is no longer primary: [EVOLUTIONARY_VS_BEAM_SEARCH_PROBLEM_STATEMENT.md](/Users/aaronyu/Desktop/TA-MPQ/EVOLUTIONARY_VS_BEAM_SEARCH_PROBLEM_STATEMENT.md)
- literature map: [PAPER_REFERENCE_NOTES.html](/Users/aaronyu/Desktop/TA-MPQ/PAPER_REFERENCE_NOTES.html)

## Active Code Path

The current mainline code path is built around:

- search entrypoint:
  [run_hierarchical_uniform8_search](/Users/aaronyu/Desktop/TA-MPQ/src/ta_mpq/modal_feasibility_app.py:2981)
- grouped search + hierarchical promotion logic:
  [src/ta_mpq/search.py](/Users/aaronyu/Desktop/TA-MPQ/src/ta_mpq/search.py)
- structured quantization search helpers:
  [src/ta_mpq/quant_search](/Users/aaronyu/Desktop/TA-MPQ/src/ta_mpq/quant_search)
- PTQ / artifact path:
  [src/ta_mpq/quantization.py](/Users/aaronyu/Desktop/TA-MPQ/src/ta_mpq/quantization.py)
  and
  [src/ta_mpq/feasibility.py](/Users/aaronyu/Desktop/TA-MPQ/src/ta_mpq/feasibility.py)
- benchmark execution:
  [src/ta_mpq/baseline.py](/Users/aaronyu/Desktop/TA-MPQ/src/ta_mpq/baseline.py)

The current contract to treat as active is:

- [configs/experiment_contract_27b_9b.json](/Users/aaronyu/Desktop/TA-MPQ/configs/experiment_contract_27b_9b.json)

## What The Mainline Is Not

Do **not** resume work from these as if they were current:

- old surrogate training / retraining loops as the primary route
- archived `uniform8-aware` closed-loop outputs
- earlier `GSM8K-first` workflow docs
- older `27B -> 9B` mainline notes

Those are historical references, not active instructions.

## Current Workflow

The active workflow is:

1. build or load a feasibility report for the source model
2. construct coarse and fine search groups
3. run hierarchical structured search under the matched uniform-int8 budget
4. export top candidates into PTQ-ready policy artifacts
5. quantize with the current backend
6. smoke-test and benchmark the saved artifact
7. compare against uniform-int8 and native baselines on the target task

## Current Benchmark Positioning

- primary benchmark: `MATH-500`
- secondary benchmark: `GSM8K` only as a sanity check

The repository already documents why `MATH-500` is the stronger discriminative benchmark for this route in [status_quo.md](/Users/aaronyu/Desktop/TA-MPQ/status_quo.md).

## Near-Term Priorities

If we continue from the current mainline, the next priorities are:

1. validate the hierarchical route on `MATH-500`
2. make the budget story externally cleaner
3. determine how much `16-bit` / BF16 rescue is justified under the fixed matched budget
4. separate true bit-allocation gains from quantization-config-only gains

## Historical Material

Old workflow notes and retired mainline documents are archived here:

- [docs/archive/legacy-mainline](/Users/aaronyu/Desktop/TA-MPQ/docs/archive/legacy-mainline)

They are kept for context, but they should not be treated as the active route.
