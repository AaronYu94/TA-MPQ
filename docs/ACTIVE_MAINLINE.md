# Active Mainline

## Scope

This file defines the only active research route in this repository after merging PR `#1` (`Greedy search`).

Everything here should be treated as the current mainline. Older workflow notes remain available for historical context, but they are archived and should not drive new experiments.

## Main Problem

The project is now framed as:

> Search for a structured `{4,8,16}` mixed-precision policy for `Qwen/Qwen3.5-27B` under a matched **uniform-int8 linear-weight budget**, then validate the saved PTQ artifact on `MATH-500`.

This is no longer primarily a surrogate/evolutionary-search project.

It is now primarily a:

- constrained bit-allocation problem
- hierarchical coarse-to-fine search problem
- PTQ artifact validation problem

## Active Assumptions

- source model: `Qwen/Qwen3.5-27B`
- primary task: `MATH-500`
- budget definition: matched **uniform-int8 linear-weight** footprint
- search space: `{4, 8, 16}`
- backend: `llmcompressor`
- high-precision tier: `16-bit` groups are effectively left higher precision

## Active Research Questions

The mainline should answer these questions:

1. Can a structured mixed-precision policy beat uniform int8 under the same matched linear-weight budget?
2. How much `16-bit` / BF16 rescue budget should be spent under that budget?
3. Is the gain coming from true bit placement, or mostly from quantization-config refinement?
4. Which benchmark gives the strongest discriminative signal for this search problem?

## Active Method

The current method is:

1. Build a feasibility report from the source model.
2. Build coarse search groups (`per_block_window_component`).
3. Run a coarse search under the matched uniform-int8 budget.
4. Expand the best coarse policy to fine groups (`per_block_component`).
5. Promote a subset of fine groups and freeze the rest.
6. Run fine search on the promoted subset.
7. Run quantization-config refinement on top finalists.
8. Export PTQ-ready candidates.
9. Quantize with the backend.
10. Smoke-test and benchmark the saved artifact.

## Active Entry Points

### Main search

- [run_hierarchical_uniform8_search](/Users/aaronyu/Desktop/TA-MPQ/src/ta_mpq/modal_feasibility_app.py:2981)

### Main search implementation

- [src/ta_mpq/search.py](/Users/aaronyu/Desktop/TA-MPQ/src/ta_mpq/search.py)
- [src/ta_mpq/quant_search](/Users/aaronyu/Desktop/TA-MPQ/src/ta_mpq/quant_search)

Important components inside `quant_search`:

- [greedy_path.py](/Users/aaronyu/Desktop/TA-MPQ/src/ta_mpq/quant_search/greedy_path.py)
- [frontier_search.py](/Users/aaronyu/Desktop/TA-MPQ/src/ta_mpq/quant_search/frontier_search.py)
- [budget.py](/Users/aaronyu/Desktop/TA-MPQ/src/ta_mpq/quant_search/budget.py)
- [group_registry.py](/Users/aaronyu/Desktop/TA-MPQ/src/ta_mpq/quant_search/group_registry.py)
- [policy_builder.py](/Users/aaronyu/Desktop/TA-MPQ/src/ta_mpq/quant_search/policy_builder.py)

### PTQ and evaluation

- [src/ta_mpq/quantization.py](/Users/aaronyu/Desktop/TA-MPQ/src/ta_mpq/quantization.py)
- [src/ta_mpq/feasibility.py](/Users/aaronyu/Desktop/TA-MPQ/src/ta_mpq/feasibility.py)
- [src/ta_mpq/baseline.py](/Users/aaronyu/Desktop/TA-MPQ/src/ta_mpq/baseline.py)

### Active contract

- [configs/experiment_contract_27b_9b.json](/Users/aaronyu/Desktop/TA-MPQ/configs/experiment_contract_27b_9b.json)

## Active Interpretation Of The Search

The right way to describe the current mainline is:

- **hierarchical**
- **coarse-to-fine**
- **structured**
- **budget-constrained**

Do **not** describe the current mainline as:

- surrogate-guided evolutionary search
- closed-loop retraining search
- no-surrogate local search

Those branches may still exist in the codebase, but they are no longer the repository’s active narrative.

## Active Benchmark Positioning

- Primary benchmark: `MATH-500`
- Secondary benchmark: `GSM8K` only as a sanity check

The reason is practical: GSM8K currently looks too flat for this model and budget, while MATH-500 has shown a clearer gap between uniform int8 and mixed candidates.

## Active Documentation

Use these supporting documents together:

- current status and caveats:
  [status_quo.md](/Users/aaronyu/Desktop/TA-MPQ/status_quo.md)
- BF16 / rescue-tier framing:
  [BF16_BUDGET_PROBLEM_STATEMENT.md](/Users/aaronyu/Desktop/TA-MPQ/BF16_BUDGET_PROBLEM_STATEMENT.md)
- why the older evolutionary framing is no longer central:
  [EVOLUTIONARY_VS_BEAM_SEARCH_PROBLEM_STATEMENT.md](/Users/aaronyu/Desktop/TA-MPQ/EVOLUTIONARY_VS_BEAM_SEARCH_PROBLEM_STATEMENT.md)
- literature / method references:
  [PAPER_REFERENCE_NOTES.html](/Users/aaronyu/Desktop/TA-MPQ/PAPER_REFERENCE_NOTES.html)

## What Is Archived

These older materials are no longer the active workflow:

- surrogate-guided uniform8-aware workflow notes
- old `27B -> 9B` mainline docs
- evolutionary workflow handoff docs
- older search analysis notes for the archived route

They now live under:

- [docs/archive/legacy-mainline](/Users/aaronyu/Desktop/TA-MPQ/docs/archive/legacy-mainline)

## Practical Rule

If a teammate or AI agent opens this repository today, the safe default is:

- follow [README.md](/Users/aaronyu/Desktop/TA-MPQ/README.md)
- then follow this file
- then work outward from [run_hierarchical_uniform8_search](/Users/aaronyu/Desktop/TA-MPQ/src/ta_mpq/modal_feasibility_app.py:2981)

That is the current mainline.
