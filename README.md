# TA-MPQ

This repository now tracks the active `TA-MPQ` workflow around:

- `Qwen/Qwen3.5-27B` as the compressed source model
- `Qwen/Qwen3.5-9B` as the native baseline
- `uniform 8-bit 27B` as the main matched-budget reference
- `MATH-500` as the current task used to train and validate task-aware search

The project is no longer on the early `9B -> 4B` scaffold. The active route is:

- search in `{4, 8, 16}`
- target the measured `uniform 8-bit` weight-footprint budget
- learn a surrogate on executed task-aware policies
- use closed-loop search to produce new mixed-precision candidates
- compare the resulting quantized `27B` against both `uniform 8-bit` and native `9B`

## Active contract

The main experiment contract is:

- `configs/experiment_contract_27b_9b_math500.json`

Key rules in the current contract:

- compare task-aware quantized `27B` against `uniform 8-bit 27B`
- also require the task-aware model to remain stronger than native `9B`
- use the measured `uniform 8-bit` footprint as the search budget
- search over `4 / 8 / 16-bit` group assignments

## Current status

The current stable takeaway is:

- task-aware quantized `27B` is stronger than native `9B`
- but it has not yet stably beaten `uniform 8-bit`

The active debugging direction is:

- run targeted precision ablations
- update group-value priors from real accuracy drops
- retrain the surrogate
- rerun the `uniform8-aware` search

## Main entrypoints

Run the current closed-loop search:

```bash
python3 -m modal run src/ta_mpq/modal_feasibility_app.py::run_surrogate_closed_loop
```

Run targeted precision ablation sensitivity:

```bash
python3 -m modal run src/ta_mpq/modal_feasibility_app.py::run_precision_ablation_sensitivity
```

Run large-sample evaluation from a closed-loop summary:

```bash
python3 -m modal run src/ta_mpq/modal_feasibility_app.py::run_large_sample_eval_from_closed_loop
```

Run unit tests:

```bash
python3 -m unittest discover -s tests
```

## Output layout

Current-route artifacts live under:

- `outputs/closed_loop/`
- `outputs/search/`
- `outputs/surrogate/`
- `outputs/policies/`
- `outputs/evaluations/`
- `outputs/feasibility/`
- `outputs/sensitivity/`

Older retired routes are being moved under:

- `outputs/archive/`

This keeps the current `uniform8-aware` line readable while preserving historical artifacts for reference.

## Practical note

If you are trying to improve the current route, the most useful next step is usually not “more generations”.
It is:

1. run targeted `16 -> 8` and `8 -> 4` ablations
2. update the group-value prior
3. retrain the surrogate
4. rerun the `uniform8-aware` `4/8/16` search
5. only then rerun the large-sample benchmark
