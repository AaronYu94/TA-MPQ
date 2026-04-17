# TA-MPQ

This repository tracks the current `TA-MPQ` workflow for:

- source model: `Qwen/Qwen3.5-27B`
- native baseline: `Qwen/Qwen3.5-9B`
- matched-budget reference: `uniform 8-bit 27B`
- active task: `MATH-500`
- active search space: `{4, 8, 16}`

This repo is no longer on the early `9B -> 4B` scaffold. If you are continuing the project, do not resume from old `uniform 4-bit`, old `9B budget`, or archived `GSM8K-first` routes. The only active route is:

- target the measured `uniform 8-bit` weight-footprint budget
- learn from executed task-aware policies on `MATH-500`
- use targeted precision ablation to decide which groups deserve `16-bit`
- retrain the surrogate
- rerun `uniform8-aware` search
- only then rerun large-sample evaluation

## Current Goal

The current research goal is:

1. train a `uniform8-aware` task-aware quantization policy for `Qwen3.5-27B`
2. make it stronger than native `Qwen3.5-9B`
3. then make it stronger than `uniform 8-bit 27B` under the same budget

In short:

- desired ranking: `TA-MPQ quantized 27B > uniform 8-bit 27B > native 9B`

## Latest Finished Result

The latest fully finished large-sample result is:

- `TA-MPQ quantized 27B`: `36%`
- `uniform 8-bit 27B`: `39%`
- native `9B`: `29%`

Canonical summary file:

- `outputs/closed_loop/math500-uniform8-route-v2-ablation-candidate-01-limit-100-large-sample-summary.json`

So the current stable conclusion is:

- `TA-MPQ > native 9B`
- but `TA-MPQ < uniform 8-bit`

## Latest In-Progress Route

The latest candidate that should be treated as the current working branch is:

- search result:
  `outputs/search/math500-uniform8-route-search-v3-ablation16.json`
- candidate:
  `outputs/policies/math500-uniform8-route-v3-ablation16/candidate-01.json`
- PTQ report:
  `outputs/feasibility/math500-qwen35-27b-vs-9b-math500-uniform8-route-v3-ablation16-candidate-01-llmcompressor-source-report.json`

Important note:

- this `v3-ablation16` candidate is not yet a new winner
- its current `limit=25` reference score is only `28%`
- we are using it as a diagnostic candidate to learn where `16-bit` and `8-bit` are being wasted

Reference evaluation files for the current diagnostic run:

- `outputs/evaluations/math500-uniform8-route-v3-candidate-01-precision-ablation-16to8-g12-reference.json`
- `outputs/evaluations/math500-uniform8-route-v3-candidate-01-precision-ablation-8to4-g12-reference.json`

## Single Source Of Truth

If you are an AI agent or a new contributor, start from these files first.

### Contract

- `configs/experiment_contract_27b_9b_math500.json`

### Executed-policy manifest

- `outputs/closed_loop/math500-uniform8-aware-executed-manifest-v3-ablation16.json`

This is the current manifest that records the real executed policies used to train the current `uniform8-aware` route.

### Current value prior and surrogate lineage

- prior:
  `outputs/surrogate/math500-uniform8-route-group-value-prior-v6-dataset-v3-ablation8to4-16to8.json`
- latest surrogate:
  `outputs/surrogate/math500-uniform8-route-surrogate-v3-ablation16.json`
- latest surrogate model:
  `outputs/surrogate/math500-uniform8-route-surrogate-v3-ablation16-model.json`

### Current targeted ablation manifests

- `16 -> 8`:
  `outputs/ablations/math500-uniform8-route-v3-candidate-01-precision-ablation-16to8-g12/manifest.json`
- `8 -> 4`:
  `outputs/ablations/math500-uniform8-route-v3-candidate-01-precision-ablation-8to4-g12/manifest.json`

When these runs finish, the expected profile outputs are:

- `outputs/sensitivity/math500-uniform8-route-v3-candidate-01-precision-ablation-16to8-g12-profile.json`
- `outputs/sensitivity/math500-uniform8-route-v3-candidate-01-precision-ablation-8to4-g12-profile.json`

## What The Repository Is Telling Us Right Now

The main bottleneck is not backend correctness.

The main bottleneck is:

- `16-bit` budget is still not being assigned to the most valuable layer groups

Why we believe this:

- backend execution is working
- current `uniform8-aware` candidates are runnable
- previous targeted ablations already showed that some groups can be downgraded with zero accuracy drop
- at least one earlier diagnostic signal showed `block:30:mlp.down_proj` is worth keeping high precision, while `block:49:mlp.down_proj` is not

So the current strategy is:

1. run targeted `16 -> 8` and `8 -> 4` ablations
2. turn real accuracy-drop signals into a better group-value prior
3. retrain the surrogate
4. rerun `uniform8-aware` `{4,8,16}` search
5. PTQ the new top-1 candidate
6. only then rerun large-sample benchmark

## Exact Next Step

If you want to continue the project from the current state, do the following in order.

### Step 1: Finish or rerun the current targeted ablations

Run `16 -> 8` targeted ablation on the current diagnostic candidate:

```bash
python3 -m modal run src/ta_mpq/modal_feasibility_app.py::run_precision_ablation_sensitivity \
  --report-path outputs/feasibility/math500-qwen35-27b-vs-9b-math500-uniform8-route-v3-ablation16-candidate-01-llmcompressor-source-report.json \
  --candidate-path outputs/policies/math500-uniform8-route-v3-ablation16/candidate-01.json \
  --contract-path configs/experiment_contract_27b_9b_math500.json \
  --ranking-profile-path outputs/sensitivity/math500-27b-task-sensitivity-v2-limit-32.json \
  --ranking-field combined_sensitivity \
  --max-groups 12 \
  --floor-bit 8 \
  --allowed-bits 8,16 \
  --reference-bits 16 \
  --policy-source llmcompressor \
  --backend-variant source \
  --task-name math500 \
  --limit 25 \
  --max-new-tokens 32 \
  --calibration-limit 4 \
  --prior-weight 0.25 \
  --output-name math500-uniform8-route-v3-candidate-01-precision-ablation-16to8-g12
```

Run `8 -> 4` targeted ablation on the same candidate:

```bash
python3 -m modal run src/ta_mpq/modal_feasibility_app.py::run_precision_ablation_sensitivity \
  --report-path outputs/feasibility/math500-qwen35-27b-vs-9b-math500-uniform8-route-v3-ablation16-candidate-01-llmcompressor-source-report.json \
  --candidate-path outputs/policies/math500-uniform8-route-v3-ablation16/candidate-01.json \
  --contract-path configs/experiment_contract_27b_9b_math500.json \
  --ranking-profile-path outputs/sensitivity/math500-27b-task-sensitivity-v2-limit-32.json \
  --ranking-field combined_sensitivity \
  --max-groups 12 \
  --floor-bit 4 \
  --allowed-bits 4,8,16 \
  --reference-bits 8 \
  --policy-source llmcompressor \
  --backend-variant source \
  --task-name math500 \
  --limit 25 \
  --max-new-tokens 32 \
  --calibration-limit 4 \
  --prior-weight 0.25 \
  --output-name math500-uniform8-route-v3-candidate-01-precision-ablation-8to4-g12
```

### Step 2: Build an ablation-adjusted group-value prior

After both profile files exist, merge them into a new prior:

```bash
python3 -m modal run src/ta_mpq/modal_feasibility_app.py::build_ablation_adjusted_group_value_prior \
  --base-prior-path outputs/surrogate/math500-uniform8-route-group-value-prior-v6-dataset-v3-ablation8to4-16to8.json \
  --ablation-profile-paths outputs/sensitivity/math500-uniform8-route-v3-candidate-01-precision-ablation-16to8-g12-profile.json,outputs/sensitivity/math500-uniform8-route-v3-candidate-01-precision-ablation-8to4-g12-profile.json \
  --output-name math500-uniform8-route-group-value-prior-v7-v3-g12-ablation-adjusted
```

### Step 3: Rebuild the surrogate dataset

```bash
python3 -m modal run src/ta_mpq/modal_feasibility_app.py::build_surrogate_dataset \
  --manifest-path outputs/closed_loop/math500-uniform8-aware-executed-manifest-v3-ablation16.json \
  --contract-path configs/experiment_contract_27b_9b_math500.json \
  --output-name math500-uniform8-route-dataset-v4-v3-g12-ablation-adjusted
```

### Step 4: Retrain the surrogate

```bash
python3 -m modal run src/ta_mpq/modal_feasibility_app.py::run_surrogate_training \
  --dataset-path outputs/surrogate/math500-uniform8-route-dataset-v4-v3-g12-ablation-adjusted.json \
  --output-name math500-uniform8-route-surrogate-v4-v3-g12-ablation-adjusted \
  --ensemble-size 8
```

### Step 5: Rerun uniform8-aware `{4,8,16}` search

```bash
python3 -m modal run src/ta_mpq/modal_feasibility_app.py::run_surrogate_guided_search \
  --report-path outputs/feasibility/math500-qwen35-27b-vs-9b-math500-uniform8-route-v3-ablation16-candidate-01-llmcompressor-source-report.json \
  --surrogate-summary-path outputs/surrogate/math500-uniform8-route-surrogate-v4-v3-g12-ablation-adjusted.json \
  --surrogate-model-path outputs/surrogate/math500-uniform8-route-surrogate-v4-v3-g12-ablation-adjusted-model.json \
  --contract-path configs/experiment_contract_27b_9b_math500.json \
  --group-value-prior-path outputs/surrogate/math500-uniform8-route-group-value-prior-v7-v3-g12-ablation-adjusted.json \
  --sensitivity-profile-path outputs/sensitivity/math500-27b-task-sensitivity-v2-limit-32.json \
  --output-name math500-uniform8-route-search-v4-v3-g12-ablation-adjusted \
  --export-dir outputs/policies/math500-uniform8-route-v4-v3-g12-ablation-adjusted
```

### Step 6: PTQ the new top-1 candidate

Use the new exported `candidate-01.json` and run the standard feasibility/PTQ path. If you are continuing from the current route, this is the next real checkpoint that matters before any new benchmark.

### Step 7: Only after PTQ succeeds, run the next large-sample evaluation

If you create a new closed-loop or summary file for the new winner, evaluate it with:

```bash
python3 -m modal run src/ta_mpq/modal_feasibility_app.py::run_large_sample_eval_from_closed_loop \
  --closed-loop-summary-path <new-summary-path> \
  --contract-path configs/experiment_contract_27b_9b_math500.json \
  --summary-record-key best_closed_loop_record \
  --limit 100 \
  --max-new-tokens 32
```

## What Not To Do

Do not resume from:

- `outputs/archive/`
- old `GSM8K`-first routes
- old `uniform 4-bit` routes
- old `9B budget` routes
- the earliest `9B -> 4B` scaffold

Those are useful for history, but not for current continuation.

## Main Entry Points

The current Modal entrypoints that matter most are:

- build targeted ablation:
  `python3 -m modal run src/ta_mpq/modal_feasibility_app.py::run_precision_ablation_sensitivity`
- build surrogate dataset:
  `python3 -m modal run src/ta_mpq/modal_feasibility_app.py::build_surrogate_dataset`
- train surrogate:
  `python3 -m modal run src/ta_mpq/modal_feasibility_app.py::run_surrogate_training`
- build ablation-adjusted prior:
  `python3 -m modal run src/ta_mpq/modal_feasibility_app.py::build_ablation_adjusted_group_value_prior`
- run guided search:
  `python3 -m modal run src/ta_mpq/modal_feasibility_app.py::run_surrogate_guided_search`
- run closed loop:
  `python3 -m modal run src/ta_mpq/modal_feasibility_app.py::run_surrogate_closed_loop`
- run large-sample eval:
  `python3 -m modal run src/ta_mpq/modal_feasibility_app.py::run_large_sample_eval_from_closed_loop`

## Quick Sanity Check

Before trusting any new route, check all three:

1. quantized `27B` still beats native `9B`
2. the candidate is actually runnable after PTQ
3. the new candidate is better than the previous `uniform8-aware` large-sample route, not just better on a single `limit=25` slice

## Tests

Run unit tests with:

```bash
python3 -m unittest discover -s tests
```

## Output Layout

Current-route artifacts live under:

- `outputs/closed_loop/`
- `outputs/search/`
- `outputs/surrogate/`
- `outputs/policies/`
- `outputs/evaluations/`
- `outputs/feasibility/`
- `outputs/sensitivity/`
- `outputs/ablations/`

Retired routes live under:

- `outputs/archive/`
