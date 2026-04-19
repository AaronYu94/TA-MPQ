# Task-Aware vs Uniform 4-bit: Current Interpretation

Date: 2026-04-11

## Bottom line

Task-aware quantization is now clearly strong enough to beat native `Qwen3.5-9B`, and it has now produced a **first narrow large-sample win** over `uniform-4bit` on `MATH-500`. That said, the margin is still small enough that we should treat it as a promising result that needs confirmation rather than a final settled claim.

The most important current facts are:

1. On GSM8K, `candidate-03` ties `uniform-4bit` at `56%`.
2. On MATH-500, the earlier guided line improved from `28%` to `36%`, and the first surrogate-guided line reached `40%` on a small `25`-example slice.
3. On the newer `closed-loop v3` large-sample comparison, the best task-aware candidate reaches `40% (40/100)`, slightly above `uniform-4bit` at `39% (39/100)` and native `9B` at `29% (29/100)`.
4. So the project has moved past “can task-aware PTQ run?” and even past “can it ever beat uniform?”, and into a narrower question: can this edge be repeated and made statistically convincing?

## Evidence

### 1. Backend projection removes all low-bit decisions

`candidate-03` was searched in the full project space, with original module counts:

- `9 x 2-bit`
- `52 x 3-bit`
- `268 x 4-bit`
- `168 x 8-bit`

But the current backend projection in [policy_export.py](/Users/aaronyu/Desktop/TA-MPQ/src/ta_mpq/policy_export.py#L19) maps:

- `2 -> 4`
- `3 -> 4`
- `4 -> 4`
- `8 -> 8`

So the executed policy for `candidate-03` becomes:

- `329 x 4-bit`
- `168 x 8-bit`

That means:

- all `9` of the `2-bit` assignments are erased
- all `52` of the `3-bit` assignments are erased
- `61 / 497 = 12.27%` of all module assignments are changed before execution

The affected modules are almost entirely `self_attn.{k,q,v,o}_proj`, plus `lm_head`.

This creates a search/execution mismatch:

- search optimizes the unprojected candidate
- real PTQ validation runs the projected candidate

Relevant code:

- [search.py](/Users/aaronyu/Desktop/TA-MPQ/src/ta_mpq/search.py#L146)
- [search.py](/Users/aaronyu/Desktop/TA-MPQ/src/ta_mpq/search.py#L217)
- [policy_export.py](/Users/aaronyu/Desktop/TA-MPQ/src/ta_mpq/policy_export.py#L19)
- [policy_export.py](/Users/aaronyu/Desktop/TA-MPQ/src/ta_mpq/policy_export.py#L159)

### 2. Projection makes candidate-03 heavier before it is ever evaluated

`candidate-03` has:

- search-time estimated average bit width: `5.4473`
- execution-time estimated average bit width: `5.5645`

So projection increases the average bit width by:

- `+0.1171` bits
- about `+2.15%`

It also increases the effective footprint away from the intended low-bit search solution.

Artifacts:

- [candidate-03.json](/Users/aaronyu/Desktop/TA-MPQ/outputs/policies/gsm8k-qwen35-27b-vs-9b-v2/candidate-03.json)
- [gsm8k-qwen35-27b-vs-9b-candidate-03-llmcompressor-source-report.json](/Users/aaronyu/Desktop/TA-MPQ/outputs/feasibility/gsm8k-qwen35-27b-vs-9b-candidate-03-llmcompressor-source-report.json)

### 3. The empirical difference vs uniform is tiny on the current sample

On GSM8K with `25` examples:

- `candidate-03`: `14/25 = 56%`
- `uniform-4bit`: `14/25 = 56%`
- native `9B`: `5/25 = 20%`

But the key detail is that `candidate-03` and `uniform-4bit` differ in correctness on only `2` examples:

- `candidate-03` gets one extra example right that uniform misses
- `uniform-4bit` gets one extra example right that candidate misses
- correctness matches on `23 / 25` examples

So the current run is enough to show:

- quantized `27B` beats native `9B`

but not enough to show:

- task-aware mixed precision beats uniform `4-bit`

Artifacts:

- [candidate-03-quantized-vs-native-9b-limit-25-quantized.json](/Users/aaronyu/Desktop/TA-MPQ/outputs/evaluations/candidate-03-quantized-vs-native-9b-limit-25-quantized.json)
- [uniform-4bit-quantized-vs-native-9b-limit-25-quantized.json](/Users/aaronyu/Desktop/TA-MPQ/outputs/evaluations/uniform-4bit-quantized-vs-native-9b-limit-25-quantized.json)

### 4. The current GSM8K setup is almost “final-answer only”

The GSM8K prompt explicitly tells the model to:

- return only the final numeric answer
- include no reasoning or extra words

See [gsm8k.py](/Users/aaronyu/Desktop/TA-MPQ/src/ta_mpq/tasks/gsm8k.py#L38).

The decoding path also uses:

- `max_new_tokens`
- `do_sample=True`
- short sampled outputs

See [baseline.py](/Users/aaronyu/Desktop/TA-MPQ/src/ta_mpq/baseline.py#L86).

In the current runs:

- `candidate-03` average completion length is `3.44` tokens
- `uniform-4bit` average completion length is `3.32` tokens

So this experiment is measuring very short answer generation, not long-form reasoning traces. That makes it harder for precision-sensitive differences to show up.

### 5. Uniform 4-bit is a stronger baseline than expected on this task

`uniform-4bit` reaches the same GSM8K accuracy as `candidate-03`, while remaining much lighter:

- `candidate-03` estimated weight footprint: `16.5977 GB`
- `uniform-4bit` estimated weight footprint: `11.9312 GB`

So at the moment, even though task-aware PTQ works technically, the value proposition is not yet proven on this task/sample.

## Current interpretation

The strongest explanation is:

- the task-aware policy is being partially flattened before execution
- the remaining executed differences are not large enough to separate from uniform `4-bit` on this GSM8K setup

## New finding: the task-aware target names themselves are correct

After adding a first task-specific sensitivity pipeline for `MATH-500`, we generated a new guided search result:

- sensitivity profile: [math500-27b-task-sensitivity-v1.json](/Users/aaronyu/Desktop/TA-MPQ/outputs/sensitivity/math500-27b-task-sensitivity-v1.json)
- guided search output: [math500-27b-guided-proxy-search-v1.json](/Users/aaronyu/Desktop/TA-MPQ/outputs/search/math500-27b-guided-proxy-search-v1.json)
- exported candidates: [manifest.json](/Users/aaronyu/Desktop/TA-MPQ/outputs/policies/math500-27b-guided-v1/manifest.json)

This first `MATH-500`-guided top-1 candidate is especially useful because:

- it already stays in pure `{4,8}` space
- it has `0` downgraded modules under the current backend projection

So it removes the earlier `2/3-bit -> 4-bit` projection loss and should be a much cleaner task-aware validation candidate.

To verify whether the problem was really our target strings, we added a dedicated live target-matching probe and ran it on the projected `llmcompressor` policy for the guided top-1 candidate:

- probe result: [math500-candidate-01-llmcompressor-target-matching.json](/Users/aaronyu/Desktop/TA-MPQ/outputs/feasibility/math500-candidate-01-llmcompressor-target-matching.json)

That probe shows:

- `196/196` exact `8-bit` module targets are present in the live `Qwen3.5-27B` module tree
- the default `Linear` target matches all `497` linear modules
- there are `0` unmatched targets overall

So the current evidence says:

- the task-aware policy export is naming targets correctly
- the live model does recognize those exact names
- the earlier `Could not match ...` warnings are likely coming from a later backend stage or a different internal model state, not from simple target-name typos in our exported policy

This is still a backend-side bottleneck, but it is now a narrower one than we first thought.

This means the current result should be stated as:

- **positive**: quantized `27B` can beat native `9B`
- **not yet proven**: task-aware mixed precision beats uniform `4-bit`

## New finding: the first task-guided `MATH-500` candidate still does not outperform the baselines

After validating that the target names are correct, we ran the first guided `MATH-500` candidate through real PTQ evaluation:

- guided candidate result: [math500-guided-candidate-01-quantized-vs-native-9b-limit-25-quantized.json](/Users/aaronyu/Desktop/TA-MPQ/outputs/evaluations/math500-guided-candidate-01-quantized-vs-native-9b-limit-25-quantized.json)
- native `9B` comparison: [math500-guided-candidate-01-quantized-vs-native-9b-limit-25-native.json](/Users/aaronyu/Desktop/TA-MPQ/outputs/evaluations/math500-guided-candidate-01-quantized-vs-native-9b-limit-25-native.json)

On `MATH-500`, `limit=25`, the current numbers are:

- guided `candidate-01`: `28%`
- native `9B`: `32%`
- previous task-aware `candidate-03`: `32%`
- `uniform-4bit`: `36%`

So the current state is:

- the new task-guided search pipeline is implemented
- the target names it emits are valid
- but the first guided policy still underperforms the strongest existing baseline in real execution

This suggests that the remaining gap is more likely due to:

- the quality of the sensitivity signal / guided search objective
- runtime behavior when loading and executing the compressed artifact
- or both together

## New finding: a larger task-specific MATH-500 subset improves the guided policy, but still only ties uniform 4-bit

We reran the `MATH-500` task-sensitivity pipeline with a larger subset:

- sensitivity profile: [math500-27b-task-sensitivity-v2-limit-32.json](/Users/aaronyu/Desktop/TA-MPQ/outputs/sensitivity/math500-27b-task-sensitivity-v2-limit-32.json)
- guided search output: [math500-27b-guided-proxy-search-v2-limit-32.json](/Users/aaronyu/Desktop/TA-MPQ/outputs/search/math500-27b-guided-proxy-search-v2-limit-32.json)
- exported candidates: [manifest.json](/Users/aaronyu/Desktop/TA-MPQ/outputs/policies/math500-27b-guided-v2-limit-32/manifest.json)

The search top-1 remained identical to the earlier guided `candidate-01`, which means the stronger task signal did **not** change the best proxy optimum. But it did surface a new clean alternative:

- `candidate-05`
- pure `{4,8}` policy
- `0` downgraded modules
- estimated weight footprint: `16.7034 GB`

Compared with the earlier guided top-1, `candidate-05` changes `50` group assignments:

- `19` groups move from `4 -> 8`
- `31` groups move from `8 -> 4`

These changes are concentrated in:

- late `linear_attn.*` groups
- `mlp.down_proj`
- `linear_attn.out_proj`

Artifacts:

- [candidate-05.json](/Users/aaronyu/Desktop/TA-MPQ/outputs/policies/math500-27b-guided-v2-limit-32/candidate-05.json)
- [math500-qwen35-27b-vs-9b-candidate-05-llmcompressor-source-report.json](/Users/aaronyu/Desktop/TA-MPQ/outputs/feasibility/math500-qwen35-27b-vs-9b-candidate-05-llmcompressor-source-report.json)
- [math500-guided-candidate-05-quantized-vs-native-9b-limit-25-quantized.json](/Users/aaronyu/Desktop/TA-MPQ/outputs/evaluations/math500-guided-candidate-05-quantized-vs-native-9b-limit-25-quantized.json)
- [math500-guided-candidate-05-quantized-vs-native-9b-limit-25-native.json](/Users/aaronyu/Desktop/TA-MPQ/outputs/evaluations/math500-guided-candidate-05-quantized-vs-native-9b-limit-25-native.json)

On `MATH-500`, `limit=25`, the current numbers are now:

- guided `candidate-01`: `28%`
- previous task-aware `candidate-03`: `32%`
- native `9B`: `32%`
- guided `candidate-05`: `36%`
- `uniform-4bit`: `36%`

So the larger task-specific subset **did** help:

- task-aware improved from `28%` to `36%` on the guided MATH-500 line
- the new guided candidate now matches the strongest current `uniform-4bit` result
- but it still does not exceed it

The tie is also still very small-sample:

- `candidate-05` and `uniform-4bit` differ on only `4` examples
- `candidate-05` wins `2`
- `uniform-4bit` wins `2`

This changes the interpretation in an important way:

- task-aware underperformance is **not** just because the earlier guided candidate was weak
- stronger task-specific sensitivity can improve the guided policy materially
- but the current guided search signal is still not strong enough to produce a clear win over a strong uniform baseline

## New finding: more real MATH-500 candidates still do not beat the current best guided policy

We then ran two more real PTQ evaluations from the `guided-v2` MATH-500 policy set:

- [math500-guided-v2-candidate-02-quantized-vs-native-9b-limit-25-quantized.json](/Users/aaronyu/Desktop/TA-MPQ/outputs/evaluations/math500-guided-v2-candidate-02-quantized-vs-native-9b-limit-25-quantized.json)
- [math500-guided-v2-candidate-04-quantized-vs-native-9b-limit-25-quantized.json](/Users/aaronyu/Desktop/TA-MPQ/outputs/evaluations/math500-guided-v2-candidate-04-quantized-vs-native-9b-limit-25-quantized.json)

Their `MATH-500`, `limit=25` results are:

- guided-v2 `candidate-02`: `28%`
- guided-v2 `candidate-04`: `32%`
- guided `candidate-05`: `36%`
- `uniform-4bit`: `36%`
- native `9B`: `32%`

So even after adding more clean `{4,8}` task-guided candidates, the current picture does not change:

- `candidate-05` remains the best task-aware `MATH-500` policy
- task-aware still does not have a clean win over `uniform-4bit`
- the bottleneck is no longer “we have not tried enough candidates,” but rather “the search signal is still not separating the right policies strongly enough”

Artifacts:

- [math500-guided-v2-candidate-02-quantized-vs-native-9b-limit-25-native.json](/Users/aaronyu/Desktop/TA-MPQ/outputs/evaluations/math500-guided-v2-candidate-02-quantized-vs-native-9b-limit-25-native.json)
- [math500-guided-v2-candidate-04-quantized-vs-native-9b-limit-25-native.json](/Users/aaronyu/Desktop/TA-MPQ/outputs/evaluations/math500-guided-v2-candidate-04-quantized-vs-native-9b-limit-25-native.json)

## New finding: the surrogate pipeline is now real, but still not trustworthy enough

We have now moved beyond heuristic-only task guidance and built the proposal-style surrogate loop:

- executed-policy dataset: [math500-executed-policy-dataset-v3.json](/Users/aaronyu/Desktop/TA-MPQ/outputs/surrogate/math500-executed-policy-dataset-v3.json)
- surrogate summary: [math500-xgboost-bootstrap-v3.json](/Users/aaronyu/Desktop/TA-MPQ/outputs/surrogate/math500-xgboost-bootstrap-v3.json)
- surrogate model: [math500-xgboost-bootstrap-v3-model.json](/Users/aaronyu/Desktop/TA-MPQ/outputs/surrogate/math500-xgboost-bootstrap-v3-model.json)
- surrogate-guided search: [math500-surrogate-guided-search-v3.json](/Users/aaronyu/Desktop/TA-MPQ/outputs/search/math500-surrogate-guided-search-v3.json)
- surrogate-guided policy manifest: [manifest.json](/Users/aaronyu/Desktop/TA-MPQ/outputs/policies/math500-surrogate-guided-v3/manifest.json)

The current bootstrap dataset has:

- `8` real executed `MATH-500` policies
- `45` features

And the `XGBoost` surrogate now reports:

- training `MAE = 0.0024`
- leave-one-out `MAE = 0.0454`
- leave-one-out `RMSE = 0.0551`
- leave-one-out `Spearman = -0.5930`
- leave-one-out `top-1 hit rate = 0.0`
- leave-one-out `top-3 hit rate = 0.0`

So the important update is:

- the surrogate pipeline is now fully connected end-to-end
- but the current surrogate is still too data-poor to rank policies reliably
- this means the next high-value step is not “add yet another heuristic candidate,” but “expand the real executed-policy dataset until the surrogate starts correlating with held-out task scores”

## New finding: the first surrogate-guided candidate produces a provisional small-sample win on MATH-500

We then took the `v3` surrogate-guided top-1 candidate through real PTQ and evaluation:

- PTQ report: [math500-surrogate-guided-v3-candidate-01-llmcompressor-source-report.json](/Users/aaronyu/Desktop/TA-MPQ/outputs/feasibility/math500-surrogate-guided-v3-candidate-01-llmcompressor-source-report.json)
- quantized evaluation: [math500-surrogate-guided-v3-candidate-01-quantized-vs-native-9b-limit-25-quantized.json](/Users/aaronyu/Desktop/TA-MPQ/outputs/evaluations/math500-surrogate-guided-v3-candidate-01-quantized-vs-native-9b-limit-25-quantized.json)
- native `9B` comparison: [math500-surrogate-guided-v3-candidate-01-quantized-vs-native-9b-limit-25-native.json](/Users/aaronyu/Desktop/TA-MPQ/outputs/evaluations/math500-surrogate-guided-v3-candidate-01-quantized-vs-native-9b-limit-25-native.json)

On `MATH-500`, `limit=25`, the updated numbers are:

- surrogate-guided `candidate-01`: `40% (10/25)`
- guided `candidate-05`: `36% (9/25)`
- `uniform-4bit`: `36% (9/25)`
- native `9B`: `32% (8/25)`

This was the first clean positive result on the main question:

- task-aware / benchmark-guided search was no longer merely tying `uniform-4bit`
- it had produced a quantized `27B` candidate that was better than both `uniform-4bit` and native `9B` on this initial `MATH-500` slice

We should still be careful not to overclaim:

- the margin over `uniform-4bit` is only `1 / 25` example
- the current sample is still small
- but this was only the first result that moved the project from “promising tie” to “provisional task-aware win”

## New finding: adding that win already improves surrogate validation

After adding the new surrogate-guided winner into the executed-policy dataset, we rebuilt the dataset and retrained the surrogate:

- dataset: [math500-executed-policy-dataset-v4.json](/Users/aaronyu/Desktop/TA-MPQ/outputs/surrogate/math500-executed-policy-dataset-v4.json)
- surrogate summary: [math500-xgboost-bootstrap-v4.json](/Users/aaronyu/Desktop/TA-MPQ/outputs/surrogate/math500-xgboost-bootstrap-v4.json)
- model: [math500-xgboost-bootstrap-v4-model.json](/Users/aaronyu/Desktop/TA-MPQ/outputs/surrogate/math500-xgboost-bootstrap-v4-model.json)

The dataset is now:

- `9` executed policies
- `45` features

And the new surrogate diagnostics improve meaningfully over `v3`:

- `v3` validation `Spearman = -0.5930`
- `v4` validation `Spearman = 0.2650`
- `v3` validation `top-3 hit rate = 0.0`
- `v4` validation `top-3 hit rate = 1.0`

So the current interpretation should now be:

- the surrogate path is not just connected
- it is beginning to learn a directionally useful ranking signal
- but it still needs more executed policies before we can trust it as the final search oracle

## Updated interpretation after the small-sample win

The strongest explanation is now:

- backend projection mismatch matters on the older `{2,3,4,8}` GSM8K candidates
- stronger task-specific signal does matter: it can produce a provisional win over `uniform-4bit` on a small `MATH-500` slice
- the surrogate path is the right direction and is starting to improve once real winners are fed back into the dataset
- the remaining bottleneck is statistical confidence and data volume, not whether benchmark-guided search can work at all

## New finding: the larger MATH-500 sample removes that provisional win

We then ran a larger `MATH-500`, `limit=100` head-to-head among the current best candidates:

- surrogate-guided quantized evaluation: [math500-surrogate-guided-v3-candidate-01-quantized-vs-native-9b-limit-100-quantized.json](/Users/aaronyu/Desktop/TA-MPQ/outputs/evaluations/math500-surrogate-guided-v3-candidate-01-quantized-vs-native-9b-limit-100-quantized.json)
- surrogate-guided native `9B` comparison: [math500-surrogate-guided-v3-candidate-01-quantized-vs-native-9b-limit-100-native.json](/Users/aaronyu/Desktop/TA-MPQ/outputs/evaluations/math500-surrogate-guided-v3-candidate-01-quantized-vs-native-9b-limit-100-native.json)
- uniform `4-bit` evaluation: [math500-uniform-4bit-quantized-vs-native-9b-limit-100-quantized.json](/Users/aaronyu/Desktop/TA-MPQ/outputs/evaluations/math500-uniform-4bit-quantized-vs-native-9b-limit-100-quantized.json)
- guided `candidate-05` evaluation: [math500-guided-candidate-05-quantized-vs-native-9b-limit-100-quantized.json](/Users/aaronyu/Desktop/TA-MPQ/outputs/evaluations/math500-guided-candidate-05-quantized-vs-native-9b-limit-100-quantized.json)

On `MATH-500`, `limit=100`, the updated numbers are:

- `uniform-4bit`: `39% (39/100)`
- surrogate-guided `v3 candidate-01`: `36% (36/100)`
- guided `candidate-05`: `33% (33/100)`
- native `9B`: `29% (29/100)`

This changes the project interpretation in an important way:

- the surrogate-guided path still clearly beats native `9B`
- but the earlier `40% vs 36%` result does not hold up as a large-sample claim
- on the current larger `MATH-500` slice, `uniform-4bit` is still the strongest baseline

So the strongest current claim is now:

- benchmark-guided search can produce quantized `27B` policies that are materially better than native `9B`
- but we still do **not** have large-sample evidence that task-aware mixed precision beats `uniform-4bit`

## Revised interpretation

The strongest explanation is now:

- backend projection mismatch is no longer the main blocker for the newer pure `{4,8}` candidates
- task-specific sensitivity and surrogate guidance are useful enough to beat native `9B`
- however, the current task-aware signal is still not strong enough to outperform `uniform-4bit` on a larger `MATH-500` sample
- the next bottleneck is not wiring, but improving the quality of the training signal and the executed-policy dataset

## New finding: closed-loop v3 produces the first narrow large-sample win over uniform 4-bit

We then moved from single-shot surrogate-guided search to a fully automated multi-iteration closed loop:

- closed-loop summary: [math500-surrogate-closed-loop-v3-summary.json](/Users/aaronyu/Desktop/TA-MPQ/outputs/closed_loop/math500-surrogate-closed-loop-v3-summary.json)
- executed manifest: [math500-surrogate-closed-loop-v3-executed-manifest.json](/Users/aaronyu/Desktop/TA-MPQ/outputs/closed_loop/math500-surrogate-closed-loop-v3-executed-manifest.json)
- large-sample summary: [math500-surrogate-closed-loop-v3-best-closed-loop-record-limit-100-large-sample-summary.json](/Users/aaronyu/Desktop/TA-MPQ/outputs/closed_loop/math500-surrogate-closed-loop-v3-best-closed-loop-record-limit-100-large-sample-summary.json)

The best closed-loop record comes from:

- `math500-surrogate-closed-loop-v3-iter-01-candidate-01`
- dev `MATH-500` accuracy: `52% (13/25)`
- policy shape: `343 x 4-bit + 154 x 8-bit`
- estimated weight footprint: `16.2693 GB`

Artifacts:

- candidate manifest entry: [candidate-01.json](/Users/aaronyu/Desktop/TA-MPQ/outputs/policies/math500-surrogate-closed-loop-v3-iter-01/candidate-01.json)
- dev quantized eval: [math500-surrogate-closed-loop-v3-iter-01-candidate-01-dev-quantized.json](/Users/aaronyu/Desktop/TA-MPQ/outputs/evaluations/math500-surrogate-closed-loop-v3-iter-01-candidate-01-dev-quantized.json)
- PTQ report: [math500-qwen35-27b-vs-9b-math500-surrogate-closed-loop-v3-iter-01-candidate-01-llmcompressor-source-report.json](/Users/aaronyu/Desktop/TA-MPQ/outputs/feasibility/math500-qwen35-27b-vs-9b-math500-surrogate-closed-loop-v3-iter-01-candidate-01-llmcompressor-source-report.json)

We then promoted that best closed-loop record to a larger `MATH-500`, `limit=100` evaluation:

- quantized eval: [math500-surrogate-closed-loop-v3-best-closed-loop-record-limit-100-quantized.json](/Users/aaronyu/Desktop/TA-MPQ/outputs/evaluations/math500-surrogate-closed-loop-v3-best-closed-loop-record-limit-100-quantized.json)
- native `9B` eval: [math500-surrogate-closed-loop-v3-best-closed-loop-record-limit-100-native.json](/Users/aaronyu/Desktop/TA-MPQ/outputs/evaluations/math500-surrogate-closed-loop-v3-best-closed-loop-record-limit-100-native.json)
- uniform `4-bit` eval: [math500-surrogate-closed-loop-v3-best-closed-loop-record-limit-100-uniform-4bit.json](/Users/aaronyu/Desktop/TA-MPQ/outputs/evaluations/math500-surrogate-closed-loop-v3-best-closed-loop-record-limit-100-uniform-4bit.json)

On `MATH-500`, `limit=100`, the updated numbers are now:

- closed-loop `v3 iter-01 candidate-01`: `40% (40/100)`
- `uniform-4bit`: `39% (39/100)`
- surrogate-guided `v3 candidate-01`: `36% (36/100)`
- guided `candidate-05`: `33% (33/100)`
- native `9B`: `29% (29/100)`

This is the most important new result in the project:

- task-aware / benchmark-guided search no longer only beats native `9B`
- it now also has a first large-sample win over `uniform-4bit`
- but the margin is only `1 / 100` example, so this is still a narrow edge, not yet a robust final victory

## Updated interpretation after closed-loop v3

The strongest explanation is now:

- backend projection mismatch was an early issue, but it is no longer the main story for the newer pure `{4,8}` candidates
- feeding benchmark outcomes back into a closed-loop surrogate-guided search does produce better policies
- the current best task-aware candidate is now strong enough to beat both native `9B` and `uniform-4bit` on one large `MATH-500` slice
- the remaining bottleneck is now confirmation and stability, not basic viability

## Best next experiments

1. Re-run the `closed-loop v3` winner on a fresh large held-out `MATH-500` slice, or a larger/full split, to test whether the `40% vs 39%` edge is stable.
2. Expand the executed-policy training set beyond `13` records and retrain the surrogate until held-out ranking improves further.
3. Compare the closed-loop winner against `uniform-4bit` on a second task that should induce a different sensitivity pattern.
4. Move from heuristic sensitivity to stronger task-score-driven supervision when building the surrogate dataset.
5. Reduce the remaining execution mismatch further by supporting real `2/3-bit` execution, or by continuing to search only over currently supported backend bits.
