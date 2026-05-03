# TA-MPQ Status Quo and What To Explore Next

## Purpose

This note is a handoff document for deeper external research. It summarizes:

- what the current repo is actually doing,
- what the latest measured results are,
- where the pipeline is likely misleading or underpowered,
- and what research directions are worth prioritizing next.

The goal is to recalibrate before investing in more search or more engineering.

## Executive Summary

- The current project is a task-aware mixed-precision search pipeline for `Qwen/Qwen3.5-27B`.
- The live search budget is still based on the measured **linear-layer-only** uniform `int8` footprint: `23.8623046875 GiB`.
- The current quantization backend is `llmcompressor`, using integer weight-only quantization for `4/8-bit` and leaving `16-bit` modules in higher precision.
- The current pipeline now includes a hierarchical coarse-to-fine route plus a quantization-config refinement stage.
- GSM8K does **not** currently show a strong gap between native BF16 and uniform `int8` for this model on the tested slice.
- MATH-500 remains the stronger benchmark for discriminating between uniform `int8` and task-aware mixed precision in the existing repo artifacts.
- There is still a major backend reliability problem: saved-artifact reload emits target-resolution warnings for some nondefault module-specific overrides, especially around `mlp.down_proj` and a few other modules.
- Because of that, measured scores on saved artifacts are valid for those exact artifacts, but some results are not yet a clean proof that the intended policy survives export/reload intact.

## Current Pipeline

### High-level flow

1. Build a feasibility report from the 27B source model.
2. Collect searchable layer groups from tracked `Linear` modules.
3. Search over `4/8/16` bit assignments under a fixed budget.
4. Export top candidates into:
   - a project-level mixed-precision policy, and
   - an `llmcompressor` backend projection.
5. Build a quantized artifact with `llmcompressor`.
6. Smoke-test the saved artifact.
7. Benchmark the saved artifact on the task.

### Main entrypoints in code

- Contract: [configs/experiment_contract_27b_9b.json](/Users/zhangletian/Projects/15642/TA-MPQ/configs/experiment_contract_27b_9b.json)
- Hierarchical search entrypoint: [run_hierarchical_uniform8_search](/Users/zhangletian/Projects/15642/TA-MPQ/src/ta_mpq/modal_feasibility_app.py:1390)
- Candidate export: [export_top_candidates](/Users/zhangletian/Projects/15642/TA-MPQ/src/ta_mpq/policy_export.py:29)
- Recipe emission: [to_llmcompressor_recipe_config](/Users/zhangletian/Projects/15642/TA-MPQ/src/ta_mpq/quantization.py:126)
- Feasibility report / weight inventory: [collect_model_weight_inventory](/Users/zhangletian/Projects/15642/TA-MPQ/src/ta_mpq/feasibility.py:41) and [build_feasibility_report](/Users/zhangletian/Projects/15642/TA-MPQ/src/ta_mpq/feasibility.py:89)

### What the current hierarchical route does

The current `run_hierarchical_uniform8_search(...)` route does the following:

1. Coarse proxy search on `per_block_window_component`.
   - `window_size = 4`
   - default coarse search settings in code:
     - `population_size = 80`
     - `generations = 24`
2. Expand the best coarse candidate to fine groups.
3. Promote a subset of fine groups and freeze the rest.
4. Fine search on `per_block_component`.
   - if surrogate is present, use surrogate search
   - otherwise, use proxy evolution search
5. Run a quantization-config refinement stage on top fine candidates.
   - currently tunes:
     - `group_size` from `{32, 64, 128}`
     - `symmetric` from `{true, false}`

### Search space today

- Bit-widths: `{4, 8, 16}`
- Main fine grouping: `per_block_component`
- Coarse grouping: `per_block_window_component`
- Config refinement parameters:
  - `group_size`
  - `symmetric`

## What Is Actually Being Counted in the Budget

### Current accounting rule

The current budget is **not** full checkpoint size. It is the tracked weight footprint of the searched `Linear` modules.

- The repo collects `Linear` weights only when building the searchable layer inventory.
- This includes modules like attention/MLP projections and `lm_head`.
- It excludes embeddings, norms, and other non-`Linear` weights from the active search budget.

Relevant code:

- [collect_model_weight_inventory](/Users/zhangletian/Projects/15642/TA-MPQ/src/ta_mpq/feasibility.py:41)
- [estimate_weight_footprint_gb](/Users/zhangletian/Projects/15642/TA-MPQ/src/ta_mpq/quantization.py)

### Current budget value

- `search_target_budget_gb = 23.8623046875`
- Source: [configs/experiment_contract_27b_9b.json](/Users/zhangletian/Projects/15642/TA-MPQ/configs/experiment_contract_27b_9b.json)

### Important fairness caveat

This is internally consistent for search, because both uniform `int8` and TA-MPQ candidates are compared under the same linear-layer accounting rule.

It is **not** a clean full-model deployment-size claim.

This should either be:

- relabeled clearly as `matched linear-weight budget`, or
- replaced by full-model accounting as the main reported budget.

The repo now has partial support for full-model accounting fields in fresh feasibility reports:

- `total_model_parameters`
- `total_non_linear_parameters`
- `estimated_non_linear_weight_footprint_gb`
- `estimated_full_model_weight_footprint_gb`

Relevant code:

- [build_feasibility_report](/Users/zhangletian/Projects/15642/TA-MPQ/src/ta_mpq/feasibility.py:89)

## Current Quantization Semantics

- Uniform `8-bit` baseline is `int8`, not `fp8`.
- Dynamic/task-aware route currently quantizes compressed modules into `int4/int8`.
- `16-bit` modules are effectively kept high precision rather than quantized into `fp8/bf8`.

Current `llmcompressor` recipe generation:

- explicit override groups first,
- `default` group last.

Relevant code:

- [to_llmcompressor_recipe_config](/Users/zhangletian/Projects/15642/TA-MPQ/src/ta_mpq/quantization.py:126)

## Current Results

### GSM8K, limit 200

Measured on saved artifacts:

- Native BF16 27B: `0.495` (`99/200`)
  - [gsm8k-qwen35-27b-bf16-native-limit-200.json](/Users/zhangletian/Projects/15642/TA-MPQ/outputs/evaluations/gsm8k-qwen35-27b-bf16-native-limit-200.json)
- Uniform `int8`: `0.495` (`99/200`)
  - [gsm8k-qwen35-27b-uniform-int8-limit-200.json](/Users/zhangletian/Projects/15642/TA-MPQ/outputs/evaluations/gsm8k-qwen35-27b-uniform-int8-limit-200.json)
- Hierarchical mixed `4/8/16` candidate-03: `0.500` (`100/200`)
  - [gsm8k-hierarchical-uniform8-route-v3-proxy-candidate-03-limit-200-quantized.json](/Users/zhangletian/Projects/15642/TA-MPQ/outputs/evaluations/gsm8k-hierarchical-uniform8-route-v3-proxy-candidate-03-limit-200-quantized.json)

Interpretation:

- GSM8K is currently too flat to be a strong primary benchmark for this search problem.
- Uniform `int8` already ties native BF16 on this slice.
- The mixed candidate is only `+1` question over both, which is not a strong signal.

### MATH-500, existing stronger reference

Existing repo large-sample summaries already show a meaningful gap:

- Uniform `int8` reference: `0.39`
- Best prior mixed-precision route in repo: `0.37`
- Native `9B`: `0.29`

Files:

- Uniform-8 reference / v2 ablation summary: [math500-uniform8-route-v2-ablation-candidate-01-limit-100-large-sample-summary.json](/Users/zhangletian/Projects/15642/TA-MPQ/outputs/closed_loop/math500-uniform8-route-v2-ablation-candidate-01-limit-100-large-sample-summary.json)
- Best prior v1 summary: [math500-uniform8-route-v1-candidate-01-limit-100-large-sample-summary.json](/Users/zhangletian/Projects/15642/TA-MPQ/outputs/closed_loop/math500-uniform8-route-v1-candidate-01-limit-100-large-sample-summary.json)

### MATH-500, current mixed candidate probe

For the same GSM8K-derived mixed candidate family, on `limit=200`:

- Uniform `int8`: `0.37`
  - [math500-qwen35-27b-uniform-int8-limit-200.json](/Users/zhangletian/Projects/15642/TA-MPQ/outputs/evaluations/math500-qwen35-27b-uniform-int8-limit-200.json)
- Hierarchical mixed `4/8/16` candidate-03: `0.35`
  - [math500-hierarchical-uniform8-route-v3-proxy-candidate-03-limit-200-quantized.json](/Users/zhangletian/Projects/15642/TA-MPQ/outputs/evaluations/math500-hierarchical-uniform8-route-v3-proxy-candidate-03-limit-200-quantized.json)

The BF16 `MATH-500 limit=200` run was intentionally stopped before completion, so there is no fresh BF16 number from that exact run.

Interpretation:

- The current GSM8K-discovered mixed candidate does not transfer well to MATH-500.
- This is more evidence that GSM8K is not a great primary optimization target for the current pipeline.

## Current Search Status

### Current top candidate family on GSM8K route

From [gsm8k-hierarchical-uniform8-route-v3-proxy.json](/Users/zhangletian/Projects/15642/TA-MPQ/outputs/search/gsm8k-hierarchical-uniform8-route-v3-proxy.json):

- candidate-01:
  - footprint `23.8623 GiB`
  - average bits `8.0`
  - provenance `aggressive_quant_config_seed`
  - effectively an all-8-bit candidate with nondefault quantization config changes
- candidate-02:
  - footprint `23.7451 GiB`
  - average bits `7.9607`
  - provenance `aggressive_quant_config_seed`
- candidate-03:
  - footprint `23.5783 GiB`
  - average bits `7.9048`
  - provenance `aggressive_quant_config_seed`

This strongly suggests the current search is being driven heavily by quantization-config refinement, not just bit allocation.

## Known Problems and Failure Modes

### 1. Saved-artifact reload still emits target-resolution warnings

This is the most important pipeline integrity problem right now.

Observed repeatedly on saved mixed artifacts:

- `Could not match re:.*model.layers....mlp.down_proj$`
- similar misses for `lm_head`, `self_attn.o_proj`, `self_attn.q_proj`, and a few others

Implication:

- source build may succeed,
- smoke test may succeed,
- evaluation may still be loading an artifact whose override policy was not fully resolved as intended.

This means current saved-artifact benchmark results are usable, but some are not a clean proof that the exact exported override set survived reload.

### 2. Budget story is still not externally clean

- Search budget is linear-only.
- Some newer feasibility reports include full-model accounting fields, but the search contract still uses the older linear-only budget.

### 3. Search objective may be too weakly anchored

On GSM8K, native BF16 and uniform `int8` are nearly indistinguishable at the tested scale.
That means the search objective does not get a strong learning signal there.

### 4. Coarse-to-fine may be adding complexity without demonstrated win

The hierarchical route is functioning, but there is not yet strong evidence that it is beating a simpler fine-grained search in the current regime.

### 5. Config refinement may be masking whether bit allocation is really improving

Current top candidates are coming from `aggressive_quant_config_seed`.
That makes it harder to answer the core research question:

- are we winning because of better `4/8/16` placement,
- or because of backend-specific config tweaks on mostly-8-bit policies?

## Recommendation: What Benchmark Should Be Primary?

### Recommendation

Use **MATH-500** as the primary research benchmark.

Why:

- The repo already shows a real gap there:
  - uniform `int8` around `0.39`
  - best prior mixed candidate around `0.37`
- That makes it a better benchmark for ranking mixed-precision policies.
- GSM8K appears too saturated / too flat for this model and budget.

### Role for GSM8K

Keep GSM8K as a secondary benchmark:

- for transfer/generalization checks,
- for sanity-checking that a MATH-500 win is not narrowly overfit,
- but not as the single primary optimization target.

## Recommendation: Keep or Remove Coarse-to-Fine Search?

### Recommendation

Do **not** remove coarse-to-fine entirely.
But also do **not** treat it as the default mainline until it proves value against a simpler baseline.

### Suggested stance

- Keep hierarchical coarse-to-fine as an experimental branch.
- Re-establish a strong flat baseline:
  - direct `per_block_component` search,
  - exact same budget,
  - exact same backend/export path.
- Compare:
  - flat fine search,
  - hierarchical coarse-to-fine,
  - hierarchical without config refinement,
  - hierarchical with config refinement.

### Why

Right now, hierarchical search may be helping with dimensionality, but it also adds:

- another grouping abstraction,
- promotion heuristics,
- freezing logic,
- and extra failure modes.

It should be justified empirically, not assumed.

## Recommendation: What To Explore Next

### A. Fix the artifact integrity problem first

This is the highest-priority engineering issue.

Research question:

- Why do some override groups match during source build but fail during saved-artifact reload?

Things to investigate:

- whether regex target encoding is still too fragile,
- whether certain module names change after save/load,
- whether override groups with same-bit config changes are being dropped or merged,
- whether `llmcompressor` save/load semantics normalize group definitions differently than one-shot build.

Deliverable:

- one candidate whose policy target set is verified both:
  - before save, and
  - after reload.

### B. Decide the budget story explicitly

Two options:

1. Keep current budget but relabel it clearly as linear-only.
2. Move the mainline search to full-model accounting.

Recommendation:

- for research clarity, move toward full-model accounting,
- but keep linear-only budget as a reproducibility baseline.

### C. Re-center on MATH-500 for policy search

Suggested main optimization loop:

- search on MATH-500 dev slice,
- evaluate top candidates on larger MATH-500 slice,
- use GSM8K only as a secondary transfer benchmark.

### D. Separate bit-allocation search from config-refinement search

Run these as distinct phases with distinct reports:

1. bit allocation only,
2. config refinement only on top of selected bit allocations,
3. joint comparison.

Why:

- current results are too entangled,
- and it is unclear what is actually creating gains.

### E. Re-establish a simpler baseline before more complexity

Run a strong flat search with:

- grouping = `per_block_component`
- no hierarchical stage
- no config refinement

Then add back complexity one piece at a time.

## Search Parameters Worth Manipulating

### Grouping / structure

- `per_block_component` vs hierarchical
- contiguous window size
  - current code fixes `window_size = 4`
  - research whether `2`, `4`, or `8` is better
- selective refinement rules
  - number of promoted groups
  - criteria for promotion

### Bit allocation

- allowed bit set
  - current `{4,8,16}`
  - consider whether a simpler `{8,16}` search or a more aggressive `{4,8}` ablation is informative
- explicit cap on number of `16-bit` groups
- explicit cap on number of `4-bit` groups

### Evolution / optimization

- `population_size`
- `generations`
- `elite_count`
- `tournament_size`
- `mutation_rate`
- number of random seeds

### Coarse-to-fine controls

- `max_promoted_fine_groups`
- promotion rule thresholds
- whether to freeze non-promoted groups at best-coarse assignment or allow small local edits outside promoted set

### Quantization-config refinement

- `group_size`
  - currently `{32,64,128}`
- `symmetric`
  - currently `{true,false}`
- `max_tunable_config_groups`
- whether config refinement should be limited to:
  - only 4-bit groups,
  - only promoted groups,
  - or only groups with strongest ablation evidence

### Evaluation protocol

- calibration sample count
- dev-limit size
- large-sample evaluation size
- whether to optimize for:
  - raw accuracy,
  - advantage over uniform `int8`,
  - or a two-stage filter

## Pipeline Changes Worth Exploring

### 1. Flat-search baseline path

Add a clean path that runs:

- exact same export/backend,
- exact same budget,
- exact same evaluation,
- but no coarse-to-fine search,
- and optionally no config refinement.

This should become the control arm.

### 2. Two-stage optimization with explicit provenance

Every evaluated candidate should be labeled as one of:

- bit-allocation-derived,
- config-refinement-derived,
- joint-derived.

This makes it much easier to interpret wins.

### 3. Full-model accounting mode

Add a contract option that explicitly switches search accounting between:

- `linear_only`
- `full_model_assumed_non_linear_bf16`

### 4. Artifact validation stage

Insert a post-export validation step:

1. inspect intended targets before save,
2. build artifact,
3. reload artifact,
4. inspect quantized module histogram and matched targets,
5. only benchmark if validation passes.

### 5. Ablation-driven priors

Use more direct ablation evidence to construct priors on:

- where `16-bit` actually helps,
- where `4-bit` is safe,
- and where config overrides are worth tuning.

## Concrete Recommendations for the Next Cycle

### Recommended mainline

1. Fix saved-artifact target integrity.
2. Choose MATH-500 as the primary benchmark.
3. Keep the exact current budget as a reproducibility baseline, but clearly label it linear-only.
4. Re-run a flat `per_block_component` search without hierarchical search.
5. Re-run the same search with hierarchical search.
6. Compare both with and without config refinement.
7. Only after that, decide whether hierarchical search is worth keeping as the default.

### Recommended benchmark plan

- Primary:
  - MATH-500 dev slice for iteration
  - larger MATH-500 slice for acceptance
- Secondary:
  - GSM8K for transfer check

### Recommended decision rule

Promote a pipeline variant only if it:

- survives export and reload cleanly,
- beats uniform `int8` on MATH-500,
- and does not rely on unclear accounting claims.

## Questions for Deep Research

1. For weight-only quantization of large reasoning models, which benchmarks most reliably expose differences between uniform `int8` and task-aware mixed precision?
2. In practice, do coarse-to-fine search strategies help for weight-only mixed-precision allocation, or do they mostly add heuristic instability versus fine-grained search with better priors?
3. What is the most defensible budget accounting scheme for mixed-precision checkpoint comparison when the search only touches linear layers?
4. For `llmcompressor`-style export/reload pipelines, what are known failure modes around target matching, regex targets, and override-group persistence?
5. For mixed-precision search, which search parameters tend to matter most:
   - grouping granularity,
   - bit set,
   - mutation/population depth,
   - or quantization-config parameters like group size and symmetry?
6. If the goal is to beat uniform `int8` on a single high-value benchmark, is it better to:
   - tune bit allocation first,
   - tune backend config first,
   - or jointly optimize both?

## Bottom Line

The project is no longer blocked on having a functioning search pipeline.
It is now blocked on **clarity**:

- clarity about the budget being claimed,
- clarity about whether wins come from bit placement or backend config tweaks,
- clarity about whether saved artifacts truly preserve intended policies,
- and clarity about which benchmark is actually discriminative enough to optimize against.

The strongest next move is not “search deeper.”
The strongest next move is to simplify the experiment matrix, fix artifact integrity, and re-anchor the optimization loop on MATH-500.
