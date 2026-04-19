# Problem Statement: Evolutionary Search vs Beam-Style Search for Mixed-Precision `{2,4,8}` Policies

## Context
This project is searching for mixed-precision quantization policies for `Qwen/Qwen3.5-9B` on `MATH-500`.

The current setup is:
- objective: maximize task accuracy under an **int4-matched linear-weight footprint**
- model family: `Qwen/Qwen3.5-9B`
- search space: per-group bit assignments in `{2,4,8}`
- grouping: `per_block_component`
- benchmark mode: `simple_evals_nonthinking`, greedy decoding, `max_new_tokens=4096`
- search split: a persistent `first300` deck, with small per-turn evals
- finalist comparison split: `last100`

The current search implementation is nominally “evolutionary,” but in practice it behaves more like:
- generate a small set of neighbors
- quantize/evaluate them
- keep the top `beam_width` survivors
- repeat

So one question is whether this should really be treated as an evolutionary algorithm at all, or whether it should be redesigned as an explicit beam search over sparse budgeted edits.

## Observed Problem
The search is producing candidates that are **too close to uniform int4**.

Even after multiple rounds, almost every candidate is:
- `~242-243` groups at `4-bit`
- only `4-5` groups at `2-bit`
- only `2-3` groups at `8-bit`

For example:

### `round-09-candidate-01`
- `243` groups at `4-bit`
- `4` groups at `2-bit`
- `2` groups at `8-bit`

Non-`4-bit` groups:
- `8-bit`
  - `block:0:linear_attn.in_proj_a`
  - `global::lm_head`
- `2-bit`
  - `block:0:linear_attn.in_proj_b`
  - `block:10:linear_attn.in_proj_a`
  - `block:10:linear_attn.in_proj_b`
  - `block:12:linear_attn.in_proj_a`

### `round-05-candidate-01`
- `242` groups at `4-bit`
- `4` groups at `2-bit`
- `3` groups at `8-bit`

Non-`4-bit` groups:
- `8-bit`
  - `block:0:linear_attn.in_proj_b`
  - `block:0:linear_attn.in_proj_z`
  - `global::lm_head`
- `2-bit`
  - `block:10:linear_attn.in_proj_a`
  - `block:10:linear_attn.in_proj_b`
  - `block:12:linear_attn.in_proj_a`
  - `block:22:mlp.gate_proj`

### Most non-uniform candidate found so far: `round-04-candidate-02`
- `241` groups at `4-bit`
- `5` groups at `2-bit`
- `3` groups at `8-bit`

This is still only `8 / 249` groups different from `4-bit`.

## Why This Looks Wrong
The search space is supposed to be a meaningful `{2,4,8}` mixed-precision space, but the actual explored policies are tiny perturbations of uniform int4.

That suggests a structural failure mode:

### Hypothesis: repair collapse
The current algorithm likely does this:
1. mutate a full assignment over all groups
2. repair the assignment back to the exact budget
3. dedupe candidates after repair

If repair is strong enough, then many different mutations collapse back to almost the same repaired policy.

That causes:
- weak effective heredity
- poor diversity
- candidates clustering near uniform int4
- the search failing to explore strongly non-uniform but still budget-feasible policies

In other words, the algorithm may be “evolutionary” in interface, but not in real search behavior.

## Evidence From Current Run
The non-staged search has produced:
- `2` seeds
- round candidates through at least round `9`

Search-slice results so far include:
- `seed-01 uniform_int4`: `5/10 = 0.50`
- `seed-02 compression_first`: `10/10 = 1.00`
- best later candidates often around `9/10 = 0.90`

But the important issue is not the exact search-slice score.
The important issue is that the produced policies remain near-uniform int4.

So even though the search is operationally progressing, it may be exploring only a very narrow subspace.

## Decision Question
Given the above behavior, what search design is more appropriate?

### Option A: Keep the evolutionary algorithm, but fix the representation
Possible direction:
- stop representing candidates as full 249-group assignments
- instead represent a candidate as a sparse delta from uniform int4:
  - a set of `4 -> 8` promotions
  - a set of `4 -> 2` demotions
- enforce budget mostly by construction, not by global repair
- use mutation operators over these sparse deltas

This would preserve the evolutionary framing while reducing repair collapse.

### Option B: Replace it with an explicit beam search
Possible direction:
- treat uniform int4 as the base policy
- define valid edit bundles that preserve budget:
  - add one promoted `8-bit` group plus compensating `2-bit` demotions
  - swap promoted groups
  - swap demoted groups
  - add/remove structured block-level bundles
- each round:
  - expand current beam
  - score children
  - keep top `k`

This may fit the problem better because:
- the space is sparse
- the constraint is strict
- crossover is probably not useful
- beam search is easier to reason about than mutation + repair

## Core Technical Question
For a budget-constrained mixed-precision search where the current algorithm collapses toward near-uniform int4, which redesign is more appropriate:

1. a repaired evolutionary algorithm over sparse deltas
2. an explicit beam search over exact-budget edit bundles

## Constraints
The answer should respect these project constraints:
- this is a 1-month class project, not a production system
- primary goal is **accuracy**, not latency
- the current runtime path is acceptable as an internal research caveat
- the search split is noisy and relatively small, so the algorithm should be robust to noisy evaluations
- implementation complexity matters; a cleaner, easier-to-debug search may be preferable even if it is less theoretically fancy

## What I Want Help Deciding
Please give a recommendation on:

1. whether the current failure mode is best understood as **repair collapse**
2. whether the search should remain “evolutionary” or be reframed as **beam search**
3. what candidate representation should be used
4. how budget feasibility should be enforced
5. what mutation / expansion operators should exist
6. how to preserve diversity without wasting too much compute

## My Current Lean
I currently lean toward:
- using **uniform int4 as the base policy**
- representing candidates as sparse exact-budget deviations from that base
- and likely using a **beam search** rather than a classic evolutionary algorithm

But I want an outside judgment on whether that is the right redesign.
