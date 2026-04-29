# TA-MPQ Poster Draft

## Title

TA-MPQ: Structured Exact-Budget Coarse-to-Fine Mixed-Precision Search for Task-Aware LLM Quantization

## One-Sentence Takeaway

We replace unstable evolutionary search with a structured exact-budget mixed-precision search, and show that task-aware mixed policies can recover a substantial portion of the quality lost by uniform INT4 under the same memory budget.

## Motivation & Problem

Running LLMs under strict memory budgets matters for local inference and multi-LLM orchestration.

Uniform quantization is simple, but it treats all linear components the same even though their importance varies widely across tasks. Under a fixed budget, assigning the same precision everywhere can waste capacity on low-sensitivity groups while under-serving groups that are much more valuable for a target benchmark.

Our goal is to fit a stronger model into an INT4-sized footprint while preserving as much task performance as possible. Instead of increasing total budget, we keep the same exact raw weight footprint as uniform INT4 and reallocate precision across groups in a task-aware way.

Key points to emphasize on the poster:

- Uniform INT4 is budget-efficient but task-agnostic.
- Different benchmarks value different groups differently.
- The problem is not just quantization, but budget-constrained precision allocation.
- We want better accuracy than uniform INT4 without exceeding the same raw weight budget.

## Core Idea

We treat mixed-precision assignment as an exact-budget allocation problem rather than a free-form policy search problem.

The method has three main ingredients:

1. **Task sensitivity profiling**
   Measure how important each linear group is for the target benchmark.
2. **Exact-budget frontier construction**
   Promote high-value groups to higher precision, demote low-value large groups to lower precision, and keep the total raw weight footprint matched to uniform INT4.
3. **Coarse-to-fine search**
   Search over an ordered spectrum of feasible policies instead of relying on random mutation and crossover.

The current implementation uses `{2, 4, 8}` mixed precision:

- `INT8` preserves the most valuable groups.
- `INT2` pays for those promotions under a strict budget.
- `INT4` fills the remainder.

## Why Not Evolutionary Search?

We originally explored an evolutionary algorithm, but it was not a good fit for the exact-budget version of this problem.

The main issue was that **crossover and exact budget enforcement conflict**:

- Crossover often produced children that exceeded the target budget.
- The repair step then rewrote large parts of the child policy.
- As a result, the final repaired policy often no longer preserved meaningful inherited structure from the parents.
- Under limited compute, this made the search unstable, noisy, and hard to interpret.

The key message for the poster:

- We are **not** claiming evolutionary search is always ineffective.
- We are saying it was unreliable in our setting: **strict exact-budget constraints + limited compute + mixed `INT2/4/8` search**.
- This motivated a switch to a more structured and reproducible search procedure.

## Method Overview

### Structured Exact-Budget Coarse-to-Fine Mixed-Precision Search

1. Build a sensitivity profile over model groups for the target task.
2. Rank groups by a value signal based on **task sensitivity / true group size**.
3. Construct an **ordered exact-budget frontier** of mixed-precision policies.
4. Split the frontier into `8` evenly spaced coarse sectors and evaluate one representative policy per sector.
5. Select the best coarse sector.
6. Generate `4` evenly spaced refined policies inside that sector and evaluate them.
7. Quantize the winning policy with real PTQ / oneshot quantization.
8. Compare the resulting artifact against `BF16`, `uniform INT8`, and `uniform INT4`.

Total policy evaluations for the final selection step:

- `8` coarse candidates
- `4` refined candidates
- `12` total evaluations

Important implementation note:

- The final mixed models are **real quantized artifacts**, not surrogate-only selections.

## Exact-Budget Policy Frontier

The search space is not random. It is an **ordered policy spectrum** under a fixed INT4-sized budget.

Interpretation of the frontier:

- Left side: more `INT8 + INT2`, more aggressive reallocation
- Right side: more `INT4`, closer to uniform INT4
- Every candidate remains within the same exact raw weight footprint budget

Two workload-family policies are currently in use:

### Code-oriented policy

- policy id: `gpf_refine_00_i0112`
- policy hash: `7caae6...2043e4`
- group counts: `57 x INT2`, `80 x INT4`, `112 x INT8`
- budget slack fraction: `0.000264`

### Math-oriented policy

- policy id: `gpf_refine_02_i0110`
- policy hash: `9bbaf3...59b9e9`
- group counts: `75 x INT2`, `64 x INT4`, `110 x INT8`
- budget slack fraction: `0.000462`

This is an important story point:

- We do **not** re-search one policy per benchmark.
- We currently use **one policy per workload family**:
  - code policy for `MMLU-coding`, `HumanEval`, `BigCodeBench`
  - math policy for `MATH-500`

## Coarse-to-Fine Search

The final policy selection procedure is intentionally simple and reproducible.

### Round 1 - Coarse

- Divide the ordered frontier into `8` evenly spaced sectors.
- Evaluate one representative policy near the center of each sector.
- Select the best sector.
- If policies tie, prefer the one with a higher `INT4` fraction.

### Round 2 - Fine

- Generate `4` evenly spaced refined policies inside the winning sector.
- Evaluate all `4`.
- Select the highest-accuracy policy as the final output.

Why this matters:

- No random crossover
- No large off-budget jumps
- No opaque repair-heavy evolution loop
- Easy to explain and easy to rerun

## Experimental Setup

### Model and policy setting

- base model: `Qwen/Qwen3.5-9B`
- mixed precision space: `{2, 4, 8}`
- budget target: exact raw weight footprint of uniform `INT4`

### Baselines

- `BF16`
- `uniform INT8`
- `uniform INT4`
- `mixed exact-INT4-budget`

### Benchmarks

- `MMLU-coding`
- `HumanEval / EvalPlus`
- `BigCodeBench-Hard`
- `MATH-500 last100`

### Evaluation note

We currently use `llmcompressor.oneshot` rather than a highly optimized INT4 serving stack.

Therefore:

- `accuracy` and `token usage` are the most trustworthy comparison points
- `latency` is still useful to report, but is more affected by backend implementation details

## Main Results

### Accuracy summary

| Benchmark | BF16 | INT8 | INT4 | Mixed | Main interpretation |
|---|---:|---:|---:|---:|---|
| `MMLU-coding last100` | `95%` | `94%` | `91%` | `91%` | mixed matches INT4, no gain yet |
| `HumanEval pass@1` | `87.8%` | `89.0%` | `81.1%` | `84.8%` | mixed clearly improves over INT4 |
| `HumanEval+ pass@1` | `82.3%` | `82.3%` | `76.8%` | `79.9%` | mixed again improves over INT4 |
| `BigCodeBench-Hard pass@1` | `29.05%` | `27.03%` | `16.89%` | `23.65%` | strongest current win over INT4 |
| `MATH-500 last100` | `86%` | `-` | `80%` | `84%` | math-oriented mixed closes much of the INT4 gap |

### BigCodeBench-Hard headline

This is currently the clearest success case for the method:

- `BF16`: `43 / 148`
- `INT4`: `25 / 148`
- `Mixed`: `35 / 148`

So the mixed policy recovers `10` of the `18` tasks lost by uniform INT4 relative to BF16.

### High-level interpretation

- Mixed precision is **not** currently beating `INT8`.
- Mixed precision **does** consistently outperform `uniform INT4` on the stronger generative/code/math benchmarks we care about.
- The strongest story is not “mixed wins everything,” but rather:
  **under an INT4-sized budget, structured task-aware reallocation is meaningfully better than uniform INT4.**

## Token Usage

### Most important token observations

| Benchmark | BF16 avg completion tokens | INT8 | INT4 | Mixed | Main interpretation |
|---|---:|---:|---:|---:|---|
| `MMLU-coding last100` | `647.8` | `696.5` | `903.8` | `787.4` | mixed uses fewer tokens than INT4, but still above BF16/INT8 |
| `HumanEval` | `360.8` | `364.8` | `412.2` | `358.9` | mixed has the lowest token usage in this set |
| `BigCodeBench-Hard` | `512.2` | `511.0` | `533.7` | `501.7` | mixed has the best token profile here |
| `MATH-500 last100` | `1167.0` | `-` | `1494.3` | `1284.5` | mixed again improves over INT4, but MATH still has a long-tail token wall |

### Token-wall note

- `HumanEval` is capped by its evaluation setup (`max_new_tokens = 768`), so its `p95` sits near the token wall.
- `MATH-500` still shows a visible token-wall effect (`p95 = 4096`), so average token usage should be interpreted together with capped counts.

### Why token usage matters here

Token usage is not a perfect systems metric, but in the current setup it is often a more stable signal than raw latency:

- lower completion length usually means less wasted generation
- unlike latency, it is less entangled with backend/kernel support quality
- it complements accuracy by showing whether the policy is making the model more or less generation-efficient

## Key Figures To Include

Recommended figure set for the poster:

1. **Uniform INT4 vs Mixed under the same exact budget**
   Show two rows with the same total width:
   - top row: all `INT4`
   - bottom row: mixed `INT8 / INT4 / INT2`
   - caption: `same exact raw weight footprint`

2. **Method pipeline**
   Left-to-right workflow:
   - sensitivity profiling
   - rank by value / size
   - build exact-budget frontier
   - coarse search
   - fine search
   - final PTQ artifact
   - evaluation

3. **Policy frontier illustration**
   Show `S1 ... S8` as feasible policies under the same budget:
   - left: more `INT8 + INT2`
   - right: more `INT4`

4. **Coarse-to-fine search panel**
   Visualize:
   - Round 1: `8` sectors
   - select best sector
   - Round 2: `4` refined policies
   - total `12` evaluations

5. **Why not evolutionary search**
   Show:
   - parent A + parent B
   - crossover child
   - child exceeds budget
   - repair rewrites policy
   - callout: `repair destroys inherited structure`

6. **Results figure**
   Two-panel result chart:
   - left: benchmark accuracy
   - right: average completion tokens

7. **Optional workload-family-aware policy figure**
   Show:
   - one code policy feeding `MMLU-coding`, `HumanEval`, `BigCodeBench`
   - one math policy feeding `MATH-500`

If space is limited, prioritize:

- motivation figure
- method pipeline
- coarse-to-fine figure
- results figure

## Current Claim Boundary

These are the claims we can currently support with the evidence in hand:

- We can support that **structured task-aware mixed precision outperforms uniform INT4** on `HumanEval`, `HumanEval+`, `BigCodeBench-Hard`, and `MATH-500 last100`.
- We can support that the method works better on stronger generative/code/math benchmarks than on a simpler multiple-choice benchmark like `MMLU-coding`.
- We can support that **one policy per workload family** is a better description of the current system than “one universal mixed policy.”

These are the claims we should **not** make yet:

- We should not claim superiority over `INT8`.
- We should not claim a single universal mixed policy across all tasks.
- We should not claim latency superiority as a backend-agnostic result.

## Limitations

- No consistent win over `INT8`
- No single universal policy across all workload families
- Some benchmark sizes are still modest, so small gaps should be interpreted carefully
- Latency comparisons depend on the current serving/runtime stack
- `MATH-500` still shows token-wall effects in the current evaluation setup
- The current evidence is stronger for **workload-family-aware** policies than for benchmark-agnostic mixed precision

## Future Work

- Evaluate more model families and larger scales
- Compare layer-level vs. component-level quantization granularity
- Improve the sensitivity profiler and candidate generator
- Optimize directly for token usage in addition to accuracy
- Run controlled cross-task transfer experiments
- Test with a stronger optimized INT4 / mixed-precision serving backend
- Explore whether one policy can generalize across multiple related coding tasks or multiple related math tasks

## Notes

### Suggested poster emphasis

If the poster needs one central message, use this:

> Under an INT4-sized budget, structured task-aware mixed precision is a practical and reproducible way to recover a meaningful portion of the quality lost by uniform INT4.

### Suggested naming

- umbrella project name: `TA-MPQ`
- current method name:
  `Structured Exact-Budget Coarse-to-Fine Mixed-Precision Search`

### Color suggestion for figures

- `BF16`: dark neutral / navy
- `INT8`: warm orange or soft red
- `INT4`: muted green
- `INT2`: light gray
- `Mixed`: crimson highlight

### Phrasing to avoid

- Avoid: “we beat INT8”
- Avoid: “one mixed policy works for every benchmark”
- Avoid: “latency proves system superiority”

### Phrasing that is safe

- “Mixed precision recovers a substantial fraction of the quality lost by uniform INT4.”
- “The current evidence supports workload-family-aware mixed precision.”
- “The structured exact-budget search is more stable and interpretable than the earlier evolutionary-search attempt in this setting.”
