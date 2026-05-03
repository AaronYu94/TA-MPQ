# BF16 Budget Selection Under a Fixed Mixed-Precision Compression Budget

## Goal

Given:

- a target mixed-precision weight budget,
- a resulting effective compression ratio,
- a set of model groups or layers with task-dependent importance scores,

determine **how much of the total budget should be spent on keeping some groups in BF16**, as opposed to spending that budget on upgrading other groups from `int4` to `int8`.

This is a **budget allocation problem**, not just a ranking problem.

The central question is:

> Under a fixed total weight budget, how much high-precision BF16 capacity should be reserved, and which groups should receive it, so that final task performance is maximized?

## Why This Matters

The current mixed-precision design allows three effective tiers:

- `int4`: default compression floor
- `int8`: medium-precision tier
- `BF16`: high-precision rescue tier

BF16 is much more expensive than `int8`, and especially expensive relative to `int4`.

Under a fixed target budget:

- every group kept in `BF16` consumes budget that could otherwise be used to upgrade several other groups from `int4` to `int8`
- as the target budget gets tighter, the opportunity cost of BF16 increases
- therefore the BF16 allocation should not be fixed globally; it should depend on the target budget and thus on the effective compression ratio

## Current Context

The project currently targets a matched linear-weight footprint budget derived from a uniform-8 baseline. In practice:

- we compare against a matched-budget uniform `int8` model
- mixed precision is chosen over groupings such as `per_block_component`
- the current pipeline supports `4/8/16` bit assignments
- high-precision groups are effectively left unquantized and therefore behave like BF16/FP16-rescued groups

Historically, broad free-form search over `{4,8,16}` has been difficult to optimize cleanly. A more structured alternative is:

1. determine a BF16 reserve or cutoff as a function of budget
2. freeze the BF16 groups
3. allocate the remaining budget across `int4` and `int8`

The missing piece is a principled algorithm for step 1.

## Core Problem Statement

We want to solve:

> Given a total target budget `B`, determine the optimal BF16 budget `B_16(B)` or equivalent BF16 selection rule, such that the final mixed-precision policy achieves the best task accuracy under the total budget.

This can be framed at either of two levels:

### Level 1: BF16 Budget Fraction

Determine a function:

`f(B) -> BF16 reserve`

where the reserve may be expressed as:

- a fraction of the total budget
- a fraction of the upgrade slack above an all-`int4` baseline
- a maximum number of bytes allowed for BF16 rescue

### Level 2: BF16 Selection Rule

Determine a rule:

`(group scores, parameter counts, target budget) -> set of BF16 groups`

This rule may or may not explicitly expose a scalar BF16 reserve.

## Budget Math

Let each group `g` have:

- parameter count `N_g`
- assigned precision `b_g in {4, 8, 16}`

Then the matched linear-weight footprint is:

`W = sum_g (N_g * b_g / 8)` bytes

Given a target budget `B`, all assignments must satisfy:

`W <= B`

### Important Opportunity-Cost Facts

Relative to a uniform-8 baseline:

- moving one group from `8 -> 16` costs `+8` bits per parameter
- moving one group from `8 -> 4` saves `-4` bits per parameter

So for equal-sized groups:

- one BF16 rescue requires roughly two `8 -> 4` demotions to offset it

Relative to an all-`int4` baseline:

- `4 -> 8` costs `+4` bits per parameter
- `8 -> 16` costs another `+8` bits per parameter

So BF16 is the most expensive marginal decision.

This means:

- under tighter budgets, BF16 becomes harder to justify
- under looser budgets, BF16 may become worthwhile for the most important groups

## Desired Output of Research

Deep research should produce an algorithm that answers:

1. **How should BF16 budget depend on target budget or compression ratio?**
2. **What statistics should drive BF16 selection?**
3. **How should BF16 compete against `int4 -> int8` upgrades?**
4. **What optimization method is appropriate?**
5. **How should this algorithm behave across different model sizes and different target budgets?**

## Candidate Inputs Available in This Project

The project can already provide the following signals:

- group-level task sensitivity scores
- group-level value priors
- group-level parameter counts
- exact matched linear-weight budget accounting
- direct benchmark evaluations on math/reasoning tasks
- ablation-derived signals for precision changes in some groups

So the research does **not** need to invent the entire pipeline from scratch. It should focus on the decision rule for allocating BF16 under a total budget.

## Key Research Questions

### 1. What Should the BF16 Budget Be a Function Of?

Possible choices:

- total target budget `B`
- effective compression ratio relative to BF16 or uniform-8
- upgrade slack above an all-`int4` baseline
- model size
- task type
- score distribution of group sensitivities

The strongest candidate may be:

- **upgrade slack above an all-`int4` baseline**

because both BF16 rescue and `int4 -> int8` upgrades spend from the same pool of extra bits.

### 2. Should BF16 Be Chosen by Count, Threshold, or Knapsack?

Possible formulations:

- top-K groups by sensitivity
- all groups above a sensitivity threshold
- greedy by value-per-byte
- full knapsack-style optimization
- two-stage optimization where BF16 is chosen first and `int4/int8` later

We want to know which formulation is theoretically and empirically best.

### 3. What Signal Should Drive BF16 Rescue?

A raw task-sensitivity score may not be sufficient.

The research should evaluate whether BF16 selection should be driven by:

- raw task sensitivity
- estimated marginal benefit of `8 -> 16`
- estimated marginal benefit per byte
- separate transition-specific ablations
- structural priors such as late-layer preference or component-family preference

### 4. How Should BF16 Compete Against Int8 Upgrades?

This is the central tradeoff.

Every byte spent on BF16 is a byte that cannot be spent upgrading more groups from `int4` to `int8`.

The research should explicitly model:

- benefit of `4 -> 8`
- benefit of `8 -> 16`
- cost of each transition
- how to compare them fairly under one shared budget

### 5. Should BF16 Be Globally Fixed or Recomputed Per Budget?

The working hypothesis is:

- BF16 allocation should be recomputed for each target budget

The research should confirm or refute this.

### 6. How Sparse Should BF16 Be?

The algorithm should ideally answer:

- under what budgets should BF16 be zero?
- when does a nonzero BF16 reserve become worthwhile?
- how quickly should BF16 share grow as budgets become looser?

### 7. How Should This Interact With Search?

Possible downstream pipelines:

- deterministic BF16 allocation, then deterministic `int4/int8` allocation
- deterministic BF16 allocation, then local search over `int4/int8`
- deterministic BF16 allocation, then evolutionary search over `int4/int8`
- joint optimization over all tiers

The research should recommend whether BF16 should be:

- fully greedy,
- optimized jointly,
- or fixed first and then excluded from later search.

## Constraints

Any proposed algorithm should respect the following practical constraints:

- must work with heterogeneous group sizes
- must operate under exact matched-budget accounting
- must be compatible with real model groups, not only equal-size toy layers
- should be interpretable enough to debug
- should be computationally feasible for repeated experiments
- should not depend on an extremely accurate surrogate model

## Non-Goals

This research question is **not** asking for:

- a new quantization backend
- activation quantization
- full-model memory accounting redesign
- generic benchmark selection advice
- a free-form search over all `4/8/16` assignments

The narrow goal is:

> determine the BF16 budget or BF16 allocation rule under a fixed total compression budget

## Strong Hypotheses To Evaluate

These are not conclusions, but promising hypotheses worth testing:

1. The optimal BF16 share is a function of **upgrade slack above all-`int4`**, not just total model size.
2. BF16 should be extremely sparse under matched uniform-8 budgets.
3. A fixed BF16 fraction across budgets is likely suboptimal.
4. BF16 selection should be based on **marginal value per byte**, not raw sensitivity alone.
5. Deterministic BF16 rescue plus search over only `{4,8}` may outperform unconstrained search over `{4,8,16}`.

## Desired Deliverable From Deep Research

The ideal answer should provide:

1. a precise optimization formulation
2. a recommended algorithm
3. justification for the objective and stopping rule
4. tradeoff discussion between greedy, threshold-based, and knapsack-style methods
5. guidance on how BF16 budget should scale with tighter vs looser budgets
6. suggestions for which empirical signals to measure in order to estimate BF16 value
7. a practical implementation sketch suitable for a real mixed-precision quantization pipeline

## One-Sentence Summary

We need an algorithm that, **given a total mixed-precision budget and thus a compression ratio, decides how much scarce high-precision BF16 capacity should be reserved, because BF16 rescue and `int4 -> int8` upgrades compete for the same limited budget and their tradeoff changes as the budget tightens or loosens.**
