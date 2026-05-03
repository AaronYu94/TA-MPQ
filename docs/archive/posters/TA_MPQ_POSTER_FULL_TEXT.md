# TA-MPQ Poster Full Text

Snapshot date: 2026-04-22

## Title

TA-MPQ: Structured Mixed Precision at an Exact INT4 Budget

## Subtitle

We moved from evolutionary crossover to a deterministic task-sensitivity frontier: promote the most valuable groups to 8-bit, demote lower-value groups to 2-bit, and keep the full model near the raw uniform-INT4 weight footprint.

## One-line takeaway

The new structured search is controllable and reproducible. The current mixed 2/4/8-bit policy matches the INT4-sized raw weight budget, improves over uniform INT4 on HumanEval/EvalPlus, but does not yet separate from INT4 on MMLU-coding.

## Background

The original TA-MPQ route used evolutionary search over mixed-precision policies. That route became hard to interpret because crossover and budget repair were pulling in opposite directions: crossover randomly combines two policies, while exact-budget quantization needs coordinated promotions and demotions. In practice, many candidates were either repaired heavily after crossover or stayed too close to mostly INT4 policies.

We replaced this with a structured search route. Instead of asking an evolutionary algorithm to discover a policy from scratch, we use task sensitivity to build an ordered frontier. The frontier moves from an INT4 anchor toward policies with more INT8 groups. Because the target budget is approximately the raw uniform-INT4 footprint, every INT8 promotion must be balanced by lower-value INT2 demotions.

## Method

The current pipeline has five stages.

1. Profile task sensitivity on MMLU-coding development prompts.
2. Rank groups by expected value for 8-bit promotion and expected cost for 4-bit to 2-bit demotion.
3. Build a greedy exact-budget frontier from the uniform INT4 anchor.
4. Coarsely sample the frontier, then refine around the best local region.
5. Quantize the selected policy into a real model artifact and evaluate it on held-out coding tasks.

This is not the old evolutionary search. There is no random crossover in the current mainline. The main control variable is the amount and placement of INT8 under an exact INT4-sized budget.

## Current mixed policy

Policy ID: `gpf_refine_00_i0112`

Policy hash: `7caae6e48fce5c6736e9464543d7f78329ceae88693fa2a6db90d47dd82043e4`

Model: `Qwen/Qwen3.5-9B`

Method: `greedy_path_frontier`

Grouping: `per_block_component`

Bit space: `{2, 4, 8}`

Total groups: `249`

Changed groups vs uniform INT4: `169`

Estimated average bit-width: `3.9989`

Estimated raw weight footprint: `3.694 GB`

Budget rule: matched raw uniform-INT4 weight footprint

Budget slack: `0.026%`

Bit distribution by group count:

| Bit width | Groups |
|---|---:|
| 2-bit | 57 |
| 4-bit | 80 |
| 8-bit | 112 |

Bit distribution by parameter mass:

| Bit width | Parameter mass |
|---|---:|
| 2-bit | 26.00% |
| 4-bit | 61.02% |
| 8-bit | 12.98% |

Interpretation: the policy is not simply “many layers in 8-bit.” It upgrades a relatively small fraction of parameter mass to 8-bit, keeps most mass at 4-bit, and uses 2-bit on low-value groups to keep the total raw weight footprint close to uniform INT4.

## MMLU-coding results

Task: MMLU-coding last100

Prompt style: simple-evals style

Max new tokens: 4096

Result type: single-run accuracy on 100 examples

| Model | Accuracy | Correct / Total | Length capped | Avg latency |
|---|---:|---:|---:|---:|
| BF16 native | 95% | 95 / 100 | 0 | 9.14s |
| Uniform INT8 | 94% | 94 / 100 | 2 | 46.21s |
| Uniform INT4 | 91% | 91 / 100 | 5 | 66.88s |
| Mixed 2/4/8 | 91% | 91 / 100 | 3 | 12.29s |

Interpretation: MMLU-coding does not currently show a win for mixed over uniform INT4. It does show that the mixed policy remains near the INT4 accuracy level while using a structured exact-budget policy. This is useful, but it is not yet the main success case.

## HumanEval / EvalPlus results

Evaluator: official EvalPlus

Dataset: HumanEval / HumanEval+

Generation: greedy

Max new tokens: 768

| Model | HumanEval base pass@1 | HumanEval+ pass@1 |
|---|---:|---:|
| BF16 native | 87.8% | 82.3% |
| Uniform INT8 | 89.0% | 82.3% |
| Uniform INT4 | 81.1% | 76.8% |
| Mixed 2/4/8 | 84.8% | 79.9% |

Interpretation: HumanEval/EvalPlus is the strongest current signal for the mixed policy. Mixed recovers `+3.7` points over uniform INT4 on HumanEval base and `+3.0` points over uniform INT4 on HumanEval+. It is still below INT8 and BF16, but it improves the INT4-sized model without increasing the raw weight budget.

## BigCodeBench status

Dataset: BigCodeBench-Hard

Split: instruct

Prompt: official BigCodeBench instruct prompt builder

Max new tokens: 1280

Mode: generation-only first; execution grading pending

| Model | Generated | Length capped | Status |
|---|---:|---:|---|
| BF16 native | 148 / 148 | 2 | Generation complete; eval pending |
| Uniform INT8 | 20 / 148 | 0 | H100 running |
| Uniform INT4 | 20 / 148 | 0 | H100 running |
| Mixed 2/4/8 | 30 / 148 | 1 | H100 running |

We intentionally do not report BigCodeBench scores yet. The previous local evaluator timed out because the execution environment was incomplete. The new runner saves generation artifacts before evaluation, so evaluator failures will not erase samples.

## Current interpretation

The current story is not “mixed beats every baseline.” The current story is narrower and more defensible:

1. The search mechanism is now stable and interpretable.
2. The selected mixed policy exactly targets the INT4-sized raw weight budget.
3. INT8 remains a very strong upper quantized baseline.
4. Uniform INT4 loses quality on generative coding tasks.
5. Mixed recovers a meaningful part of the INT4 loss on HumanEval/EvalPlus.
6. MMLU-coding does not yet separate mixed from INT4.
7. BigCodeBench will be the next major test because it is generative, dependency-heavy, and closer to realistic code synthesis than multiple-choice coding questions.

## Limitations

The current evaluation is still not the final paper-grade result. MMLU-coding last100 is small. HumanEval has only 164 tasks. BigCodeBench execution scoring is pending. The current budget is raw weight footprint, not peak VRAM, throughput, or end-to-end serving cost. These should be reported separately.

## Next steps

1. Finish BigCodeBench generation for INT8, INT4, and mixed.
2. Run official BigCodeBench execution grading in a stable environment.
3. Report peak VRAM and tokens/sec separately from raw weight footprint.
4. Use BigCodeBench results to decide whether MMLU-coding sensitivity is sufficient or needs task-specific refinement.
5. If BigCodeBench shows mixed gains, promote this structured exact-budget route as the main TA-MPQ story.
6. If BigCodeBench does not show mixed gains, improve the sensitivity profiler and candidate generator before claiming task-aware benefit.

## Recommended poster conclusion

Structured exact-budget mixed precision is a better mainline than evolutionary crossover. The current 2/4/8-bit policy is not a universal win, but it is a real, reproducible INT4-sized quantized model that recovers part of the INT4 degradation on generative code evaluation. The next decisive result is BigCodeBench execution accuracy.
