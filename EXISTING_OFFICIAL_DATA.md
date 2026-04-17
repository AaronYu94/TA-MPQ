# Existing Official Data for TA-MPQ

Snapshot date: 2026-04-08

This note collects public, official data we can reuse before running our own Modal experiments.

## Current experiment contract

- Compressed source model: `Qwen/Qwen3.5-9B`
- Native baseline model: `Qwen/Qwen3.5-4B`
- Comparison rule: match approximate VRAM budget, not parameter count
- First task we plan to run ourselves: `GSM8K`

## Official model facts

| Item | Qwen3.5-9B | Qwen3.5-4B |
| --- | --- | --- |
| Parameters | 9B | 4B |
| Layers | 32 | 32 |
| Context length | 262,144 native, extendable to 1,010,000 | 262,144 native, extendable to 1,010,000 |
| License | Apache-2.0 | Apache-2.0 |

Source:
- https://huggingface.co/Qwen/Qwen3.5-9B
- https://huggingface.co/Qwen/Qwen3.5-4B

## Official benchmark snapshot

These numbers come from the official Qwen Hugging Face model cards. They are useful as public prior evidence that the native `9B` is generally stronger than the native `4B`, but they are not a substitute for our own `GSM8K`, latency, or VRAM measurements.

### Language benchmarks

| Benchmark | Qwen3.5-9B | Qwen3.5-4B | Delta |
| --- | ---: | ---: | ---: |
| MMLU-Pro | 82.5 | 79.1 | +3.4 |
| MMLU-Redux | 91.1 | 88.8 | +2.3 |
| C-Eval | 88.2 | 85.1 | +3.1 |
| SuperGPQA | 58.2 | 52.9 | +5.3 |
| GPQA Diamond | 81.7 | 76.2 | +5.5 |
| IFEval | 91.5 | 89.8 | +1.7 |
| IFBench | 64.5 | 59.2 | +5.3 |
| MultiChallenge | 54.5 | 49.0 | +5.5 |
| AA-LCR | 63.0 | 57.0 | +6.0 |
| LongBench v2 | 55.2 | 50.0 | +5.2 |
| HMMT Feb 25 | 83.2 | 74.0 | +9.2 |
| HMMT Nov 25 | 82.9 | 76.8 | +6.1 |
| LiveCodeBench v6 | 65.6 | 55.8 | +9.8 |
| OJBench | 29.2 | 24.1 | +5.1 |
| BFCL-V4 | 66.1 | 50.3 | +15.8 |
| TAU2-Bench | 79.1 | 79.9 | -0.8 |
| VITA-Bench | 29.8 | 22.0 | +7.8 |
| DeepPlanning | 18.0 | 17.6 | +0.4 |
| MMMLU | 81.2 | 76.1 | +5.1 |
| MMLU-ProX | 76.3 | 71.5 | +4.8 |
| NOVA-63 | 55.9 | 54.3 | +1.6 |
| INCLUDE | 75.6 | 71.0 | +4.6 |
| Global PIQA | 83.2 | 78.9 | +4.3 |
| PolyMATH | 57.3 | 51.1 | +6.2 |
| WMT24++ | 72.6 | 66.6 | +6.0 |
| MAXIFE | 83.4 | 78.0 | +5.4 |

Interpretation:
- The public table supports the general hypothesis that native `9B` is usually stronger than native `4B`.
- The biggest public deltas that look closest to our target domains are on `HMMT`, `LiveCodeBench`, and `PolyMATH`.
- `TAU2-Bench` is a reminder that bigger is not uniformly better on every benchmark.

Source:
- https://huggingface.co/Qwen/Qwen3.5-4B
- https://huggingface.co/Qwen/Qwen3.5-9B

## Mixed-precision feasibility evidence

The official LLM Compressor docs explicitly support non-uniform quantization and mixed-precision recipes:

- mixed precision examples include `int8 + int4`
- a higher-precision config group can be assigned to specific layers such as `down_proj`
- the resulting model is marked as `mixed-precision` in `config.json`

This is good evidence that per-layer mixed precision is a real supported path in the serving stack we are considering. It does not yet prove that our exact `Qwen3.5-9B` setup, our target bit set `{2, 3, 4, 8}`, or our target tasks will work end to end.

Source:
- https://docs.vllm.ai/projects/llm-compressor/en/stable/examples/quantization_non_uniform/

## Data that is still missing and must be measured by us

The official cards do not give the exact numbers we need for the first project milestone:

- `GSM8K` is not listed in either official model card
- `HumanEval` is not listed in either official model card
- Modal-specific `latency` is not listed
- Modal-specific `peak VRAM` is not listed
- our target comparison, "quantized 9B at roughly 4B budget vs native 4B," is not publicly reported in the official cards

This means we can reuse official benchmark tables for motivation and context, but we still need to run:

1. native `4B` baseline on our harness
2. native `9B` baseline on our harness
3. mixed-precision feasibility on `9B`
4. later, quantized `9B` vs native `4B`

## Practical takeaway

What we can safely treat as already known:

- the official `9B` is generally stronger than the official `4B`
- the serving ecosystem already supports mixed-precision, non-uniform quantization recipes

What we still need to establish ourselves:

- whether the native `4B` defines a realistic VRAM budget on Modal for our setup
- whether `9B` can be compressed to that budget while staying competitive on our chosen task
- whether our chosen quantization backend can support the full policy space we want to search
