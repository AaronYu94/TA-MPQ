# Search Analysis

## Current Inputs

- Mainline contract: [configs/experiment_contract_27b_9b.json](/Users/aaronyu/Desktop/TA-MPQ/configs/experiment_contract_27b_9b.json)
- Feasibility report with layer stats: [outputs/feasibility/gsm8k-qwen35-27b-vs-9b-report.json](/Users/aaronyu/Desktop/TA-MPQ/outputs/feasibility/gsm8k-qwen35-27b-vs-9b-report.json)
- Native 9B baseline budget source: [outputs/baselines/gsm8k-qwen35-27b-vs-9b-native-baseline-qwen3.5-9b.json](/Users/aaronyu/Desktop/TA-MPQ/outputs/baselines/gsm8k-qwen35-27b-vs-9b-native-baseline-qwen3.5-9b.json)
- First proxy search: [outputs/search/gsm8k-qwen35-27b-vs-9b-proxy-search.json](/Users/aaronyu/Desktop/TA-MPQ/outputs/search/gsm8k-qwen35-27b-vs-9b-proxy-search.json)
- Second proxy search: [outputs/search/gsm8k-qwen35-27b-vs-9b-proxy-search-v2.json](/Users/aaronyu/Desktop/TA-MPQ/outputs/search/gsm8k-qwen35-27b-vs-9b-proxy-search-v2.json)

## What Changed In V2

- Added stronger low-bit exploration bias in [src/ta_mpq/search.py](/Users/aaronyu/Desktop/TA-MPQ/src/ta_mpq/search.py)
- Added an exploratory seed that assigns `2/3-bit` to low-sensitivity groups before budget repair
- Relaxed repair so candidates do not always refill to nearly all-`8-bit`
- Added compression and low-bit bonuses to the proxy fitness
- Increased search pressure with larger population and more generations

## Budget

- Native `Qwen3.5-9B` target budget used by search: about `16.748 GiB`
- Search grouping: `per_block_component`
- Number of search groups: `497`

## V1 Summary

- V1 top candidates mostly stayed in `{4, 8}` only
- Best candidate:
  - average bit-width: `5.615`
  - estimated footprint: `16.748 GiB`
  - bit counts: `234 x 8-bit`, `263 x 4-bit`

## V2 Summary

- V2 top candidates now use all four project bit-widths: `{2, 3, 4, 8}`
- Best candidate:
  - fitness: `1.0366`
  - average bit-width: `5.5291`
  - estimated footprint: `16.4921 GiB`
  - provenance: `generation_17`
  - bit counts:
    - `178 x 8-bit`
    - `242 x 4-bit`
    - `63 x 3-bit`
    - `14 x 2-bit`

## Consistent Patterns In Top-5

- Frequently preserved at `8-bit`:
  - many `linear_attn.in_proj_qkv`
  - many `linear_attn.out_proj`
  - many `mlp.down_proj`
- Frequently pushed to `2/3-bit`:
  - several `self_attn.k_proj`
  - several `self_attn.q_proj`
  - some `self_attn.o_proj`
  - some `self_attn.v_proj`

This is the first sign that the search is learning a non-uniform allocation pattern rather than just reproducing the initial heuristic.

## Takeaway

- The proposal pipeline is now past simple feasibility.
- We have a working proxy evolutionary search loop.
- The search is now producing budget-feasible, genuinely mixed-bit candidates for the `27B -> 9B` mainline.
- The next missing step is to validate these top policies with a real quantization backend once the `Qwen3.5` toolchain issue is resolved.

