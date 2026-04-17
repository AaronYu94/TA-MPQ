# TA-MPQ

This workspace contains the first executable scaffold for the TA-MPQ project.

Current scope:

- lock the experiment contract around `Qwen/Qwen3.5-9B` vs `Qwen/Qwen3.5-4B`
- run a GSM8K baseline harness on Modal
- track exact-match accuracy, latency, and GPU memory
- probe mixed-precision feasibility before building surrogate search

## Experiment contract

The active contract is stored in [configs/experiment_contract.json](/Users/aaronyu/Desktop/TA-MPQ/configs/experiment_contract.json).

Key rule:

- compare a task-aware quantized `9B` checkpoint against the native `4B` baseline at approximately matched VRAM budget, not matched parameter count

## Quickstart

Run the native baselines on Modal:

```bash
modal run src/ta_mpq/modal_app.py::main --limit 10
```

Probe mixed-precision feasibility with the default smoke-test policy:

```bash
modal run src/ta_mpq/modal_feasibility_app.py::run_feasibility_probe --calibration-limit 16
```

Run local unit tests:

```bash
python3 -m unittest discover -s tests
```

## Notes

- The baseline harness is wired for `GSM8K` first because it is cheap to evaluate and stable to debug.
- The mixed-precision policy layer supports the project search space `{2, 3, 4, 8}`.
- The current `llm-compressor` runtime bridge is intentionally stricter and only emits backend configs for `{4, 8}` until we verify whether lower-bit mixed precision is actually deployable for this model family.
