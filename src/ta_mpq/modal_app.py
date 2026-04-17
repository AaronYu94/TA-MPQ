from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import modal


CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2] if len(CURRENT_FILE.parents) >= 3 else Path.cwd()
SRC_ROOT = PROJECT_ROOT / "src"
if SRC_ROOT.exists() and str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ta_mpq.baseline import save_summary
from ta_mpq.contracts import load_contract


app = modal.App("ta-mpq-experiments")

cache_volume = modal.Volume.from_name("ta-mpq-hf-cache", create_if_missing=True)

baseline_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.6.0",
        "transformers>=4.51.0",
        "datasets>=3.5.0",
        "accelerate>=1.6.0",
        "huggingface_hub>=0.30.0",
        "safetensors>=0.5.0",
        "sentencepiece>=0.2.0",
    )
    .add_local_python_source("ta_mpq")
)


@app.function(
    image=baseline_image,
    gpu="A100-80GB",
    timeout=60 * 60 * 4,
    volumes={"/cache": cache_volume},
)
def run_gsm8k_baseline(
    contract_data: dict[str, Any],
    model_role: str,
    limit: int | None = None,
) -> dict[str, Any]:
    import os

    os.environ["HF_HOME"] = "/cache/hf"
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    from ta_mpq.baseline import evaluate_task_baseline
    from ta_mpq.contracts import ExperimentContract

    contract = ExperimentContract.from_dict(contract_data)
    actual_limit = limit if limit is not None else contract.baseline_eval_limit
    model_id = contract.resolve_model_role(model_role)
    summary = evaluate_task_baseline(
        model_ref=model_id,
        task_name=contract.task_name,
        limit=actual_limit,
        max_new_tokens=contract.generation_max_new_tokens,
        tokenizer_source=model_id,
        model_label=model_id,
        load_dtype="bfloat16",
    )
    summary["model_role"] = model_role
    summary["contract_name"] = contract.name
    return summary


@app.local_entrypoint()
def main(
    contract_path: str = "configs/experiment_contract.json",
    limit: int | None = None,
) -> None:
    contract = load_contract(PROJECT_ROOT / contract_path)
    native_summary = run_gsm8k_baseline.remote(contract.to_dict(), "native_baseline", limit)
    upper_bound_summary = run_gsm8k_baseline.remote(contract.to_dict(), "upper_bound", limit)

    save_summary(
        PROJECT_ROOT / "outputs" / "baselines" / _summary_filename(contract.name, native_summary),
        native_summary,
    )
    save_summary(
        PROJECT_ROOT / "outputs" / "baselines" / _summary_filename(contract.name, upper_bound_summary),
        upper_bound_summary,
    )


def _summary_filename(contract_name: str, summary: dict[str, Any]) -> str:
    role = str(summary["model_role"]).replace("_", "-")
    model_slug = str(summary["model_id"]).split("/")[-1].lower()
    return f"{contract_name}-{role}-{model_slug}.json"
