from __future__ import annotations

import json
import importlib.machinery
from pathlib import Path
import random
import re
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


app = modal.App("ta-mpq-feasibility")

cache_volume = modal.Volume.from_name("ta-mpq-hf-cache", create_if_missing=True)
artifact_volume = modal.Volume.from_name("ta-mpq-artifacts", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")
REFERENCE_ACCURACY_SENTINEL = -999.0
TORCH_VERSION = "2.10.0"
TORCH_PACKAGE = f"torch=={TORCH_VERSION}"
ACCELERATE_PACKAGE = "accelerate==1.6.0"
DATASETS_PACKAGE = "datasets==4.0.0"
EINOPS_PACKAGE = "einops==0.8.2"
HF_HUB_PACKAGE = "huggingface_hub==1.11.0"
LLMCOMPRESSOR_PACKAGE = "llmcompressor==0.8.0"
SAFETENSORS_PACKAGE = "safetensors==0.5.0"
SENTENCEPIECE_PACKAGE = "sentencepiece==0.2.0"
NINJA_PACKAGE = "ninja==1.11.1.4"
EVALPLUS_PACKAGE = "evalplus==0.3.1"
BIGCODEBENCH_PACKAGE = "bigcodebench==0.2.2.dev2"
DEFAULT_MODAL_GPU = "B200"
A100_40GB_MODAL_GPU = "A100-40GB"
H100_MODAL_GPU = "H100"
DEFAULT_STAGE1_TURN_LIMIT = 10
DEFAULT_STAGE2_TURN_LIMIT = 15
DEFAULT_ROUND_PROPOSAL_EVAL_COUNT = 4
DEFAULT_ROUND_SURVIVOR_EVAL_COUNT = 2
FLASH_LINEAR_ATTENTION_INSTALL = (
    "python -m pip uninstall -y fla-core flash-linear-attention || true && "
    "python -m pip install -U fla-core==0.4.2 flash-linear-attention==0.4.2"
)

report_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        TORCH_PACKAGE,
        # Qwen3.5 loading currently requires a Transformers build that
        # recognizes the qwen3_5 architecture.
        "git+https://github.com/huggingface/transformers.git",
        DATASETS_PACKAGE,
        ACCELERATE_PACKAGE,
        HF_HUB_PACKAGE,
        SAFETENSORS_PACKAGE,
        SENTENCEPIECE_PACKAGE,
        EINOPS_PACKAGE,
        NINJA_PACKAGE,
    )
    .run_commands(FLASH_LINEAR_ATTENTION_INSTALL)
    .add_local_python_source("ta_mpq")
)

quant_source_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        # Keep the source quantization path on the same Torch build that the
        # Modal GPU driver stack supports instead of letting pip float to a
        # newer CUDA runtime and silently falling back to CPU.
        TORCH_PACKAGE,
        "git+https://github.com/huggingface/transformers.git",
        DATASETS_PACKAGE,
        ACCELERATE_PACKAGE,
        HF_HUB_PACKAGE,
        SAFETENSORS_PACKAGE,
        SENTENCEPIECE_PACKAGE,
        "auto-round>=0.10.2",
        "frozendict==2.4.0",
        "loguru>=0.7.2",
        "nvidia-ml-py>=12.560.30",
        "Pillow>=10.0.0",
        "pydantic>=2.0",
        "PyYAML>=6.0.1",
        "requests>=2.32.2",
        "tqdm>=4.66.3",
        EINOPS_PACKAGE,
        NINJA_PACKAGE,
    )
    .run_commands(
        # Qwen3.5 fast-path support is newer than the old pinned PyPI wheels we
        # were using, so track the upstream FLA package while keeping the base
        # layer in control of shared deps like Transformers and PyTorch.
        FLASH_LINEAR_ATTENTION_INSTALL,
        "python -m pip install --no-deps git+https://github.com/vllm-project/compressed-tensors.git",
        "python -m pip install --no-deps git+https://github.com/vllm-project/llm-compressor.git"
    )
    .add_local_python_source("ta_mpq")
)

# Keep the legacy "stable" quantization functions on the same working source
# stack so Modal does not need to build an incompatible llmcompressor+torch
# image that this project no longer relies on.
quant_image = quant_source_image

evalplus_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        TORCH_PACKAGE,
        DATASETS_PACKAGE,
        ACCELERATE_PACKAGE,
        HF_HUB_PACKAGE,
        SAFETENSORS_PACKAGE,
        SENTENCEPIECE_PACKAGE,
        EINOPS_PACKAGE,
        NINJA_PACKAGE,
        EVALPLUS_PACKAGE,
        "auto-round>=0.10.2",
        "frozendict==2.4.0",
        "loguru>=0.7.2",
        "nvidia-ml-py>=12.560.30",
        "Pillow>=10.0.0",
        "pydantic>=2.0",
        "PyYAML>=6.0.1",
        "requests>=2.32.2",
        "tqdm>=4.66.3",
    )
    .pip_install("git+https://github.com/huggingface/transformers.git")
    .run_commands(
        FLASH_LINEAR_ATTENTION_INSTALL,
        "python -m pip install --no-deps git+https://github.com/vllm-project/compressed-tensors.git",
        "python -m pip install --no-deps git+https://github.com/vllm-project/llm-compressor.git",
    )
    .add_local_python_source("ta_mpq")
)

bigcodebench_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        TORCH_PACKAGE,
        DATASETS_PACKAGE,
        ACCELERATE_PACKAGE,
        HF_HUB_PACKAGE,
        SAFETENSORS_PACKAGE,
        SENTENCEPIECE_PACKAGE,
        EINOPS_PACKAGE,
        NINJA_PACKAGE,
        "auto-round>=0.10.2",
        "frozendict==2.4.0",
        "loguru>=0.7.2",
        "nvidia-ml-py>=12.560.30",
        "Pillow>=10.0.0",
        "pydantic>=2.0",
        "PyYAML>=6.0.1",
        "requests>=2.32.2",
        "tqdm>=4.66.3",
        "appdirs>=1.4.4",
        "fire>=0.6.0",
        "pqdm>=0.2.0",
        "tempdir>=0.7.1",
        "termcolor>=2.0.0",
        "tree-sitter>=0.22.0",
        "tree-sitter-python>=0.21.0",
        "wget>=3.2",
        "gradio-client>=1.0.0",
        "e2b>=1.0.0",
    )
    .pip_install("git+https://github.com/huggingface/transformers.git")
    .run_commands(
        FLASH_LINEAR_ATTENTION_INSTALL,
        "python -m pip install --no-deps git+https://github.com/vllm-project/compressed-tensors.git",
        "python -m pip install --no-deps git+https://github.com/vllm-project/llm-compressor.git",
    )
    .run_commands(f"python -m pip install --no-deps {BIGCODEBENCH_PACKAGE}")
    .add_local_python_source("ta_mpq")
)

surrogate_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("xgboost>=2.1.0")
    .add_local_python_source("ta_mpq")
)


@app.function(
    image=quant_source_image,
    gpu=DEFAULT_MODAL_GPU,
    timeout=60 * 10,
    secrets=[hf_secret],
)
def probe_quant_source_runtime() -> str:
    import importlib
    import os
    import subprocess

    import torch

    report: dict[str, Any] = {
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }

    try:
        report["cuda_is_available"] = bool(torch.cuda.is_available())
        report["cuda_device_count"] = int(torch.cuda.device_count())
    except Exception as exc:
        report["cuda_probe_error_type"] = type(exc).__name__
        report["cuda_probe_error_message"] = str(exc)
        report["cuda_is_available"] = False
        report["cuda_device_count"] = 0
        return report

    if report["cuda_is_available"]:
        try:
            report["cuda_devices"] = [
                torch.cuda.get_device_name(index) for index in range(report["cuda_device_count"])
            ]
            report["current_device_index"] = int(torch.cuda.current_device())
        except Exception as exc:
            report["cuda_device_query_error_type"] = type(exc).__name__
            report["cuda_device_query_error_message"] = str(exc)

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
        )
        report["nvidia_smi_returncode"] = int(result.returncode)
        if result.stdout:
            report["nvidia_smi"] = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if result.stderr:
            report["nvidia_smi_stderr"] = result.stderr.strip()
    except Exception as exc:
        report["nvidia_smi_error_type"] = type(exc).__name__
        report["nvidia_smi_error_message"] = str(exc)

    optional_modules = ("fla", "causal_conv1d")
    report["optional_module_specs"] = {
        module_name: bool(importlib.machinery.PathFinder.find_spec(module_name))
        for module_name in optional_modules
    }
    report["optional_module_imports"] = {}
    for module_name in optional_modules:
        try:
            importlib.import_module(module_name)
            report["optional_module_imports"][module_name] = {"available": True}
        except Exception as exc:
            report["optional_module_imports"][module_name] = {
                "available": False,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }

    try:
        from transformers.utils import import_utils as transformers_import_utils

        transformers_flags: dict[str, Any] = {}
        for attribute_name in (
            "is_flash_linear_attention_available",
            "is_causal_conv1d_available",
        ):
            checker = getattr(transformers_import_utils, attribute_name, None)
            if checker is None:
                transformers_flags[attribute_name] = None
                continue
            try:
                transformers_flags[attribute_name] = bool(checker())
            except Exception as exc:
                transformers_flags[attribute_name] = {
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
        report["transformers_optional_kernels"] = transformers_flags
    except Exception as exc:
        report["transformers_optional_kernels_error_type"] = type(exc).__name__
        report["transformers_optional_kernels_error_message"] = str(exc)

    try:
        qwen3_5_module = importlib.import_module("transformers.models.qwen3_5.modeling_qwen3_5")
        report["qwen3_5_fast_path_available"] = getattr(
            qwen3_5_module,
            "is_fast_path_available",
            None,
        )
    except Exception as exc:
        report["qwen3_5_fast_path_error_type"] = type(exc).__name__
        report["qwen3_5_fast_path_error_message"] = str(exc)

    return json.dumps(report, sort_keys=True)


def _resolve_remote_model_ref(
    model_ref: str,
    *,
    reload_attempts: int = 30,
    sleep_seconds: float = 4.0,
) -> str:
    import time

    if not model_ref.startswith("/artifacts/"):
        return model_ref

    model_path = Path(model_ref)
    for _ in range(reload_attempts):
        try:
            artifact_volume.reload()
        except Exception:
            pass
        if model_path.exists():
            return str(model_path)
        time.sleep(sleep_seconds)

    raise FileNotFoundError(f"Artifact model path did not appear on the mounted volume: {model_ref}")


def _sync_artifact_volume() -> None:
    try:
        artifact_volume.commit()
    except Exception:
        pass


def _configure_hf_environment() -> None:
    import os

    os.environ["HF_HOME"] = "/cache/hf"
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    token = os.environ.get("HF_TOKEN")
    if token:
        os.environ.setdefault("HF_HUB_TOKEN", token)
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", token)


@app.function(
    image=report_image,
    gpu=DEFAULT_MODAL_GPU,
    timeout=60 * 60 * 4,
    volumes={"/cache": cache_volume, "/artifacts": artifact_volume},
    secrets=[hf_secret],
)
def probe_mixed_precision_report(
    contract_data: dict[str, Any],
    calibration_limit: int = 16,
    policy_payload: dict[str, Any] | None = None,
    policy_label: str = "default-policy",
    precomputed_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    _configure_hf_environment()

    from ta_mpq.contracts import ExperimentContract
    from ta_mpq.feasibility import maybe_run_llmcompressor_oneshot
    from ta_mpq.quantization import MixedPrecisionPolicy

    contract = ExperimentContract.from_dict(contract_data)
    policy = MixedPrecisionPolicy.from_dict(policy_payload) if policy_payload else None
    output_dir = f"/artifacts/{contract.name}-feasibility-{_slugify(policy_label)}"
    report = maybe_run_llmcompressor_oneshot(
        model_id=contract.compressed_source_model_id,
        output_dir=output_dir,
        policy=policy,
        calibration_limit=calibration_limit,
        dry_run=True,
        precomputed_report=precomputed_report,
    )
    report["contract_name"] = contract.name
    report["model_id"] = contract.compressed_source_model_id
    report["probe_mode"] = "dry_run_policy_report"
    report["policy_label"] = policy_label
    return report


@app.function(
    image=report_image,
    gpu=DEFAULT_MODAL_GPU,
    timeout=60 * 60 * 4,
    volumes={"/cache": cache_volume, "/artifacts": artifact_volume},
    secrets=[hf_secret],
)
def probe_live_policy_target_matching(
    contract_data: dict[str, Any],
    policy_payload: dict[str, Any],
    policy_label: str = "default-policy",
) -> dict[str, Any]:
    _configure_hf_environment()

    from ta_mpq.contracts import ExperimentContract
    from ta_mpq.feasibility import inspect_live_policy_target_matching
    from ta_mpq.quantization import MixedPrecisionPolicy

    contract = ExperimentContract.from_dict(contract_data)
    policy = MixedPrecisionPolicy.from_dict(policy_payload)
    report = inspect_live_policy_target_matching(
        model_id=contract.compressed_source_model_id,
        policy=policy,
    )
    report["contract_name"] = contract.name
    report["policy_label"] = policy_label
    report["probe_mode"] = "live_policy_target_matching"
    return report


def _probe_loaded_artifact_quantization_state_impl(
    model_ref: str,
    policy_payload: dict[str, Any] | None = None,
    policy_label: str = "default-policy",
) -> dict[str, Any]:
    _configure_hf_environment()

    _apply_transformers_token_compat_patch()
    _apply_compressed_tensors_distributed_compat_patch()

    from ta_mpq.feasibility import inspect_loaded_model_quantization_state
    from ta_mpq.quantization import MixedPrecisionPolicy

    policy = MixedPrecisionPolicy.from_dict(policy_payload) if policy_payload else None
    report = inspect_loaded_model_quantization_state(
        model_ref=model_ref,
        policy=policy,
    )
    report["policy_label"] = policy_label
    report["probe_mode"] = "loaded_model_quantization_state"
    return report


@app.function(
    image=quant_source_image,
    gpu=DEFAULT_MODAL_GPU,
    timeout=60 * 60 * 4,
    volumes={"/cache": cache_volume, "/artifacts": artifact_volume},
    secrets=[hf_secret],
)
def probe_loaded_artifact_quantization_state(
    model_ref: str,
    policy_payload: dict[str, Any] | None = None,
    policy_label: str = "default-policy",
) -> dict[str, Any]:
    return _probe_loaded_artifact_quantization_state_impl(
        model_ref=model_ref,
        policy_payload=policy_payload,
        policy_label=policy_label,
    )


@app.function(
    image=quant_source_image,
    gpu=A100_40GB_MODAL_GPU,
    timeout=60 * 60 * 4,
    volumes={"/cache": cache_volume, "/artifacts": artifact_volume},
    secrets=[hf_secret],
)
def probe_loaded_artifact_quantization_state_a100(
    model_ref: str,
    policy_payload: dict[str, Any] | None = None,
    policy_label: str = "default-policy",
) -> dict[str, Any]:
    return _probe_loaded_artifact_quantization_state_impl(
        model_ref=model_ref,
        policy_payload=policy_payload,
        policy_label=policy_label,
    )


@app.function(
    image=quant_image,
    gpu=DEFAULT_MODAL_GPU,
    timeout=60 * 60 * 8,
    volumes={"/cache": cache_volume, "/artifacts": artifact_volume},
    secrets=[hf_secret],
)
def probe_mixed_precision_feasibility(
    contract_data: dict[str, Any],
    calibration_limit: int = 16,
    dry_run: bool = True,
    policy_payload: dict[str, Any] | None = None,
    policy_label: str = "default-policy",
    precomputed_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    _configure_hf_environment()

    from ta_mpq.contracts import ExperimentContract
    from ta_mpq.feasibility import maybe_run_llmcompressor_oneshot
    from ta_mpq.quantization import MixedPrecisionPolicy

    contract = ExperimentContract.from_dict(contract_data)
    policy = MixedPrecisionPolicy.from_dict(policy_payload) if policy_payload else None
    output_dir = f"/artifacts/{contract.name}-feasibility-{_slugify(policy_label)}"
    report = maybe_run_llmcompressor_oneshot(
        model_id=contract.compressed_source_model_id,
        output_dir=output_dir,
        policy=policy,
        calibration_limit=calibration_limit,
        dry_run=dry_run,
        precomputed_report=precomputed_report,
        processor_strategy="tokenizer",
        oneshot_device="cuda:0",
    )
    if not dry_run and output_dir.startswith("/artifacts/"):
        _sync_artifact_volume()
    report["contract_name"] = contract.name
    report["model_id"] = contract.compressed_source_model_id
    report["policy_label"] = policy_label
    report["backend_variant"] = "stable"
    return report


def _probe_mixed_precision_feasibility_source_impl(
    contract_data: dict[str, Any],
    calibration_limit: int = 16,
    dry_run: bool = True,
    policy_payload: dict[str, Any] | None = None,
    policy_label: str = "default-policy",
    precomputed_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    _configure_hf_environment()

    _apply_transformers_token_compat_patch()
    _apply_compressed_tensors_distributed_compat_patch()

    from ta_mpq.contracts import ExperimentContract
    from ta_mpq.feasibility import maybe_run_llmcompressor_oneshot
    from ta_mpq.quantization import MixedPrecisionPolicy

    contract = ExperimentContract.from_dict(contract_data)
    policy = MixedPrecisionPolicy.from_dict(policy_payload) if policy_payload else None
    output_dir = f"/artifacts/{contract.name}-feasibility-{_slugify(policy_label)}-source"
    report = maybe_run_llmcompressor_oneshot(
        model_id=contract.compressed_source_model_id,
        output_dir=output_dir,
        policy=policy,
        calibration_limit=calibration_limit,
        dry_run=dry_run,
        precomputed_report=precomputed_report,
        processor_strategy="tokenizer",
        oneshot_device="cuda:0",
    )
    if not dry_run and output_dir.startswith("/artifacts/"):
        _sync_artifact_volume()
    report["contract_name"] = contract.name
    report["model_id"] = contract.compressed_source_model_id
    report["policy_label"] = policy_label
    report["backend_variant"] = "source"
    return report


@app.function(
    image=quant_source_image,
    gpu=DEFAULT_MODAL_GPU,
    timeout=60 * 60 * 8,
    volumes={"/cache": cache_volume, "/artifacts": artifact_volume},
    secrets=[hf_secret],
)
def probe_mixed_precision_feasibility_source(
    contract_data: dict[str, Any],
    calibration_limit: int = 16,
    dry_run: bool = True,
    policy_payload: dict[str, Any] | None = None,
    policy_label: str = "default-policy",
    precomputed_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return _probe_mixed_precision_feasibility_source_impl(
        contract_data=contract_data,
        calibration_limit=calibration_limit,
        dry_run=dry_run,
        policy_payload=policy_payload,
        policy_label=policy_label,
        precomputed_report=precomputed_report,
    )


@app.function(
    image=quant_source_image,
    gpu=A100_40GB_MODAL_GPU,
    timeout=60 * 60 * 8,
    volumes={"/cache": cache_volume, "/artifacts": artifact_volume},
    secrets=[hf_secret],
)
def probe_mixed_precision_feasibility_source_a100(
    contract_data: dict[str, Any],
    calibration_limit: int = 16,
    dry_run: bool = True,
    policy_payload: dict[str, Any] | None = None,
    policy_label: str = "default-policy",
    precomputed_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return _probe_mixed_precision_feasibility_source_impl(
        contract_data=contract_data,
        calibration_limit=calibration_limit,
        dry_run=dry_run,
        policy_payload=policy_payload,
        policy_label=policy_label,
        precomputed_report=precomputed_report,
    )


@app.function(
    image=quant_source_image,
    gpu=DEFAULT_MODAL_GPU,
    timeout=60 * 60 * 2,
    volumes={"/cache": cache_volume, "/artifacts": artifact_volume},
    secrets=[hf_secret],
)
def smoke_test_quantized_artifact_source(
    artifact_dir: str,
    tokenizer_source: str,
    max_new_tokens: int = 8,
) -> dict[str, Any]:
    _configure_hf_environment()

    _apply_transformers_token_compat_patch()
    _apply_compressed_tensors_distributed_compat_patch()

    from ta_mpq.feasibility import run_quantized_smoke_test

    report = run_quantized_smoke_test(
        model_path=artifact_dir,
        tokenizer_source=tokenizer_source,
        max_new_tokens=max_new_tokens,
    )
    report["artifact_dir"] = artifact_dir
    report["tokenizer_source"] = tokenizer_source
    report["backend_variant"] = "source"
    return report


def _evaluate_task_source_model_impl(
    model_ref: str,
    tokenizer_source: str,
    model_label: str,
    task_name: str,
    limit: int,
    max_new_tokens: int,
    split: str = "test",
    example_ids: list[str] | None = None,
    load_dtype: str = "auto",
    task_prompt_style: str = "",
) -> dict[str, Any]:
    _configure_hf_environment()

    _apply_transformers_token_compat_patch()
    _apply_compressed_tensors_distributed_compat_patch()

    from ta_mpq.baseline import evaluate_task_model

    resolved_model_ref = _resolve_remote_model_ref(model_ref)
    summary = evaluate_task_model(
        model_ref=resolved_model_ref,
        task_name=task_name,
        tokenizer_source=tokenizer_source,
        model_label=model_label,
        limit=limit,
        max_new_tokens=max_new_tokens,
        split=split,
        example_ids=example_ids,
        load_dtype=load_dtype,
        task_prompt_style=task_prompt_style,
    )
    summary["backend_variant"] = "source"
    return summary


@app.function(
    image=quant_source_image,
    gpu=DEFAULT_MODAL_GPU,
    timeout=60 * 60 * 4,
    volumes={"/cache": cache_volume, "/artifacts": artifact_volume},
    secrets=[hf_secret],
)
def evaluate_task_source_model(
    model_ref: str,
    tokenizer_source: str,
    model_label: str,
    task_name: str,
    limit: int,
    max_new_tokens: int,
    split: str = "test",
    example_ids: list[str] | None = None,
    load_dtype: str = "auto",
    task_prompt_style: str = "",
) -> dict[str, Any]:
    return _evaluate_task_source_model_impl(
        model_ref=model_ref,
        tokenizer_source=tokenizer_source,
        model_label=model_label,
        task_name=task_name,
        limit=limit,
        max_new_tokens=max_new_tokens,
        split=split,
        example_ids=example_ids,
        load_dtype=load_dtype,
        task_prompt_style=task_prompt_style,
    )


@app.function(
    image=quant_source_image,
    gpu=A100_40GB_MODAL_GPU,
    timeout=60 * 60 * 4,
    volumes={"/cache": cache_volume, "/artifacts": artifact_volume},
    secrets=[hf_secret],
)
def evaluate_task_source_model_a100(
    model_ref: str,
    tokenizer_source: str,
    model_label: str,
    task_name: str,
    limit: int,
    max_new_tokens: int,
    split: str = "test",
    example_ids: list[str] | None = None,
    load_dtype: str = "auto",
    task_prompt_style: str = "",
) -> dict[str, Any]:
    return _evaluate_task_source_model_impl(
        model_ref=model_ref,
        tokenizer_source=tokenizer_source,
        model_label=model_label,
        task_name=task_name,
        limit=limit,
        max_new_tokens=max_new_tokens,
        split=split,
        example_ids=example_ids,
        load_dtype=load_dtype,
        task_prompt_style=task_prompt_style,
    )


def _run_evalplus_hf_model_impl(
    model_ref: str,
    model_label: str,
    dataset: str,
    greedy: bool,
    n_samples: int,
    dtype: str,
    backend: str,
    force_base_prompt: bool,
    attn_implementation: str,
    version: str,
    parallel: int,
    max_new_tokens: int,
    task_limit: int,
    runner_mode: str,
    output_slug: str,
) -> dict[str, Any]:
    _configure_hf_environment()
    _apply_transformers_token_compat_patch()
    _apply_compressed_tensors_distributed_compat_patch()

    import contextlib
    import io
    import json
    import time

    from evalplus.evaluate import evaluate
    from evalplus.data.humaneval import get_human_eval_plus
    from evalplus.data.mbpp import get_mbpp_plus
    from evalplus.provider.utility import (
        extra_eos_for_direct_completion,
        make_raw_chat_prompt,
    )
    from evalplus.sanitize import sanitize
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    resolved_model_ref = _resolve_remote_model_ref(model_ref)
    started_at = time.time()
    eval_root = Path("/tmp") / "ta-mpq-evalplus" / output_slug
    eval_root.mkdir(parents=True, exist_ok=True)

    log_buffer = io.StringIO()
    if runner_mode not in {"direct_hf", "evalplus_hf"}:
        raise ValueError(f"Unknown EvalPlus runner_mode: {runner_mode}")

    if runner_mode == "evalplus_hf":
        with contextlib.redirect_stdout(log_buffer), contextlib.redirect_stderr(log_buffer):
            evaluate(
                dataset=dataset,
                model=resolved_model_ref,
                root=str(eval_root),
                backend=backend,
                greedy=greedy,
                n_samples=n_samples,
                bs=1 if greedy else None,
                temperature=0.0 if greedy else 0.2,
                resume=False,
                trust_remote_code=True,
                dtype=dtype,
                force_base_prompt=force_base_prompt,
                attn_implementation=attn_implementation,
                parallel=parallel,
                version=version,
            )
    else:
        dataset_dir = eval_root / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)
        identifier = resolved_model_ref.strip("./").replace("/", "--")
        identifier += f"_{backend}_direct_temp_{0.0 if greedy else 0.2}"
        samples_path = dataset_dir / f"{identifier}.jsonl"
        raw_samples_path = dataset_dir / f"{identifier}.raw.jsonl"
        samples_path.unlink(missing_ok=True)
        raw_samples_path.unlink(missing_ok=True)

        if dataset == "humaneval":
            problems = get_human_eval_plus(version=version)
        elif dataset == "mbpp":
            problems = get_mbpp_plus(version=version)
        else:
            raise ValueError(f"Unsupported EvalPlus dataset for direct_hf: {dataset}")
        items = list(problems.items())
        if task_limit > 0:
            items = items[:task_limit]

        direct_completion = force_base_prompt
        tokenizer = AutoTokenizer.from_pretrained(resolved_model_ref, use_fast=False)
        direct_completion = direct_completion or tokenizer.chat_template is None
        eos_strings = list(getattr(tokenizer, "additional_special_tokens", []) or [])
        if tokenizer.eos_token:
            eos_strings.append(tokenizer.eos_token)
        if direct_completion:
            eos_strings.extend(extra_eos_for_direct_completion(dataset))
        else:
            eos_strings.append("\n```\n")

        torch_dtype = getattr(torch, dtype)
        model_kwargs = {
            "device_map": "auto",
            "trust_remote_code": True,
            "torch_dtype": torch_dtype,
            "attn_implementation": attn_implementation,
        }
        with contextlib.redirect_stdout(log_buffer), contextlib.redirect_stderr(log_buffer):
            print(f"direct_hf model_kwargs={model_kwargs}")
            print(f"direct_hf task_count={len(items)} max_new_tokens={max_new_tokens}")
            model = AutoModelForCausalLM.from_pretrained(resolved_model_ref, **model_kwargs)
            model.eval()
            device = next(model.parameters()).device
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id

            instruction_prefix = (
                "Please provide a self-contained Python script that solves the following "
                "problem in a markdown code block:"
            )
            response_prefix = (
                "Below is a Python script with a self-contained function that solves the "
                "problem and passes corresponding tests:"
            )
            for task_index, (task_id, task) in enumerate(items, start=1):
                print(f"direct_hf codegen {task_index}/{len(items)} {task_id}", flush=True)
                prompt = task["prompt"].strip() + "\n"
                if direct_completion:
                    generation_prompt = prompt
                else:
                    generation_prompt = make_raw_chat_prompt(
                        prompt,
                        instruction_prefix,
                        response_prefix,
                        tokenizer,
                    )
                encoded = tokenizer(generation_prompt, return_tensors="pt").to(device)
                generation_kwargs: dict[str, Any] = {
                    "max_new_tokens": max_new_tokens,
                    "do_sample": not greedy,
                    "num_return_sequences": 1,
                    "pad_token_id": tokenizer.pad_token_id,
                }
                if not greedy:
                    generation_kwargs["temperature"] = 0.2
                    generation_kwargs["top_p"] = 0.95
                for _sample_index in range(n_samples):
                    output_ids = model.generate(**encoded, **generation_kwargs)
                    generated = tokenizer.decode(
                        output_ids[0, encoded["input_ids"].shape[-1] :],
                        skip_special_tokens=True,
                    )
                    cut_index = len(generated)
                    for eos in eos_strings:
                        if eos and eos in generated:
                            cut_index = min(cut_index, generated.index(eos))
                    implementation = generated[:cut_index].replace("\t", "    ")
                    solution = prompt + implementation if direct_completion else implementation
                    sanitized_solution = sanitize(solution, entrypoint=task["entry_point"])
                    with samples_path.open("a", encoding="utf-8") as handle:
                        handle.write(
                            json.dumps(
                                {"task_id": task_id, "solution": sanitized_solution},
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                    with raw_samples_path.open("a", encoding="utf-8") as handle:
                        handle.write(
                            json.dumps(
                                {"task_id": task_id, "solution": solution},
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
            del model
            torch.cuda.empty_cache()

        if task_limit > 0:
            samples_paths = [samples_path]
            raw_samples_paths = [raw_samples_path]
            summary = {
                "status": "generated_subset",
                "model_id": model_label or model_ref,
                "model_ref": model_ref,
                "resolved_model_ref": resolved_model_ref,
                "dataset": dataset,
                "num_tasks": len(items),
                "num_samples": len(items) * n_samples,
                "base_correct": None,
                "plus_correct": None,
                "base_pass_at_1": None,
                "plus_pass_at_1": None,
                "remote_result_path": None,
                "remote_samples_paths": [str(samples_path)],
                "remote_raw_samples_paths": [str(raw_samples_path)],
                "elapsed_sec": time.time() - started_at,
                "note": "task_limit was set, so this run only validates EvalPlus code generation.",
            }
            summary["evalplus_config"] = {
                "backend": backend,
                "greedy": greedy,
                "n_samples": n_samples,
                "dtype": dtype,
                "force_base_prompt": force_base_prompt,
                "attn_implementation": attn_implementation,
                "version": version,
                "parallel": parallel,
                "max_new_tokens": max_new_tokens,
                "task_limit": task_limit,
                "runner_mode": runner_mode,
            }
            summary["remote_log_tail"] = log_buffer.getvalue()[-12000:]
            summary["artifact_texts"] = {
                str(path.relative_to(eval_root)): path.read_text(encoding="utf-8")
                for path in [*samples_paths, *raw_samples_paths]
                if path.exists()
            }
            return summary

        with contextlib.redirect_stdout(log_buffer), contextlib.redirect_stderr(log_buffer):
            evaluate(
                dataset=dataset,
                samples=str(samples_path),
                parallel=parallel,
                version=version,
            )

    result_paths = sorted(eval_root.rglob("*_eval_results.json"))
    if not result_paths:
        raise FileNotFoundError(f"EvalPlus did not produce an eval_results file under {eval_root}")
    result_path = result_paths[-1]
    results = _load_json(result_path) if result_path.exists() else {}
    samples_paths = sorted(
        path for path in eval_root.rglob("*.jsonl") if not path.name.endswith(".raw.jsonl")
    )
    raw_samples_paths = sorted(eval_root.rglob("*.raw.jsonl"))

    summary = _summarize_evalplus_payload(
        results,
        dataset=dataset,
        model_ref=model_ref,
        model_label=model_label or model_ref,
        resolved_model_ref=resolved_model_ref,
        result_path=result_path,
        samples_paths=samples_paths,
        raw_samples_paths=raw_samples_paths,
        elapsed_sec=time.time() - started_at,
    )
    summary["evalplus_config"] = {
        "backend": backend,
        "greedy": greedy,
        "n_samples": n_samples,
        "dtype": dtype,
        "force_base_prompt": force_base_prompt,
        "attn_implementation": attn_implementation,
        "version": version,
        "parallel": parallel,
        "max_new_tokens": max_new_tokens,
        "task_limit": task_limit,
        "runner_mode": runner_mode,
    }
    summary["remote_log_tail"] = log_buffer.getvalue()[-12000:]
    summary["artifact_texts"] = {
        str(path.relative_to(eval_root)): path.read_text(encoding="utf-8")
        for path in [*samples_paths, *raw_samples_paths, result_path]
        if path.exists()
    }
    return summary


@app.function(
    image=evalplus_image,
    gpu=DEFAULT_MODAL_GPU,
    timeout=60 * 60 * 6,
    volumes={"/cache": cache_volume, "/artifacts": artifact_volume},
    secrets=[hf_secret],
)
def run_evalplus_hf_model(
    model_ref: str,
    model_label: str,
    dataset: str = "humaneval",
    greedy: bool = True,
    n_samples: int = 1,
    dtype: str = "bfloat16",
    backend: str = "hf",
    force_base_prompt: bool = False,
    attn_implementation: str = "eager",
    version: str = "default",
    parallel: int = 0,
    max_new_tokens: int = 768,
    task_limit: int = 0,
    runner_mode: str = "direct_hf",
    output_slug: str = "evalplus-run",
) -> dict[str, Any]:
    return _run_evalplus_hf_model_impl(
        model_ref=model_ref,
        model_label=model_label,
        dataset=dataset,
        greedy=greedy,
        n_samples=n_samples,
        dtype=dtype,
        backend=backend,
        force_base_prompt=force_base_prompt,
        attn_implementation=attn_implementation,
        version=version,
        parallel=parallel,
        max_new_tokens=max_new_tokens,
        task_limit=task_limit,
        runner_mode=runner_mode,
        output_slug=output_slug,
    )


@app.function(
    image=evalplus_image,
    gpu=A100_40GB_MODAL_GPU,
    timeout=60 * 60 * 6,
    volumes={"/cache": cache_volume, "/artifacts": artifact_volume},
    secrets=[hf_secret],
)
def run_evalplus_hf_model_a100(
    model_ref: str,
    model_label: str,
    dataset: str = "humaneval",
    greedy: bool = True,
    n_samples: int = 1,
    dtype: str = "bfloat16",
    backend: str = "hf",
    force_base_prompt: bool = False,
    attn_implementation: str = "eager",
    version: str = "default",
    parallel: int = 0,
    max_new_tokens: int = 768,
    task_limit: int = 0,
    runner_mode: str = "direct_hf",
    output_slug: str = "evalplus-run",
) -> dict[str, Any]:
    return _run_evalplus_hf_model_impl(
        model_ref=model_ref,
        model_label=model_label,
        dataset=dataset,
        greedy=greedy,
        n_samples=n_samples,
        dtype=dtype,
        backend=backend,
        force_base_prompt=force_base_prompt,
        attn_implementation=attn_implementation,
        version=version,
        parallel=parallel,
        max_new_tokens=max_new_tokens,
        task_limit=task_limit,
        runner_mode=runner_mode,
        output_slug=output_slug,
    )


def _run_bigcodebench_hf_model_impl(
    model_ref: str,
    model_label: str,
    split: str,
    subset: str,
    greedy: bool,
    n_samples: int,
    dtype: str,
    attn_implementation: str,
    parallel: int,
    max_new_tokens: int,
    task_limit: int,
    output_slug: str,
    no_gt: bool,
    evaluation_execution: str,
    generation_only: bool,
    checkpoint_interval: int,
) -> dict[str, Any]:
    _configure_hf_environment()
    _apply_transformers_token_compat_patch()
    _apply_compressed_tensors_distributed_compat_patch()

    import contextlib
    import io
    import json
    import time
    import traceback

    from bigcodebench.data import get_bigcodebench
    from bigcodebench.evaluate import evaluate
    from bigcodebench.provider.utility import EOS, make_raw_chat_prompt
    from bigcodebench.sanitize import sanitize
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    resolved_model_ref = _resolve_remote_model_ref(model_ref)
    started_at = time.time()
    eval_root = Path("/artifacts") / "bigcodebench" / output_slug
    eval_root.mkdir(parents=True, exist_ok=True)

    log_buffer = io.StringIO()
    benchmark_dir = eval_root / f"{split}-{subset}"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    identifier = resolved_model_ref.strip("./").replace("/", "--")
    identifier += f"_hf_direct_temp_{0.0 if greedy else 0.2}"
    samples_path = benchmark_dir / f"{identifier}.jsonl"
    raw_samples_path = benchmark_dir / f"{identifier}.raw.jsonl"
    generation_summary_path = eval_root / "generation_summary.json"
    result_path = Path(str(samples_path).replace(".jsonl", "_eval_results.json"))
    pass_at_k_path = Path(str(result_path).replace("eval_results.json", "pass_at_k.json"))
    for path in (samples_path, raw_samples_path, generation_summary_path, result_path, pass_at_k_path):
        path.unlink(missing_ok=True)

    with contextlib.redirect_stdout(log_buffer), contextlib.redirect_stderr(log_buffer):
        problems = get_bigcodebench(subset=subset)
    items = list(problems.items())
    if task_limit > 0:
        items = items[:task_limit]
    selected_task_ids = [task_id for task_id, _task in items]
    if task_limit > 0 and evaluation_execution == "gradio":
        selective_evaluate = ",".join(task_id.split("/")[-1] for task_id in selected_task_ids)
    else:
        selective_evaluate = ",".join(selected_task_ids) if task_limit > 0 else ""

    direct_completion = False
    tokenizer = AutoTokenizer.from_pretrained(resolved_model_ref, use_fast=False)
    direct_completion = direct_completion or tokenizer.chat_template is None
    eos_strings = list(EOS)
    if direct_completion:
        eos_strings.extend(["\ndef ", "\nclass ", "\nimport ", "\nfrom ", "\nassert "])
    else:
        eos_strings.append("\n```\n")
    if tokenizer.eos_token:
        eos_strings.append(tokenizer.eos_token)

    torch_dtype = getattr(torch, dtype)
    model_kwargs = {
        "device_map": "auto",
        "trust_remote_code": True,
        "torch_dtype": torch_dtype,
        "attn_implementation": attn_implementation,
    }
    generation_records: list[dict[str, Any]] = []

    def write_generation_checkpoint(checkpoint_label: str) -> None:
        completion_token_values = [
            int(record.get("completion_tokens", 0)) for record in generation_records
        ]
        generation_summary = {
            "model_id": model_label or model_ref,
            "model_ref": model_ref,
            "resolved_model_ref": resolved_model_ref,
            "benchmark": "bigcodebench",
            "split": split,
            "subset": subset,
            "checkpoint_label": checkpoint_label,
            "generation_complete": len(generation_records) == len(items),
            "num_generated_tasks": len(generation_records),
            "total_task_count": len(items),
            "selected_task_ids": selected_task_ids,
            "generated_task_ids": [
                str(record.get("task_id", "")) for record in generation_records
            ],
            "length_capped_count": sum(
                1 for record in generation_records if record.get("length_capped")
            ),
            "mean_completion_tokens": _mean(completion_token_values),
            "p95_completion_tokens": _percentile(completion_token_values, 95.0),
            "remote_samples_paths": [str(samples_path)],
            "remote_raw_samples_paths": [str(raw_samples_path)],
            "elapsed_generation_sec": time.time() - started_at,
        }
        generation_summary_path.write_text(
            json.dumps(generation_summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _sync_artifact_volume()

    with contextlib.redirect_stdout(log_buffer), contextlib.redirect_stderr(log_buffer):
        print(f"bigcodebench direct_hf model_kwargs={model_kwargs}")
        print(
            (
                "bigcodebench direct_hf "
                f"split={split} subset={subset} task_count={len(items)} "
                f"max_new_tokens={max_new_tokens}"
            ),
            flush=True,
        )
        model = AutoModelForCausalLM.from_pretrained(resolved_model_ref, **model_kwargs)
        model.eval()
        device = next(model.parameters()).device
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"

        instruction_prefix = (
            "Please provide a self-contained Python script that solves the following "
            "problem in a markdown code block:"
        )
        response_prefix = (
            "Below is a Python script with a self-contained function that solves the "
            "problem and passes corresponding tests:"
        )
        for task_index, (task_id, task) in enumerate(items, start=1):
            print(
                f"[bigcodebench:{split}/{subset}] codegen {task_index}/{len(items)} {task_id}",
                flush=True,
            )
            task_prompt = str(task[f"{split}_prompt"])
            if direct_completion:
                generation_prompt = task_prompt
            else:
                generation_prompt = make_raw_chat_prompt(
                    task_prompt=task_prompt,
                    subset=subset,
                    split=split,
                    instruction_prefix=instruction_prefix,
                    response_prefix=response_prefix,
                    tokenizer=tokenizer,
                    direct_completion=direct_completion,
                )
            encoded = tokenizer(generation_prompt, return_tensors="pt").to(device)
            generation_kwargs: dict[str, Any] = {
                "max_new_tokens": max_new_tokens,
                "do_sample": not greedy,
                "num_return_sequences": 1,
                "pad_token_id": tokenizer.pad_token_id,
            }
            if not greedy:
                generation_kwargs["temperature"] = 0.2
                generation_kwargs["top_p"] = 0.95
            try:
                generation_kwargs["stop_strings"] = eos_strings
                generation_kwargs["tokenizer"] = tokenizer
                start = time.perf_counter()
                output_ids = model.generate(**encoded, **generation_kwargs)
            except ValueError:
                generation_kwargs.pop("stop_strings", None)
                generation_kwargs.pop("tokenizer", None)
                start = time.perf_counter()
                output_ids = model.generate(**encoded, **generation_kwargs)
            latency_sec = time.perf_counter() - start
            generated_tokens = output_ids[0, encoded["input_ids"].shape[-1] :]
            generated = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            cut_index = len(generated)
            for eos in eos_strings:
                if eos and eos in generated:
                    cut_index = min(cut_index, generated.index(eos))
            implementation = generated[:cut_index].replace("\t", "    ")
            raw_solution = task_prompt + implementation if direct_completion else implementation
            sanitized_solution = sanitize(raw_solution, entrypoint=task["entry_point"])
            completion_tokens = int(generated_tokens.shape[-1])
            length_capped = completion_tokens >= max_new_tokens
            with samples_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {"task_id": task_id, "solution": sanitized_solution},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            with raw_samples_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "task_id": task_id,
                            "solution": raw_solution,
                            "completion_tokens": completion_tokens,
                            "latency_sec": latency_sec,
                            "length_capped": length_capped,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            generation_records.append(
                {
                    "task_id": task_id,
                    "latency_sec": latency_sec,
                    "prompt_tokens": int(encoded["input_ids"].shape[-1]),
                    "completion_tokens": completion_tokens,
                    "length_capped": length_capped,
                    "sanitized_chars": len(sanitized_solution),
                    "raw_chars": len(raw_solution),
                }
            )
            print(
                (
                    f"[bigcodebench:{split}/{subset}] completed {task_index}/{len(items)} "
                    f"last_latency_sec={latency_sec:.2f} "
                    f"last_completion_tokens={completion_tokens}"
                ),
                flush=True,
            )
            if checkpoint_interval > 0 and task_index % checkpoint_interval == 0:
                write_generation_checkpoint(f"task-{task_index}")
        del model
        torch.cuda.empty_cache()

        write_generation_checkpoint("final")

        evaluation_error: dict[str, Any] | None = None
        if generation_only or evaluation_execution == "none":
            print(
                "[bigcodebench] generation complete; skipping evaluator because "
                f"generation_only={generation_only} evaluation_execution={evaluation_execution}",
                flush=True,
            )
        else:
            try:
                evaluate(
                    split=split,
                    subset=subset,
                    samples=str(samples_path),
                    execution=evaluation_execution,
                    selective_evaluate=selective_evaluate,
                    pass_k="1",
                    save_pass_rate=True,
                    calibrated=True,
                    parallel=parallel,
                    no_gt=no_gt,
                )
            except Exception as exc:
                evaluation_error = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback_tail": traceback.format_exc()[-6000:],
                }
                print(
                    "[bigcodebench] evaluator failed after samples were saved: "
                    f"{evaluation_error['type']}: {evaluation_error['message']}",
                    flush=True,
                )
            _sync_artifact_volume()

    results = _load_json(result_path) if result_path.exists() else {}
    pass_at_k = _load_json(pass_at_k_path) if pass_at_k_path.exists() else {}
    summary = _summarize_bigcodebench_payload(
        results=results,
        pass_at_k=pass_at_k,
        generation_records=generation_records,
        split=split,
        subset=subset,
        model_ref=model_ref,
        model_label=model_label or model_ref,
        resolved_model_ref=resolved_model_ref,
        samples_path=samples_path,
        raw_samples_path=raw_samples_path,
        result_path=result_path,
        pass_at_k_path=pass_at_k_path,
        elapsed_sec=time.time() - started_at,
        no_gt=no_gt,
    )
    summary["evaluation_status"] = (
        "skipped"
        if generation_only or evaluation_execution == "none"
        else "failed"
        if evaluation_error
        else "completed"
    )
    if generation_only or evaluation_execution == "none" or evaluation_error:
        summary["num_tasks"] = len(generation_records)
        summary["pass_at_1"] = None
        summary["pass_correct"] = None
    if evaluation_error:
        summary["evaluation_error"] = evaluation_error
    summary["num_generated_tasks"] = len(generation_records)
    summary["remote_generation_summary_path"] = str(generation_summary_path)
    summary["bigcodebench_config"] = {
        "greedy": greedy,
        "n_samples": n_samples,
        "dtype": dtype,
        "attn_implementation": attn_implementation,
        "parallel": parallel,
        "max_new_tokens": max_new_tokens,
        "task_limit": task_limit,
        "split": split,
        "subset": subset,
        "no_gt": no_gt,
        "evaluation_execution": evaluation_execution,
        "generation_only": generation_only,
        "checkpoint_interval": checkpoint_interval,
    }
    summary["selected_task_ids"] = selected_task_ids
    summary["remote_log_tail"] = log_buffer.getvalue()[-12000:]
    summary["artifact_texts"] = {
        str(path.relative_to(eval_root)): path.read_text(encoding="utf-8")
        for path in [
            samples_path,
            raw_samples_path,
            generation_summary_path,
            result_path,
            pass_at_k_path,
        ]
        if path.exists()
    }
    return summary


@app.function(
    image=bigcodebench_image,
    gpu=DEFAULT_MODAL_GPU,
    timeout=60 * 60 * 8,
    volumes={"/cache": cache_volume, "/artifacts": artifact_volume},
    secrets=[hf_secret],
)
def run_bigcodebench_hf_model(
    model_ref: str,
    model_label: str,
    split: str = "instruct",
    subset: str = "hard",
    greedy: bool = True,
    n_samples: int = 1,
    dtype: str = "bfloat16",
    attn_implementation: str = "eager",
    parallel: int = 8,
    max_new_tokens: int = 1280,
    task_limit: int = 0,
    output_slug: str = "bigcodebench-run",
    no_gt: bool = False,
    evaluation_execution: str = "gradio",
    generation_only: bool = False,
    checkpoint_interval: int = 10,
) -> dict[str, Any]:
    return _run_bigcodebench_hf_model_impl(
        model_ref=model_ref,
        model_label=model_label,
        split=split,
        subset=subset,
        greedy=greedy,
        n_samples=n_samples,
        dtype=dtype,
        attn_implementation=attn_implementation,
        parallel=parallel,
        max_new_tokens=max_new_tokens,
        task_limit=task_limit,
        output_slug=output_slug,
        no_gt=no_gt,
        evaluation_execution=evaluation_execution,
        generation_only=generation_only,
        checkpoint_interval=checkpoint_interval,
    )


@app.function(
    image=bigcodebench_image,
    gpu=A100_40GB_MODAL_GPU,
    timeout=60 * 60 * 8,
    volumes={"/cache": cache_volume, "/artifacts": artifact_volume},
    secrets=[hf_secret],
)
def run_bigcodebench_hf_model_a100(
    model_ref: str,
    model_label: str,
    split: str = "instruct",
    subset: str = "hard",
    greedy: bool = True,
    n_samples: int = 1,
    dtype: str = "bfloat16",
    attn_implementation: str = "eager",
    parallel: int = 8,
    max_new_tokens: int = 1280,
    task_limit: int = 0,
    output_slug: str = "bigcodebench-run",
    no_gt: bool = False,
    evaluation_execution: str = "gradio",
    generation_only: bool = False,
    checkpoint_interval: int = 10,
) -> dict[str, Any]:
    return _run_bigcodebench_hf_model_impl(
        model_ref=model_ref,
        model_label=model_label,
        split=split,
        subset=subset,
        greedy=greedy,
        n_samples=n_samples,
        dtype=dtype,
        attn_implementation=attn_implementation,
        parallel=parallel,
        max_new_tokens=max_new_tokens,
        task_limit=task_limit,
        output_slug=output_slug,
        no_gt=no_gt,
        evaluation_execution=evaluation_execution,
        generation_only=generation_only,
        checkpoint_interval=checkpoint_interval,
    )


@app.function(
    image=bigcodebench_image,
    gpu=H100_MODAL_GPU,
    timeout=60 * 60 * 8,
    volumes={"/cache": cache_volume, "/artifacts": artifact_volume},
    secrets=[hf_secret],
)
def run_bigcodebench_hf_model_h100(
    model_ref: str,
    model_label: str,
    split: str = "instruct",
    subset: str = "hard",
    greedy: bool = True,
    n_samples: int = 1,
    dtype: str = "bfloat16",
    attn_implementation: str = "eager",
    parallel: int = 8,
    max_new_tokens: int = 1280,
    task_limit: int = 0,
    output_slug: str = "bigcodebench-run",
    no_gt: bool = False,
    evaluation_execution: str = "gradio",
    generation_only: bool = False,
    checkpoint_interval: int = 10,
) -> dict[str, Any]:
    return _run_bigcodebench_hf_model_impl(
        model_ref=model_ref,
        model_label=model_label,
        split=split,
        subset=subset,
        greedy=greedy,
        n_samples=n_samples,
        dtype=dtype,
        attn_implementation=attn_implementation,
        parallel=parallel,
        max_new_tokens=max_new_tokens,
        task_limit=task_limit,
        output_slug=output_slug,
        no_gt=no_gt,
        evaluation_execution=evaluation_execution,
        generation_only=generation_only,
        checkpoint_interval=checkpoint_interval,
    )


@app.function(
    image=bigcodebench_image,
    cpu=4,
    timeout=60 * 60 * 8,
    volumes={"/artifacts": artifact_volume},
)
def grade_bigcodebench_samples_remote(
    output_slug: str,
    split: str = "instruct",
    subset: str = "hard",
    samples_relative_path: str = "",
    raw_samples_relative_path: str = "",
    evaluation_execution: str = "gradio",
    parallel: int = 4,
    no_gt: bool = False,
    overwrite: bool = True,
) -> dict[str, Any]:
    import contextlib
    import io
    import json
    import time
    import traceback

    from bigcodebench.evaluate import evaluate

    started_at = time.time()
    try:
        artifact_volume.reload()
    except Exception:
        pass

    root = Path("/artifacts") / "bigcodebench" / _slugify(output_slug)
    if not root.exists():
        raise FileNotFoundError(f"BigCodeBench artifact directory does not exist: {root}")

    benchmark_dir = root / f"{split}-{subset}"
    if samples_relative_path:
        samples_path = root / samples_relative_path
    else:
        candidates = sorted(
            path
            for path in benchmark_dir.glob("*.jsonl")
            if not path.name.endswith(".raw.jsonl")
        )
        if not candidates:
            raise FileNotFoundError(f"No BigCodeBench samples .jsonl found under {benchmark_dir}")
        samples_path = candidates[0]
    if not samples_path.exists():
        raise FileNotFoundError(f"BigCodeBench samples file does not exist: {samples_path}")

    if raw_samples_relative_path:
        raw_samples_path = root / raw_samples_relative_path
    else:
        raw_candidate = Path(str(samples_path).replace(".jsonl", ".raw.jsonl"))
        raw_samples_path = raw_candidate if raw_candidate.exists() else samples_path

    result_path = Path(str(samples_path).replace(".jsonl", "_eval_results.json"))
    pass_at_k_path = Path(str(result_path).replace("eval_results.json", "pass_at_k.json"))
    if overwrite:
        result_path.unlink(missing_ok=True)
        pass_at_k_path.unlink(missing_ok=True)

    generation_summary_path = root / "generation_summary.json"
    generation_summary = _load_json(generation_summary_path) if generation_summary_path.exists() else {}
    generation_records: list[dict[str, Any]] = []
    if raw_samples_path.exists():
        with raw_samples_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                generation_records.append(
                    {
                        "task_id": item.get("task_id"),
                        "latency_sec": float(item.get("latency_sec", 0.0) or 0.0),
                        "completion_tokens": int(item.get("completion_tokens", 0) or 0),
                        "length_capped": bool(item.get("length_capped", False)),
                        "raw_chars": len(str(item.get("solution", ""))),
                        "sanitized_chars": 0,
                    }
                )

    log_buffer = io.StringIO()
    evaluation_error: dict[str, Any] | None = None
    with contextlib.redirect_stdout(log_buffer), contextlib.redirect_stderr(log_buffer):
        try:
            evaluate(
                split=split,
                subset=subset,
                samples=str(samples_path),
                execution=evaluation_execution,
                pass_k="1",
                save_pass_rate=True,
                calibrated=True,
                parallel=parallel,
                no_gt=no_gt,
            )
        except Exception as exc:
            evaluation_error = {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback_tail": traceback.format_exc()[-6000:],
            }
            print(
                "[bigcodebench] sample evaluator failed: "
                f"{evaluation_error['type']}: {evaluation_error['message']}",
                flush=True,
            )
    _sync_artifact_volume()

    results = _load_json(result_path) if result_path.exists() else {}
    pass_at_k = _load_json(pass_at_k_path) if pass_at_k_path.exists() else {}
    model_label = str(generation_summary.get("model_id") or output_slug)
    model_ref = str(generation_summary.get("model_ref") or output_slug)
    resolved_model_ref = str(generation_summary.get("resolved_model_ref") or model_ref)
    summary = _summarize_bigcodebench_payload(
        results=results,
        pass_at_k=pass_at_k,
        generation_records=generation_records,
        split=split,
        subset=subset,
        model_ref=model_ref,
        model_label=model_label,
        resolved_model_ref=resolved_model_ref,
        samples_path=samples_path,
        raw_samples_path=raw_samples_path,
        result_path=result_path,
        pass_at_k_path=pass_at_k_path,
        elapsed_sec=time.time() - started_at,
        no_gt=no_gt,
    )
    summary["evaluation_status"] = "failed" if evaluation_error else "completed"
    if evaluation_error:
        summary["evaluation_error"] = evaluation_error
    summary["bigcodebench_eval_config"] = {
        "split": split,
        "subset": subset,
        "evaluation_execution": evaluation_execution,
        "parallel": parallel,
        "no_gt": no_gt,
        "samples_relative_path": str(samples_path.relative_to(root)),
        "raw_samples_relative_path": str(raw_samples_path.relative_to(root))
        if raw_samples_path.exists()
        else None,
    }
    summary["remote_generation_summary_path"] = str(generation_summary_path)
    summary["remote_log_tail"] = log_buffer.getvalue()[-12000:]
    summary["artifact_texts"] = {
        str(path.relative_to(root)): path.read_text(encoding="utf-8")
        for path in [
            generation_summary_path,
            samples_path,
            raw_samples_path,
            result_path,
            pass_at_k_path,
        ]
        if path.exists()
    }
    return summary


@app.function(
    image=report_image,
    timeout=60 * 10,
    volumes={"/artifacts": artifact_volume},
)
def collect_bigcodebench_artifacts(output_slug: str) -> dict[str, Any]:
    try:
        artifact_volume.reload()
    except Exception:
        pass

    root = Path("/artifacts") / "bigcodebench" / _slugify(output_slug)
    if not root.exists():
        raise FileNotFoundError(f"BigCodeBench artifact directory does not exist: {root}")

    artifact_texts: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        artifact_texts[str(path.relative_to(root))] = path.read_text(encoding="utf-8")

    return {
        "remote_root": str(root),
        "output_slug": _slugify(output_slug),
        "artifact_texts": artifact_texts,
        "file_count": len(artifact_texts),
    }


@app.function(
    image=report_image,
    gpu=DEFAULT_MODAL_GPU,
    timeout=60 * 60 * 4,
    volumes={"/cache": cache_volume, "/artifacts": artifact_volume},
    secrets=[hf_secret],
)
def collect_task_sensitivity_profile_remote(
    contract_data: dict[str, Any],
    model_role: str = "compressed_source",
    task_name: str = "",
    limit: int = 8,
    grouping: str = "per_block_component",
    max_prompt_tokens: int = 1024,
    activation_weight: float = 0.55,
) -> dict[str, Any]:
    _configure_hf_environment()

    from ta_mpq.contracts import ExperimentContract
    from ta_mpq.feasibility import LinearLayerStat
    from ta_mpq.sensitivity import (
        build_task_sensitivity_profile,
        collect_task_activation_stats,
    )

    contract = ExperimentContract.from_dict(contract_data)
    resolved_task_name = task_name or contract.task_name
    model_id = contract.resolve_model_role(model_role)
    activation_stats = collect_task_activation_stats(
        model_id=model_id,
        task_name=resolved_task_name,
        limit=limit,
        max_prompt_tokens=max_prompt_tokens,
    )
    layer_stats = [
        LinearLayerStat(
            name=stat.name,
            parameter_count=stat.parameter_count,
            in_features=0,
            out_features=0,
        )
        for stat in activation_stats
    ]
    profile = build_task_sensitivity_profile(
        layer_stats=layer_stats,
        activation_stats=activation_stats,
        grouping=grouping,
        activation_weight=activation_weight,
    )
    profile["contract_name"] = contract.name
    profile["task_name"] = resolved_task_name
    profile["model_id"] = model_id
    profile["model_role"] = model_role
    profile["num_examples"] = limit
    profile["max_prompt_tokens"] = max_prompt_tokens
    return profile


@app.function(
    image=report_image,
    timeout=60 * 10,
    volumes={"/cache": cache_volume},
    secrets=[hf_secret],
)
def load_task_example_ids_remote(
    task_name: str,
    split: str,
) -> list[str]:
    _configure_hf_environment()

    from ta_mpq.tasks import load_task_adapter

    task = load_task_adapter(task_name)
    examples = task.load_examples(limit=None, split=split)
    return [str(example.example_id) for example in examples]


def _collect_taq_kl_lite_profile_remote_impl(
    model_id: str,
    task_name: str,
    split: str,
    num_prompts: int,
    grouping: str = "per_block_component",
    temperature: float = 1.0,
    seed: int = 42,
    noise_bits: tuple[int, ...] = (2, 4, 8),
    max_prompt_tokens: int = 1024,
    task_prompt_style: str = "simple_evals_nonthinking",
    batch_size: int = 8,
) -> dict[str, Any]:
    _configure_hf_environment()

    from ta_mpq.quant_search.group_registry import build_group_registry_from_model
    from ta_mpq.quant_search.sensitivity import profile_taq_kl_lite

    groups = build_group_registry_from_model(model_id=model_id, grouping=grouping)
    payload = profile_taq_kl_lite(
        model_id=model_id,
        task_name=task_name,
        groups=groups,
        split=split,
        num_prompts=num_prompts,
        output_path=None,
        temperature=temperature,
        seed=seed,
        noise_bits=tuple(int(bit) for bit in noise_bits),
        max_prompt_tokens=max_prompt_tokens,
        task_prompt_style=task_prompt_style,
        resume=False,
        batch_size=batch_size,
    )
    return {
        "group_registry": [group.to_dict() for group in groups],
        "sensitivity_profile": payload,
    }


def _collect_paper_task_sensitivity_profile_remote_impl(
    model_id: str,
    task_name: str,
    split: str,
    num_prompts: int,
    grouping: str = "per_block_component",
    activation_weight: float = 0.55,
    max_prompt_tokens: int = 1024,
    task_prompt_style: str = "simple_evals_nonthinking",
) -> dict[str, Any]:
    _configure_hf_environment()

    from ta_mpq.quant_search.group_registry import build_group_registry_from_model
    from ta_mpq.quant_search.sensitivity import profile_paper_task_sensitivity

    groups = build_group_registry_from_model(model_id=model_id, grouping=grouping)
    payload = profile_paper_task_sensitivity(
        model_id=model_id,
        task_name=task_name,
        groups=groups,
        split=split,
        num_prompts=num_prompts,
        output_path=None,
        activation_weight=activation_weight,
        max_prompt_tokens=max_prompt_tokens,
        task_prompt_style=task_prompt_style,
    )
    return {
        "group_registry": [group.to_dict() for group in groups],
        "sensitivity_profile": payload,
    }


@app.function(
    image=quant_source_image,
    gpu=DEFAULT_MODAL_GPU,
    timeout=60 * 60 * 8,
    volumes={"/cache": cache_volume, "/artifacts": artifact_volume},
    secrets=[hf_secret],
)
def collect_taq_kl_lite_profile_remote(
    model_id: str,
    task_name: str,
    split: str,
    num_prompts: int,
    grouping: str = "per_block_component",
    temperature: float = 1.0,
    seed: int = 42,
    noise_bits: tuple[int, ...] = (2, 4, 8),
    max_prompt_tokens: int = 1024,
    task_prompt_style: str = "simple_evals_nonthinking",
    batch_size: int = 8,
) -> dict[str, Any]:
    return _collect_taq_kl_lite_profile_remote_impl(
        model_id=model_id,
        task_name=task_name,
        split=split,
        num_prompts=num_prompts,
        grouping=grouping,
        temperature=temperature,
        seed=seed,
        noise_bits=noise_bits,
        max_prompt_tokens=max_prompt_tokens,
        task_prompt_style=task_prompt_style,
        batch_size=batch_size,
    )


@app.function(
    image=quant_source_image,
    gpu=A100_40GB_MODAL_GPU,
    timeout=60 * 60 * 8,
    volumes={"/cache": cache_volume, "/artifacts": artifact_volume},
    secrets=[hf_secret],
)
def collect_taq_kl_lite_profile_remote_a100(
    model_id: str,
    task_name: str,
    split: str,
    num_prompts: int,
    grouping: str = "per_block_component",
    temperature: float = 1.0,
    seed: int = 42,
    noise_bits: tuple[int, ...] = (2, 4, 8),
    max_prompt_tokens: int = 1024,
    task_prompt_style: str = "simple_evals_nonthinking",
    batch_size: int = 8,
) -> dict[str, Any]:
    return _collect_taq_kl_lite_profile_remote_impl(
        model_id=model_id,
        task_name=task_name,
        split=split,
        num_prompts=num_prompts,
        grouping=grouping,
        temperature=temperature,
        seed=seed,
        noise_bits=noise_bits,
        max_prompt_tokens=max_prompt_tokens,
        task_prompt_style=task_prompt_style,
        batch_size=batch_size,
    )


@app.function(
    image=quant_source_image,
    gpu=DEFAULT_MODAL_GPU,
    timeout=60 * 60 * 8,
    volumes={"/cache": cache_volume, "/artifacts": artifact_volume},
    secrets=[hf_secret],
)
def collect_paper_task_sensitivity_profile_remote(
    model_id: str,
    task_name: str,
    split: str,
    num_prompts: int,
    grouping: str = "per_block_component",
    activation_weight: float = 0.55,
    max_prompt_tokens: int = 1024,
    task_prompt_style: str = "simple_evals_nonthinking",
) -> dict[str, Any]:
    return _collect_paper_task_sensitivity_profile_remote_impl(
        model_id=model_id,
        task_name=task_name,
        split=split,
        num_prompts=num_prompts,
        grouping=grouping,
        activation_weight=activation_weight,
        max_prompt_tokens=max_prompt_tokens,
        task_prompt_style=task_prompt_style,
    )


@app.function(
    image=quant_source_image,
    gpu=A100_40GB_MODAL_GPU,
    timeout=60 * 60 * 8,
    volumes={"/cache": cache_volume, "/artifacts": artifact_volume},
    secrets=[hf_secret],
)
def collect_paper_task_sensitivity_profile_remote_a100(
    model_id: str,
    task_name: str,
    split: str,
    num_prompts: int,
    grouping: str = "per_block_component",
    activation_weight: float = 0.55,
    max_prompt_tokens: int = 1024,
    task_prompt_style: str = "simple_evals_nonthinking",
) -> dict[str, Any]:
    return _collect_paper_task_sensitivity_profile_remote_impl(
        model_id=model_id,
        task_name=task_name,
        split=split,
        num_prompts=num_prompts,
        grouping=grouping,
        activation_weight=activation_weight,
        max_prompt_tokens=max_prompt_tokens,
        task_prompt_style=task_prompt_style,
    )


@app.function(
    image=surrogate_image,
    cpu=4,
    timeout=60 * 30,
)
def train_surrogate_remote(
    dataset_payload: dict[str, Any],
    target_metric: str = "accuracy",
    random_seed: int = 0,
    ensemble_size: int = 8,
) -> dict[str, Any]:
    from ta_mpq.surrogate import train_surrogate_model

    return train_surrogate_model(
        dataset_payload=dataset_payload,
        target_metric=target_metric,
        random_seed=random_seed,
        ensemble_size=ensemble_size,
    )


@app.function(
    image=surrogate_image,
    cpu=4,
    timeout=60 * 30,
)
def run_surrogate_search_remote(
    report_payload: dict[str, Any],
    surrogate_summary_payload: dict[str, Any],
    surrogate_model_json: str,
    target_budget_gb: float,
    allowed_bits: tuple[int, ...],
    grouping: str = "per_block_component",
    group_value_prior_payload: dict[str, Any] | None = None,
    sensitivity_profile_payload: dict[str, Any] | None = None,
    sensitivity_field: str = "combined_sensitivity",
    population_size: int = 32,
    generations: int = 15,
    elite_count: int = 4,
    tournament_size: int = 3,
    mutation_rate: float = 0.1,
    uncertainty_penalty: float = 0.5,
    reference_accuracy: float | None = None,
    top_k: int = 5,
    seed: int = 0,
    search_group_names: list[str] | None = None,
    fixed_assignments: dict[str, int] | None = None,
    extra_seed_assignments: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    from ta_mpq.search import (
        build_search_groups,
        layer_stats_from_report,
        resolve_group_value_scores,
        resolve_sensitivity_overrides,
        run_surrogate_evolution_search,
    )

    layer_stats = layer_stats_from_report(report_payload)
    sensitivity_overrides = resolve_sensitivity_overrides(
        layer_stats=layer_stats,
        grouping=grouping,
        sensitivity_profile_payload=sensitivity_profile_payload,
        field=sensitivity_field,
    )
    groups = build_search_groups(
        layer_stats,
        grouping=grouping,
        sensitivity_overrides=sensitivity_overrides,
    )
    group_value_scores = resolve_group_value_scores(
        groups,
        group_value_prior_payload,
        layer_stats=layer_stats,
        target_grouping=grouping,
    )
    search_groups = None
    if search_group_names:
        requested_names = {str(name) for name in search_group_names}
        search_groups = [group for group in groups if group.name in requested_names]
        missing_group_names = sorted(requested_names - {group.name for group in search_groups})
        if missing_group_names:
            raise ValueError(f"Unknown search groups: {missing_group_names[:5]}")
    normalized_extra_seeds = None
    if extra_seed_assignments:
        normalized_extra_seeds = []
        for item in extra_seed_assignments:
            normalized_extra_seeds.append(
                {
                    "provenance": str(item.get("provenance", "extra_seed")),
                    "assignments": {
                        str(group_name): int(bit_width)
                        for group_name, bit_width in dict(item.get("assignments", {})).items()
                    },
                }
            )
    result = run_surrogate_evolution_search(
        groups=groups,
        report_payload=report_payload,
        surrogate_summary_payload=surrogate_summary_payload,
        surrogate_model_json=surrogate_model_json,
        target_budget_gb=target_budget_gb,
        allowed_bits=allowed_bits,
        grouping=grouping,
        population_size=population_size,
        generations=generations,
        elite_count=elite_count,
        tournament_size=tournament_size,
        mutation_rate=mutation_rate,
        uncertainty_penalty=uncertainty_penalty,
        group_value_scores=group_value_scores,
        sensitivity_profile_payload=sensitivity_profile_payload,
        reference_accuracy=reference_accuracy,
        top_k=top_k,
        seed=seed,
        search_groups=search_groups,
        fixed_assignments=fixed_assignments,
        extra_seed_assignments=[
            (item["provenance"], item["assignments"])
            for item in normalized_extra_seeds or []
        ],
    )
    return result.to_dict()


@app.local_entrypoint()
def run_quant_source_runtime_probe(
    output_name: str = "quant-source-runtime-probe",
) -> None:
    report = json.loads(probe_quant_source_runtime.remote())
    save_summary(
        PROJECT_ROOT / "outputs" / "feasibility" / f"{_slugify(output_name)}.json",
        report,
    )


@app.local_entrypoint()
def run_feasibility_probe(
    contract_path: str = "configs/experiment_contract.json",
    calibration_limit: int = 16,
    dry_run: bool = True,
    candidate_path: str = "",
    report_path: str = "",
    policy_source: str = "llmcompressor",
    backend_variant: str = "stable",
) -> None:
    contract = load_contract(PROJECT_ROOT / contract_path)
    policy_payload: dict[str, Any] | None = None
    policy_label = "default-policy"
    if candidate_path:
        candidate_payload = _load_json(PROJECT_ROOT / candidate_path)
        if policy_source == "project":
            policy_payload = candidate_payload["project_policy"]
        elif policy_source == "llmcompressor":
            policy_payload = candidate_payload["backend_projections"]["llmcompressor"][
                "projected_policy"
            ]
        else:
            raise ValueError(f"Unknown policy_source: {policy_source}")
        policy_label = _candidate_policy_label(candidate_path, policy_source)

    precomputed_report = _load_json(PROJECT_ROOT / report_path) if report_path else None

    if dry_run:
        report = probe_mixed_precision_report.remote(
            contract.to_dict(),
            calibration_limit=calibration_limit,
            policy_payload=policy_payload,
            policy_label=policy_label,
            precomputed_report=precomputed_report,
        )
    else:
        if backend_variant == "stable":
            report = probe_mixed_precision_feasibility.remote(
                contract.to_dict(),
                calibration_limit=calibration_limit,
                dry_run=dry_run,
                policy_payload=policy_payload,
                policy_label=policy_label,
                precomputed_report=precomputed_report,
            )
        elif backend_variant == "source":
            report = probe_mixed_precision_feasibility_source.remote(
                contract.to_dict(),
                calibration_limit=calibration_limit,
                dry_run=dry_run,
                policy_payload=policy_payload,
                policy_label=policy_label,
                precomputed_report=precomputed_report,
            )
        else:
            raise ValueError(f"Unknown backend_variant: {backend_variant}")

    if (
        not dry_run
        and backend_variant == "source"
        and report.get("status") == "smoke_test_failed"
        and report.get("artifact_dir_exists")
        and report.get("smoke_test_error_type") == "OutOfMemoryError"
    ):
        clean_smoke_test = smoke_test_quantized_artifact_source.remote(
            artifact_dir=report["output_dir"],
            tokenizer_source=contract.compressed_source_model_id,
            max_new_tokens=4,
        )
        report["same_process_smoke_test"] = {
            "status": "failed",
            "error_type": report.get("smoke_test_error_type"),
            "error_message": report.get("smoke_test_error_message"),
        }
        report["smoke_test"] = clean_smoke_test
        report["smoke_test_mode"] = "clean_process_retry"
        report["quantized_model_runnable"] = True
        report["status"] = "smoke_test_succeeded"

    report["candidate_path"] = candidate_path or None
    report["policy_source"] = policy_source
    report["report_path"] = report_path or None
    report["calibration_limit"] = calibration_limit
    report["contract_path"] = contract_path

    save_summary(
        PROJECT_ROOT
        / "outputs"
        / "feasibility"
        / f"{contract.name}-{_slugify(policy_label)}-{backend_variant}-report.json",
        report,
    )


@app.local_entrypoint()
def run_policy_target_probe(
    candidate_path: str,
    contract_path: str = "configs/experiment_contract_27b_9b.json",
    policy_source: str = "llmcompressor",
    output_name: str = "",
) -> None:
    contract = load_contract(PROJECT_ROOT / contract_path)
    candidate_payload = _load_json(PROJECT_ROOT / candidate_path)
    if policy_source == "project":
        policy_payload = candidate_payload["project_policy"]
    elif policy_source == "llmcompressor":
        policy_payload = candidate_payload["backend_projections"]["llmcompressor"][
            "projected_policy"
        ]
    else:
        raise ValueError(f"Unknown policy_source: {policy_source}")

    policy_label = _candidate_policy_label(candidate_path, policy_source)
    report = probe_live_policy_target_matching.remote(
        contract.to_dict(),
        policy_payload=policy_payload,
        policy_label=policy_label,
    )
    report["candidate_path"] = candidate_path
    report["contract_path"] = contract_path
    report["policy_source"] = policy_source

    effective_output_name = output_name or (
        f"{contract.name}-{Path(candidate_path).stem}-{policy_source}-target-matching"
    )
    save_summary(
        PROJECT_ROOT / "outputs" / "feasibility" / f"{_slugify(effective_output_name)}.json",
        report,
    )


@app.local_entrypoint()
def run_loaded_artifact_probe(
    model_ref: str,
    candidate_path: str = "",
    policy_source: str = "llmcompressor",
    output_name: str = "",
) -> None:
    policy_payload: dict[str, Any] | None = None
    policy_label = "no-policy"
    if candidate_path:
        candidate_payload = _load_json(PROJECT_ROOT / candidate_path)
        if policy_source == "project":
            policy_payload = candidate_payload["project_policy"]
        elif policy_source == "llmcompressor":
            policy_payload = candidate_payload["backend_projections"]["llmcompressor"][
                "projected_policy"
            ]
        else:
            raise ValueError(f"Unknown policy_source: {policy_source}")
        policy_label = _candidate_policy_label(candidate_path, policy_source)

    report = probe_loaded_artifact_quantization_state.remote(
        model_ref=model_ref,
        policy_payload=policy_payload,
        policy_label=policy_label,
    )
    report["candidate_path"] = candidate_path or None
    report["policy_source"] = policy_source if candidate_path else None
    effective_output_name = output_name or f"{_slugify(Path(model_ref).name)}-loaded-artifact-probe"
    save_summary(
        PROJECT_ROOT / "outputs" / "feasibility" / f"{_slugify(effective_output_name)}.json",
        report,
    )


@app.local_entrypoint()
def run_saved_smoke_test(
    artifact_dir: str,
    tokenizer_source: str,
    output_name: str = "saved-artifact-smoke-test",
    max_new_tokens: int = 8,
) -> None:
    report = smoke_test_quantized_artifact_source.remote(
        artifact_dir=artifact_dir,
        tokenizer_source=tokenizer_source,
        max_new_tokens=max_new_tokens,
    )
    save_summary(
        PROJECT_ROOT / "outputs" / "feasibility" / f"{_slugify(output_name)}.json",
        report,
    )


@app.local_entrypoint()
def run_quantized_vs_native_eval(
    artifact_dir: str,
    contract_path: str = "configs/experiment_contract_27b_9b.json",
    task_name: str = "",
    limit: int = 5,
    max_new_tokens: int = 0,
    split: str = "test",
    output_prefix: str = "candidate-01-quantized-vs-native-9b",
) -> None:
    contract = load_contract(PROJECT_ROOT / contract_path)
    resolved_task_name = task_name or contract.task_name
    resolved_max_new_tokens = _resolve_default_max_new_tokens(resolved_task_name, max_new_tokens)

    quantized_summary = evaluate_task_source_model.remote(
        model_ref=artifact_dir,
        tokenizer_source=contract.compressed_source_model_id,
        model_label=f"{contract.compressed_source_model_id}-quantized",
        task_name=resolved_task_name,
        limit=limit,
        max_new_tokens=resolved_max_new_tokens,
        split=split,
        load_dtype="auto",
    )
    native_summary = evaluate_task_source_model.remote(
        model_ref=contract.native_baseline_model_id,
        tokenizer_source=contract.native_baseline_model_id,
        model_label=contract.native_baseline_model_id,
        task_name=resolved_task_name,
        limit=limit,
        max_new_tokens=resolved_max_new_tokens,
        split=split,
        load_dtype="bfloat16",
    )

    save_summary(
        PROJECT_ROOT / "outputs" / "evaluations" / f"{_slugify(output_prefix)}-quantized.json",
        quantized_summary,
    )
    save_summary(
        PROJECT_ROOT / "outputs" / "evaluations" / f"{_slugify(output_prefix)}-native.json",
        native_summary,
    )


@app.local_entrypoint()
def run_build_no_surrogate_local_search_round(
    report_path: str,
    base_candidate_paths: str,
    contract_path: str = "configs/experiment_contract_27b_9b_math500.json",
    target_budget_gb: float = 0.0,
    allowed_bits: str = "4,8,16",
    group_value_prior_path: str = "",
    ablation_profile_paths: str = "",
    beam_size: int = 3,
    max_candidates: int = 8,
    output_name: str = "",
    output_dir: str = "",
) -> None:
    from ta_mpq.local_search import build_no_surrogate_local_search_round

    contract = load_contract(PROJECT_ROOT / contract_path)
    resolved_target_budget = (
        float(target_budget_gb)
        if target_budget_gb > 0
        else float(contract.search_target_budget_gb or 0.0)
    )
    if resolved_target_budget <= 0.0:
        raise ValueError("target_budget_gb must be positive or present in the contract")

    report_payload = _load_json(PROJECT_ROOT / report_path)
    report_payload["report_path"] = report_path
    candidate_paths = [
        item.strip()
        for item in base_candidate_paths.replace("\n", ",").split(",")
        if item.strip()
    ]
    if not candidate_paths:
        raise ValueError("base_candidate_paths must include at least one candidate path")
    candidate_payloads = [_load_json(PROJECT_ROOT / candidate_path) for candidate_path in candidate_paths]
    normalized_allowed_bits = tuple(
        sorted({int(part.strip()) for part in allowed_bits.split(",") if part.strip()})
    )
    group_value_prior_payload = (
        _load_json(PROJECT_ROOT / group_value_prior_path) if group_value_prior_path else None
    )
    profile_paths = [
        item.strip()
        for item in ablation_profile_paths.replace("\n", ",").split(",")
        if item.strip()
    ]
    ablation_profile_payloads = [_load_json(PROJECT_ROOT / path) for path in profile_paths]

    effective_output_name = output_name or (
        f"{Path(candidate_paths[0]).stem}-no-surrogate-local-search"
    )
    effective_output_dir = output_dir or (
        PROJECT_ROOT / "outputs" / "policies" / _slugify(effective_output_name)
    )
    manifest = build_no_surrogate_local_search_round(
        report_payload=report_payload,
        base_candidate_payloads=candidate_payloads,
        base_candidate_paths=candidate_paths,
        target_budget_gb=resolved_target_budget,
        output_dir=effective_output_dir,
        allowed_bits=normalized_allowed_bits,
        beam_size=beam_size,
        max_candidates=max_candidates,
        group_value_prior_payload=group_value_prior_payload,
        group_value_prior_path=group_value_prior_path or None,
        ablation_profile_payloads=ablation_profile_payloads,
        ablation_profile_paths=profile_paths,
    )
    manifest["contract_path"] = contract_path
    manifest["report_path"] = report_path
    save_summary(
        PROJECT_ROOT / "outputs" / "search" / f"{_slugify(effective_output_name)}.json",
        manifest,
    )


@app.local_entrypoint()
def run_named_task_model_eval(
    model_ref: str,
    tokenizer_source: str = "",
    model_label: str = "",
    task_name: str = "gsm8k",
    limit: int = 25,
    max_new_tokens: int = 0,
    split: str = "test",
    load_dtype: str = "bfloat16",
    task_prompt_style: str = "",
    gpu_type: str = DEFAULT_MODAL_GPU,
    output_name: str = "",
) -> None:
    resolved_tokenizer_source = tokenizer_source or model_ref
    resolved_model_label = model_label or model_ref
    resolved_max_new_tokens = _resolve_default_max_new_tokens(task_name, max_new_tokens)
    eval_remote = _resolve_eval_remote(gpu_type)
    summary = eval_remote.remote(
        model_ref=model_ref,
        tokenizer_source=resolved_tokenizer_source,
        model_label=resolved_model_label,
        task_name=task_name,
        limit=limit,
        max_new_tokens=resolved_max_new_tokens,
        split=split,
        load_dtype=load_dtype,
        task_prompt_style=task_prompt_style,
    )
    effective_output_name = output_name or (
        f"{task_name}-{Path(model_ref).name}-native-eval"
    )
    save_summary(
        PROJECT_ROOT / "outputs" / "evaluations" / f"{_slugify(effective_output_name)}.json",
        summary,
    )


@app.local_entrypoint()
def run_evaluate_candidate_manifest(
    candidate_manifest_path: str,
    contract_path: str = "configs/experiment_contract_27b_9b_math500.json",
    task_name: str = "",
    limit: int = 25,
    max_new_tokens: int = 32,
    calibration_limit: int = 4,
    policy_source: str = "llmcompressor",
    backend_variant: str = "source",
    output_name: str = "",
) -> None:
    from ta_mpq.local_search import select_best_candidate_from_evaluation_manifest

    contract = load_contract(PROJECT_ROOT / contract_path)
    resolved_task_name = task_name or contract.task_name
    manifest_payload = _load_json(PROJECT_ROOT / candidate_manifest_path)
    manifest_label = _manifest_output_label(PROJECT_ROOT / candidate_manifest_path)
    report_payload_path = manifest_payload.get("report_path")

    candidate_records: list[dict[str, Any]] = []
    for entry in manifest_payload.get("candidates", []):
        candidate_path = str(entry["path"])
        artifact_dir, report_output_path = _resolve_candidate_artifact_dir(
            contract=contract,
            candidate_path=candidate_path,
            policy_source=policy_source,
            backend_variant=backend_variant,
            calibration_limit=calibration_limit,
            report_payload_path=report_payload_path,
        )
        evaluation_summary = evaluate_task_source_model.remote(
            model_ref=artifact_dir,
            tokenizer_source=contract.compressed_source_model_id,
            model_label=f"{contract.compressed_source_model_id}-{Path(candidate_path).stem}",
            task_name=resolved_task_name,
            limit=limit,
            max_new_tokens=max_new_tokens,
            load_dtype="auto",
        )
        evaluation_output_path = (
            PROJECT_ROOT
            / "outputs"
            / "evaluations"
            / f"{_slugify(manifest_label)}-{Path(candidate_path).stem}.json"
        )
        save_summary(evaluation_output_path, evaluation_summary)
        candidate_records.append(
            {
                **entry,
                "candidate_path": candidate_path,
                "report_path": to_relative_path(Path(report_output_path), PROJECT_ROOT),
                "evaluation_path": to_relative_path(evaluation_output_path, PROJECT_ROOT),
                "accuracy": float(evaluation_summary.get("accuracy", 0.0)),
                "num_correct": int(evaluation_summary.get("num_correct", 0)),
            }
        )

    evaluation_manifest = {
        "candidate_manifest_path": candidate_manifest_path,
        "contract_path": contract_path,
        "task_name": resolved_task_name,
        "limit": limit,
        "max_new_tokens": max_new_tokens,
        "policy_source": policy_source,
        "backend_variant": backend_variant,
        "candidates": candidate_records,
    }
    best_candidate = select_best_candidate_from_evaluation_manifest(evaluation_manifest)
    evaluation_manifest["best_candidate"] = best_candidate
    effective_output_name = output_name or f"{manifest_label}-eval"
    save_summary(
        PROJECT_ROOT / "outputs" / "evaluations" / f"{_slugify(effective_output_name)}.json",
        evaluation_manifest,
    )


@app.local_entrypoint()
def run_evalplus_model_eval(
    model_ref: str,
    model_label: str = "",
    dataset: str = "humaneval",
    greedy: bool = True,
    n_samples: int = 1,
    dtype: str = "bfloat16",
    backend: str = "hf",
    force_base_prompt: bool = False,
    attn_implementation: str = "eager",
    version: str = "default",
    parallel: int = 0,
    max_new_tokens: int = 768,
    task_limit: int = 0,
    runner_mode: str = "direct_hf",
    gpu_type: str = A100_40GB_MODAL_GPU,
    output_name: str = "",
) -> None:
    resolved_model_label = model_label or model_ref
    effective_output_name = output_name or (
        f"evalplus-{dataset}-{Path(model_ref).name}-{backend}-greedy"
    )
    output_slug = _slugify(effective_output_name)
    evalplus_remote = _resolve_evalplus_remote(gpu_type)
    summary = evalplus_remote.remote(
        model_ref=model_ref,
        model_label=resolved_model_label,
        dataset=dataset,
        greedy=greedy,
        n_samples=n_samples,
        dtype=dtype,
        backend=backend,
        force_base_prompt=force_base_prompt,
        attn_implementation=attn_implementation,
        version=version,
        parallel=parallel,
        max_new_tokens=max_new_tokens,
        task_limit=task_limit,
        runner_mode=runner_mode,
        output_slug=output_slug,
    )
    artifact_texts = dict(summary.pop("artifact_texts", {}))
    output_dir = PROJECT_ROOT / "outputs" / "evalplus" / output_slug
    output_dir.mkdir(parents=True, exist_ok=True)
    local_artifact_paths: dict[str, str] = {}
    for relative_path, text in artifact_texts.items():
        artifact_path = output_dir / relative_path
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(text, encoding="utf-8")
        local_artifact_paths[relative_path] = to_relative_path(artifact_path, PROJECT_ROOT)
    summary["local_artifact_paths"] = local_artifact_paths
    save_summary(output_dir / "summary.json", summary)
    print(output_dir / "summary.json")


@app.local_entrypoint()
def run_bigcodebench_model_eval(
    model_ref: str,
    model_label: str = "",
    split: str = "instruct",
    subset: str = "hard",
    greedy: bool = True,
    n_samples: int = 1,
    dtype: str = "bfloat16",
    attn_implementation: str = "eager",
    parallel: int = 8,
    max_new_tokens: int = 1280,
    task_limit: int = 0,
    gpu_type: str = A100_40GB_MODAL_GPU,
    output_name: str = "",
    no_gt: bool = False,
    evaluation_execution: str = "gradio",
    generation_only: bool = False,
    checkpoint_interval: int = 10,
) -> None:
    resolved_model_label = model_label or model_ref
    effective_output_name = output_name or (
        f"bigcodebench-{split}-{subset}-{Path(model_ref).name}-hf-greedy"
    )
    output_slug = _slugify(effective_output_name)
    bigcodebench_remote = _resolve_bigcodebench_remote(gpu_type)
    summary = bigcodebench_remote.remote(
        model_ref=model_ref,
        model_label=resolved_model_label,
        split=split,
        subset=subset,
        greedy=greedy,
        n_samples=n_samples,
        dtype=dtype,
        attn_implementation=attn_implementation,
        parallel=parallel,
        max_new_tokens=max_new_tokens,
        task_limit=task_limit,
        output_slug=output_slug,
        no_gt=no_gt,
        evaluation_execution=evaluation_execution,
        generation_only=generation_only,
        checkpoint_interval=checkpoint_interval,
    )
    artifact_texts = dict(summary.pop("artifact_texts", {}))
    output_dir = PROJECT_ROOT / "outputs" / "bigcodebench" / output_slug
    output_dir.mkdir(parents=True, exist_ok=True)
    local_artifact_paths: dict[str, str] = {}
    for relative_path, text in artifact_texts.items():
        artifact_path = output_dir / relative_path
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(text, encoding="utf-8")
        local_artifact_paths[relative_path] = to_relative_path(artifact_path, PROJECT_ROOT)
    summary["local_artifact_paths"] = local_artifact_paths
    save_summary(output_dir / "summary.json", summary)
    print(output_dir / "summary.json")


@app.local_entrypoint()
def run_bigcodebench_samples_eval(
    output_name: str,
    split: str = "instruct",
    subset: str = "hard",
    samples_relative_path: str = "",
    raw_samples_relative_path: str = "",
    evaluation_execution: str = "gradio",
    parallel: int = 4,
    no_gt: bool = False,
    overwrite: bool = True,
) -> None:
    output_slug = _slugify(output_name)
    summary = grade_bigcodebench_samples_remote.remote(
        output_slug=output_slug,
        split=split,
        subset=subset,
        samples_relative_path=samples_relative_path,
        raw_samples_relative_path=raw_samples_relative_path,
        evaluation_execution=evaluation_execution,
        parallel=parallel,
        no_gt=no_gt,
        overwrite=overwrite,
    )
    artifact_texts = dict(summary.pop("artifact_texts", {}))
    output_dir = PROJECT_ROOT / "outputs" / "bigcodebench" / output_slug
    output_dir.mkdir(parents=True, exist_ok=True)
    local_artifact_paths: dict[str, str] = {}
    for relative_path, text in artifact_texts.items():
        artifact_path = output_dir / relative_path
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(text, encoding="utf-8")
        local_artifact_paths[relative_path] = to_relative_path(artifact_path, PROJECT_ROOT)
    summary["local_artifact_paths"] = local_artifact_paths
    save_summary(output_dir / "grading_summary.json", summary)
    print(output_dir / "grading_summary.json")


@app.local_entrypoint()
def download_bigcodebench_artifacts(output_name: str) -> None:
    output_slug = _slugify(output_name)
    payload = collect_bigcodebench_artifacts.remote(output_slug)
    artifact_texts = dict(payload.get("artifact_texts", {}))
    output_dir = PROJECT_ROOT / "outputs" / "bigcodebench" / output_slug
    output_dir.mkdir(parents=True, exist_ok=True)

    local_artifact_paths: dict[str, str] = {}
    for relative_path, text in artifact_texts.items():
        artifact_path = output_dir / relative_path
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(text, encoding="utf-8")
        local_artifact_paths[relative_path] = to_relative_path(artifact_path, PROJECT_ROOT)

    manifest = {
        "output_slug": output_slug,
        "remote_root": payload.get("remote_root"),
        "file_count": payload.get("file_count", 0),
        "local_artifact_paths": local_artifact_paths,
    }
    save_summary(output_dir / "artifact_manifest.json", manifest)
    print(output_dir / "artifact_manifest.json")


@app.local_entrypoint()
def run_uniform_int4_task_baseline(
    report_path: str,
    contract_path: str = "configs/experiment_contract_9b_math500_int8.json",
    task_name: str = "",
    task_prompt_style: str = "simple_evals_nonthinking",
    split: str = "last100",
    limit: int = 100,
    max_new_tokens: int = 4096,
    calibration_limit: int = 4,
    gpu_type: str = A100_40GB_MODAL_GPU,
    output_name: str = "",
) -> None:
    from ta_mpq.search import build_search_groups, estimate_candidate_weight_footprint_gb, layer_stats_from_report

    contract = load_contract(PROJECT_ROOT / contract_path)
    resolved_task_name = task_name or contract.task_name
    report_payload = _load_json(PROJECT_ROOT / report_path)
    layer_stats = layer_stats_from_report(report_payload)
    groups = build_search_groups(layer_stats, grouping="per_block_component")
    fixed_assignments = _resolve_sprint_fixed_assignments(groups)
    group_bits = {group.name: 4 for group in groups}
    group_bits.update(fixed_assignments)
    target_budget_gb = estimate_candidate_weight_footprint_gb(groups, group_bits)
    effective_output_name = output_name or (
        f"{contract.name}-{resolved_task_name}-uniform-int4-{split}"
    )
    output_slug = _slugify(effective_output_name)
    policy_output_dir = PROJECT_ROOT / "outputs" / "policies" / output_slug
    policy_output_dir.mkdir(parents=True, exist_ok=True)

    record = _run_surrogate_free_candidate_eval(
        candidate_key="baseline-uniform-int4",
        report_payload=report_payload,
        group_bits=group_bits,
        proposal_score=0.0,
        provenance="uniform_int4_seed",
        contract=contract,
        task_name=resolved_task_name,
        task_prompt_style=task_prompt_style,
        calibration_limit=calibration_limit,
        target_budget_gb=target_budget_gb,
        dev_split=split,
        dev_limit=limit,
        max_new_tokens=max_new_tokens,
        output_slug=output_slug,
        policy_output_dir=policy_output_dir,
        gpu_type=gpu_type,
    )
    save_summary(
        PROJECT_ROOT / "outputs" / "search" / f"{output_slug}.json",
        {
            "strategy": "uniform_int4_task_baseline",
            "contract_path": contract_path,
            "report_path": report_path,
            "task_name": resolved_task_name,
            "task_prompt_style": task_prompt_style,
            "split": split,
            "limit": limit,
            "max_new_tokens": max_new_tokens,
            "gpu_type": gpu_type,
            "target_budget_gb": target_budget_gb,
            "candidate": _candidate_result_snapshot(record),
        },
    )


@app.local_entrypoint()
def run_uniform_int8_task_baseline(
    report_path: str,
    contract_path: str = "configs/experiment_contract_9b_math500_int8.json",
    task_name: str = "",
    task_prompt_style: str = "simple_evals_nonthinking",
    split: str = "last100",
    limit: int = 100,
    max_new_tokens: int = 4096,
    calibration_limit: int = 4,
    gpu_type: str = A100_40GB_MODAL_GPU,
    output_name: str = "",
) -> None:
    from ta_mpq.search import build_search_groups, estimate_candidate_weight_footprint_gb, layer_stats_from_report

    contract = load_contract(PROJECT_ROOT / contract_path)
    resolved_task_name = task_name or contract.task_name
    report_payload = _load_json(PROJECT_ROOT / report_path)
    layer_stats = layer_stats_from_report(report_payload)
    groups = build_search_groups(layer_stats, grouping="per_block_component")
    fixed_assignments = _resolve_sprint_fixed_assignments(groups)
    group_bits = {group.name: 8 for group in groups}
    group_bits.update(fixed_assignments)
    target_budget_gb = estimate_candidate_weight_footprint_gb(groups, group_bits)
    effective_output_name = output_name or (
        f"{contract.name}-{resolved_task_name}-uniform-int8-{split}"
    )
    output_slug = _slugify(effective_output_name)
    policy_output_dir = PROJECT_ROOT / "outputs" / "policies" / output_slug
    policy_output_dir.mkdir(parents=True, exist_ok=True)

    record = _run_surrogate_free_candidate_eval(
        candidate_key="baseline-uniform-int8",
        report_payload=report_payload,
        group_bits=group_bits,
        proposal_score=0.0,
        provenance="uniform_int8_seed",
        contract=contract,
        task_name=resolved_task_name,
        task_prompt_style=task_prompt_style,
        calibration_limit=calibration_limit,
        target_budget_gb=target_budget_gb,
        dev_split=split,
        dev_limit=limit,
        max_new_tokens=max_new_tokens,
        output_slug=output_slug,
        policy_output_dir=policy_output_dir,
        gpu_type=gpu_type,
    )
    save_summary(
        PROJECT_ROOT / "outputs" / "search" / f"{output_slug}.json",
        {
            "strategy": "uniform_int8_task_baseline",
            "contract_path": contract_path,
            "report_path": report_path,
            "task_name": resolved_task_name,
            "task_prompt_style": task_prompt_style,
            "split": split,
            "limit": limit,
            "max_new_tokens": max_new_tokens,
            "gpu_type": gpu_type,
            "target_budget_gb": target_budget_gb,
            "candidate": _candidate_result_snapshot(record),
        },
    )


@app.local_entrypoint()
def run_surrogate_free_math500_sprint(
    report_path: str,
    contract_path: str = "configs/experiment_contract_27b_9b_math500.json",
    task_name: str = "",
    task_prompt_style: str = "",
    group_value_prior_path: str = "",
    sensitivity_profile_path: str = "",
    sensitivity_field: str = "combined_sensitivity",
    target_budget_gb: float = 0.0,
    allowed_bits: str = "4,8,16",
    calibration_limit: int = 4,
    max_new_tokens: int = 2048,
    search_split: str = "",
    dev_split: str = "dev100",
    accept_split: str = "accept200",
    final_split: str = "full500",
    turn_limit: int = 0,
    stage1_turn_limit: int = 0,
    stage2_turn_limit: int = 0,
    dev_limit_override: int = 0,
    accept_limit_override: int = 0,
    final_limit_override: int = 0,
    promotable_count: int = 30,
    demotable_count: int = 60,
    seed_provenances: str = "",
    seed_mode: str = "",
    max_seed_candidates: int = 0,
    beam_width: int = 3,
    rounds: int = 4,
    round_eval_count: int = 3,
    round_proposal_eval_count: int = 0,
    round_survivor_eval_count: int = 0,
    min_budget_utilization: float = 0.99,
    use_sensitivity_selection: bool = True,
    baseline_mode: str = "default",
    gpu_type: str = DEFAULT_MODAL_GPU,
    defer_accept_eval: bool = False,
    enable_finalist_config_refinement: bool = False,
    evaluate_full500: bool = False,
    resume_output_slug: str = "",
    output_name: str = "",
) -> None:
    from ta_mpq.search import (
        build_search_groups,
        build_surrogate_free_seed_assignments,
        estimate_assignment_search_score,
        estimate_candidate_weight_footprint_gb,
        generate_surrogate_free_neighbor_assignments,
        layer_stats_from_report,
        resolve_group_value_scores,
        resolve_sensitivity_overrides,
        resolve_surrogate_free_priority_lists,
        resolve_surrogate_free_priority_scores,
    )

    contract = load_contract(PROJECT_ROOT / contract_path)
    resolved_task_name = task_name or contract.task_name
    if resolved_task_name.lower() not in {"math500", "math-500"}:
        raise ValueError("run_surrogate_free_math500_sprint only supports MATH-500")
    resolved_max_new_tokens = _resolve_default_max_new_tokens(resolved_task_name, max_new_tokens)

    report_payload = _load_json(PROJECT_ROOT / report_path)
    group_value_prior_payload = (
        _load_json(PROJECT_ROOT / group_value_prior_path) if group_value_prior_path else None
    )
    sensitivity_profile_payload = (
        _load_json(PROJECT_ROOT / sensitivity_profile_path) if sensitivity_profile_path else None
    )
    resolved_search_split = search_split or dev_split
    normalized_allowed_bits = tuple(
        sorted({int(part.strip()) for part in allowed_bits.split(",") if part.strip()})
    )
    if normalized_allowed_bits not in {(4, 8, 16), (2, 4, 8)}:
        raise ValueError("The sprint runner currently expects allowed_bits to be exactly 4,8,16 or 2,4,8")

    layer_stats = layer_stats_from_report(report_payload)
    groups = build_search_groups(
        layer_stats,
        grouping="per_block_component",
        sensitivity_overrides=resolve_sensitivity_overrides(
            layer_stats=layer_stats,
            grouping="per_block_component",
            sensitivity_profile_payload=sensitivity_profile_payload,
            field=sensitivity_field,
        ),
    )
    fixed_assignments = _resolve_sprint_fixed_assignments(groups)
    if target_budget_gb > 0:
        resolved_target_budget_gb = target_budget_gb
    elif normalized_allowed_bits == (2, 4, 8):
        uniform_int4_budget = {
            group.name: 4
            for group in groups
        }
        uniform_int4_budget.update(fixed_assignments)
        resolved_target_budget_gb = float(
            estimate_candidate_weight_footprint_gb(groups, uniform_int4_budget)
        )
    else:
        resolved_target_budget_gb = contract.search_target_budget_gb or 0.0
    if resolved_target_budget_gb <= 0:
        raise ValueError("target_budget_gb must be positive")
    group_value_scores = resolve_group_value_scores(
        groups,
        group_value_prior_payload,
        layer_stats=layer_stats,
        target_grouping="per_block_component",
    )
    if use_sensitivity_selection:
        group_priority_scores = resolve_surrogate_free_priority_scores(
            groups,
            group_value_scores=group_value_scores,
        )
        priority_lists = resolve_surrogate_free_priority_lists(
            groups=groups,
            group_priority_scores=group_priority_scores,
            promotable_count=promotable_count,
            demotable_count=demotable_count,
            excluded_group_names=set(fixed_assignments),
        )
        promotable_group_names = priority_lists["promotable_group_names"]
        demotable_group_names = priority_lists["demotable_group_names"]
    else:
        eligible_groups = [group for group in groups if group.name not in set(fixed_assignments)]
        group_priority_scores = {group.name: 0.0 for group in groups}
        promotable_group_names = [group.name for group in eligible_groups]
        demotable_group_names = [group.name for group in eligible_groups]
    selected_seed_provenances = tuple(
        part.strip() for part in seed_provenances.split(",") if part.strip()
    ) or None
    if not selected_seed_provenances and seed_mode:
        normalized_seed_mode = tuple(part.strip() for part in seed_mode.split(",") if part.strip())
        if normalized_seed_mode:
            selected_seed_provenances = normalized_seed_mode

    effective_output_name = (
        output_name
        or resume_output_slug
        or f"{contract.name}-{resolved_task_name}-surrogate-free"
    )
    output_slug = _slugify(resume_output_slug or effective_output_name)
    policy_output_dir = PROJECT_ROOT / "outputs" / "policies" / output_slug
    policy_output_dir.mkdir(parents=True, exist_ok=True)

    stage1_limit = (
        stage1_turn_limit
        or turn_limit
        or dev_limit_override
        or DEFAULT_STAGE1_TURN_LIMIT
    )
    stage2_limit = stage2_turn_limit or DEFAULT_STAGE2_TURN_LIMIT
    dev_limit = stage1_limit
    accept_limit = accept_limit_override or _task_limit_for_split(accept_split)
    final_limit = final_limit_override or _task_limit_for_split(final_split)
    survivor_eval_count = round_survivor_eval_count or round_eval_count or DEFAULT_ROUND_SURVIVOR_EVAL_COUNT
    provisional_eval_count = max(
        survivor_eval_count,
        round_proposal_eval_count or max(
            DEFAULT_ROUND_PROPOSAL_EVAL_COUNT,
            round_eval_count,
        ),
    )
    search_deck = (
        _initialize_search_deck(resolved_task_name, resolved_search_split, output_slug)
        if stage1_limit > 0
        else None
    )

    resume_state = None
    if resume_output_slug:
        resume_state = _load_surrogate_free_resume_state(
            output_slug=output_slug,
            split_name=resolved_search_split,
            beam_width=beam_width,
        )
        if search_deck is not None:
            _advance_search_deck_for_resume(
                deck_state=search_deck,
                resumed_records=resume_state["ordered_records"],
                stage1_turn_limit=stage1_limit,
                stage2_turn_limit=stage2_limit,
                split_name=resolved_search_split,
            )

    baseline_records = {
        "native_bf16": {},
    }
    eval_remote = _resolve_eval_remote(gpu_type)
    if resume_state is not None:
        baseline_records = {}
    elif baseline_mode == "default":
        baseline_model_ref = contract.upper_bound_model_id or contract.compressed_source_model_id
        for split_name, split_limit in (
            (resolved_search_split, dev_limit),
            (accept_split, accept_limit),
        ):
            summary = eval_remote.remote(
                model_ref=baseline_model_ref,
                tokenizer_source=contract.compressed_source_model_id,
                model_label=f"{baseline_model_ref}-bf16",
                task_name=resolved_task_name,
                limit=split_limit,
                max_new_tokens=resolved_max_new_tokens,
                split=split_name,
                load_dtype="bfloat16",
                task_prompt_style=task_prompt_style,
            )
            summary_path = (
                PROJECT_ROOT
                / "outputs"
                / "evaluations"
                / f"{output_slug}-native-bf16-{_slugify(split_name)}.json"
            )
            save_summary(summary_path, summary)
            baseline_records["native_bf16"][split_name] = {
                "path": to_relative_path(summary_path, PROJECT_ROOT),
                "accuracy": float(summary.get("accuracy", 0.0)),
                "task_split": split_name,
            }
    else:
        baseline_records = {}

    all_candidate_records: list[dict[str, Any]]
    seen_signatures: set[tuple[tuple[str, int], ...]]
    beam_records: list[dict[str, Any]]
    round_summaries: list[dict[str, Any]]
    start_round_index = 1
    uniform_record = None
    if resume_state is not None:
        all_candidate_records = list(resume_state["records"])
        seen_signatures = set(resume_state["seen_signatures"])
        beam_records = list(resume_state["beam_records"])
        round_summaries = list(resume_state["round_summaries"])
        start_round_index = int(resume_state["next_round_index"])
    else:
        seed_assignments = build_surrogate_free_seed_assignments(
            groups=groups,
            target_budget_gb=resolved_target_budget_gb,
            allowed_bits=normalized_allowed_bits,
            group_priority_scores=group_priority_scores,
            promotable_group_names=promotable_group_names,
            demotable_group_names=demotable_group_names,
            fixed_assignments=fixed_assignments,
            min_budget_utilization=min_budget_utilization,
            max_seed_count=max_seed_candidates,
            selected_seed_provenances=selected_seed_provenances,
        )
        all_candidate_records = []
        seen_signatures = set()
        for seed_index, (provenance, assignments) in enumerate(seed_assignments, start=1):
            search_turn = (
                _consume_search_turn_examples(search_deck, stage1_limit)
                if search_deck is not None
                else None
            )
            record = _run_surrogate_free_candidate_eval(
                candidate_key=f"seed-{seed_index:02d}-{_slugify(provenance)}",
                report_payload=report_payload,
                group_bits=assignments,
                proposal_score=estimate_assignment_search_score(
                    groups=groups,
                    assignments=assignments,
                    target_budget_gb=resolved_target_budget_gb,
                    group_priority_scores=group_priority_scores,
                ),
                provenance=provenance,
                contract=contract,
                task_name=resolved_task_name,
                task_prompt_style=task_prompt_style,
                calibration_limit=calibration_limit,
                target_budget_gb=resolved_target_budget_gb,
                dev_split=resolved_search_split,
                dev_limit=stage1_limit,
                max_new_tokens=resolved_max_new_tokens,
                output_slug=output_slug,
                policy_output_dir=policy_output_dir,
                example_ids=(search_turn["example_ids"] if search_deck is not None else None),
                evaluation_metadata=(
                    {
                        "deck_segments": search_turn["segments"],
                        "deck_cursor_after": search_turn["cursor_after"],
                        "reshuffle_count_after": search_turn["reshuffle_count_after"],
                    }
                    if search_deck is not None
                    else None
                ),
                gpu_type=gpu_type,
            )
            all_candidate_records.append(record)
            seen_signatures.add(_assignment_signature(assignments))

        beam_records = _select_best_direct_eval_records(
            all_candidate_records,
            split_name=resolved_search_split,
            limit=beam_width,
        )
        round_summaries = []

    if resume_state is None and baseline_mode == "default":
        uniform_record = next(
            (
                record
                for record in all_candidate_records
                if str(record.get("provenance")) == "uniform_int8_seed"
            ),
            None,
        )
        if uniform_record is None:
            uniform_seed_assignments = build_surrogate_free_seed_assignments(
                groups=groups,
                target_budget_gb=resolved_target_budget_gb,
                allowed_bits=normalized_allowed_bits,
                group_priority_scores=group_priority_scores,
                promotable_group_names=promotable_group_names,
                demotable_group_names=demotable_group_names,
                fixed_assignments=fixed_assignments,
                min_budget_utilization=min_budget_utilization,
                max_seed_count=1,
                selected_seed_provenances=("uniform_int8_seed",),
            )
            uniform_assignments = uniform_seed_assignments[0][1]
            seen_signatures.add(_assignment_signature(uniform_assignments))
            search_turn = (
                _consume_search_turn_examples(search_deck, stage1_limit)
                if search_deck is not None
                else None
            )
            uniform_record = _run_surrogate_free_candidate_eval(
                candidate_key="baseline-uniform-int8",
                report_payload=report_payload,
                group_bits=uniform_assignments,
                proposal_score=estimate_assignment_search_score(
                    groups=groups,
                    assignments=uniform_assignments,
                    target_budget_gb=resolved_target_budget_gb,
                    group_priority_scores=group_priority_scores,
                ),
                provenance="uniform_int8_seed",
                contract=contract,
                task_name=resolved_task_name,
                task_prompt_style=task_prompt_style,
                calibration_limit=calibration_limit,
                target_budget_gb=resolved_target_budget_gb,
                dev_split=resolved_search_split,
                dev_limit=stage1_limit,
                max_new_tokens=resolved_max_new_tokens,
                output_slug=output_slug,
                policy_output_dir=policy_output_dir,
                example_ids=(search_turn["example_ids"] if search_turn is not None else None),
                evaluation_metadata=(
                    {
                        "deck_segments": search_turn["segments"],
                        "deck_cursor_after": search_turn["cursor_after"],
                        "reshuffle_count_after": search_turn["reshuffle_count_after"],
                    }
                    if search_turn is not None
                    else None
                ),
                gpu_type=gpu_type,
            )
            all_candidate_records.append(uniform_record)

    for round_index in range(start_round_index, rounds + 1):
        proposals: list[dict[str, Any]] = []
        for beam_record in beam_records:
            neighbors = generate_surrogate_free_neighbor_assignments(
                groups=groups,
                base_assignments=dict(beam_record["group_bit_assignments"]),
                target_budget_gb=resolved_target_budget_gb,
                allowed_bits=normalized_allowed_bits,
                group_priority_scores=group_priority_scores,
                promotable_group_names=promotable_group_names,
                demotable_group_names=demotable_group_names,
                fixed_assignments=fixed_assignments,
                min_budget_utilization=min_budget_utilization,
            )
            for provenance, assignments in neighbors:
                signature = _assignment_signature(assignments)
                if signature in seen_signatures:
                    continue
                proposals.append(
                    {
                        "provenance": provenance,
                        "group_bits": assignments,
                        "proposal_score": estimate_assignment_search_score(
                            groups=groups,
                            assignments=assignments,
                            target_budget_gb=resolved_target_budget_gb,
                            group_priority_scores=group_priority_scores,
                        ),
                        "parent_candidate_key": beam_record["candidate_key"],
                        "assignment_signature": signature,
                    }
                )
        if not proposals:
            break

        selected_proposals = sorted(
            proposals,
            key=lambda item: (float(item["proposal_score"]), item["parent_candidate_key"]),
            reverse=True,
        )[:provisional_eval_count]
        stage1_turn = (
            _consume_search_turn_examples(search_deck, stage1_limit)
            if search_deck is not None
            else None
        )
        evaluated_round_records: list[dict[str, Any]] = []
        for candidate_index, proposal in enumerate(selected_proposals, start=1):
            record = _run_surrogate_free_candidate_eval(
                candidate_key=f"round-{round_index:02d}-candidate-{candidate_index:02d}",
                report_payload=report_payload,
                group_bits=proposal["group_bits"],
                proposal_score=float(proposal["proposal_score"]),
                provenance=str(proposal["provenance"]),
                contract=contract,
                task_name=resolved_task_name,
                task_prompt_style=task_prompt_style,
                calibration_limit=calibration_limit,
                target_budget_gb=resolved_target_budget_gb,
                dev_split=resolved_search_split,
                dev_limit=stage1_limit,
                max_new_tokens=resolved_max_new_tokens,
                output_slug=output_slug,
                policy_output_dir=policy_output_dir,
                parent_candidate_key=str(proposal["parent_candidate_key"]),
                example_ids=(stage1_turn["example_ids"] if stage1_turn is not None else None),
                evaluation_metadata=(
                    {
                        "deck_segments": stage1_turn["segments"],
                        "deck_cursor_after": stage1_turn["cursor_after"],
                        "reshuffle_count_after": stage1_turn["reshuffle_count_after"],
                    }
                    if stage1_turn is not None
                    else None
                ),
                evaluation_stage="stage1",
                gpu_type=gpu_type,
            )
            evaluated_round_records.append(record)
            all_candidate_records.append(record)
            seen_signatures.add(proposal["assignment_signature"])

        provisional_ranked_records = _select_best_direct_eval_records(
            evaluated_round_records,
            split_name=resolved_search_split,
            limit=len(evaluated_round_records),
        )
        rechecked_records: list[dict[str, Any]] = []
        stage2_turn = None
        for record in provisional_ranked_records[:survivor_eval_count]:
            if stage2_turn is None and search_deck is not None:
                stage2_turn = _consume_search_turn_examples(search_deck, stage2_limit)
            _ensure_candidate_task_eval(
                record=record,
                contract=contract,
                task_name=resolved_task_name,
                task_prompt_style=task_prompt_style,
                split_name=resolved_search_split,
                split_limit=stage2_limit,
                max_new_tokens=resolved_max_new_tokens,
                output_slug=output_slug,
                example_ids=(stage2_turn["example_ids"] if stage2_turn is not None else None),
                evaluation_metadata=(
                    {
                        "deck_segments": stage2_turn["segments"],
                        "deck_cursor_after": stage2_turn["cursor_after"],
                        "reshuffle_count_after": stage2_turn["reshuffle_count_after"],
                    }
                    if stage2_turn is not None
                    else None
                ),
                evaluation_stage="stage2",
                gpu_type=gpu_type,
            )
            rechecked_records.append(record)

        stage2_keys = {str(record["candidate_key"]) for record in rechecked_records}
        round_rank_records = list(rechecked_records) + [
            record for record in provisional_ranked_records if str(record["candidate_key"]) not in stage2_keys
        ]
        beam_records = _select_best_direct_eval_records(
            round_rank_records,
            split_name=resolved_search_split,
            limit=beam_width,
        )
        round_summaries.append(
            {
                "round_index": round_index,
                "proposal_count": len(proposals),
                "provisional_pool_size": len(selected_proposals),
                "evaluated_candidates": [
                    _candidate_round_snapshot(record, resolved_search_split)
                    for record in evaluated_round_records
                ],
                "provisional_candidates": [
                    _candidate_round_snapshot(record, resolved_search_split)
                    for record in provisional_ranked_records
                ],
                "rechecked_candidates": [
                    _candidate_round_snapshot(record, resolved_search_split)
                    for record in rechecked_records
                ],
                "beam_candidates": [
                    _candidate_round_snapshot(record, resolved_search_split)
                    for record in beam_records
                ],
            }
        )

    if enable_finalist_config_refinement:
        _run_optional_finalist_config_refinement(
            all_candidate_records=all_candidate_records,
            report_payload=report_payload,
            groups=groups,
            layer_stats=layer_stats,
            group_value_scores=group_value_scores,
            resolved_target_budget_gb=resolved_target_budget_gb,
            contract=contract,
            task_name=resolved_task_name,
            task_prompt_style=task_prompt_style,
            calibration_limit=calibration_limit,
            dev_split=resolved_search_split,
            dev_limit=dev_limit,
            max_new_tokens=max_new_tokens,
            output_slug=output_slug,
            policy_output_dir=policy_output_dir,
        )

    if uniform_record is not None and not defer_accept_eval:
        _ensure_candidate_task_eval(
            record=uniform_record,
            contract=contract,
            task_name=resolved_task_name,
            task_prompt_style=task_prompt_style,
            split_name=accept_split,
            split_limit=accept_limit,
            max_new_tokens=resolved_max_new_tokens,
            output_slug=output_slug,
            gpu_type=gpu_type,
        )

    top_mixed_candidates = [
        record
        for record in _select_best_direct_eval_records(
            [
                record
                for record in all_candidate_records
                if bool(record.get("is_mixed"))
            ],
            split_name=resolved_search_split,
            limit=2,
        )
        if bool(record.get("integrity_clean"))
    ]
    if not defer_accept_eval:
        for record in top_mixed_candidates:
            _ensure_candidate_task_eval(
                record=record,
                contract=contract,
                task_name=resolved_task_name,
                task_prompt_style=task_prompt_style,
                split_name=accept_split,
                split_limit=accept_limit,
                max_new_tokens=resolved_max_new_tokens,
                output_slug=output_slug,
                gpu_type=gpu_type,
            )

    best_mixed_candidate = next(
        (
            record
            for record in _select_best_direct_eval_records(
                top_mixed_candidates,
                split_name=accept_split,
                limit=1,
            )
            if record.get("evaluations", {}).get(accept_split)
        ),
        None,
    )
    if (
        evaluate_full500
        and not defer_accept_eval
        and best_mixed_candidate is not None
        and uniform_record is not None
        and _candidate_accuracy(best_mixed_candidate, accept_split)
        >= _candidate_accuracy(uniform_record, accept_split)
    ):
        _ensure_candidate_task_eval(
            record=best_mixed_candidate,
            contract=contract,
            task_name=resolved_task_name,
            task_prompt_style=task_prompt_style,
            split_name=final_split,
            split_limit=final_limit,
            max_new_tokens=resolved_max_new_tokens,
            output_slug=output_slug,
        )
        _ensure_candidate_task_eval(
            record=uniform_record,
            contract=contract,
            task_name=resolved_task_name,
            task_prompt_style=task_prompt_style,
            split_name=final_split,
            split_limit=final_limit,
            max_new_tokens=resolved_max_new_tokens,
            output_slug=output_slug,
        )
        summary = eval_remote.remote(
            model_ref=baseline_model_ref,
            tokenizer_source=contract.compressed_source_model_id,
            model_label=f"{baseline_model_ref}-bf16",
            task_name=resolved_task_name,
            limit=final_limit,
            max_new_tokens=resolved_max_new_tokens,
            split=final_split,
            load_dtype="bfloat16",
            task_prompt_style=task_prompt_style,
        )
        summary_path = (
            PROJECT_ROOT
            / "outputs"
            / "evaluations"
            / f"{output_slug}-native-bf16-{_slugify(final_split)}.json"
        )
        save_summary(summary_path, summary)
        baseline_records["native_bf16"][final_split] = {
            "path": to_relative_path(summary_path, PROJECT_ROOT),
            "accuracy": float(summary.get("accuracy", 0.0)),
            "task_split": final_split,
        }

    search_summary = {
        "strategy": "surrogate_free_direct_eval",
        "task_name": resolved_task_name,
        "report_path": report_path,
        "contract_path": contract_path,
        "grouping": "per_block_component",
        "allowed_bits": list(normalized_allowed_bits),
        "target_budget_gb": resolved_target_budget_gb,
        "budget_accounting_mode": "matched_linear_weight_budget",
        "matched_linear_budget_gb": resolved_target_budget_gb,
        "estimated_non_linear_weight_footprint_gb": float(
            report_payload.get("estimated_non_linear_weight_footprint_gb", 0.0)
        ),
        "estimated_uniform_full_model_weight_footprint_gb": float(
            report_payload.get("estimated_full_model_weight_footprint_gb", 0.0)
        ),
        "fixed_assignments": fixed_assignments,
        "promotable_group_names": promotable_group_names,
        "demotable_group_names": demotable_group_names,
        "use_sensitivity_selection": use_sensitivity_selection,
        "seed_provenances": list(selected_seed_provenances or []),
        "seed_mode": seed_mode or None,
        "max_seed_candidates": int(max_seed_candidates),
        "search_split": resolved_search_split,
        "turn_limit": int(stage1_limit),
        "stage1_turn_limit": int(stage1_limit),
        "stage2_turn_limit": int(stage2_limit),
        "round_proposal_eval_count": int(provisional_eval_count),
        "round_survivor_eval_count": int(survivor_eval_count),
        "gpu_type": gpu_type,
        "baseline_mode": baseline_mode,
        "defer_accept_eval": defer_accept_eval,
        "resume_output_slug": resume_output_slug or None,
        "dev_split": resolved_search_split,
        "dev_limit": dev_limit,
        "accept_split": accept_split,
        "accept_limit": accept_limit,
        "task_prompt_style": task_prompt_style or None,
        "final_split": final_split if evaluate_full500 else None,
        "final_limit": final_limit if evaluate_full500 else None,
        "rounds": round_summaries,
        "baselines": baseline_records,
        "uniform_int8_record": (
            _candidate_result_snapshot(uniform_record) if uniform_record is not None else None
        ),
        "top_mixed_candidates": [
            _candidate_result_snapshot(record)
            for record in top_mixed_candidates
        ],
        "search_deck": (
            {
                "split_name": search_deck["split_name"],
                "deck_size": len(search_deck["ordered_example_ids"]),
                "cursor": int(search_deck["cursor"]),
                "reshuffle_count": int(search_deck["reshuffle_count"]),
                "shuffle_seed": search_deck["shuffle_seed"],
                "ordered_example_ids": list(search_deck["ordered_example_ids"]),
            }
            if search_deck is not None
            else None
        ),
        "resume_state": (
            {
                "resumed_candidate_count": len(resume_state["records"]),
                "next_round_index": int(resume_state["next_round_index"]),
                "completed_turns": int(resume_state["completed_turns"]),
                "resumed_candidate_keys": [record["candidate_key"] for record in resume_state["ordered_records"]],
            }
            if resume_state is not None
            else None
        ),
        "all_candidate_records": [
            _candidate_result_snapshot(record)
            for record in _select_best_direct_eval_records(
                all_candidate_records,
                split_name=resolved_search_split,
                limit=min(20, len(all_candidate_records)),
            )
        ],
    }
    save_summary(
        PROJECT_ROOT / "outputs" / "search" / f"{output_slug}.json",
        search_summary,
    )


@app.local_entrypoint()
def run_task_sensitivity_probe(
    contract_path: str = "configs/experiment_contract_27b_9b.json",
    task_name: str = "",
    model_role: str = "compressed_source",
    limit: int = 8,
    grouping: str = "per_block_component",
    max_prompt_tokens: int = 1024,
    activation_weight: float = 0.55,
    output_name: str = "",
) -> None:
    contract = load_contract(PROJECT_ROOT / contract_path)
    resolved_task_name = task_name or contract.task_name
    profile = collect_task_sensitivity_profile_remote.remote(
        contract.to_dict(),
        model_role=model_role,
        task_name=resolved_task_name,
        limit=limit,
        grouping=grouping,
        max_prompt_tokens=max_prompt_tokens,
        activation_weight=activation_weight,
    )
    effective_output_name = output_name or (
        f"{contract.name}-{resolved_task_name}-{model_role}-{grouping}-sensitivity"
    )
    save_summary(
        PROJECT_ROOT / "outputs" / "sensitivity" / f"{_slugify(effective_output_name)}.json",
        profile,
    )


@app.local_entrypoint()
def run_build_task_kl_sensitivity_profile(
    report_path: str,
    kl_stats_path: str,
    grouping: str = "per_block_component",
    kl_weight: float = 0.55,
    output_name: str = "",
) -> None:
    from ta_mpq.search import layer_stats_from_report
    from ta_mpq.sensitivity import ModuleKLDivergenceStat, build_task_kl_sensitivity_profile

    report_payload = _load_json(PROJECT_ROOT / report_path)
    kl_payload = _load_json(PROJECT_ROOT / kl_stats_path)
    raw_stats = kl_payload.get("module_kl_stats", kl_payload)
    if not isinstance(raw_stats, list):
        raise ValueError("kl_stats_path must point to a JSON list or {'module_kl_stats': [...]} payload")

    kl_stats = [
        ModuleKLDivergenceStat(
            name=str(item["name"]),
            parameter_count=int(item["parameter_count"]),
            mean_output_kl=float(item["mean_output_kl"]),
            num_observations=int(item.get("num_observations", 0)),
        )
        for item in raw_stats
    ]
    profile = build_task_kl_sensitivity_profile(
        layer_stats=layer_stats_from_report(report_payload),
        kl_stats=kl_stats,
        grouping=grouping,
        kl_weight=kl_weight,
    )
    profile["source_report_path"] = report_path
    profile["source_kl_stats_path"] = kl_stats_path
    effective_output_name = output_name or (
        f"{Path(report_path).stem}-{Path(kl_stats_path).stem}-{grouping}-kl-sensitivity"
    )
    save_summary(
        PROJECT_ROOT / "outputs" / "sensitivity" / f"{_slugify(effective_output_name)}.json",
        profile,
    )


@app.local_entrypoint()
def run_build_precision_ablation_manifest(
    report_path: str,
    candidate_path: str,
    ranking_profile_path: str = "",
    ranking_field: str = "combined_sensitivity",
    max_groups: int = 16,
    floor_bit: int = 4,
    allowed_bits: str = "4,8,16",
    reference_bits: str = "",
    output_name: str = "",
    output_dir: str = "",
) -> None:
    from ta_mpq.ablation import build_precision_ablation_manifest

    report_payload = _load_json(PROJECT_ROOT / report_path)
    candidate_payload = _load_json(PROJECT_ROOT / candidate_path)
    ranking_profile_payload = (
        _load_json(PROJECT_ROOT / ranking_profile_path) if ranking_profile_path else None
    )
    normalized_allowed_bits = tuple(
        sorted({int(part.strip()) for part in allowed_bits.split(",") if part.strip()})
    )
    normalized_reference_bits = (
        tuple(sorted({int(part.strip()) for part in reference_bits.split(",") if part.strip()}))
        if reference_bits
        else None
    )
    effective_output_name = output_name or (
        f"{Path(candidate_path).stem}-precision-ablation-manifest"
    )
    effective_output_dir = output_dir or (
        PROJECT_ROOT / "outputs" / "ablations" / _slugify(effective_output_name)
    )
    manifest = build_precision_ablation_manifest(
        report_payload=report_payload,
        reference_candidate_payload=candidate_payload,
        output_dir=effective_output_dir,
        allowed_bits=normalized_allowed_bits,
        floor_bit=floor_bit,
        max_groups=max_groups,
        reference_bit_widths=normalized_reference_bits,
        ranking_profile_payload=ranking_profile_payload,
        ranking_field=ranking_field,
    )
    manifest["report_path"] = report_path
    manifest["reference_candidate_path"] = candidate_path
    manifest["ranking_profile_path"] = ranking_profile_path or None
    manifest["ranking_field"] = ranking_field
    save_summary(Path(effective_output_dir) / "manifest.json", manifest)


@app.local_entrypoint()
def run_precision_ablation_eval_manifest(
    ablation_manifest_path: str,
    contract_path: str = "configs/experiment_contract_27b_9b_math500.json",
    reference_candidate_path: str = "",
    reference_eval_path: str = "",
    reference_report_path: str = "",
    policy_source: str = "llmcompressor",
    backend_variant: str = "source",
    task_name: str = "",
    limit: int = 25,
    max_new_tokens: int = 32,
    calibration_limit: int = 4,
    output_name: str = "",
) -> None:
    contract = load_contract(PROJECT_ROOT / contract_path)
    resolved_task_name = task_name or contract.task_name
    manifest_payload = _load_json(PROJECT_ROOT / ablation_manifest_path)
    manifest_label = _manifest_output_label(PROJECT_ROOT / ablation_manifest_path)
    reference_candidate = reference_candidate_path or str(
        manifest_payload.get("reference_candidate_path", "")
    )
    if not reference_candidate:
        raise ValueError("reference_candidate_path is required")

    if reference_eval_path:
        reference_summary = _load_json(PROJECT_ROOT / reference_eval_path)
        reference_eval_output_path = PROJECT_ROOT / reference_eval_path
    else:
        reference_artifact_dir, resolved_reference_report_path = _resolve_candidate_artifact_dir(
            contract=contract,
            candidate_path=reference_candidate,
            policy_source=policy_source,
            backend_variant=backend_variant,
            calibration_limit=calibration_limit,
            report_path=reference_report_path,
            report_payload_path=manifest_payload.get("report_path"),
        )
        reference_summary = evaluate_task_source_model.remote(
            model_ref=reference_artifact_dir,
            tokenizer_source=contract.compressed_source_model_id,
            model_label=f"{contract.compressed_source_model_id}-reference-ablation",
            task_name=resolved_task_name,
            limit=limit,
            max_new_tokens=max_new_tokens,
            load_dtype="auto",
        )
        reference_eval_output_path = (
            PROJECT_ROOT
            / "outputs"
            / "evaluations"
            / f"{_slugify(manifest_label)}-reference.json"
        )
        save_summary(reference_eval_output_path, reference_summary)
        reference_report_path = resolved_reference_report_path

    ablation_records: list[dict[str, Any]] = []
    for entry in manifest_payload.get("ablations", []):
        candidate_path = str(entry["candidate_path"])
        artifact_dir, ablation_report_path = _resolve_candidate_artifact_dir(
            contract=contract,
            candidate_path=candidate_path,
            policy_source=policy_source,
            backend_variant=backend_variant,
            calibration_limit=calibration_limit,
            report_payload_path=manifest_payload.get("report_path"),
        )
        evaluation_summary = evaluate_task_source_model.remote(
            model_ref=artifact_dir,
            tokenizer_source=contract.compressed_source_model_id,
            model_label=f"{contract.compressed_source_model_id}-{entry['group_name']}-ablation",
            task_name=resolved_task_name,
            limit=limit,
            max_new_tokens=max_new_tokens,
            load_dtype="auto",
        )
        evaluation_output_path = (
            PROJECT_ROOT
            / "outputs"
            / "evaluations"
            / f"{_slugify(manifest_label)}-{_slugify(entry['group_name'])}.json"
        )
        save_summary(evaluation_output_path, evaluation_summary)
        ablation_records.append(
            {
                **entry,
                "report_path": to_relative_path(Path(ablation_report_path), PROJECT_ROOT),
                "evaluation_path": to_relative_path(evaluation_output_path, PROJECT_ROOT),
                "accuracy": float(evaluation_summary.get("accuracy", 0.0)),
            }
        )

    evaluation_manifest = {
        "ablation_manifest_path": ablation_manifest_path,
        "contract_path": contract_path,
        "task_name": resolved_task_name,
        "limit": limit,
        "max_new_tokens": max_new_tokens,
        "reference_candidate_path": reference_candidate,
        "reference_report_path": reference_report_path or None,
        "reference_evaluation_path": to_relative_path(reference_eval_output_path, PROJECT_ROOT),
        "reference_accuracy": float(reference_summary.get("accuracy", 0.0)),
        "policy_source": policy_source,
        "backend_variant": backend_variant,
        "ablations": ablation_records,
    }
    effective_output_name = output_name or f"{manifest_label}-eval-manifest"
    save_summary(
        PROJECT_ROOT / "outputs" / "ablations" / f"{_slugify(effective_output_name)}.json",
        evaluation_manifest,
    )


@app.local_entrypoint()
def run_build_precision_ablation_profile(
    report_path: str,
    reference_candidate_path: str,
    reference_evaluation_path: str,
    ablation_manifest_path: str,
    evaluation_manifest_path: str,
    prior_weight: float = 0.25,
    output_name: str = "",
) -> None:
    from ta_mpq.ablation import (
        build_precision_ablation_profile,
        load_evaluation_payloads_from_manifest,
    )

    report_payload = _load_json(PROJECT_ROOT / report_path)
    candidate_payload = _load_json(PROJECT_ROOT / reference_candidate_path)
    reference_summary_payload = _load_json(PROJECT_ROOT / reference_evaluation_path)
    ablation_manifest_payload = _load_json(PROJECT_ROOT / ablation_manifest_path)
    evaluation_manifest_payload = _load_json(PROJECT_ROOT / evaluation_manifest_path)
    ablation_evaluation_payloads = load_evaluation_payloads_from_manifest(
        evaluation_manifest_payload,
        base_dir=PROJECT_ROOT,
    )
    profile = build_precision_ablation_profile(
        report_payload=report_payload,
        reference_candidate_payload=candidate_payload,
        reference_summary_payload=reference_summary_payload,
        ablation_manifest_payload=ablation_manifest_payload,
        ablation_evaluation_payloads=ablation_evaluation_payloads,
        prior_weight=prior_weight,
    )
    profile["report_path"] = report_path
    profile["reference_candidate_path"] = reference_candidate_path
    profile["reference_evaluation_path"] = reference_evaluation_path
    profile["ablation_manifest_path"] = ablation_manifest_path
    profile["evaluation_manifest_path"] = evaluation_manifest_path
    effective_output_name = output_name or (
        f"{Path(reference_candidate_path).stem}-precision-ablation-sensitivity"
    )
    save_summary(
        PROJECT_ROOT / "outputs" / "sensitivity" / f"{_slugify(effective_output_name)}.json",
        profile,
    )


@app.local_entrypoint()
def run_precision_ablation_sensitivity(
    report_path: str,
    candidate_path: str,
    contract_path: str = "configs/experiment_contract_27b_9b_math500.json",
    ranking_profile_path: str = "",
    ranking_field: str = "combined_sensitivity",
    max_groups: int = 16,
    floor_bit: int = 4,
    allowed_bits: str = "4,8,16",
    reference_bits: str = "",
    policy_source: str = "llmcompressor",
    backend_variant: str = "source",
    task_name: str = "",
    limit: int = 25,
    max_new_tokens: int = 32,
    calibration_limit: int = 4,
    prior_weight: float = 0.25,
    output_name: str = "",
) -> None:
    loop_slug = _slugify(output_name or f"{Path(candidate_path).stem}-precision-ablation")
    manifest_dir = PROJECT_ROOT / "outputs" / "ablations" / loop_slug
    manifest_dir.mkdir(parents=True, exist_ok=True)

    run_build_precision_ablation_manifest(
        report_path=report_path,
        candidate_path=candidate_path,
        ranking_profile_path=ranking_profile_path,
        ranking_field=ranking_field,
        max_groups=max_groups,
        floor_bit=floor_bit,
        allowed_bits=allowed_bits,
        reference_bits=reference_bits,
        output_dir=str(manifest_dir),
        output_name=loop_slug,
    )
    manifest_path = manifest_dir / "manifest.json"
    manifest_label = _manifest_output_label(manifest_path)
    eval_manifest_output_name = f"{loop_slug}-eval-manifest"
    run_precision_ablation_eval_manifest(
        ablation_manifest_path=to_relative_path(manifest_path, PROJECT_ROOT),
        contract_path=contract_path,
        reference_candidate_path=candidate_path,
        policy_source=policy_source,
        backend_variant=backend_variant,
        task_name=task_name,
        limit=limit,
        max_new_tokens=max_new_tokens,
        calibration_limit=calibration_limit,
        output_name=eval_manifest_output_name,
    )
    evaluation_manifest_path = (
        PROJECT_ROOT / "outputs" / "ablations" / f"{_slugify(eval_manifest_output_name)}.json"
    )
    reference_evaluation_path = (
        PROJECT_ROOT / "outputs" / "evaluations" / f"{_slugify(manifest_label)}-reference.json"
    )
    run_build_precision_ablation_profile(
        report_path=report_path,
        reference_candidate_path=candidate_path,
        reference_evaluation_path=to_relative_path(reference_evaluation_path, PROJECT_ROOT),
        ablation_manifest_path=to_relative_path(manifest_path, PROJECT_ROOT),
        evaluation_manifest_path=to_relative_path(evaluation_manifest_path, PROJECT_ROOT),
        prior_weight=prior_weight,
        output_name=f"{loop_slug}-profile",
    )


@app.local_entrypoint()
def run_guided_proxy_search(
    report_path: str,
    sensitivity_profile_path: str,
    target_budget_gb: float,
    contract_path: str = "configs/experiment_contract_27b_9b.json",
    grouping: str = "per_block_component",
    sensitivity_field: str = "combined_sensitivity",
    population_size: int = 32,
    generations: int = 15,
    elite_count: int = 4,
    tournament_size: int = 3,
    mutation_rate: float = 0.1,
    top_k: int = 5,
    seed: int = 0,
    output_name: str = "",
) -> None:
    from ta_mpq.search import run_proxy_search_from_report

    contract = load_contract(PROJECT_ROOT / contract_path)
    result = run_proxy_search_from_report(
        report_path=PROJECT_ROOT / report_path,
        target_budget_gb=target_budget_gb,
        allowed_bits=contract.quantization_bits,
        grouping=grouping,
        sensitivity_profile_path=PROJECT_ROOT / sensitivity_profile_path,
        sensitivity_field=sensitivity_field,
        population_size=population_size,
        generations=generations,
        elite_count=elite_count,
        tournament_size=tournament_size,
        mutation_rate=mutation_rate,
        top_k=top_k,
        seed=seed,
    )
    effective_output_name = output_name or (
        f"{contract.name}-{grouping}-{Path(sensitivity_profile_path).stem}-guided-search"
    )
    save_summary(
        PROJECT_ROOT / "outputs" / "search" / f"{_slugify(effective_output_name)}.json",
        result.to_dict(),
    )


@app.local_entrypoint()
def run_budgeted_bf16_allocator_search(
    report_path: str,
    contract_path: str = "configs/experiment_contract_27b_9b_math500.json",
    target_budget_gb: float = 0.0,
    grouping: str = "per_block_component",
    group_value_prior_path: str = "",
    sensitivity_profile_path: str = "",
    sensitivity_field: str = "combined_sensitivity",
    bf16_candidate_fraction: float = 0.15,
    bf16_rescue_scale: float = 0.5,
    bf16_sensitivity_weight: float = 0.35,
    output_name: str = "",
    export_dir: str = "",
) -> None:
    from ta_mpq.policy_export import export_top_candidates
    from ta_mpq.search import (
        build_search_groups,
        layer_stats_from_report,
        resolve_group_value_scores,
        resolve_sensitivity_overrides,
        run_budgeted_bf16_allocator,
    )

    contract = load_contract(PROJECT_ROOT / contract_path)
    report_payload = _load_json(PROJECT_ROOT / report_path)
    group_value_prior_payload = (
        _load_json(PROJECT_ROOT / group_value_prior_path) if group_value_prior_path else None
    )
    sensitivity_profile_payload = (
        _load_json(PROJECT_ROOT / sensitivity_profile_path) if sensitivity_profile_path else None
    )
    resolved_target_budget_gb = target_budget_gb or (contract.search_target_budget_gb or 0.0)
    if resolved_target_budget_gb <= 0:
        raise ValueError("target_budget_gb must be positive")

    layer_stats = layer_stats_from_report(report_payload)
    groups = build_search_groups(
        layer_stats,
        grouping=grouping,
        sensitivity_overrides=resolve_sensitivity_overrides(
            layer_stats=layer_stats,
            grouping=grouping,
            sensitivity_profile_payload=sensitivity_profile_payload,
            field=sensitivity_field,
        ),
    )
    fixed_assignments = _resolve_sprint_fixed_assignments(groups)
    group_value_scores = resolve_group_value_scores(
        groups,
        group_value_prior_payload,
        layer_stats=layer_stats,
        target_grouping=grouping,
    )
    result, manifest = run_budgeted_bf16_allocator(
        groups=groups,
        target_budget_gb=resolved_target_budget_gb,
        allowed_bits=(4, 8, 16),
        grouping=grouping,
        group_value_scores=group_value_scores,
        fixed_assignments=fixed_assignments,
        bf16_candidate_fraction=bf16_candidate_fraction,
        bf16_rescue_scale=bf16_rescue_scale,
        bf16_sensitivity_weight=bf16_sensitivity_weight,
    )

    effective_output_name = output_name or (
        f"{contract.name}-{Path(report_path).stem}-budgeted-bf16-allocator"
    )
    output_slug = _slugify(effective_output_name)
    search_output_path = PROJECT_ROOT / "outputs" / "search" / f"{output_slug}.json"
    manifest_output_path = PROJECT_ROOT / "outputs" / "search" / f"{output_slug}-manifest.json"
    save_summary(search_output_path, result.to_dict())
    manifest.update(
        {
            "report_path": report_path,
            "contract_path": contract_path,
            "group_value_prior_path": group_value_prior_path or None,
            "sensitivity_profile_path": sensitivity_profile_path or None,
            "sensitivity_field": sensitivity_field,
            "fixed_assignments": fixed_assignments,
        }
    )
    save_summary(manifest_output_path, manifest)

    effective_export_dir = export_dir or (
        str(PROJECT_ROOT / "outputs" / "policies" / output_slug)
    )
    export_top_candidates(
        report_path=PROJECT_ROOT / report_path,
        search_result_path=search_output_path,
        output_dir=effective_export_dir,
        top_k=1,
    )


@app.local_entrypoint()
def build_surrogate_dataset(
    manifest_path: str,
    target_metric: str = "",
    contract_path: str = "configs/experiment_contract_27b_9b_math500.json",
    uniform_baseline_bit_width: int = 0,
    output_name: str = "",
) -> None:
    from ta_mpq.surrogate import (
        build_surrogate_dataset_from_manifest,
        resolve_manifest_paths,
    )

    contract = load_contract(PROJECT_ROOT / contract_path)
    manifest_full_path = PROJECT_ROOT / manifest_path
    manifest_payload = resolve_manifest_paths(
        _load_json(manifest_full_path),
        base_dir=PROJECT_ROOT,
    )
    resolved_target_metric = target_metric or contract.surrogate_target_metric or "accuracy"
    resolved_uniform_baseline_bit_width = (
        uniform_baseline_bit_width
        if uniform_baseline_bit_width > 0
        else (contract.surrogate_uniform_baseline_bit_width or 0)
    )
    dataset = build_surrogate_dataset_from_manifest(
        manifest_payload=manifest_payload,
        target_metric=resolved_target_metric,
        uniform_baseline_bit_width=(
            resolved_uniform_baseline_bit_width if resolved_uniform_baseline_bit_width > 0 else None
        ),
    )
    effective_output_name = output_name or (
        f"{Path(manifest_path).stem}-{resolved_target_metric}-dataset"
    )
    save_summary(
        PROJECT_ROOT / "outputs" / "surrogate" / f"{_slugify(effective_output_name)}.json",
        dataset.to_dict(),
    )


@app.local_entrypoint()
def run_surrogate_training(
    dataset_path: str,
    target_metric: str = "",
    output_name: str = "",
    random_seed: int = 0,
    ensemble_size: int = 8,
) -> None:
    dataset_payload = _load_json(PROJECT_ROOT / dataset_path)
    resolved_target_metric = target_metric or str(dataset_payload.get("target_metric", "accuracy"))
    training_result = train_surrogate_remote.remote(
        dataset_payload,
        target_metric=resolved_target_metric,
        random_seed=random_seed,
        ensemble_size=ensemble_size,
    )
    effective_output_name = output_name or (
        f"{Path(dataset_path).stem}-{resolved_target_metric}-surrogate"
    )
    model_json = str(training_result.pop("model_json", ""))
    save_summary(
        PROJECT_ROOT / "outputs" / "surrogate" / f"{_slugify(effective_output_name)}.json",
        training_result,
    )
    if model_json:
        model_path = (
            PROJECT_ROOT / "outputs" / "surrogate" / f"{_slugify(effective_output_name)}-model.json"
        )
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_text(model_json, encoding="utf-8")


@app.local_entrypoint()
def build_ablation_adjusted_group_value_prior(
    base_prior_path: str,
    ablation_profile_paths: str,
    output_name: str = "",
    zero_drop_tolerance: float = 1e-9,
    no_evidence_four_bit_penalty: float = 0.05,
    no_evidence_sixteen_bit_cap: float = 0.01,
) -> None:
    from ta_mpq.surrogate import build_ablation_adjusted_group_value_prior

    base_prior_payload = _load_json(PROJECT_ROOT / base_prior_path)
    resolved_profile_paths = [
        PROJECT_ROOT / path.strip()
        for path in ablation_profile_paths.split(",")
        if path.strip()
    ]
    if not resolved_profile_paths:
        raise ValueError("ablation_profile_paths must include at least one profile path")

    ablation_profiles = [_load_json(path) for path in resolved_profile_paths]
    adjusted_prior = build_ablation_adjusted_group_value_prior(
        base_prior_payload=base_prior_payload,
        ablation_profile_payloads=ablation_profiles,
        zero_drop_tolerance=zero_drop_tolerance,
        no_evidence_four_bit_penalty=no_evidence_four_bit_penalty,
        no_evidence_sixteen_bit_cap=no_evidence_sixteen_bit_cap,
    )
    effective_output_name = output_name or f"{Path(base_prior_path).stem}-ablation-adjusted"
    save_summary(
        PROJECT_ROOT / "outputs" / "surrogate" / f"{_slugify(effective_output_name)}.json",
        adjusted_prior,
    )


@app.local_entrypoint()
def run_surrogate_guided_search(
    report_path: str,
    surrogate_summary_path: str,
    surrogate_model_path: str,
    target_budget_gb: float = 0.0,
    allowed_bits: str = "",
    contract_path: str = "configs/experiment_contract_27b_9b_math500.json",
    group_value_prior_path: str = "",
    sensitivity_profile_path: str = "",
    sensitivity_field: str = "combined_sensitivity",
    grouping: str = "per_block_component",
    population_size: int = 32,
    generations: int = 15,
    elite_count: int = 4,
    tournament_size: int = 3,
    mutation_rate: float = 0.1,
    uncertainty_penalty: float = 0.5,
    reference_accuracy: float = REFERENCE_ACCURACY_SENTINEL,
    top_k: int = 5,
    seed: int = 0,
    output_name: str = "",
    export_dir: str = "",
) -> None:
    from ta_mpq.policy_export import export_top_candidates

    contract = load_contract(PROJECT_ROOT / contract_path)
    report_payload = _load_json(PROJECT_ROOT / report_path)
    surrogate_summary_payload = _load_json(PROJECT_ROOT / surrogate_summary_path)
    surrogate_model_json = (PROJECT_ROOT / surrogate_model_path).read_text(encoding="utf-8")
    group_value_prior_payload = (
        _load_json(PROJECT_ROOT / group_value_prior_path) if group_value_prior_path else None
    )
    sensitivity_profile_payload = (
        _load_json(PROJECT_ROOT / sensitivity_profile_path) if sensitivity_profile_path else None
    )
    resolved_target_budget_gb = target_budget_gb or (contract.search_target_budget_gb or 0.0)
    if resolved_target_budget_gb <= 0:
        raise ValueError("target_budget_gb must be positive")
    normalized_allowed_bits = (
        tuple(sorted({int(part.strip()) for part in allowed_bits.split(",") if part.strip()}))
        if allowed_bits
        else tuple(contract.quantization_bits)
    )
    resolved_target_metric = str(
        surrogate_summary_payload.get("target_metric")
        or contract.surrogate_target_metric
        or "accuracy"
    )
    resolved_reference_accuracy = _resolve_reference_accuracy(
        reference_accuracy,
        resolved_target_metric,
    )

    search_result = run_surrogate_search_remote.remote(
        report_payload=report_payload,
        surrogate_summary_payload=surrogate_summary_payload,
        surrogate_model_json=surrogate_model_json,
        target_budget_gb=resolved_target_budget_gb,
        allowed_bits=normalized_allowed_bits,
        grouping=grouping,
        group_value_prior_payload=group_value_prior_payload,
        sensitivity_profile_payload=sensitivity_profile_payload,
        sensitivity_field=sensitivity_field,
        population_size=population_size,
        generations=generations,
        elite_count=elite_count,
        tournament_size=tournament_size,
        mutation_rate=mutation_rate,
        uncertainty_penalty=uncertainty_penalty,
        reference_accuracy=resolved_reference_accuracy,
        top_k=top_k,
        seed=seed,
    )

    effective_output_name = output_name or (
        f"{Path(surrogate_summary_path).stem}-{Path(report_path).stem}-search"
    )
    search_output_path = (
        PROJECT_ROOT / "outputs" / "search" / f"{_slugify(effective_output_name)}.json"
    )
    save_summary(search_output_path, search_result)

    effective_export_dir = export_dir or (
        str(PROJECT_ROOT / "outputs" / "policies" / _slugify(effective_output_name))
    )
    export_top_candidates(
        report_path=PROJECT_ROOT / report_path,
        search_result_path=search_output_path,
        output_dir=effective_export_dir,
        top_k=top_k,
    )


@app.local_entrypoint()
def run_hierarchical_uniform8_search(
    report_path: str,
    surrogate_summary_path: str = "",
    surrogate_model_path: str = "",
    target_budget_gb: float = 0.0,
    allowed_bits: str = "",
    contract_path: str = "configs/experiment_contract_27b_9b.json",
    group_value_prior_path: str = "",
    sensitivity_profile_path: str = "",
    sensitivity_field: str = "combined_sensitivity",
    window_size: int = 4,
    max_promoted_fine_groups: int = 160,
    coarse_population_size: int = 80,
    coarse_generations: int = 24,
    fine_population_size: int = 64,
    fine_generations: int = 24,
    config_group_sizes: str = "32,64,128",
    config_symmetric_options: str = "true,false",
    max_tunable_config_groups: int = 48,
    config_population_size: int = 40,
    config_generations: int = 16,
    config_seed_candidate_count: int = 3,
    elite_count: int = 4,
    tournament_size: int = 3,
    mutation_rate: float = 0.1,
    uncertainty_penalty: float = 0.5,
    reference_accuracy: float = REFERENCE_ACCURACY_SENTINEL,
    top_k: int = 5,
    seed: int = 0,
    output_name: str = "",
    export_dir: str = "",
) -> None:
    from ta_mpq.policy_export import export_top_candidates
    from ta_mpq.search import (
        SearchCandidate,
        build_hierarchical_promotion_manifest,
        build_search_groups,
        expand_group_assignments,
        layer_stats_from_report,
        refine_candidate_quantization_configs,
        resolve_group_value_scores,
        resolve_sensitivity_overrides,
        run_proxy_evolution_search,
    )

    if window_size != 4:
        raise ValueError("window_size is fixed to 4 for v1")

    contract = load_contract(PROJECT_ROOT / contract_path)
    report_payload = _load_json(PROJECT_ROOT / report_path)
    if bool(surrogate_summary_path) != bool(surrogate_model_path):
        raise ValueError(
            "surrogate_summary_path and surrogate_model_path must both be provided "
            "or both be omitted"
        )
    surrogate_summary_payload = (
        _load_json(PROJECT_ROOT / surrogate_summary_path) if surrogate_summary_path else None
    )
    surrogate_model_json = (
        (PROJECT_ROOT / surrogate_model_path).read_text(encoding="utf-8")
        if surrogate_model_path
        else ""
    )
    group_value_prior_payload = (
        _load_json(PROJECT_ROOT / group_value_prior_path) if group_value_prior_path else None
    )
    sensitivity_profile_payload = (
        _load_json(PROJECT_ROOT / sensitivity_profile_path) if sensitivity_profile_path else None
    )
    resolved_target_budget_gb = target_budget_gb or (contract.search_target_budget_gb or 0.0)
    if resolved_target_budget_gb <= 0:
        raise ValueError("target_budget_gb must be positive")
    normalized_allowed_bits = (
        tuple(sorted({int(part.strip()) for part in allowed_bits.split(",") if part.strip()}))
        if allowed_bits
        else tuple(contract.quantization_bits)
    )
    normalized_config_group_sizes = tuple(
        sorted({int(part.strip()) for part in config_group_sizes.split(",") if part.strip()})
    )
    normalized_config_symmetric_options = tuple(
        dict.fromkeys(
            part.strip().lower() not in {"false", "0", "no"}
            for part in config_symmetric_options.split(",")
            if part.strip()
        )
    )
    resolved_target_metric = str(
        (surrogate_summary_payload or {}).get("target_metric")
        or contract.surrogate_target_metric
        or "accuracy"
    )
    resolved_reference_accuracy = _resolve_reference_accuracy(
        reference_accuracy,
        resolved_target_metric,
    )

    layer_stats = layer_stats_from_report(report_payload)
    coarse_grouping = "per_block_window_component"
    fine_grouping = "per_block_component"
    coarse_groups = build_search_groups(
        layer_stats,
        grouping=coarse_grouping,
        sensitivity_overrides=resolve_sensitivity_overrides(
            layer_stats=layer_stats,
            grouping=coarse_grouping,
            sensitivity_profile_payload=sensitivity_profile_payload,
            field=sensitivity_field,
        ),
    )
    fine_groups = build_search_groups(
        layer_stats,
        grouping=fine_grouping,
        sensitivity_overrides=resolve_sensitivity_overrides(
            layer_stats=layer_stats,
            grouping=fine_grouping,
            sensitivity_profile_payload=sensitivity_profile_payload,
            field=sensitivity_field,
        ),
    )
    coarse_group_value_scores = resolve_group_value_scores(
        coarse_groups,
        group_value_prior_payload,
        layer_stats=layer_stats,
        target_grouping=coarse_grouping,
    )
    fine_group_value_scores = resolve_group_value_scores(
        fine_groups,
        group_value_prior_payload,
        layer_stats=layer_stats,
        target_grouping=fine_grouping,
    )
    coarse_result = run_proxy_evolution_search(
        groups=coarse_groups,
        target_budget_gb=resolved_target_budget_gb,
        allowed_bits=normalized_allowed_bits,
        grouping=coarse_grouping,
        population_size=coarse_population_size,
        generations=coarse_generations,
        elite_count=min(elite_count, coarse_population_size),
        tournament_size=tournament_size,
        mutation_rate=mutation_rate,
        top_k=8,
        seed=seed,
    )
    if not coarse_result.top_candidates:
        raise RuntimeError("Coarse proxy search did not produce any candidates")

    best_coarse_assignments = coarse_result.top_candidates[0].bits_dict()
    expanded_best_assignments = expand_group_assignments(
        best_coarse_assignments,
        target_groups=fine_groups,
        source_grouping=coarse_grouping,
    )
    promotion_manifest = build_hierarchical_promotion_manifest(
        coarse_groups=coarse_groups,
        coarse_candidates=list(coarse_result.top_candidates),
        fine_groups=fine_groups,
        coarse_group_value_scores=coarse_group_value_scores,
        source_grouping=coarse_grouping,
        max_promoted_fine_groups=max_promoted_fine_groups,
    )
    promotion_manifest["best_coarse_candidate_bits"] = best_coarse_assignments
    promotion_manifest["expanded_best_fine_assignments"] = expanded_best_assignments

    promoted_fine_group_names = set(promotion_manifest["promoted_fine_group_names"])
    frozen_fine_group_names = set(promotion_manifest["frozen_fine_group_names"])
    active_search_groups = [
        group for group in fine_groups if group.name in promoted_fine_group_names
    ]
    fixed_assignments = {
        group_name: expanded_best_assignments[group_name]
        for group_name in frozen_fine_group_names
    }

    effective_output_name = output_name or (
        f"{Path(surrogate_summary_path).stem}-{Path(report_path).stem}-hierarchical-search"
    )
    output_slug = _slugify(effective_output_name)
    coarse_output_path = PROJECT_ROOT / "outputs" / "search" / f"{output_slug}-coarse.json"
    promotion_manifest_path = (
        PROJECT_ROOT / "outputs" / "search" / f"{output_slug}-promotion-manifest.json"
    )
    config_refinement_manifest_path = (
        PROJECT_ROOT / "outputs" / "search" / f"{output_slug}-config-refinement.json"
    )
    search_output_path = PROJECT_ROOT / "outputs" / "search" / f"{output_slug}.json"
    save_summary(coarse_output_path, coarse_result.to_dict())
    save_summary(promotion_manifest_path, promotion_manifest)

    if surrogate_summary_payload is not None:
        fine_result = run_surrogate_search_remote.remote(
            report_payload=report_payload,
            surrogate_summary_payload=surrogate_summary_payload,
            surrogate_model_json=surrogate_model_json,
            target_budget_gb=resolved_target_budget_gb,
            allowed_bits=normalized_allowed_bits,
            grouping=fine_grouping,
            group_value_prior_payload=group_value_prior_payload,
            sensitivity_profile_payload=sensitivity_profile_payload,
            sensitivity_field=sensitivity_field,
            population_size=fine_population_size,
            generations=fine_generations,
            elite_count=min(elite_count, fine_population_size),
            tournament_size=tournament_size,
            mutation_rate=mutation_rate,
            uncertainty_penalty=uncertainty_penalty,
            reference_accuracy=resolved_reference_accuracy,
            top_k=top_k,
            seed=seed,
            search_group_names=[group.name for group in active_search_groups],
            fixed_assignments=fixed_assignments,
            extra_seed_assignments=[
                {
                    "provenance": "expanded_best_coarse_candidate",
                    "assignments": expanded_best_assignments,
                }
            ],
        )
        fine_result["fine_search_mode"] = "surrogate"
    else:
        fine_result = run_proxy_evolution_search(
            groups=fine_groups,
            target_budget_gb=resolved_target_budget_gb,
            allowed_bits=normalized_allowed_bits,
            grouping=fine_grouping,
            population_size=fine_population_size,
            generations=fine_generations,
            elite_count=min(elite_count, fine_population_size),
            tournament_size=tournament_size,
            mutation_rate=mutation_rate,
            top_k=top_k,
            seed=seed,
            search_groups=active_search_groups,
            fixed_assignments=fixed_assignments,
            extra_seed_assignments=[
                ("expanded_best_coarse_candidate", expanded_best_assignments)
            ],
        ).to_dict()
        fine_result["fine_search_mode"] = "proxy"
    fine_top_candidates = [
        SearchCandidate.from_dict(candidate_payload)
        for candidate_payload in fine_result.get("top_candidates", [])
    ]
    config_refinement = refine_candidate_quantization_configs(
        groups=fine_groups,
        layer_stats=layer_stats,
        base_candidates=fine_top_candidates,
        group_value_scores=fine_group_value_scores,
        allowed_group_names=promoted_fine_group_names,
        group_size_options=normalized_config_group_sizes,
        symmetric_options=normalized_config_symmetric_options,
        max_tunable_groups=max_tunable_config_groups,
        population_size=config_population_size,
        generations=config_generations,
        top_k=top_k,
        seed=seed,
        seed_candidate_count=config_seed_candidate_count,
    )
    save_summary(config_refinement_manifest_path, config_refinement)
    fine_result["top_candidates"] = list(config_refinement["top_candidates"])
    fine_result["config_refinement"] = {
        "manifest_path": str(config_refinement_manifest_path),
        "group_size_options": list(normalized_config_group_sizes),
        "symmetric_options": list(normalized_config_symmetric_options),
        "max_tunable_groups": max_tunable_config_groups,
        "population_size": config_population_size,
        "generations": config_generations,
        "seed_candidate_count": config_seed_candidate_count,
    }
    save_summary(search_output_path, fine_result)

    effective_export_dir = export_dir or (
        str(PROJECT_ROOT / "outputs" / "policies" / output_slug)
    )
    export_top_candidates(
        report_path=PROJECT_ROOT / report_path,
        search_result_path=search_output_path,
        output_dir=effective_export_dir,
        top_k=top_k,
    )


@app.local_entrypoint()
def run_surrogate_closed_loop(
    executed_manifest_path: str,
    report_path: str,
    sensitivity_profile_path: str,
    contract_path: str = "configs/experiment_contract_27b_9b_math500.json",
    target_budget_gb: float = 0.0,
    target_metric: str = "",
    allowed_bits: str = "",
    grouping: str = "per_block_component",
    sensitivity_field: str = "combined_sensitivity",
    population_size: int = 32,
    generations: int = 15,
    elite_count: int = 4,
    tournament_size: int = 3,
    mutation_rate: float = 0.1,
    uncertainty_penalty: float = 0.5,
    reference_accuracy: float = REFERENCE_ACCURACY_SENTINEL,
    acquisition_exploration_weight: float = 0.5,
    acquisition_diversity_weight: float = 0.15,
    top_k: int = 5,
    seed: int = 0,
    ensemble_size: int = 8,
    iterations: int = 1,
    candidates_per_iteration: int = 1,
    calibration_limit: int = 4,
    dev_limit: int = 25,
    eval_max_new_tokens: int = 32,
    policy_source: str = "llmcompressor",
    backend_variant: str = "source",
    output_name: str = "",
    output_manifest_path: str = "",
) -> None:
    from ta_mpq.closed_loop import (
        append_record_if_novel,
        best_record_by_accuracy,
        build_executed_record,
        select_novel_candidates,
        to_relative_path,
    )
    from ta_mpq.policy_export import export_top_candidates
    from ta_mpq.surrogate import (
        build_surrogate_dataset_from_manifest,
        build_group_value_prior_from_dataset,
        resolve_manifest_paths,
    )

    contract = load_contract(PROJECT_ROOT / contract_path)
    loop_slug = _slugify(output_name or f"{contract.name}-{contract.task_name}-closed-loop")
    resolved_target_metric = target_metric or contract.surrogate_target_metric or "accuracy"
    resolved_target_budget_gb = target_budget_gb or (contract.search_target_budget_gb or 0.0)
    if resolved_target_budget_gb <= 0:
        raise ValueError("target_budget_gb must be positive")
    normalized_allowed_bits = (
        tuple(sorted({int(part.strip()) for part in allowed_bits.split(",") if part.strip()}))
        if allowed_bits
        else tuple(contract.quantization_bits)
    )
    resolved_reference_accuracy = _resolve_reference_accuracy(
        reference_accuracy,
        resolved_target_metric,
    )
    executed_manifest_payload = _load_json(PROJECT_ROOT / executed_manifest_path)
    manifest_output_path = (
        PROJECT_ROOT / output_manifest_path
        if output_manifest_path
        else PROJECT_ROOT / "outputs" / "closed_loop" / f"{loop_slug}-executed-manifest.json"
    )
    manifest_output_path.parent.mkdir(parents=True, exist_ok=True)

    base_report_payload = _load_json(PROJECT_ROOT / report_path)
    sensitivity_profile_payload = _load_json(PROJECT_ROOT / sensitivity_profile_path)
    iteration_summaries: list[dict[str, Any]] = []

    for iteration in range(1, iterations + 1):
        iteration_slug = f"{loop_slug}-iter-{iteration:02d}"

        dataset = build_surrogate_dataset_from_manifest(
            manifest_payload=resolve_manifest_paths(
                executed_manifest_payload,
                base_dir=PROJECT_ROOT,
            ),
            target_metric=resolved_target_metric,
            uniform_baseline_bit_width=contract.surrogate_uniform_baseline_bit_width,
        )
        dataset_path = PROJECT_ROOT / "outputs" / "surrogate" / f"{iteration_slug}-dataset.json"
        save_summary(dataset_path, dataset.to_dict())
        group_value_prior = build_group_value_prior_from_dataset(
            dataset.to_dict(),
            target_metric=resolved_target_metric,
        )
        group_value_prior_path = (
            PROJECT_ROOT / "outputs" / "surrogate" / f"{iteration_slug}-group-value-prior.json"
        )
        save_summary(group_value_prior_path, group_value_prior)

        training_result = train_surrogate_remote.remote(
            dataset.to_dict(),
            target_metric=resolved_target_metric,
            random_seed=seed + iteration - 1,
            ensemble_size=ensemble_size,
        )
        model_json = str(training_result.pop("model_json", ""))
        surrogate_summary_path = (
            PROJECT_ROOT / "outputs" / "surrogate" / f"{iteration_slug}-surrogate.json"
        )
        save_summary(surrogate_summary_path, training_result)
        surrogate_model_path = (
            PROJECT_ROOT / "outputs" / "surrogate" / f"{iteration_slug}-surrogate-model.json"
        )
        if model_json:
            surrogate_model_path.write_text(model_json, encoding="utf-8")

        search_result = run_surrogate_search_remote.remote(
            report_payload=base_report_payload,
            surrogate_summary_payload=training_result,
            surrogate_model_json=model_json,
            target_budget_gb=resolved_target_budget_gb,
            allowed_bits=normalized_allowed_bits,
            grouping=grouping,
            group_value_prior_payload=group_value_prior,
            sensitivity_profile_payload=sensitivity_profile_payload,
            sensitivity_field=sensitivity_field,
            population_size=population_size,
            generations=generations,
            elite_count=elite_count,
            tournament_size=tournament_size,
            mutation_rate=mutation_rate,
            uncertainty_penalty=uncertainty_penalty,
            reference_accuracy=resolved_reference_accuracy,
            top_k=top_k,
            seed=seed + iteration - 1,
        )
        search_output_path = PROJECT_ROOT / "outputs" / "search" / f"{iteration_slug}.json"
        save_summary(search_output_path, search_result)

        export_dir = PROJECT_ROOT / "outputs" / "policies" / iteration_slug
        export_top_candidates(
            report_path=PROJECT_ROOT / report_path,
            search_result_path=search_output_path,
            output_dir=export_dir,
            top_k=top_k,
        )
        candidate_manifest_payload = _load_json(export_dir / "manifest.json")
        selected_candidates = select_novel_candidates(
            executed_manifest_payload=executed_manifest_payload,
            candidate_manifest_payload=candidate_manifest_payload,
            base_dir=PROJECT_ROOT,
            policy_source=policy_source,
            limit=candidates_per_iteration,
            exploration_weight=acquisition_exploration_weight,
            diversity_weight=acquisition_diversity_weight,
        )

        executed_records: list[dict[str, Any]] = []
        for selected_index, selected_candidate in enumerate(selected_candidates, start=1):
            candidate_path = Path(selected_candidate["path"])
            candidate_payload = _load_json(candidate_path)
            if policy_source == "project":
                policy_payload = candidate_payload["project_policy"]
            elif policy_source == "llmcompressor":
                policy_payload = candidate_payload["backend_projections"]["llmcompressor"][
                    "projected_policy"
                ]
            else:
                raise ValueError(f"Unknown policy_source: {policy_source}")

            policy_id = f"{iteration_slug}-{candidate_path.stem}"
            policy_label = f"{iteration_slug}-{candidate_path.stem}-{policy_source}"
            report_output_path = (
                PROJECT_ROOT
                / "outputs"
                / "feasibility"
                / f"{contract.name}-{_slugify(policy_label)}-{backend_variant}-report.json"
            )
            if report_output_path.exists():
                report = _load_json(report_output_path)
            else:
                if backend_variant == "stable":
                    report = probe_mixed_precision_feasibility.remote(
                        contract.to_dict(),
                        calibration_limit=calibration_limit,
                        dry_run=False,
                        policy_payload=policy_payload,
                        policy_label=policy_label,
                        precomputed_report=base_report_payload,
                    )
                elif backend_variant == "source":
                    report = probe_mixed_precision_feasibility_source.remote(
                        contract.to_dict(),
                        calibration_limit=calibration_limit,
                        dry_run=False,
                        policy_payload=policy_payload,
                        policy_label=policy_label,
                        precomputed_report=base_report_payload,
                    )
                else:
                    raise ValueError(f"Unknown backend_variant: {backend_variant}")

                if (
                    backend_variant == "source"
                    and report.get("status") == "smoke_test_failed"
                    and report.get("artifact_dir_exists")
                    and report.get("smoke_test_error_type") == "OutOfMemoryError"
                ):
                    clean_smoke_test = smoke_test_quantized_artifact_source.remote(
                        artifact_dir=report["output_dir"],
                        tokenizer_source=contract.compressed_source_model_id,
                        max_new_tokens=4,
                    )
                    report["same_process_smoke_test"] = {
                        "status": "failed",
                        "error_type": report.get("smoke_test_error_type"),
                        "error_message": report.get("smoke_test_error_message"),
                    }
                    report["smoke_test"] = clean_smoke_test
                    report["smoke_test_mode"] = "clean_process_retry"
                    report["quantized_model_runnable"] = True
                    report["status"] = "smoke_test_succeeded"

                report["candidate_path"] = to_relative_path(candidate_path, PROJECT_ROOT)
                report["policy_source"] = policy_source
                report["contract_path"] = contract_path
                save_summary(report_output_path, report)

            artifact_dir = report.get("output_dir")
            if not artifact_dir:
                raise RuntimeError(f"Quantization report missing output_dir for {candidate_path}")

            eval_prefix = f"{iteration_slug}-{candidate_path.stem}-dev"
            quantized_eval_path = (
                PROJECT_ROOT / "outputs" / "evaluations" / f"{_slugify(eval_prefix)}-quantized.json"
            )
            native_eval_path = (
                PROJECT_ROOT / "outputs" / "evaluations" / f"{_slugify(eval_prefix)}-native.json"
            )
            if quantized_eval_path.exists() and native_eval_path.exists():
                quantized_summary = _load_json(quantized_eval_path)
                native_summary = _load_json(native_eval_path)
            else:
                quantized_summary = evaluate_task_source_model.remote(
                    model_ref=artifact_dir,
                    tokenizer_source=contract.compressed_source_model_id,
                    model_label=f"{contract.compressed_source_model_id}-quantized",
                    task_name=contract.task_name,
                    limit=dev_limit,
                    max_new_tokens=eval_max_new_tokens,
                    load_dtype="auto",
                )
                native_summary = evaluate_task_source_model.remote(
                    model_ref=contract.native_baseline_model_id,
                    tokenizer_source=contract.native_baseline_model_id,
                    model_label=contract.native_baseline_model_id,
                    task_name=contract.task_name,
                    limit=dev_limit,
                    max_new_tokens=eval_max_new_tokens,
                    load_dtype="bfloat16",
                )
                save_summary(quantized_eval_path, quantized_summary)
                save_summary(native_eval_path, native_summary)

            record = build_executed_record(
                policy_id=policy_id,
                task_name=contract.task_name,
                candidate_path=to_relative_path(candidate_path, PROJECT_ROOT),
                report_path=to_relative_path(report_output_path, PROJECT_ROOT),
                evaluation_path=to_relative_path(quantized_eval_path, PROJECT_ROOT),
                sensitivity_profile_path=to_relative_path(
                    PROJECT_ROOT / sensitivity_profile_path,
                    PROJECT_ROOT,
                ),
                sensitivity_field=sensitivity_field,
                provenance="surrogate_guided_closed_loop",
                search_result_path=to_relative_path(search_output_path, PROJECT_ROOT),
                surrogate_summary_path=to_relative_path(surrogate_summary_path, PROJECT_ROOT),
            )
            executed_manifest_payload = append_record_if_novel(
                manifest_payload=executed_manifest_payload,
                record_payload=record,
                base_dir=PROJECT_ROOT,
                policy_source=policy_source,
            )
            executed_records.append(
                {
                    "policy_id": policy_id,
                    "candidate_path": record["candidate_path"],
                    "report_path": record["report_path"],
                    "evaluation_path": record["evaluation_path"],
                    "quantized_accuracy": float(quantized_summary["accuracy"]),
                    "native_accuracy": float(native_summary["accuracy"]),
                    "artifact_dir": artifact_dir,
                    "prediction_uncertainty": selected_candidate.get("prediction_uncertainty"),
                }
            )

        save_summary(manifest_output_path, executed_manifest_payload)
        iteration_summaries.append(
            {
                "iteration": iteration,
                "dataset_path": to_relative_path(dataset_path, PROJECT_ROOT),
                "surrogate_summary_path": to_relative_path(surrogate_summary_path, PROJECT_ROOT),
                "surrogate_model_path": to_relative_path(surrogate_model_path, PROJECT_ROOT),
                "group_value_prior_path": to_relative_path(group_value_prior_path, PROJECT_ROOT),
                "search_result_path": to_relative_path(search_output_path, PROJECT_ROOT),
                "policy_manifest_path": to_relative_path(export_dir / "manifest.json", PROJECT_ROOT),
                "selected_candidates": selected_candidates,
                "executed_records": executed_records,
            }
        )
        if not selected_candidates:
            break

    best_closed_loop_record = best_record_by_accuracy(
        executed_manifest_payload,
        base_dir=PROJECT_ROOT,
        task_name=contract.task_name,
        provenance_prefix="surrogate_guided_closed_loop",
    )
    best_surrogate_record = best_record_by_accuracy(
        executed_manifest_payload,
        base_dir=PROJECT_ROOT,
        task_name=contract.task_name,
        provenance_prefix="surrogate_guided",
    )

    summary_payload = {
        "loop_name": loop_slug,
        "contract_path": contract_path,
        "task_name": contract.task_name,
        "iterations_requested": iterations,
        "iterations_completed": len(iteration_summaries),
        "target_metric": resolved_target_metric,
        "target_budget_gb": resolved_target_budget_gb,
        "ensemble_size": ensemble_size,
        "uncertainty_penalty": uncertainty_penalty,
        "reference_accuracy": resolved_reference_accuracy,
        "acquisition_exploration_weight": acquisition_exploration_weight,
        "acquisition_diversity_weight": acquisition_diversity_weight,
        "dev_limit": dev_limit,
        "eval_max_new_tokens": eval_max_new_tokens,
        "executed_manifest_path": to_relative_path(manifest_output_path, PROJECT_ROOT),
        "best_closed_loop_record": best_closed_loop_record,
        "best_surrogate_record": best_surrogate_record,
        "large_sample_ready": bool(best_closed_loop_record),
        "iteration_summaries": iteration_summaries,
    }
    save_summary(
        PROJECT_ROOT / "outputs" / "closed_loop" / f"{loop_slug}-summary.json",
        summary_payload,
    )


@app.local_entrypoint()
def run_large_sample_eval_from_closed_loop(
    closed_loop_summary_path: str,
    contract_path: str = "configs/experiment_contract_27b_9b_math500.json",
    summary_record_key: str = "best_closed_loop_record",
    limit: int = 100,
    max_new_tokens: int = 32,
    output_prefix: str = "",
    reference_report_path: str = "",
    reference_label: str = "",
) -> None:
    from ta_mpq.closed_loop import artifact_dir_from_record, to_relative_path

    contract = load_contract(PROJECT_ROOT / contract_path)
    summary_payload = _load_json(PROJECT_ROOT / closed_loop_summary_path)
    selected_record = summary_payload.get(summary_record_key)
    if not selected_record:
        raise ValueError(
            f"Summary {closed_loop_summary_path} does not contain {summary_record_key}"
        )

    artifact_dir = artifact_dir_from_record(selected_record, base_dir=PROJECT_ROOT)
    if not artifact_dir:
        raise RuntimeError(
            f"Could not resolve artifact_dir from {summary_record_key} in {closed_loop_summary_path}"
        )

    candidate_label = Path(str(selected_record.get("candidate_path", "winner"))).stem
    effective_output_prefix = output_prefix or (
        f"{Path(closed_loop_summary_path).stem}-{candidate_label}-limit-{limit}"
    )

    quantized_summary = evaluate_task_source_model.remote(
        model_ref=artifact_dir,
        tokenizer_source=contract.compressed_source_model_id,
        model_label=f"{contract.compressed_source_model_id}-quantized",
        task_name=contract.task_name,
        limit=limit,
        max_new_tokens=max_new_tokens,
        load_dtype="auto",
    )
    native_summary = evaluate_task_source_model.remote(
        model_ref=contract.native_baseline_model_id,
        tokenizer_source=contract.native_baseline_model_id,
        model_label=contract.native_baseline_model_id,
        task_name=contract.task_name,
        limit=limit,
        max_new_tokens=max_new_tokens,
        load_dtype="bfloat16",
    )

    quantized_output_path = (
        PROJECT_ROOT / "outputs" / "evaluations" / f"{_slugify(effective_output_prefix)}-quantized.json"
    )
    native_output_path = (
        PROJECT_ROOT / "outputs" / "evaluations" / f"{_slugify(effective_output_prefix)}-native.json"
    )
    save_summary(quantized_output_path, quantized_summary)
    save_summary(native_output_path, native_summary)

    comparison_summary: dict[str, Any] = {
        "closed_loop_summary_path": closed_loop_summary_path,
        "summary_record_key": summary_record_key,
        "candidate_path": selected_record.get("candidate_path"),
        "artifact_dir": artifact_dir,
        "task_name": contract.task_name,
        "limit": limit,
        "max_new_tokens": max_new_tokens,
        "quantized_evaluation_path": to_relative_path(quantized_output_path, PROJECT_ROOT),
        "native_evaluation_path": to_relative_path(native_output_path, PROJECT_ROOT),
        "quantized_accuracy": float(quantized_summary.get("accuracy", 0.0)),
        "native_accuracy": float(native_summary.get("accuracy", 0.0)),
    }

    if reference_report_path:
        reference_report_payload = _load_json(PROJECT_ROOT / reference_report_path)
        reference_artifact_dir = reference_report_payload.get("output_dir")
        if not reference_artifact_dir:
            raise RuntimeError(f"Reference report {reference_report_path} missing output_dir")
        resolved_reference_label = reference_label or Path(reference_report_path).stem
        reference_summary = evaluate_task_source_model.remote(
            model_ref=reference_artifact_dir,
            tokenizer_source=contract.compressed_source_model_id,
            model_label=resolved_reference_label,
            task_name=contract.task_name,
            limit=limit,
            max_new_tokens=max_new_tokens,
            load_dtype="auto",
        )
        reference_output_path = (
            PROJECT_ROOT
            / "outputs"
            / "evaluations"
            / f"{_slugify(effective_output_prefix)}-{_slugify(resolved_reference_label)}.json"
        )
        save_summary(reference_output_path, reference_summary)
        comparison_summary["reference_report_path"] = reference_report_path
        comparison_summary["reference_evaluation_path"] = to_relative_path(
            reference_output_path,
            PROJECT_ROOT,
        )
        comparison_summary["reference_accuracy"] = float(reference_summary.get("accuracy", 0.0))

    save_summary(
        PROJECT_ROOT
        / "outputs"
        / "closed_loop"
        / f"{_slugify(effective_output_prefix)}-large-sample-summary.json",
        comparison_summary,
    )


def _load_json(path: str | Path) -> dict[str, Any]:
    json_path = Path(path)
    with json_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _slugify(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-") or "default-policy"


def _resolve_reference_accuracy(
    explicit_reference_accuracy: float,
    target_metric: str,
) -> float | None:
    if explicit_reference_accuracy != REFERENCE_ACCURACY_SENTINEL:
        return explicit_reference_accuracy
    if "advantage_over" in target_metric:
        return 0.0
    return None


def _candidate_policy_label(candidate_path: str, policy_source: str) -> str:
    candidate = Path(candidate_path)
    parent_label = _slugify(candidate.parent.name) if candidate.parent.name else "root"
    return f"{parent_label}-{candidate.stem}-{policy_source}"


def _manifest_output_label(path: str | Path) -> str:
    manifest_path = Path(path)
    if manifest_path.stem == "manifest" and manifest_path.parent.name:
        return manifest_path.parent.name
    return manifest_path.stem


def to_relative_path(path: str | Path, base_dir: str | Path) -> str:
    resolved_path = Path(path).resolve()
    resolved_base = Path(base_dir).resolve()
    try:
        return str(resolved_path.relative_to(resolved_base))
    except ValueError:
        return str(resolved_path)


def _task_limit_for_split(split_name: str) -> int:
    normalized = str(split_name or "test").lower()
    if normalized == "dev100":
        return 100
    if normalized == "accept200":
        return 200
    if normalized == "train200":
        return 200
    if normalized == "test300":
        return 300
    if normalized == "first300":
        return 300
    if normalized == "last100":
        return 100
    if normalized == "full500":
        return 500
    return 25


def _resolve_default_max_new_tokens(task_name: str, max_new_tokens: int) -> int:
    if max_new_tokens > 0:
        return int(max_new_tokens)
    normalized_task_name = str(task_name or "").strip().lower()
    if normalized_task_name in {"math500", "math-500"}:
        return 2048
    return 64


def _resolve_sprint_fixed_assignments(groups: list[Any]) -> dict[str, int]:
    fixed_assignments: dict[str, int] = {}
    for group in groups:
        if "lm_head" in str(group.name):
            fixed_assignments[str(group.name)] = 8
    return fixed_assignments


def _assignment_signature(assignments: dict[str, int]) -> tuple[tuple[str, int], ...]:
    return tuple(sorted((str(group_name), int(bit_width)) for group_name, bit_width in assignments.items()))


def _resolve_eval_remote(gpu_type: str):
    if gpu_type == A100_40GB_MODAL_GPU:
        return evaluate_task_source_model_a100
    return evaluate_task_source_model


def _resolve_evalplus_remote(gpu_type: str):
    if gpu_type == A100_40GB_MODAL_GPU:
        return run_evalplus_hf_model_a100
    return run_evalplus_hf_model


def _resolve_bigcodebench_remote(gpu_type: str):
    if gpu_type == A100_40GB_MODAL_GPU:
        return run_bigcodebench_hf_model_a100
    if gpu_type == H100_MODAL_GPU:
        return run_bigcodebench_hf_model_h100
    return run_bigcodebench_hf_model


def _summarize_evalplus_payload(
    results: dict[str, Any],
    *,
    dataset: str,
    model_ref: str,
    model_label: str,
    resolved_model_ref: str,
    result_path: Path,
    samples_paths: list[Path],
    raw_samples_paths: list[Path],
    elapsed_sec: float,
) -> dict[str, Any]:
    task_results = dict(results.get("eval", {}))
    num_tasks = len(task_results)
    num_samples = sum(len(list(records)) for records in task_results.values())
    base_correct = 0
    plus_correct = 0
    base_status_counts: dict[str, int] = {}
    plus_status_counts: dict[str, int] = {}
    for records in task_results.values():
        task_base_passed = False
        task_plus_passed = False
        for record in records:
            base_status = str(record.get("base_status"))
            plus_status = str(record.get("plus_status"))
            base_status_counts[base_status] = base_status_counts.get(base_status, 0) + 1
            plus_status_counts[plus_status] = plus_status_counts.get(plus_status, 0) + 1
            if base_status == "pass":
                task_base_passed = True
            if base_status == "pass" and plus_status == "pass":
                task_plus_passed = True
        if task_base_passed:
            base_correct += 1
        if task_plus_passed:
            plus_correct += 1

    return {
        "model_id": model_label,
        "model_ref": model_ref,
        "resolved_model_ref": resolved_model_ref,
        "dataset": dataset,
        "num_tasks": num_tasks,
        "num_samples": num_samples,
        "base_correct": base_correct,
        "plus_correct": plus_correct,
        "base_pass_at_1": base_correct / num_tasks if num_tasks else 0.0,
        "plus_pass_at_1": plus_correct / num_tasks if num_tasks else 0.0,
        "base_status_counts": base_status_counts,
        "plus_status_counts": plus_status_counts,
        "eval_hash": results.get("hash"),
        "eval_date": results.get("date"),
        "remote_result_path": str(result_path),
        "remote_samples_paths": [str(path) for path in samples_paths],
        "remote_raw_samples_paths": [str(path) for path in raw_samples_paths],
        "elapsed_sec": elapsed_sec,
    }


def _mean(values: list[float] | list[int]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _percentile(values: list[float] | list[int], percentile: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(float(value) for value in values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * (percentile / 100.0)
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = rank - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def _summarize_bigcodebench_payload(
    *,
    results: dict[str, Any],
    pass_at_k: dict[str, Any],
    generation_records: list[dict[str, Any]],
    split: str,
    subset: str,
    model_ref: str,
    model_label: str,
    resolved_model_ref: str,
    samples_path: Path,
    raw_samples_path: Path,
    result_path: Path,
    pass_at_k_path: Path,
    elapsed_sec: float,
    no_gt: bool,
) -> dict[str, Any]:
    task_results = dict(results.get("eval", {}))
    num_tasks = len(task_results)
    num_samples = sum(len(list(records)) for records in task_results.values())
    status_counts: dict[str, int] = {}
    passed_tasks = 0
    failed_tasks: list[str] = []
    for task_id, records in task_results.items():
        task_passed = False
        for record in records:
            status = str(record.get("status"))
            status_counts[status] = status_counts.get(status, 0) + 1
            if status == "pass":
                task_passed = True
        if task_passed:
            passed_tasks += 1
        else:
            failed_tasks.append(str(task_id))

    completion_tokens = [int(record.get("completion_tokens", 0)) for record in generation_records]
    latencies = [float(record.get("latency_sec", 0.0)) for record in generation_records]

    return {
        "model_id": model_label,
        "model_ref": model_ref,
        "resolved_model_ref": resolved_model_ref,
        "benchmark": "bigcodebench",
        "split": split,
        "subset": subset,
        "num_tasks": num_tasks,
        "num_samples": num_samples,
        "pass_correct": passed_tasks,
        "pass_at_1": float(pass_at_k.get("pass@1", passed_tasks / num_tasks if num_tasks else 0.0)),
        "status_counts": status_counts,
        "failed_tasks": failed_tasks,
        "length_capped_count": sum(1 for record in generation_records if record.get("length_capped")),
        "mean_latency_sec": _mean(latencies),
        "median_latency_sec": _percentile(latencies, 50.0),
        "p95_latency_sec": _percentile(latencies, 95.0),
        "mean_completion_tokens": _mean(completion_tokens),
        "median_completion_tokens": _percentile(completion_tokens, 50.0),
        "p95_completion_tokens": _percentile(completion_tokens, 95.0),
        "generation_records": generation_records,
        "pass_at_k": pass_at_k,
        "eval_date": results.get("date"),
        "no_gt": no_gt,
        "remote_result_path": str(result_path),
        "remote_pass_at_k_path": str(pass_at_k_path),
        "remote_samples_paths": [str(samples_path)],
        "remote_raw_samples_paths": [str(raw_samples_path)],
        "elapsed_sec": elapsed_sec,
    }


def _resolve_feasibility_remote(gpu_type: str):
    if gpu_type == A100_40GB_MODAL_GPU:
        return probe_mixed_precision_feasibility_source_a100
    return probe_mixed_precision_feasibility_source


def _resolve_loaded_probe_remote(gpu_type: str):
    if gpu_type == A100_40GB_MODAL_GPU:
        return probe_loaded_artifact_quantization_state_a100
    return probe_loaded_artifact_quantization_state


def _initialize_search_deck(
    task_name: str,
    split_name: str,
    run_slug: str,
) -> dict[str, Any]:
    ordered_example_ids = list(load_task_example_ids_remote.remote(task_name=task_name, split=split_name))
    if not ordered_example_ids:
        raise ValueError(f"No examples available for search deck split {split_name}")
    return {
        "run_slug": run_slug,
        "task_name": task_name,
        "split_name": split_name,
        "ordered_example_ids": ordered_example_ids,
        "active_example_ids": list(ordered_example_ids),
        "cursor": 0,
        "reshuffle_count": 0,
        "shuffle_seed": f"{run_slug}:deck:0",
    }


def _consume_search_turn_examples(
    deck_state: dict[str, Any],
    turn_limit: int,
) -> dict[str, Any]:
    if turn_limit <= 0:
        raise ValueError("turn_limit must be positive")

    selected_ids: list[str] = []
    segment_details: list[dict[str, Any]] = []
    cursor = int(deck_state["cursor"])
    reshuffle_count = int(deck_state["reshuffle_count"])
    active_ids = list(deck_state["active_example_ids"])
    ordered_ids = list(deck_state["ordered_example_ids"])

    while len(selected_ids) < turn_limit:
        remaining_in_deck = len(active_ids) - cursor
        if remaining_in_deck == 0:
            reshuffle_count += 1
            rng = random.Random(f"{deck_state['run_slug']}:deck:{reshuffle_count}")
            active_ids = list(ordered_ids)
            rng.shuffle(active_ids)
            cursor = 0
            remaining_in_deck = len(active_ids)
        take_count = min(turn_limit - len(selected_ids), remaining_in_deck)
        start = cursor
        end = cursor + take_count
        segment_ids = active_ids[start:end]
        segment_details.append(
            {
                "reshuffle_count": reshuffle_count,
                "deck_start": start,
                "deck_end": end,
                "example_ids": segment_ids,
            }
        )
        selected_ids.extend(segment_ids)
        cursor = end

    deck_state["active_example_ids"] = active_ids
    deck_state["cursor"] = cursor
    deck_state["reshuffle_count"] = reshuffle_count
    deck_state["shuffle_seed"] = f"{deck_state['run_slug']}:deck:{reshuffle_count}"
    return {
        "example_ids": selected_ids,
        "segments": segment_details,
        "cursor_after": cursor,
        "reshuffle_count_after": reshuffle_count,
    }


def _candidate_key_sort_key(candidate_key: str) -> tuple[int, int, int, str]:
    seed_match = re.match(r"^seed-(\d+)-", candidate_key)
    if seed_match:
        return (0, int(seed_match.group(1)), 0, candidate_key)
    round_match = re.match(r"^round-(\d+)-candidate-(\d+)$", candidate_key)
    if round_match:
        return (1, int(round_match.group(1)), int(round_match.group(2)), candidate_key)
    return (2, 0, 0, candidate_key)


def _round_index_from_candidate_key(candidate_key: str) -> int | None:
    match = re.match(r"^round-(\d+)-candidate-\d+$", candidate_key)
    if not match:
        return None
    return int(match.group(1))


def _load_resumed_candidate_record(
    output_slug: str,
    candidate_key: str,
    split_name: str,
) -> dict[str, Any] | None:
    candidate_path = PROJECT_ROOT / "outputs" / "policies" / output_slug / f"{candidate_key}.json"
    feasibility_report_path = (
        PROJECT_ROOT / "outputs" / "feasibility" / f"{output_slug}-{candidate_key}-source-report.json"
    )
    loaded_artifact_probe_path = (
        PROJECT_ROOT / "outputs" / "feasibility" / f"{output_slug}-{candidate_key}-loaded-artifact-probe.json"
    )
    integrity_manifest_path = (
        PROJECT_ROOT / "outputs" / "feasibility" / f"{output_slug}-{candidate_key}-integrity.json"
    )
    evaluation_path = _evaluation_output_path(output_slug, candidate_key, split_name)
    stage1_path = _evaluation_output_path(output_slug, candidate_key, split_name, "stage1")
    stage2_path = _evaluation_output_path(output_slug, candidate_key, split_name, "stage2")
    required_paths = [candidate_path, feasibility_report_path, integrity_manifest_path]
    if not all(path.exists() for path in required_paths):
        return None
    if not evaluation_path.exists() and not stage1_path.exists():
        return None

    candidate_payload = _load_json(candidate_path)
    source_report = _load_json(feasibility_report_path)
    integrity_manifest = _load_json(integrity_manifest_path)
    if stage1_path.exists():
        stage1_summary = _load_json(stage1_path)
        split_payload = _merge_staged_evaluation_payload(
            None,
            "stage1",
            _build_evaluation_payload(stage1_summary, split_name, stage1_path),
        )
        if stage2_path.exists():
            stage2_summary = _load_json(stage2_path)
            split_payload = _merge_staged_evaluation_payload(
                split_payload,
                "stage2",
                _build_evaluation_payload(stage2_summary, split_name, stage2_path),
            )
    else:
        evaluation_summary = _load_json(evaluation_path)
        split_payload = _build_evaluation_payload(evaluation_summary, split_name, evaluation_path)

    evaluations: dict[str, Any] = {
        split_name: split_payload,
    }
    record: dict[str, Any] = {
        "candidate_key": candidate_key,
        "candidate_path": to_relative_path(candidate_path, PROJECT_ROOT),
        "artifact_dir": str(source_report.get("output_dir") or "") or None,
        "group_bit_assignments": {
            str(group_name): int(bit_width)
            for group_name, bit_width in dict(
                candidate_payload.get("group_bit_assignments", {})
            ).items()
        },
        "bit_counts": candidate_payload.get("bit_counts", {}),
        "matched_linear_weight_footprint_gb": float(
            candidate_payload.get("matched_linear_weight_footprint_gb") or 0.0
        ),
        "estimated_full_model_weight_footprint_gb": float(
            candidate_payload.get("estimated_full_model_weight_footprint_gb") or 0.0
        ),
        "estimated_average_bit_width": float(candidate_payload.get("estimated_average_bit_width") or 0.0),
        "proposal_score": float(candidate_payload.get("fitness") or 0.0),
        "budget_alignment_score": float(candidate_payload.get("budget_alignment_score") or 0.0),
        "provenance": str(candidate_payload.get("provenance", candidate_key)),
        "parent_candidate_key": None,
        "feasibility_report_path": to_relative_path(feasibility_report_path, PROJECT_ROOT),
        "loaded_artifact_probe_path": (
            to_relative_path(loaded_artifact_probe_path, PROJECT_ROOT)
            if loaded_artifact_probe_path.exists()
            else None
        ),
        "integrity_manifest_path": to_relative_path(integrity_manifest_path, PROJECT_ROOT),
        "integrity_clean": bool(integrity_manifest.get("is_clean")),
        "unresolved_target_warnings": integrity_manifest.get("unresolved_target_warnings", []),
        "smoke_test_passed": bool(source_report.get("quantized_model_runnable")),
        "is_mixed": any(
            int(bit_width) != 8
            for bit_width in dict(candidate_payload.get("group_bit_assignments", {})).values()
        ),
        "evaluations": evaluations,
    }
    return record


def _load_surrogate_free_resume_state(
    output_slug: str,
    split_name: str,
    beam_width: int,
) -> dict[str, Any]:
    policy_dir = PROJECT_ROOT / "outputs" / "policies" / output_slug
    if not policy_dir.exists():
        raise FileNotFoundError(f"Resume policy directory not found: {policy_dir}")

    candidate_keys = [
        path.stem
        for path in policy_dir.glob("*.json")
        if path.stem.startswith("seed-") or path.stem.startswith("round-")
    ]
    records: list[dict[str, Any]] = []
    ordered_records: list[dict[str, Any]] = []
    for candidate_key in sorted(candidate_keys, key=_candidate_key_sort_key):
        record = _load_resumed_candidate_record(output_slug, candidate_key, split_name)
        if record is None:
            continue
        ordered_records.append(record)
        records.append(record)
    if not ordered_records:
        raise ValueError(f"No completed search candidates found to resume for slug {output_slug}")

    seen_signatures = {
        _assignment_signature(dict(record.get("group_bit_assignments", {})))
        for record in ordered_records
    }
    beam_records = _select_best_direct_eval_records(records, split_name=split_name, limit=beam_width)
    completed_round_indices = [
        round_index
        for record in ordered_records
        if (round_index := _round_index_from_candidate_key(str(record["candidate_key"]))) is not None
    ]
    next_round_index = (max(completed_round_indices) + 1) if completed_round_indices else 1

    round_summaries: list[dict[str, Any]] = []
    grouped_round_records: dict[int, list[dict[str, Any]]] = {}
    for record in ordered_records:
        round_index = _round_index_from_candidate_key(str(record["candidate_key"]))
        if round_index is None:
            continue
        grouped_round_records.setdefault(round_index, []).append(record)
    for round_index in sorted(grouped_round_records):
        provisional_records = _select_best_direct_eval_records(
            grouped_round_records[round_index],
            split_name=split_name,
            limit=len(grouped_round_records[round_index]),
        )
        rechecked_records = [
            record
            for record in provisional_records
            if _candidate_stage_accuracy(record, split_name, "stage2") is not None
        ]
        round_summaries.append(
            {
                "round_index": round_index,
                "proposal_count": None,
                "provisional_pool_size": len(grouped_round_records[round_index]),
                "evaluated_candidates": [
                    _candidate_round_snapshot(record, split_name)
                    for record in grouped_round_records[round_index]
                ],
                "provisional_candidates": [
                    _candidate_round_snapshot(record, split_name)
                    for record in provisional_records
                ],
                "rechecked_candidates": [
                    _candidate_round_snapshot(record, split_name)
                    for record in rechecked_records
                ],
                "beam_candidates": [],
            }
        )

    return {
        "records": records,
        "ordered_records": ordered_records,
        "seen_signatures": seen_signatures,
        "beam_records": beam_records,
        "round_summaries": round_summaries,
        "next_round_index": next_round_index,
        "completed_turns": len(ordered_records),
    }


def _advance_search_deck_for_resume(
    deck_state: dict[str, Any],
    resumed_records: list[dict[str, Any]],
    stage1_turn_limit: int,
    stage2_turn_limit: int,
    split_name: str,
) -> None:
    replayed_stage_keys: set[tuple[int, str]] = set()
    for record in resumed_records:
        evaluation = dict(record.get("evaluations", {})).get(split_name, {})
        round_index = _round_index_from_candidate_key(str(record["candidate_key"]))
        stage1_ids = list(evaluation.get("stage1_evaluated_example_ids", []))
        stage2_ids = list(evaluation.get("stage2_evaluated_example_ids", []))
        if stage1_ids or stage2_ids:
            if round_index is None:
                raise ValueError(
                    "Staged evaluation metadata is only supported for round candidates during resume"
                )
            stage1_key = (round_index, "stage1")
            if stage1_ids and stage1_key not in replayed_stage_keys:
                consumed = _consume_search_turn_examples(deck_state, stage1_turn_limit)
                if stage1_ids != list(consumed["example_ids"]):
                    raise ValueError(
                        "Resume deck replay diverged from saved stage1 example ids for "
                        f"{record['candidate_key']}"
                    )
                replayed_stage_keys.add(stage1_key)
            stage2_key = (round_index, "stage2")
            if stage2_ids and stage2_key not in replayed_stage_keys:
                consumed = _consume_search_turn_examples(deck_state, stage2_turn_limit)
                if stage2_ids != list(consumed["example_ids"]):
                    raise ValueError(
                        "Resume deck replay diverged from saved stage2 example ids for "
                        f"{record['candidate_key']}"
                    )
                replayed_stage_keys.add(stage2_key)
            continue

        expected_ids = _evaluation_example_ids(evaluation)
        if not expected_ids:
            continue
        consumed = _consume_search_turn_examples(deck_state, stage1_turn_limit)
        if expected_ids != list(consumed["example_ids"]):
            raise ValueError(
                "Resume deck replay diverged from saved example ids for "
                f"{record['candidate_key']}"
            )


def _normalize_source_smoke_result(report: dict[str, Any], tokenizer_source: str) -> dict[str, Any]:
    if (
        report.get("status") == "smoke_test_failed"
        and report.get("artifact_dir_exists")
        and report.get("smoke_test_error_type") == "OutOfMemoryError"
    ):
        clean_smoke_test = smoke_test_quantized_artifact_source.remote(
            artifact_dir=report["output_dir"],
            tokenizer_source=tokenizer_source,
            max_new_tokens=4,
        )
        report["same_process_smoke_test"] = {
            "status": "failed",
            "error_type": report.get("smoke_test_error_type"),
            "error_message": report.get("smoke_test_error_message"),
        }
        report["smoke_test"] = clean_smoke_test
        report["smoke_test_mode"] = "clean_process_retry"
        report["quantized_model_runnable"] = True
        report["status"] = "smoke_test_succeeded"
    return report


def _build_candidate_integrity_manifest(
    candidate_payload: dict[str, Any],
    source_report: dict[str, Any],
    loaded_probe: dict[str, Any] | None,
) -> dict[str, Any]:
    from ta_mpq.feasibility import build_policy_target_integrity_manifest
    from ta_mpq.quantization import MixedPrecisionPolicy

    policy = MixedPrecisionPolicy.from_dict(
        candidate_payload["backend_projections"]["llmcompressor"]["projected_policy"]
    )
    integrity_manifest = build_policy_target_integrity_manifest(
        policy=policy,
        source_target_matching=source_report.get("policy_target_matching"),
        reload_target_matching=(loaded_probe or {}).get("policy_target_matching"),
    )
    integrity_manifest["candidate_rank"] = candidate_payload.get("rank")
    integrity_manifest["candidate_provenance"] = candidate_payload.get("provenance")
    integrity_manifest["candidate_path"] = candidate_payload.get("candidate_path")
    integrity_manifest["matched_linear_weight_footprint_gb"] = candidate_payload.get(
        "matched_linear_weight_footprint_gb"
    )
    integrity_manifest["estimated_full_model_weight_footprint_gb"] = candidate_payload.get(
        "estimated_full_model_weight_footprint_gb"
    )
    return integrity_manifest


def _candidate_accuracy(record: dict[str, Any], split_name: str) -> float:
    evaluations = dict(record.get("evaluations", {}))
    payload = evaluations.get(split_name) or {}
    if payload.get("combined_accuracy") is not None:
        return float(payload["combined_accuracy"])
    if payload.get("accuracy") is not None:
        return float(payload["accuracy"])
    if payload.get("stage1_accuracy") is not None:
        return float(payload["stage1_accuracy"])
    return -1.0


def _candidate_stage_accuracy(record: dict[str, Any], split_name: str, stage_name: str) -> float | None:
    evaluations = dict(record.get("evaluations", {}))
    payload = evaluations.get(split_name) or {}
    value = payload.get(f"{stage_name}_accuracy")
    if value is None:
        return None
    return float(value)


def _evaluation_output_path(
    output_slug: str,
    candidate_key: str,
    split_name: str,
    evaluation_stage: str | None = None,
) -> Path:
    suffix = f"-{evaluation_stage}" if evaluation_stage else ""
    return (
        PROJECT_ROOT
        / "outputs"
        / "evaluations"
        / f"{output_slug}-{candidate_key}-{_slugify(split_name)}{suffix}-quantized.json"
    )


def _build_evaluation_payload(
    summary: dict[str, Any],
    split_name: str,
    evaluation_path: Path,
    evaluation_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "path": to_relative_path(evaluation_path, PROJECT_ROOT),
        "accuracy": float(summary.get("accuracy", 0.0)),
        "num_correct": int(summary.get("num_correct", 0)),
        "num_examples": int(summary.get("num_examples", 0)),
        "task_split": split_name,
        "evaluated_example_ids": list(summary.get("evaluated_example_ids", [])),
    }
    if evaluation_metadata:
        payload.update(evaluation_metadata)
    return payload


def _merge_staged_evaluation_payload(
    existing_payload: dict[str, Any] | None,
    stage_name: str,
    stage_payload: dict[str, Any],
) -> dict[str, Any]:
    merged = dict(existing_payload or {})
    for key in ("path", "accuracy", "num_correct", "num_examples", "evaluated_example_ids"):
        merged[f"{stage_name}_{key}"] = stage_payload.get(key)
    merged["task_split"] = stage_payload["task_split"]
    for metadata_key, metadata_value in stage_payload.items():
        if metadata_key in {"path", "accuracy", "num_correct", "num_examples", "task_split", "evaluated_example_ids"}:
            continue
        merged[f"{stage_name}_{metadata_key}"] = metadata_value

    stage1_correct = merged.get("stage1_num_correct")
    stage1_examples = merged.get("stage1_num_examples")
    stage1_accuracy = merged.get("stage1_accuracy")
    stage1_ids = list(merged.get("stage1_evaluated_example_ids", []))
    stage2_correct = merged.get("stage2_num_correct")
    stage2_examples = merged.get("stage2_num_examples")
    stage2_accuracy = merged.get("stage2_accuracy")
    stage2_ids = list(merged.get("stage2_evaluated_example_ids", []))

    if stage2_correct is not None and stage2_examples is not None and stage1_correct is not None and stage1_examples is not None:
        total_correct = int(stage1_correct) + int(stage2_correct)
        total_examples = int(stage1_examples) + int(stage2_examples)
        merged["combined_num_correct"] = total_correct
        merged["combined_num_examples"] = total_examples
        merged["combined_accuracy"] = 0.0 if total_examples == 0 else total_correct / total_examples
        merged["accuracy"] = float(merged["combined_accuracy"])
        merged["num_correct"] = total_correct
        merged["num_examples"] = total_examples
        merged["evaluated_example_ids"] = stage1_ids + stage2_ids
        merged["path"] = merged.get("stage2_path") or merged.get("stage1_path")
    else:
        merged["combined_num_correct"] = None
        merged["combined_num_examples"] = None
        merged["combined_accuracy"] = None
        merged["accuracy"] = float(stage1_accuracy if stage1_accuracy is not None else stage_payload["accuracy"])
        merged["num_correct"] = int(stage1_correct if stage1_correct is not None else stage_payload["num_correct"])
        merged["num_examples"] = int(stage1_examples if stage1_examples is not None else stage_payload["num_examples"])
        merged["evaluated_example_ids"] = stage1_ids or list(stage_payload.get("evaluated_example_ids", []))
        merged["path"] = merged.get("stage1_path") or stage_payload.get("path")
    return merged


def _evaluation_example_ids(payload: dict[str, Any]) -> list[str]:
    stage1_ids = payload.get("stage1_evaluated_example_ids")
    stage2_ids = payload.get("stage2_evaluated_example_ids")
    if stage1_ids is not None or stage2_ids is not None:
        return list(stage1_ids or []) + list(stage2_ids or [])
    return list(payload.get("evaluated_example_ids", []))


def _direct_eval_sort_key(record: dict[str, Any], split_name: str) -> tuple[Any, ...]:
    return (
        _candidate_accuracy(record, split_name),
        1 if bool(record.get("integrity_clean")) else 0,
        float(record.get("budget_alignment_score", 0.0)),
        1 if bool(record.get("smoke_test_passed")) else 0,
        float(record.get("proposal_score", 0.0)),
    )


def _select_best_direct_eval_records(
    records: list[dict[str, Any]],
    split_name: str,
    limit: int,
) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen_signatures: set[tuple[tuple[str, int], ...]] = set()
    for record in sorted(records, key=lambda item: _direct_eval_sort_key(item, split_name), reverse=True):
        signature = _assignment_signature(dict(record.get("group_bit_assignments", {})))
        if signature in seen_signatures:
            continue
        deduped.append(record)
        seen_signatures.add(signature)
        if len(deduped) == limit:
            break
    return deduped


def _candidate_round_snapshot(record: dict[str, Any], split_name: str) -> dict[str, Any]:
    return {
        "candidate_key": record["candidate_key"],
        "provenance": record["provenance"],
        "proposal_score": record.get("proposal_score"),
        "accuracy": _candidate_accuracy(record, split_name),
        "stage1_accuracy": _candidate_stage_accuracy(record, split_name, "stage1"),
        "stage2_accuracy": _candidate_stage_accuracy(record, split_name, "stage2"),
        "integrity_clean": record.get("integrity_clean"),
        "smoke_test_passed": record.get("smoke_test_passed"),
        "matched_linear_weight_footprint_gb": record.get("matched_linear_weight_footprint_gb"),
        "estimated_full_model_weight_footprint_gb": record.get(
            "estimated_full_model_weight_footprint_gb"
        ),
        "parent_candidate_key": record.get("parent_candidate_key"),
    }


def _candidate_result_snapshot(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_key": record["candidate_key"],
        "candidate_path": record.get("candidate_path"),
        "artifact_dir": record.get("artifact_dir"),
        "provenance": record.get("provenance"),
        "proposal_score": record.get("proposal_score"),
        "is_mixed": record.get("is_mixed"),
        "integrity_clean": record.get("integrity_clean"),
        "unresolved_target_warnings": record.get("unresolved_target_warnings"),
        "smoke_test_passed": record.get("smoke_test_passed"),
        "budget_alignment_score": record.get("budget_alignment_score"),
        "matched_linear_weight_footprint_gb": record.get("matched_linear_weight_footprint_gb"),
        "estimated_full_model_weight_footprint_gb": record.get(
            "estimated_full_model_weight_footprint_gb"
        ),
        "bit_counts": record.get("bit_counts"),
        "evaluations": record.get("evaluations", {}),
        "feasibility_report_path": record.get("feasibility_report_path"),
        "loaded_artifact_probe_path": record.get("loaded_artifact_probe_path"),
        "integrity_manifest_path": record.get("integrity_manifest_path"),
    }


def _run_surrogate_free_candidate_eval(
    candidate_key: str,
    report_payload: dict[str, Any],
    group_bits: dict[str, int],
    proposal_score: float,
    provenance: str,
    contract: Any,
    task_name: str,
    task_prompt_style: str,
    calibration_limit: int,
    target_budget_gb: float,
    dev_split: str,
    dev_limit: int,
    max_new_tokens: int,
    output_slug: str,
    policy_output_dir: Path,
    parent_candidate_key: str | None = None,
    group_quantization_overrides: dict[str, dict[str, Any]] | None = None,
    example_ids: list[str] | None = None,
    evaluation_metadata: dict[str, Any] | None = None,
    evaluation_stage: str | None = None,
    gpu_type: str = DEFAULT_MODAL_GPU,
) -> dict[str, Any]:
    from ta_mpq.policy_export import export_candidate_from_group_bits
    from ta_mpq.search import estimate_budget_alignment_score

    candidate_payload = export_candidate_from_group_bits(
        report_payload=report_payload,
        grouping="per_block_component",
        group_bits=group_bits,
        group_quantization_overrides=group_quantization_overrides,
        rank=0,
        metadata={
            "fitness": proposal_score,
            "proxy_quality_score": proposal_score,
            "estimated_average_bit_width": None,
            "estimated_weight_footprint_gb": None,
            "provenance": provenance,
        },
    )
    candidate_path = policy_output_dir / f"{candidate_key}.json"
    candidate_payload["candidate_path"] = to_relative_path(candidate_path, PROJECT_ROOT)
    save_summary(candidate_path, candidate_payload)

    policy_payload = candidate_payload["backend_projections"]["llmcompressor"]["projected_policy"]
    feasibility_remote = _resolve_feasibility_remote(gpu_type)
    loaded_probe_remote = _resolve_loaded_probe_remote(gpu_type)

    report = feasibility_remote.remote(
        contract.to_dict(),
        calibration_limit=calibration_limit,
        dry_run=False,
        policy_payload=policy_payload,
        policy_label=f"{output_slug}-{candidate_key}",
        precomputed_report=report_payload,
    )
    report = _normalize_source_smoke_result(report, contract.compressed_source_model_id)
    report["candidate_path"] = candidate_payload["candidate_path"]
    feasibility_report_path = (
        PROJECT_ROOT / "outputs" / "feasibility" / f"{output_slug}-{candidate_key}-source-report.json"
    )
    save_summary(feasibility_report_path, report)

    artifact_dir = str(report.get("output_dir") or "")
    loaded_probe: dict[str, Any] | None = None
    loaded_probe_path: Path | None = None
    if artifact_dir and report.get("artifact_dir_exists"):
        loaded_probe = loaded_probe_remote.remote(
            model_ref=artifact_dir,
            policy_payload=policy_payload,
            policy_label=f"{output_slug}-{candidate_key}",
        )
        loaded_probe["candidate_path"] = candidate_payload["candidate_path"]
        loaded_probe_path = (
            PROJECT_ROOT
            / "outputs"
            / "feasibility"
            / f"{output_slug}-{candidate_key}-loaded-artifact-probe.json"
        )
        save_summary(loaded_probe_path, loaded_probe)

    integrity_manifest = _build_candidate_integrity_manifest(
        candidate_payload=candidate_payload,
        source_report=report,
        loaded_probe=loaded_probe,
    )
    integrity_manifest_path = (
        PROJECT_ROOT / "outputs" / "feasibility" / f"{output_slug}-{candidate_key}-integrity.json"
    )
    save_summary(integrity_manifest_path, integrity_manifest)

    record: dict[str, Any] = {
        "candidate_key": candidate_key,
        "candidate_path": to_relative_path(candidate_path, PROJECT_ROOT),
        "artifact_dir": artifact_dir or None,
        "group_bit_assignments": {
            str(group_name): int(bit_width)
            for group_name, bit_width in group_bits.items()
        },
        "bit_counts": candidate_payload.get("bit_counts", {}),
        "matched_linear_weight_footprint_gb": float(
            candidate_payload.get("matched_linear_weight_footprint_gb", 0.0)
        ),
        "estimated_full_model_weight_footprint_gb": float(
            candidate_payload.get("estimated_full_model_weight_footprint_gb", 0.0)
        ),
        "estimated_average_bit_width": float(candidate_payload.get("estimated_average_bit_width", 0.0)),
        "proposal_score": float(proposal_score),
        "budget_alignment_score": estimate_budget_alignment_score(
            footprint_gb=float(candidate_payload.get("matched_linear_weight_footprint_gb", 0.0)),
            target_budget_gb=float(target_budget_gb),
        ),
        "provenance": provenance,
        "parent_candidate_key": parent_candidate_key,
        "feasibility_report_path": to_relative_path(feasibility_report_path, PROJECT_ROOT),
        "loaded_artifact_probe_path": (
            to_relative_path(loaded_probe_path, PROJECT_ROOT) if loaded_probe_path else None
        ),
        "integrity_manifest_path": to_relative_path(integrity_manifest_path, PROJECT_ROOT),
        "integrity_clean": bool(integrity_manifest.get("is_clean")),
        "unresolved_target_warnings": integrity_manifest.get("unresolved_target_warnings", []),
        "smoke_test_passed": bool(report.get("quantized_model_runnable")),
        "is_mixed": any(int(bit_width) != 8 for bit_width in group_bits.values()),
        "evaluations": {},
    }
    _ensure_candidate_task_eval(
        record=record,
        contract=contract,
        task_name=task_name,
        task_prompt_style=task_prompt_style,
        split_name=dev_split,
        split_limit=dev_limit,
        max_new_tokens=max_new_tokens,
        output_slug=output_slug,
        example_ids=example_ids,
        evaluation_metadata=evaluation_metadata,
        evaluation_stage=evaluation_stage,
        gpu_type=gpu_type,
    )
    return record


def _ensure_candidate_task_eval(
    record: dict[str, Any],
    contract: Any,
    task_name: str,
    task_prompt_style: str,
    split_name: str,
    split_limit: int,
    max_new_tokens: int,
    output_slug: str,
    example_ids: list[str] | None = None,
    evaluation_metadata: dict[str, Any] | None = None,
    evaluation_stage: str | None = None,
    gpu_type: str = DEFAULT_MODAL_GPU,
) -> None:
    evaluations = dict(record.get("evaluations", {}))
    existing_payload = dict(evaluations.get(split_name, {}))
    if evaluation_stage:
        if f"{evaluation_stage}_path" in existing_payload:
            record["evaluations"] = evaluations
            return
    elif split_name in evaluations:
        record["evaluations"] = evaluations
        return
    if not record.get("artifact_dir") or not record.get("integrity_clean") or not record.get("smoke_test_passed"):
        skipped_payload = {
            "status": "skipped_integrity_gate",
            "accuracy": -1.0,
            "num_correct": 0,
            "num_examples": 0,
            "task_split": split_name,
        }
        evaluations[split_name] = (
            _merge_staged_evaluation_payload(existing_payload, evaluation_stage, skipped_payload)
            if evaluation_stage
            else skipped_payload
        )
        record["evaluations"] = evaluations
        return

    eval_remote = _resolve_eval_remote(gpu_type)
    summary = eval_remote.remote(
        model_ref=str(record["artifact_dir"]),
        tokenizer_source=contract.compressed_source_model_id,
        model_label=f"{contract.compressed_source_model_id}-{record['candidate_key']}",
        task_name=task_name,
        limit=split_limit,
        max_new_tokens=max_new_tokens,
        split=split_name,
        example_ids=example_ids,
        load_dtype="auto",
        task_prompt_style=task_prompt_style,
    )
    evaluation_path = _evaluation_output_path(
        output_slug=output_slug,
        candidate_key=str(record["candidate_key"]),
        split_name=split_name,
        evaluation_stage=evaluation_stage,
    )
    save_summary(evaluation_path, summary)
    evaluation_payload = _build_evaluation_payload(
        summary=summary,
        split_name=split_name,
        evaluation_path=evaluation_path,
        evaluation_metadata=evaluation_metadata,
    )
    evaluations[split_name] = (
        _merge_staged_evaluation_payload(existing_payload, evaluation_stage, evaluation_payload)
        if evaluation_stage
        else evaluation_payload
    )
    record["evaluations"] = evaluations


def _run_optional_finalist_config_refinement(
    all_candidate_records: list[dict[str, Any]],
    report_payload: dict[str, Any],
    groups: list[Any],
    layer_stats: list[Any],
    group_value_scores: dict[str, float],
    resolved_target_budget_gb: float,
    contract: Any,
    task_name: str,
    task_prompt_style: str,
    calibration_limit: int,
    dev_split: str,
    dev_limit: int,
    max_new_tokens: int,
    output_slug: str,
    policy_output_dir: Path,
) -> None:
    from ta_mpq.search import SearchCandidate, refine_candidate_quantization_configs

    top_mixed = _select_best_direct_eval_records(
        [
            record
            for record in all_candidate_records
            if bool(record.get("is_mixed")) and bool(record.get("integrity_clean"))
        ],
        split_name=dev_split,
        limit=1,
    )
    if not top_mixed:
        return
    uniform_records = [
        record
        for record in all_candidate_records
        if str(record.get("provenance")) == "uniform_int8_seed"
    ]
    if uniform_records and _candidate_accuracy(top_mixed[0], dev_split) < _candidate_accuracy(
        uniform_records[0],
        dev_split,
    ):
        return

    base_record = top_mixed[0]
    base_candidate = SearchCandidate(
        group_bits=tuple(sorted(dict(base_record["group_bit_assignments"]).items())),
        estimated_average_bit_width=float(base_record.get("estimated_average_bit_width", 0.0)),
        estimated_weight_footprint_gb=float(base_record.get("matched_linear_weight_footprint_gb", 0.0)),
        proxy_quality_score=float(base_record.get("proposal_score", 0.0)),
        fitness=float(base_record.get("proposal_score", 0.0)),
        provenance=str(base_record["candidate_key"]),
    )
    refinement = refine_candidate_quantization_configs(
        groups=groups,
        layer_stats=layer_stats,
        base_candidates=[base_candidate],
        group_value_scores=group_value_scores,
        group_size_options=(64, 128),
        symmetric_options=(True, False),
        max_tunable_groups=8,
        population_size=8,
        generations=4,
        top_k=4,
        seed=0,
        seed_candidate_count=1,
    )
    for candidate_index, candidate_payload in enumerate(refinement.get("top_candidates", [])[:4], start=1):
        record = _run_surrogate_free_candidate_eval(
            candidate_key=f"refinement-{candidate_index:02d}",
            report_payload=report_payload,
            group_bits={
                str(group_name): int(bit_width)
                for group_name, bit_width in dict(candidate_payload.get("group_bits", {})).items()
            },
            group_quantization_overrides=dict(candidate_payload.get("group_quantization_overrides", {})),
            proposal_score=float(candidate_payload.get("fitness", 0.0)),
            provenance=f"finalist_config_refinement_{candidate_index}",
            contract=contract,
            task_name=task_name,
            task_prompt_style=task_prompt_style,
            calibration_limit=calibration_limit,
            target_budget_gb=resolved_target_budget_gb,
            dev_split=dev_split,
            dev_limit=dev_limit,
            max_new_tokens=max_new_tokens,
            output_slug=output_slug,
            policy_output_dir=policy_output_dir,
            parent_candidate_key=str(base_record["candidate_key"]),
        )
        all_candidate_records.append(record)


def _resolve_candidate_artifact_dir(
    contract: Any,
    candidate_path: str,
    policy_source: str,
    backend_variant: str,
    calibration_limit: int,
    report_payload_path: str | None = None,
    report_path: str = "",
) -> tuple[str, str]:
    resolved_report_path = PROJECT_ROOT / report_path if report_path else None
    if resolved_report_path is None:
        policy_label = _candidate_policy_label(candidate_path, policy_source)
        resolved_report_path = (
            PROJECT_ROOT
            / "outputs"
            / "feasibility"
            / f"{contract.name}-{_slugify(policy_label)}-{backend_variant}-report.json"
        )

    if resolved_report_path.exists():
        report = _load_json(resolved_report_path)
    else:
        candidate_payload = _load_json(PROJECT_ROOT / candidate_path)
        if policy_source == "project":
            policy_payload = candidate_payload["project_policy"]
        elif policy_source == "llmcompressor":
            policy_payload = candidate_payload["backend_projections"]["llmcompressor"][
                "projected_policy"
            ]
        else:
            raise ValueError(f"Unknown policy_source: {policy_source}")

        precomputed_report = _load_json(PROJECT_ROOT / report_payload_path) if report_payload_path else None
        policy_label = _candidate_policy_label(candidate_path, policy_source)
        if backend_variant == "stable":
            report = probe_mixed_precision_feasibility.remote(
                contract.to_dict(),
                calibration_limit=calibration_limit,
                dry_run=False,
                policy_payload=policy_payload,
                policy_label=policy_label,
                precomputed_report=precomputed_report,
            )
        elif backend_variant == "source":
            report = probe_mixed_precision_feasibility_source.remote(
                contract.to_dict(),
                calibration_limit=calibration_limit,
                dry_run=False,
                policy_payload=policy_payload,
                policy_label=policy_label,
                precomputed_report=precomputed_report,
            )
        else:
            raise ValueError(f"Unknown backend_variant: {backend_variant}")

        if backend_variant == "source":
            report = _normalize_source_smoke_result(report, contract.compressed_source_model_id)

        report["candidate_path"] = candidate_path
        report["policy_source"] = policy_source
        save_summary(resolved_report_path, report)

    artifact_dir = report.get("output_dir")
    if not artifact_dir:
        raise RuntimeError(f"Quantization report missing output_dir for {candidate_path}")
    return str(artifact_dir), str(resolved_report_path)


def _apply_transformers_token_compat_patch() -> None:
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    for cls in (AutoConfig, AutoModelForCausalLM, AutoTokenizer):
        original = cls.from_pretrained
        if getattr(original, "_ta_mpq_token_compat_patched", False):
            continue

        original_func = original.__func__

        @classmethod
        def patched(class_ref, *args: Any, _original_func=original_func, **kwargs: Any):
            if "use_auth_token" in kwargs:
                if "token" not in kwargs:
                    kwargs["token"] = kwargs["use_auth_token"]
                kwargs.pop("use_auth_token", None)
            return _original_func(class_ref, *args, **kwargs)

        setattr(patched, "_ta_mpq_token_compat_patched", True)
        cls.from_pretrained = patched


def _apply_compressed_tensors_distributed_compat_patch() -> None:
    import sys
    import types
    from typing import Callable, Hashable, TypeVar

    try:
        import compressed_tensors
    except Exception:
        return

    if "compressed_tensors.distributed" in sys.modules:
        return

    T = TypeVar("T", bound=Hashable)

    def greedy_bin_packing(
        items: list[T],
        num_bins: int,
        item_weight_fn: Callable[[T], int | float] = lambda x: 1,
    ) -> tuple[list[T], list[list[T]], dict[T, int]]:
        items.sort(key=item_weight_fn, reverse=True)
        bin_to_items: list[list[T]] = [[] for _ in range(num_bins)]
        item_to_bin: dict[T, int] = {}
        bin_weights: list[float] = [0 for _ in range(num_bins)]
        for item in items:
            target_bin = bin_weights.index(min(bin_weights))
            bin_to_items[target_bin].append(item)
            item_to_bin[item] = target_bin
            bin_weights[target_bin] += float(item_weight_fn(item))
        return items, bin_to_items, item_to_bin

    def wait_for_comms(pending_comms) -> None:
        for comm in list(pending_comms):
            comm.wait()
        pending_comms.clear()

    distributed_module = types.ModuleType("compressed_tensors.distributed")
    distributed_module.greedy_bin_packing = greedy_bin_packing
    distributed_module.wait_for_comms = wait_for_comms
    distributed_module.__all__ = ["greedy_bin_packing", "wait_for_comms"]

    assign_module = types.ModuleType("compressed_tensors.distributed.assign")
    assign_module.greedy_bin_packing = greedy_bin_packing
    assign_module.__all__ = ["greedy_bin_packing"]

    utils_module = types.ModuleType("compressed_tensors.distributed.utils")
    utils_module.wait_for_comms = wait_for_comms
    utils_module.__all__ = ["wait_for_comms"]

    sys.modules["compressed_tensors.distributed"] = distributed_module
    sys.modules["compressed_tensors.distributed.assign"] = assign_module
    sys.modules["compressed_tensors.distributed.utils"] = utils_module
    setattr(compressed_tensors, "distributed", distributed_module)
