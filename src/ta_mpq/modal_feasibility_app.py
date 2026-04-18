from __future__ import annotations

import json
from pathlib import Path
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
REFERENCE_ACCURACY_SENTINEL = -999.0

report_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch>=2.10.0",
        # Qwen3.5 loading currently requires a Transformers build that
        # recognizes the qwen3_5 architecture.
        "git+https://github.com/huggingface/transformers.git",
        "datasets>=4.0.0",
        "accelerate>=1.6.0",
        "huggingface_hub>=0.30.0",
        "safetensors>=0.5.0",
        "sentencepiece>=0.2.0",
    )
    .add_local_python_source("ta_mpq")
)

quant_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.10.0",
        "transformers>=4.56.1",
        "datasets>=4.0.0",
        "accelerate>=1.6.0",
        "huggingface_hub>=0.30.0",
        "safetensors>=0.5.0",
        "sentencepiece>=0.2.0",
        "llmcompressor>=0.8.0",
    )
    .add_local_python_source("ta_mpq")
)

quant_source_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch>=2.10.0",
        "git+https://github.com/huggingface/transformers.git",
        "datasets>=4.0.0",
        "accelerate>=1.6.0",
        "huggingface_hub>=0.30.0",
        "safetensors>=0.5.0",
        "sentencepiece>=0.2.0",
        "compressed-tensors>=0.14.0",
        "loguru>=0.7.2",
        "Pillow>=10.0.0",
        "requests>=2.32.2",
        "torchvision>=0.21.0",
    )
    .run_commands(
        "python -m pip install --no-deps git+https://github.com/vllm-project/llm-compressor.git"
    )
    .add_local_python_source("ta_mpq")
)

surrogate_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("xgboost>=2.1.0")
    .add_local_python_source("ta_mpq")
)


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


@app.function(
    image=report_image,
    gpu="A100-80GB",
    timeout=60 * 60 * 4,
    volumes={"/cache": cache_volume, "/artifacts": artifact_volume},
)
def probe_mixed_precision_report(
    contract_data: dict[str, Any],
    calibration_limit: int = 16,
    policy_payload: dict[str, Any] | None = None,
    policy_label: str = "default-policy",
    precomputed_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    import os

    os.environ["HF_HOME"] = "/cache/hf"
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

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
    gpu="A100-80GB",
    timeout=60 * 60 * 4,
    volumes={"/cache": cache_volume, "/artifacts": artifact_volume},
)
def probe_live_policy_target_matching(
    contract_data: dict[str, Any],
    policy_payload: dict[str, Any],
    policy_label: str = "default-policy",
) -> dict[str, Any]:
    import os

    os.environ["HF_HOME"] = "/cache/hf"
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

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


@app.function(
    image=quant_source_image,
    gpu="A100-80GB",
    timeout=60 * 60 * 4,
    volumes={"/cache": cache_volume, "/artifacts": artifact_volume},
)
def probe_loaded_artifact_quantization_state(
    model_ref: str,
    policy_payload: dict[str, Any] | None = None,
    policy_label: str = "default-policy",
) -> dict[str, Any]:
    import os

    os.environ["HF_HOME"] = "/cache/hf"
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    _apply_transformers_token_compat_patch()

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
    image=quant_image,
    gpu="A100-80GB",
    timeout=60 * 60 * 8,
    volumes={"/cache": cache_volume, "/artifacts": artifact_volume},
)
def probe_mixed_precision_feasibility(
    contract_data: dict[str, Any],
    calibration_limit: int = 16,
    dry_run: bool = True,
    policy_payload: dict[str, Any] | None = None,
    policy_label: str = "default-policy",
    precomputed_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    import os

    os.environ["HF_HOME"] = "/cache/hf"
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

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
    )
    if not dry_run and output_dir.startswith("/artifacts/"):
        _sync_artifact_volume()
    report["contract_name"] = contract.name
    report["model_id"] = contract.compressed_source_model_id
    report["policy_label"] = policy_label
    report["backend_variant"] = "stable"
    return report


@app.function(
    image=quant_source_image,
    gpu="A100-80GB",
    timeout=60 * 60 * 8,
    volumes={"/cache": cache_volume, "/artifacts": artifact_volume},
)
def probe_mixed_precision_feasibility_source(
    contract_data: dict[str, Any],
    calibration_limit: int = 16,
    dry_run: bool = True,
    policy_payload: dict[str, Any] | None = None,
    policy_label: str = "default-policy",
    precomputed_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    import os

    os.environ["HF_HOME"] = "/cache/hf"
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    _apply_transformers_token_compat_patch()

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
    gpu="A100-80GB",
    timeout=60 * 60 * 2,
    volumes={"/cache": cache_volume, "/artifacts": artifact_volume},
)
def smoke_test_quantized_artifact_source(
    artifact_dir: str,
    tokenizer_source: str,
    max_new_tokens: int = 8,
) -> dict[str, Any]:
    import os

    os.environ["HF_HOME"] = "/cache/hf"
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    _apply_transformers_token_compat_patch()

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


@app.function(
    image=quant_source_image,
    gpu="A100-80GB",
    timeout=60 * 60 * 4,
    volumes={"/cache": cache_volume, "/artifacts": artifact_volume},
)
def evaluate_task_source_model(
    model_ref: str,
    tokenizer_source: str,
    model_label: str,
    task_name: str,
    limit: int,
    max_new_tokens: int,
    load_dtype: str = "auto",
) -> dict[str, Any]:
    import os

    os.environ["HF_HOME"] = "/cache/hf"
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    _apply_transformers_token_compat_patch()

    from ta_mpq.baseline import evaluate_task_model

    resolved_model_ref = _resolve_remote_model_ref(model_ref)
    summary = evaluate_task_model(
        model_ref=resolved_model_ref,
        task_name=task_name,
        tokenizer_source=tokenizer_source,
        model_label=model_label,
        limit=limit,
        max_new_tokens=max_new_tokens,
        load_dtype=load_dtype,
    )
    summary["backend_variant"] = "source"
    return summary


@app.function(
    image=report_image,
    gpu="A100-80GB",
    timeout=60 * 60 * 4,
    volumes={"/cache": cache_volume, "/artifacts": artifact_volume},
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
    import os

    os.environ["HF_HOME"] = "/cache/hf"
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

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
) -> dict[str, Any]:
    from ta_mpq.search import (
        build_search_groups,
        layer_stats_from_report,
        resolve_group_value_scores,
        run_surrogate_evolution_search,
    )

    sensitivity_overrides = None
    if sensitivity_profile_payload is not None:
        from ta_mpq.sensitivity import group_sensitivity_overrides_from_profile

        sensitivity_overrides = group_sensitivity_overrides_from_profile(
            sensitivity_profile_payload,
            field=sensitivity_field,
        )

    groups = build_search_groups(
        layer_stats_from_report(report_payload),
        grouping=grouping,
        sensitivity_overrides=sensitivity_overrides,
    )
    group_value_scores = resolve_group_value_scores(groups, group_value_prior_payload)
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
    )
    return result.to_dict()


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
    max_new_tokens: int = 32,
    output_prefix: str = "candidate-01-quantized-vs-native-9b",
) -> None:
    contract = load_contract(PROJECT_ROOT / contract_path)
    resolved_task_name = task_name or contract.task_name

    quantized_summary = evaluate_task_source_model.remote(
        model_ref=artifact_dir,
        tokenizer_source=contract.compressed_source_model_id,
        model_label=f"{contract.compressed_source_model_id}-quantized",
        task_name=resolved_task_name,
        limit=limit,
        max_new_tokens=max_new_tokens,
        load_dtype="auto",
    )
    native_summary = evaluate_task_source_model.remote(
        model_ref=contract.native_baseline_model_id,
        tokenizer_source=contract.native_baseline_model_id,
        model_label=contract.native_baseline_model_id,
        task_name=resolved_task_name,
        limit=limit,
        max_new_tokens=max_new_tokens,
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
