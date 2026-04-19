from __future__ import annotations

from dataclasses import asdict, dataclass
import gc
from pathlib import Path
import re
import shutil
import traceback
from typing import Any

from ta_mpq.quantization import (
    MixedPrecisionPolicy,
    assign_bits_to_modules,
    default_feasibility_policy,
    estimate_average_bit_width,
    estimate_weight_footprint_gb,
    to_llmcompressor_recipe_config,
)
from ta_mpq.transformers_compat import apply_qwen3_5_fast_path_compat_patch


@dataclass(frozen=True, slots=True)
class LinearLayerStat:
    name: str
    parameter_count: int
    in_features: int
    out_features: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LinearLayerStat":
        return cls(
            name=str(payload["name"]),
            parameter_count=int(payload["parameter_count"]),
            in_features=int(payload["in_features"]),
            out_features=int(payload["out_features"]),
        )


@dataclass(frozen=True, slots=True)
class ModelWeightInventory:
    layer_stats: tuple[LinearLayerStat, ...]
    total_parameter_count: int
    total_linear_parameters: int
    total_non_linear_parameters: int


def collect_model_weight_inventory(model_id: str) -> ModelWeightInventory:
    import torch

    apply_qwen3_5_fast_path_compat_patch()
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map="auto",
    )

    try:
        total_parameter_count = sum(int(parameter.numel()) for parameter in model.parameters())
        layer_stats: list[LinearLayerStat] = []
        for name, module in model.named_modules():
            if module.__class__.__name__ != "Linear":
                continue
            weight = getattr(module, "weight", None)
            if weight is None:
                continue
            out_features = int(getattr(module, "out_features", weight.shape[0]))
            in_features = int(getattr(module, "in_features", weight.shape[1]))
            layer_stats.append(
                LinearLayerStat(
                    name=name,
                    parameter_count=int(weight.numel()),
                    in_features=in_features,
                    out_features=out_features,
                )
            )
        total_linear_parameters = sum(layer.parameter_count for layer in layer_stats)
        total_non_linear_parameters = max(total_parameter_count - total_linear_parameters, 0)
        return ModelWeightInventory(
            layer_stats=tuple(layer_stats),
            total_parameter_count=total_parameter_count,
            total_linear_parameters=total_linear_parameters,
            total_non_linear_parameters=total_non_linear_parameters,
        )
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def collect_linear_layer_stats(model_id: str) -> list[LinearLayerStat]:
    return list(collect_model_weight_inventory(model_id).layer_stats)


def layer_stats_from_report_payload(payload: dict[str, Any]) -> list[LinearLayerStat]:
    raw_layer_stats = payload.get("layer_stats")
    if not raw_layer_stats:
        raise ValueError("report payload does not include layer_stats")
    return [LinearLayerStat.from_dict(item) for item in raw_layer_stats]


def build_feasibility_report(
    layer_stats: list[LinearLayerStat],
    policy: MixedPrecisionPolicy | None = None,
    total_model_parameters: int | None = None,
    non_linear_parameter_count: int | None = None,
    non_linear_bit_width: int = 16,
) -> dict[str, Any]:
    active_policy = policy or default_feasibility_policy()
    module_names = [layer.name for layer in layer_stats]
    param_counts = {layer.name: layer.parameter_count for layer in layer_stats}
    assignments = assign_bits_to_modules(module_names, active_policy)
    recipe_config = to_llmcompressor_recipe_config(active_policy)
    total_linear_parameters = sum(layer.parameter_count for layer in layer_stats)
    tracked_weight_footprint_gb = estimate_weight_footprint_gb(param_counts, assignments)

    rule_hits: dict[str, int] = {}
    for rule in active_policy.rules:
        rule_hits[rule.name] = sum(
            1 for layer_name, bit_width in assignments.items() if bit_width == rule.bit_width
        )

    report = {
        "num_linear_layers": len(layer_stats),
        "total_linear_parameters": total_linear_parameters,
        "layer_stats": [layer.to_dict() for layer in layer_stats],
        "policy": active_policy.to_dict(),
        "llmcompressor_recipe_config": recipe_config,
        "budget_accounting_mode": "matched_linear_weight_budget",
        "estimated_average_bit_width": estimate_average_bit_width(param_counts, assignments),
        "estimated_weight_footprint_gb": tracked_weight_footprint_gb,
        "matched_linear_weight_footprint_gb": tracked_weight_footprint_gb,
        "rule_hits": rule_hits,
        "sample_assignments": dict(list(assignments.items())[:20]),
    }
    if total_model_parameters is not None:
        resolved_non_linear_parameter_count = (
            int(non_linear_parameter_count)
            if non_linear_parameter_count is not None
            else max(int(total_model_parameters) - total_linear_parameters, 0)
        )
        non_linear_weight_footprint_gb = (
            resolved_non_linear_parameter_count * int(non_linear_bit_width)
        ) / 8 / (1024**3)
        report["total_model_parameters"] = int(total_model_parameters)
        report["total_non_linear_parameters"] = resolved_non_linear_parameter_count
        report["non_linear_weight_bit_width_assumption"] = int(non_linear_bit_width)
        report["estimated_non_linear_weight_footprint_gb"] = non_linear_weight_footprint_gb
        report["estimated_full_model_weight_footprint_gb"] = (
            tracked_weight_footprint_gb + non_linear_weight_footprint_gb
        )
    return report


def inspect_policy_targets_against_named_modules(
    named_module_types: dict[str, str],
    recipe_config: dict[str, Any],
) -> dict[str, Any]:
    config_groups = dict(recipe_config.get("config_groups", {}))
    all_targets: list[str] = []
    all_matched_modules: set[str] = set()
    all_unmatched_targets: list[str] = []
    kind_totals = {
        "exact_name": 0,
        "regex": 0,
        "class_name": 0,
        "unknown": 0,
    }

    group_summaries: dict[str, Any] = {}
    for group_name, group_config in config_groups.items():
        targets = [str(target) for target in group_config.get("targets", [])]
        all_targets.extend(targets)

        matched_targets: list[str] = []
        unmatched_targets: list[str] = []
        matched_modules: set[str] = set()
        per_target_match_counts: dict[str, int] = {}
        kind_counts = {
            "exact_name": 0,
            "regex": 0,
            "class_name": 0,
            "unknown": 0,
        }

        for target in targets:
            target_kind = _target_kind(target, named_module_types)
            kind_counts[target_kind] += 1
            kind_totals[target_kind] += 1

            matched_names = _match_target_to_module_names(target, named_module_types)
            per_target_match_counts[target] = len(matched_names)
            if matched_names:
                matched_targets.append(target)
                matched_modules.update(matched_names)
                all_matched_modules.update(matched_names)
            else:
                unmatched_targets.append(target)
                all_unmatched_targets.append(target)

        group_summaries[group_name] = {
            "num_targets": len(targets),
            "matched_target_count": len(matched_targets),
            "unmatched_target_count": len(unmatched_targets),
            "matched_module_count": len(matched_modules),
            "target_kind_counts": kind_counts,
            "matched_targets": matched_targets,
            "unmatched_targets": unmatched_targets,
            "matched_targets_sample": matched_targets[:20],
            "matched_modules_sample": sorted(matched_modules)[:20],
            "per_target_match_counts": per_target_match_counts,
            "per_target_match_counts_sample": {
                target: per_target_match_counts[target]
                for target in list(per_target_match_counts)[:20]
            },
        }

    ignore_targets = [str(target) for target in recipe_config.get("ignore", [])]
    ignore_matched_targets: list[str] = []
    ignore_unmatched_targets: list[str] = []
    ignore_matched_modules: set[str] = set()
    ignore_per_target_match_counts: dict[str, int] = {}
    ignore_kind_counts = {
        "exact_name": 0,
        "regex": 0,
        "class_name": 0,
        "unknown": 0,
    }
    for target in ignore_targets:
        target_kind = _target_kind(target, named_module_types)
        ignore_kind_counts[target_kind] += 1
        matched_names = _match_target_to_module_names(target, named_module_types)
        ignore_per_target_match_counts[target] = len(matched_names)
        if matched_names:
            ignore_matched_targets.append(target)
            ignore_matched_modules.update(matched_names)
        else:
            ignore_unmatched_targets.append(target)

    return {
        "num_named_modules": len(named_module_types),
        "num_recipe_targets": len(all_targets),
        "all_targets": all_targets,
        "matched_target_count": len(all_targets) - len(all_unmatched_targets),
        "unmatched_target_count": len(all_unmatched_targets),
        "matched_module_count": len(all_matched_modules),
        "matched_targets": sorted(
            {
                target
                for group_summary in group_summaries.values()
                for target in group_summary.get("matched_targets", [])
            }
        ),
        "target_kind_counts": kind_totals,
        "all_unmatched_targets": all_unmatched_targets,
        "all_matched_modules_sample": sorted(all_matched_modules)[:40],
        "ignore_targets": ignore_targets,
        "ignore_matched_target_count": len(ignore_matched_targets),
        "ignore_unmatched_target_count": len(ignore_unmatched_targets),
        "ignore_matched_targets": ignore_matched_targets,
        "ignore_unmatched_targets": ignore_unmatched_targets,
        "ignore_matched_module_count": len(ignore_matched_modules),
        "ignore_target_kind_counts": ignore_kind_counts,
        "ignore_per_target_match_counts": ignore_per_target_match_counts,
        "ignore_matched_modules_sample": sorted(ignore_matched_modules)[:20],
        "group_summaries": group_summaries,
    }


def build_policy_target_integrity_manifest(
    policy: MixedPrecisionPolicy,
    source_target_matching: dict[str, Any] | None,
    reload_target_matching: dict[str, Any] | None,
) -> dict[str, Any]:
    recipe_config = to_llmcompressor_recipe_config(policy)
    intended_rule_targets: list[dict[str, Any]] = []
    for group_name, group_config in dict(recipe_config.get("config_groups", {})).items():
        if group_name == "default":
            continue
        for target in group_config.get("targets", []):
            intended_rule_targets.append(
                {
                    "origin": "config_group",
                    "group_name": str(group_name),
                    "target": str(target),
                }
            )

    intended_ignore_targets = [
        {
            "origin": "ignore",
            "group_name": "ignore",
            "target": str(target),
        }
        for target in recipe_config.get("ignore", [])
    ]
    intended_targets = intended_rule_targets + intended_ignore_targets

    source_matched_targets = _resolved_target_matches(source_target_matching)
    source_unmatched_targets = _resolved_target_misses(source_target_matching)
    reload_matched_targets = _resolved_target_matches(reload_target_matching)
    reload_unmatched_targets = _resolved_target_misses(reload_target_matching)

    target_records: list[dict[str, Any]] = []
    unresolved_target_warnings: list[str] = []
    for item in intended_targets:
        target = str(item["target"])
        source_matched = target in source_matched_targets
        reload_matched = target in reload_matched_targets
        resolved = source_matched and reload_matched
        if not resolved:
            unresolved_target_warnings.append(target)
        target_records.append(
            {
                **item,
                "source_matched": source_matched,
                "reload_matched": reload_matched,
                "source_unmatched": target in source_unmatched_targets,
                "reload_unmatched": target in reload_unmatched_targets,
                "resolved": resolved,
            }
        )

    return {
        "num_intended_non_default_targets": len(target_records),
        "intended_non_default_targets": target_records,
        "source_matched_targets": sorted(source_matched_targets),
        "source_unmatched_targets": sorted(source_unmatched_targets),
        "reload_matched_targets": sorted(reload_matched_targets),
        "reload_unmatched_targets": sorted(reload_unmatched_targets),
        "unresolved_target_warnings": sorted(set(unresolved_target_warnings)),
        "is_clean": len(set(unresolved_target_warnings)) == 0,
    }


def inspect_live_policy_target_matching(
    model_id: str,
    policy: MixedPrecisionPolicy,
) -> dict[str, Any]:
    import torch

    apply_qwen3_5_fast_path_compat_patch()
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    try:
        named_module_types = {
            name: module.__class__.__name__ for name, module in model.named_modules()
        }
        summary = inspect_policy_targets_against_named_modules(
            named_module_types=named_module_types,
            recipe_config=to_llmcompressor_recipe_config(policy),
        )
        summary["model_id"] = model_id
        summary["num_linear_named_modules"] = sum(
            1 for class_name in named_module_types.values() if _module_class_matches(class_name, "Linear")
        )
        return summary
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def inspect_loaded_model_quantization_state(
    model_ref: str,
    policy: MixedPrecisionPolicy | None = None,
) -> dict[str, Any]:
    import torch

    apply_qwen3_5_fast_path_compat_patch()
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_ref,
        trust_remote_code=True,
        dtype="auto",
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    try:
        named_module_types = {
            name: module.__class__.__name__ for name, module in model.named_modules()
        }
        report: dict[str, Any] = {
            "model_ref": model_ref,
            "num_named_modules": len(named_module_types),
            "num_linear_named_modules": sum(
                1
                for class_name in named_module_types.values()
                if _module_class_matches(class_name, "Linear")
            ),
        }

        if policy is not None:
            report["policy_target_matching"] = inspect_policy_targets_against_named_modules(
                named_module_types=named_module_types,
                recipe_config=to_llmcompressor_recipe_config(policy),
            )

        quantized_modules: list[dict[str, Any]] = []
        bit_histogram: dict[str, int] = {}
        for name, module in model.named_modules():
            scheme = getattr(module, "quantization_scheme", None)
            if scheme is None:
                continue
            weights = getattr(scheme, "weights", None)
            weight_bits = getattr(weights, "num_bits", None) if weights is not None else None
            if weight_bits is not None:
                bit_histogram[str(int(weight_bits))] = bit_histogram.get(str(int(weight_bits)), 0) + 1
            quantized_modules.append(
                {
                    "name": name,
                    "module_class": module.__class__.__name__,
                    "weight_bits": int(weight_bits) if weight_bits is not None else None,
                }
            )

        report["quantized_module_count"] = len(quantized_modules)
        report["quantized_weight_bit_histogram"] = bit_histogram
        report["quantized_modules_sample"] = quantized_modules[:40]
        report["quantized_8bit_modules_sample"] = [
            item["name"] for item in quantized_modules if item["weight_bits"] == 8
        ][:60]
        return report
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def maybe_run_llmcompressor_oneshot(
    model_id: str,
    output_dir: str | Path,
    policy: MixedPrecisionPolicy | None = None,
    calibration_limit: int = 16,
    max_seq_length: int = 2048,
    dry_run: bool = True,
    precomputed_report: dict[str, Any] | None = None,
    processor_strategy: str = "auto",
    oneshot_device: str | None = None,
) -> dict[str, Any]:
    apply_qwen3_5_fast_path_compat_patch()
    active_policy = policy or default_feasibility_policy()
    output_path = Path(output_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        if precomputed_report is not None:
            layer_stats = layer_stats_from_report_payload(precomputed_report)
            total_model_parameters = precomputed_report.get("total_model_parameters")
            non_linear_parameter_count = precomputed_report.get("total_non_linear_parameters")
        else:
            weight_inventory = collect_model_weight_inventory(model_id)
            layer_stats = list(weight_inventory.layer_stats)
            total_model_parameters = weight_inventory.total_parameter_count
            non_linear_parameter_count = weight_inventory.total_non_linear_parameters
        report = build_feasibility_report(
            layer_stats,
            active_policy,
            total_model_parameters=(
                int(total_model_parameters) if total_model_parameters is not None else None
            ),
            non_linear_parameter_count=(
                int(non_linear_parameter_count) if non_linear_parameter_count is not None else None
            ),
        )
        report["policy_target_matching"] = inspect_policy_targets_against_named_modules(
            named_module_types={layer.name: "Linear" for layer in layer_stats},
            recipe_config=report["llmcompressor_recipe_config"],
        )
    except Exception as exc:
        return {
            "model_id": model_id,
            "output_dir": str(output_path),
            "dry_run": dry_run,
            "policy": active_policy.to_dict(),
            "status": "report_build_failed",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
        }

    report["output_dir"] = str(output_path)
    report["dry_run"] = dry_run
    report["oneshot_device"] = oneshot_device

    if dry_run:
        report["status"] = "dry_run_report_ready"
        return report

    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier

    try:
        if output_path.exists():
            shutil.rmtree(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        oneshot_kwargs = _build_oneshot_kwargs(
            model_id=model_id,
            processor_strategy=processor_strategy,
            oneshot_device=oneshot_device,
        )

        modifier = QuantizationModifier(
            config_groups=report["llmcompressor_recipe_config"]["config_groups"],
            ignore=report["llmcompressor_recipe_config"]["ignore"],
        )
        oneshot(
            model=model_id,
            dataset="gsm8k",
            dataset_config_name="main",
            recipe=[modifier],
            output_dir=str(output_path),
            max_seq_length=max_seq_length,
            num_calibration_samples=calibration_limit,
            **oneshot_kwargs,
        )
    except Exception as exc:
        report["status"] = "oneshot_failed"
        report["error_type"] = type(exc).__name__
        report["error_message"] = str(exc)
        report["traceback"] = traceback.format_exc()
        return report

    report["status"] = "oneshot_succeeded"
    report["artifact_dir_exists"] = output_path.exists()
    report["artifact_sample"] = sorted(
        str(path.relative_to(output_path))
        for path in output_path.rglob("*")
        if path.is_file()
    )[:20]

    try:
        report["smoke_test"] = run_quantized_smoke_test(
            model_path=output_path,
            tokenizer_source=model_id,
        )
        report["status"] = "smoke_test_succeeded"
        report["quantized_model_runnable"] = True
    except Exception as exc:
        report["status"] = "smoke_test_failed"
        report["quantized_model_runnable"] = False
        report["smoke_test_error_type"] = type(exc).__name__
        report["smoke_test_error_message"] = str(exc)
        report["smoke_test_traceback"] = traceback.format_exc()
    return report


def _build_oneshot_kwargs(
    model_id: str,
    processor_strategy: str,
    oneshot_device: str | None,
) -> dict[str, Any]:
    oneshot_kwargs: dict[str, Any] = {}
    if processor_strategy == "tokenizer":
        apply_qwen3_5_fast_path_compat_patch()
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
        )
        oneshot_kwargs["tokenizer"] = tokenizer
    elif processor_strategy != "auto":
        raise ValueError(f"Unknown processor_strategy: {processor_strategy}")

    return oneshot_kwargs


def _target_kind(target: str, named_module_types: dict[str, str]) -> str:
    if target.startswith("re:"):
        return "regex"
    if target in named_module_types:
        return "exact_name"
    if any(_module_class_matches(class_name, target) for class_name in named_module_types.values()):
        return "class_name"
    return "unknown"


def _match_target_to_module_names(
    target: str,
    named_module_types: dict[str, str],
) -> list[str]:
    if target.startswith("re:"):
        pattern = re.compile(target.removeprefix("re:"))
        return [name for name in named_module_types if pattern.match(name)]
    if target in named_module_types:
        return [target]
    return [
        name
        for name, class_name in named_module_types.items()
        if _module_class_matches(class_name, target)
    ]


def _module_class_matches(class_name: str, target: str) -> bool:
    return class_name == target or (class_name == "LinearBase" and target == "Linear")


def _resolved_target_matches(target_matching: dict[str, Any] | None) -> set[str]:
    if not target_matching:
        return set()
    return {
        str(target)
        for target in [
            *list(target_matching.get("matched_targets", [])),
            *list(target_matching.get("ignore_matched_targets", [])),
        ]
    }


def _resolved_target_misses(target_matching: dict[str, Any] | None) -> set[str]:
    if not target_matching:
        return set()
    return {
        str(target)
        for target in [
            *list(target_matching.get("all_unmatched_targets", [])),
            *list(target_matching.get("ignore_unmatched_targets", [])),
        ]
    }


def run_quantized_smoke_test(
    model_path: str | Path,
    tokenizer_source: str,
    prompt: str = "What is 2 + 2? Answer with a number only.",
    max_new_tokens: int = 8,
) -> dict[str, Any]:
    import torch
    model_ref = str(model_path)
    try:
        result = _run_quantized_smoke_test_once(
            model_ref=model_ref,
            tokenizer_source=tokenizer_source,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            device_strategy="gpu_auto",
        )
        result["execution_strategy"] = "gpu_auto"
        return result
    except torch.OutOfMemoryError as gpu_exc:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        cpu_result = _run_quantized_smoke_test_once(
            model_ref=model_ref,
            tokenizer_source=tokenizer_source,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            device_strategy="cpu",
        )
        cpu_result["execution_strategy"] = "cpu_fallback"
        cpu_result["fallback_reason"] = "gpu_oom"
        cpu_result["gpu_oom_error_type"] = type(gpu_exc).__name__
        cpu_result["gpu_oom_error_message"] = str(gpu_exc)
        return cpu_result


def _run_quantized_smoke_test_once(
    model_ref: str,
    tokenizer_source: str,
    prompt: str,
    max_new_tokens: int,
    device_strategy: str,
) -> dict[str, Any]:
    import torch

    apply_qwen3_5_fast_path_compat_patch()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        trust_remote_code=True,
    )

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "dtype": "auto",
        "low_cpu_mem_usage": True,
    }
    if device_strategy == "gpu_auto":
        model_kwargs["device_map"] = "auto"
    elif device_strategy == "cpu":
        model_kwargs["device_map"] = {"": "cpu"}
    else:
        raise ValueError(f"Unknown device_strategy: {device_strategy}")

    model = AutoModelForCausalLM.from_pretrained(
        model_ref,
        **model_kwargs,
    )

    try:
        input_device = _infer_input_device(model)
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {name: tensor.to(input_device) for name, tensor in inputs.items()}
        generated = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
        )
        prompt_length = int(inputs["input_ids"].shape[1])
        completion_ids = generated[0][prompt_length:]
        completion = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
        return {
            "prompt": prompt,
            "completion": completion,
            "max_new_tokens": max_new_tokens,
            "device_strategy": device_strategy,
            "input_device": str(input_device),
        }
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _infer_input_device(model: Any) -> Any:
    import torch

    if hasattr(model, "hf_device_map") and isinstance(model.hf_device_map, dict):
        for device in model.hf_device_map.values():
            if isinstance(device, str) and device not in {"cpu", "disk", "meta"}:
                return torch.device(device)

    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device

    return torch.device("cpu")
