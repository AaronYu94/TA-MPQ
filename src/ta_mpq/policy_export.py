from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re
from typing import Any

from ta_mpq.quantization import (
    LLM_COMPRESSOR_SUPPORTED_BITS,
    HIGH_PRECISION_BIT,
    BitRule,
    MixedPrecisionPolicy,
    estimate_average_bit_width,
    estimate_weight_footprint_gb,
    to_llmcompressor_recipe_config,
    validate_policy,
)
from ta_mpq.search import build_search_groups, layer_stats_from_report


BACKEND_BIT_PROJECTION = {
    "llmcompressor": {
        2: 4,
        3: 4,
        4: 4,
        8: 8,
        16: 16,
    }
}

DEFAULT_GROUP_SIZE = 128
DEFAULT_SYMMETRIC = True


def export_top_candidates(
    report_path: str | Path,
    search_result_path: str | Path,
    output_dir: str | Path,
    top_k: int | None = None,
) -> dict[str, Any]:
    report_payload = _load_json(report_path)
    search_payload = _load_json(search_result_path)
    grouping = str(search_payload["grouping"])

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    top_candidates = list(search_payload["top_candidates"])
    if top_k is not None:
        top_candidates = top_candidates[:top_k]

    manifest_candidates: list[dict[str, Any]] = []
    for index, candidate_payload in enumerate(top_candidates, start=1):
        exported = export_candidate_payload(
            report_payload=report_payload,
            grouping=grouping,
            candidate_payload=candidate_payload,
            rank=index,
        )
        candidate_path = output_root / f"candidate-{index:02d}.json"
        _save_json(candidate_path, exported)
        manifest_candidates.append(
            {
                "rank": index,
                "path": str(candidate_path),
                "fitness": exported["fitness"],
                "estimated_average_bit_width": exported["estimated_average_bit_width"],
                "estimated_weight_footprint_gb": exported["estimated_weight_footprint_gb"],
                "matched_linear_weight_footprint_gb": exported["matched_linear_weight_footprint_gb"],
                "estimated_full_model_weight_footprint_gb": exported[
                    "estimated_full_model_weight_footprint_gb"
                ],
                "conservative_prediction": exported.get("conservative_prediction"),
                "budget_alignment_score": exported.get("budget_alignment_score"),
                "group_value_alignment_score": exported.get("group_value_alignment_score"),
                "reference_accuracy": exported.get("reference_accuracy"),
                "reference_advantage_score": exported.get("reference_advantage_score"),
                "prediction_uncertainty": exported.get("prediction_uncertainty"),
                "bit_counts": exported["bit_counts"],
                "llmcompressor_downgraded_module_count": exported["backend_projections"][
                    "llmcompressor"
                ]["downgraded_module_count"],
            }
        )

    manifest = {
        "report_path": str(Path(report_path)),
        "search_result_path": str(Path(search_result_path)),
        "grouping": grouping,
        "num_exported_candidates": len(manifest_candidates),
        "candidates": manifest_candidates,
    }
    _save_json(output_root / "manifest.json", manifest)
    return manifest


def load_policy_from_candidate(
    candidate_path: str | Path,
    source: str = "project",
) -> MixedPrecisionPolicy:
    payload = _load_json(candidate_path)

    if source == "project":
        policy_payload = payload["project_policy"]
    elif source == "llmcompressor":
        policy_payload = payload["backend_projections"]["llmcompressor"]["projected_policy"]
    else:
        raise ValueError(f"Unknown candidate policy source: {source}")

    return MixedPrecisionPolicy.from_dict(policy_payload)


def export_candidate_payload(
    report_payload: dict[str, Any],
    grouping: str,
    candidate_payload: dict[str, Any],
    rank: int,
) -> dict[str, Any]:
    group_bits = {str(name): int(bit) for name, bit in candidate_payload["group_bits"].items()}
    group_quantization_overrides = _normalize_group_quantization_overrides(
        candidate_payload.get("group_quantization_overrides")
    )
    return export_candidate_from_group_bits(
        report_payload=report_payload,
        grouping=grouping,
        group_bits=group_bits,
        group_quantization_overrides=group_quantization_overrides,
        rank=rank,
        metadata={
            "fitness": float(candidate_payload["fitness"]),
            "proxy_quality_score": float(candidate_payload["proxy_quality_score"]),
            "conservative_prediction": (
                float(candidate_payload["conservative_prediction"])
                if candidate_payload.get("conservative_prediction") is not None
                else None
            ),
            "budget_alignment_score": (
                float(candidate_payload["budget_alignment_score"])
                if candidate_payload.get("budget_alignment_score") is not None
                else None
            ),
            "group_value_alignment_score": (
                float(candidate_payload["group_value_alignment_score"])
                if candidate_payload.get("group_value_alignment_score") is not None
                else None
            ),
            "reference_accuracy": (
                float(candidate_payload["reference_accuracy"])
                if candidate_payload.get("reference_accuracy") is not None
                else None
            ),
            "reference_advantage_score": (
                float(candidate_payload["reference_advantage_score"])
                if candidate_payload.get("reference_advantage_score") is not None
                else None
            ),
            "prediction_uncertainty": (
                float(candidate_payload["prediction_uncertainty"])
                if candidate_payload.get("prediction_uncertainty") is not None
                else None
            ),
            "estimated_average_bit_width": float(candidate_payload["estimated_average_bit_width"]),
            "estimated_weight_footprint_gb": float(candidate_payload["estimated_weight_footprint_gb"]),
            "provenance": str(candidate_payload["provenance"]),
            "quantization_config_score": (
                float(candidate_payload["quantization_config_score"])
                if candidate_payload.get("quantization_config_score") is not None
                else None
            ),
        },
    )


def export_candidate_from_group_bits(
    report_payload: dict[str, Any],
    grouping: str,
    group_bits: dict[str, int],
    rank: int,
    group_quantization_overrides: dict[str, dict[str, Any]] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    layer_stats = layer_stats_from_report(report_payload)
    param_counts = {stat.name: stat.parameter_count for stat in layer_stats}
    module_assignments = expand_group_bits_to_module_assignments(
        report_payload=report_payload,
        grouping=grouping,
        group_bits=group_bits,
    )
    group_quantization_configs = build_group_quantization_configs(
        group_bits=group_bits,
        group_quantization_overrides=group_quantization_overrides,
    )
    module_quantization_overrides = expand_group_quantization_overrides(
        report_payload=report_payload,
        grouping=grouping,
        group_quantization_overrides=group_quantization_overrides or {},
    )
    module_quantization_configs = build_module_quantization_configs(
        module_assignments=module_assignments,
        module_quantization_overrides=module_quantization_overrides,
    )
    project_policy = build_project_policy_from_module_quantization_configs(module_quantization_configs)
    llmcompressor_projection = build_backend_projection(
        module_assignments=module_assignments,
        backend="llmcompressor",
        module_quantization_overrides=module_quantization_overrides,
    )
    non_linear_footprint_gb = float(report_payload.get("estimated_non_linear_weight_footprint_gb", 0.0))
    matched_linear_weight_footprint_gb = (
        float(metadata["estimated_weight_footprint_gb"])
        if metadata and metadata.get("estimated_weight_footprint_gb") is not None
        else estimate_weight_footprint_gb(param_counts, module_assignments)
    )
    estimated_full_model_weight_footprint_gb = (
        matched_linear_weight_footprint_gb + non_linear_footprint_gb
    )

    exported = {
        "rank": rank,
        "contract_name": report_payload.get("contract_name"),
        "model_id": report_payload.get("model_id"),
        "grouping": grouping,
        "budget_accounting_mode": "matched_linear_weight_budget",
        "fitness": None,
        "proxy_quality_score": None,
        "conservative_prediction": None,
        "budget_alignment_score": None,
        "group_value_alignment_score": None,
        "reference_accuracy": None,
        "reference_advantage_score": None,
        "prediction_uncertainty": None,
        "estimated_average_bit_width": (
            float(metadata["estimated_average_bit_width"])
            if metadata and metadata.get("estimated_average_bit_width") is not None
            else estimate_average_bit_width(param_counts, module_assignments)
        ),
        "estimated_weight_footprint_gb": matched_linear_weight_footprint_gb,
        "matched_linear_weight_footprint_gb": matched_linear_weight_footprint_gb,
        "estimated_full_model_weight_footprint_gb": estimated_full_model_weight_footprint_gb,
        "provenance": str(metadata["provenance"]) if metadata and metadata.get("provenance") else "manual",
        "bit_counts": _bit_counts(group_bits.values()),
        "group_bit_assignments": group_bits,
        "group_quantization_overrides": _serialize_group_quantization_overrides(
            group_quantization_overrides or {}
        ),
        "group_quantization_configs": _serialize_quantization_configs(group_quantization_configs),
        "module_assignment_count": len(module_assignments),
        "module_bit_assignments": module_assignments,
        "module_quantization_overrides": _serialize_group_quantization_overrides(
            module_quantization_overrides
        ),
        "module_quantization_configs": _serialize_quantization_configs(module_quantization_configs),
        "project_policy": project_policy.to_dict(),
        "backend_projections": {
            "llmcompressor": llmcompressor_projection,
        },
    }
    if metadata:
        for key, value in metadata.items():
            if key in {"estimated_average_bit_width", "estimated_weight_footprint_gb", "provenance"}:
                continue
            exported[key] = value
    return exported


def expand_group_bits_to_module_assignments(
    report_payload: dict[str, Any],
    grouping: str,
    group_bits: dict[str, int],
) -> dict[str, int]:
    layer_stats = layer_stats_from_report(report_payload)
    search_groups = build_search_groups(layer_stats, grouping=grouping)
    group_lookup = {group.name: group for group in search_groups}

    module_assignments: dict[str, int] = {}
    for group_name, bit_width in group_bits.items():
        group = group_lookup.get(group_name)
        if group is None:
            raise KeyError(f"Unknown search group in candidate export: {group_name}")
        for layer_name in group.layer_names:
            module_assignments[layer_name] = bit_width
    return dict(sorted(module_assignments.items()))


def expand_group_quantization_overrides(
    report_payload: dict[str, Any],
    grouping: str,
    group_quantization_overrides: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    if not group_quantization_overrides:
        return {}

    normalized_overrides = _normalize_group_quantization_overrides(group_quantization_overrides)
    layer_stats = layer_stats_from_report(report_payload)
    search_groups = build_search_groups(layer_stats, grouping=grouping)
    group_lookup = {group.name: group for group in search_groups}

    module_quantization_overrides: dict[str, dict[str, Any]] = {}
    for group_name, override in normalized_overrides.items():
        group = group_lookup.get(group_name)
        if group is None:
            raise KeyError(f"Unknown search group in candidate export: {group_name}")
        for layer_name in group.layer_names:
            module_quantization_overrides[layer_name] = dict(override)
    return {
        module_name: module_quantization_overrides[module_name]
        for module_name in sorted(module_quantization_overrides)
    }


def build_group_quantization_configs(
    group_bits: dict[str, int],
    group_quantization_overrides: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    normalized_overrides = _normalize_group_quantization_overrides(group_quantization_overrides)
    return {
        group_name: {
            "bit_width": int(bit_width),
            "group_size": int(normalized_overrides.get(group_name, {}).get("group_size", DEFAULT_GROUP_SIZE)),
            "symmetric": bool(normalized_overrides.get(group_name, {}).get("symmetric", DEFAULT_SYMMETRIC)),
        }
        for group_name, bit_width in sorted(group_bits.items())
    }


def build_module_quantization_configs(
    module_assignments: dict[str, int],
    module_quantization_overrides: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    normalized_overrides = _normalize_group_quantization_overrides(module_quantization_overrides)
    return {
        module_name: {
            "bit_width": int(bit_width),
            "group_size": int(
                normalized_overrides.get(module_name, {}).get("group_size", DEFAULT_GROUP_SIZE)
            ),
            "symmetric": bool(
                normalized_overrides.get(module_name, {}).get("symmetric", DEFAULT_SYMMETRIC)
            ),
        }
        for module_name, bit_width in sorted(module_assignments.items())
    }


def build_project_policy(
    module_assignments: dict[str, int],
    default_bit_width: int | None = None,
) -> MixedPrecisionPolicy:
    chosen_default = (
        {
            "bit_width": int(default_bit_width),
            "group_size": DEFAULT_GROUP_SIZE,
            "symmetric": DEFAULT_SYMMETRIC,
        }
        if default_bit_width is not None
        else None
    )
    module_quantization_configs = build_module_quantization_configs(module_assignments)
    return build_project_policy_from_module_quantization_configs(
        module_quantization_configs,
        default_quantization_config=chosen_default,
    )


def build_project_policy_from_module_quantization_configs(
    module_quantization_configs: dict[str, dict[str, Any]],
    default_quantization_config: dict[str, Any] | None = None,
    target_compaction_strategy: str | None = None,
) -> MixedPrecisionPolicy:
    if not module_quantization_configs:
        raise ValueError("module_quantization_configs must not be empty")

    normalized_configs = _normalize_quantization_configs(module_quantization_configs)
    ignored_modules = tuple(
        sorted(
            module_name
            for module_name, config in normalized_configs.items()
            if int(config["bit_width"]) == HIGH_PRECISION_BIT
        )
    )
    quantized_configs = {
        module_name: config
        for module_name, config in normalized_configs.items()
        if int(config["bit_width"]) != HIGH_PRECISION_BIT
    }
    if not quantized_configs:
        raise ValueError("At least one quantized module assignment is required")

    chosen_default = (
        _normalize_quantization_config(default_quantization_config)
        if default_quantization_config is not None
        else _most_common_quantization_config(quantized_configs.values())
    )
    rules = _rules_for_module_quantization_configs(
        quantized_configs,
        chosen_default,
        target_compaction_strategy=target_compaction_strategy,
    )
    policy = MixedPrecisionPolicy(
        default_bit_width=int(chosen_default["bit_width"]),
        default_targets=("Linear",),
        ignore=ignored_modules,
        rules=rules,
    )
    validate_policy(policy)
    return policy


def build_backend_projection(
    module_assignments: dict[str, int],
    backend: str,
    module_quantization_overrides: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if backend not in BACKEND_BIT_PROJECTION:
        raise ValueError(f"Unknown backend projection: {backend}")

    projection_map = BACKEND_BIT_PROJECTION[backend]
    projected_module_assignments = {
        module_name: projection_map[int(bit_width)]
        for module_name, bit_width in module_assignments.items()
    }
    projected_module_quantization_configs = build_module_quantization_configs(
        module_assignments=projected_module_assignments,
        module_quantization_overrides=module_quantization_overrides,
    )
    # The llmcompressor source backend accepts simple regex targets reliably,
    # but live runs showed misses for exact-name targets and for one giant
    # regex union. Emit one anchored suffix regex per module instead.
    target_compaction_strategy = "per_target_regex" if backend == "llmcompressor" else None
    projected_policy = build_project_policy_from_module_quantization_configs(
        projected_module_quantization_configs,
        target_compaction_strategy=target_compaction_strategy,
    )

    downgraded_modules = [
        {
            "module_name": module_name,
            "original_bit_width": original_bit,
            "projected_bit_width": projected_module_assignments[module_name],
        }
        for module_name, original_bit in module_assignments.items()
        if projected_module_assignments[module_name] != original_bit
    ]

    return {
        "supported_bits": list(LLM_COMPRESSOR_SUPPORTED_BITS),
        "projected_bit_counts": _bit_counts(projected_module_assignments.values()),
        "projected_quantization_config_counts": _quantization_config_counts(
            projected_module_quantization_configs.values()
        ),
        "downgraded_module_count": len(downgraded_modules),
        "downgraded_modules": downgraded_modules,
        "projected_module_bit_assignments": projected_module_assignments,
        "projected_module_quantization_configs": _serialize_quantization_configs(
            projected_module_quantization_configs
        ),
        "projected_policy": projected_policy.to_dict(),
        "target_compaction": {
            "strategy": target_compaction_strategy or "none",
            "pre_compaction_target_count": sum(
                len(config["targets"])
                for config in to_llmcompressor_recipe_config(
                    build_project_policy_from_module_quantization_configs(
                        projected_module_quantization_configs
                    )
                )["config_groups"].values()
            ),
            "post_compaction_target_count": sum(
                len(config["targets"])
                for config in to_llmcompressor_recipe_config(projected_policy)["config_groups"].values()
            ),
        },
        "recipe_config": to_llmcompressor_recipe_config(projected_policy),
    }


def _rules_for_module_quantization_configs(
    module_quantization_configs: dict[str, dict[str, Any]],
    default_quantization_config: dict[str, Any],
    target_compaction_strategy: str | None = None,
) -> tuple[BitRule, ...]:
    config_to_targets: dict[tuple[int, int, bool], list[str]] = {}
    default_key = _quantization_config_key(default_quantization_config)
    for module_name, config in module_quantization_configs.items():
        config_key = _quantization_config_key(config)
        if config_key == default_key:
            continue
        config_to_targets.setdefault(config_key, []).append(module_name)

    rules: list[BitRule] = []
    for bit_width, group_size, symmetric in sorted(config_to_targets):
        raw_targets = tuple(sorted(config_to_targets[(bit_width, group_size, symmetric)]))
        rules.append(
            BitRule(
                name=_rule_name_for_quantization_config(bit_width, group_size, symmetric),
                targets=_compact_rule_targets(raw_targets, strategy=target_compaction_strategy),
                bit_width=bit_width,
                group_size=group_size,
                symmetric=symmetric,
            )
        )
    return tuple(rules)


def _most_common_bit(bit_widths: Any) -> int:
    counts = Counter(int(bit_width) for bit_width in bit_widths)
    return counts.most_common(1)[0][0]


def _most_common_quantization_config(configs: Any) -> dict[str, Any]:
    normalized_configs = [_normalize_quantization_config(config) for config in configs]
    counts = Counter(_quantization_config_key(config) for config in normalized_configs)
    bit_width, group_size, symmetric = counts.most_common(1)[0][0]
    return {
        "bit_width": bit_width,
        "group_size": group_size,
        "symmetric": symmetric,
    }


def _bit_counts(bit_widths: Any) -> dict[str, int]:
    counts = Counter(int(bit_width) for bit_width in bit_widths)
    return {str(bit_width): counts[bit_width] for bit_width in sorted(counts)}


def _quantization_config_counts(configs: Any) -> dict[str, int]:
    counts = Counter(_quantization_config_key(config) for config in configs)
    return {
        _rule_name_for_quantization_config(bit_width, group_size, symmetric): counts[
            (bit_width, group_size, symmetric)
        ]
        for bit_width, group_size, symmetric in sorted(counts)
    }


def _load_json(path: str | Path) -> dict[str, Any]:
    json_path = Path(path)
    with json_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _save_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _normalize_group_quantization_overrides(
    group_quantization_overrides: dict[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    if not group_quantization_overrides:
        return {}
    normalized: dict[str, dict[str, Any]] = {}
    for group_name, override in group_quantization_overrides.items():
        normalized[str(group_name)] = {
            "group_size": int(override.get("group_size", DEFAULT_GROUP_SIZE)),
            "symmetric": bool(override.get("symmetric", DEFAULT_SYMMETRIC)),
        }
    return normalized


def _normalize_quantization_configs(
    quantization_configs: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    return {
        str(module_name): _normalize_quantization_config(config)
        for module_name, config in quantization_configs.items()
    }


def _normalize_quantization_config(config: dict[str, Any] | None) -> dict[str, Any]:
    if config is None:
        return {
            "bit_width": 4,
            "group_size": DEFAULT_GROUP_SIZE,
            "symmetric": DEFAULT_SYMMETRIC,
        }
    return {
        "bit_width": int(config["bit_width"]),
        "group_size": int(config.get("group_size", DEFAULT_GROUP_SIZE)),
        "symmetric": bool(config.get("symmetric", DEFAULT_SYMMETRIC)),
    }


def _quantization_config_key(config: dict[str, Any]) -> tuple[int, int, bool]:
    normalized = _normalize_quantization_config(config)
    return (
        int(normalized["bit_width"]),
        int(normalized["group_size"]),
        bool(normalized["symmetric"]),
    )


def _serialize_group_quantization_overrides(
    overrides: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    normalized = _normalize_group_quantization_overrides(overrides)
    return {
        name: {
            "group_size": int(config["group_size"]),
            "symmetric": bool(config["symmetric"]),
        }
        for name, config in sorted(normalized.items())
    }


def _serialize_quantization_configs(
    quantization_configs: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    normalized = _normalize_quantization_configs(quantization_configs)
    return {
        name: {
            "bit_width": int(config["bit_width"]),
            "group_size": int(config["group_size"]),
            "symmetric": bool(config["symmetric"]),
        }
        for name, config in sorted(normalized.items())
    }


def _rule_name_for_quantization_config(
    bit_width: int,
    group_size: int,
    symmetric: bool,
) -> str:
    symmetry_label = "sym" if symmetric else "asym"
    return f"bit_{bit_width}_gs_{group_size}_{symmetry_label}"


def _compact_rule_targets(
    targets: tuple[str, ...],
    strategy: str | None,
) -> tuple[str, ...]:
    if strategy == "per_target_regex":
        return tuple(_module_name_to_target_regex(target) for target in targets)
    if strategy != "regex_union" or len(targets) <= 1:
        return targets
    if any(target.startswith("re:") for target in targets):
        return targets
    escaped_targets = "|".join(re.escape(target) for target in sorted(targets))
    return (f"re:^(?:{escaped_targets})$",)


def _module_name_to_target_regex(module_name: str) -> str:
    if module_name.startswith("re:"):
        return module_name
    return f"re:.*{re.escape(module_name)}$"
