from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
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
    return export_candidate_from_group_bits(
        report_payload=report_payload,
        grouping=grouping,
        group_bits=group_bits,
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
        },
    )


def export_candidate_from_group_bits(
    report_payload: dict[str, Any],
    grouping: str,
    group_bits: dict[str, int],
    rank: int,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    layer_stats = layer_stats_from_report(report_payload)
    param_counts = {stat.name: stat.parameter_count for stat in layer_stats}
    module_assignments = expand_group_bits_to_module_assignments(
        report_payload=report_payload,
        grouping=grouping,
        group_bits=group_bits,
    )
    project_policy = build_project_policy(module_assignments)
    llmcompressor_projection = build_backend_projection(
        module_assignments=module_assignments,
        backend="llmcompressor",
    )

    exported = {
        "rank": rank,
        "contract_name": report_payload.get("contract_name"),
        "model_id": report_payload.get("model_id"),
        "grouping": grouping,
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
        "estimated_weight_footprint_gb": (
            float(metadata["estimated_weight_footprint_gb"])
            if metadata and metadata.get("estimated_weight_footprint_gb") is not None
            else estimate_weight_footprint_gb(param_counts, module_assignments)
        ),
        "provenance": str(metadata["provenance"]) if metadata and metadata.get("provenance") else "manual",
        "bit_counts": _bit_counts(group_bits.values()),
        "group_bit_assignments": group_bits,
        "module_assignment_count": len(module_assignments),
        "module_bit_assignments": module_assignments,
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


def build_project_policy(
    module_assignments: dict[str, int],
    default_bit_width: int | None = None,
) -> MixedPrecisionPolicy:
    if not module_assignments:
        raise ValueError("module_assignments must not be empty")

    ignored_modules = tuple(
        sorted(
            module_name
            for module_name, bit_width in module_assignments.items()
            if int(bit_width) == HIGH_PRECISION_BIT
        )
    )
    quantized_assignments = {
        module_name: int(bit_width)
        for module_name, bit_width in module_assignments.items()
        if int(bit_width) != HIGH_PRECISION_BIT
    }
    if not quantized_assignments:
        raise ValueError("At least one quantized module assignment is required")

    chosen_default = default_bit_width or _most_common_bit(quantized_assignments.values())
    rules = _rules_for_module_assignments(quantized_assignments, chosen_default)
    policy = MixedPrecisionPolicy(
        default_bit_width=chosen_default,
        default_targets=("Linear",),
        ignore=ignored_modules,
        rules=rules,
    )
    validate_policy(policy)
    return policy


def build_backend_projection(
    module_assignments: dict[str, int],
    backend: str,
) -> dict[str, Any]:
    if backend not in BACKEND_BIT_PROJECTION:
        raise ValueError(f"Unknown backend projection: {backend}")

    projection_map = BACKEND_BIT_PROJECTION[backend]
    projected_module_assignments = {
        module_name: projection_map[int(bit_width)]
        for module_name, bit_width in module_assignments.items()
    }
    projected_policy = build_project_policy(projected_module_assignments)

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
        "downgraded_module_count": len(downgraded_modules),
        "downgraded_modules": downgraded_modules,
        "projected_module_bit_assignments": projected_module_assignments,
        "projected_policy": projected_policy.to_dict(),
        "recipe_config": to_llmcompressor_recipe_config(projected_policy),
    }


def _rules_for_module_assignments(
    module_assignments: dict[str, int],
    default_bit_width: int,
) -> tuple[BitRule, ...]:
    bit_to_targets: dict[int, list[str]] = {}
    for module_name, bit_width in module_assignments.items():
        if bit_width == default_bit_width:
            continue
        bit_to_targets.setdefault(bit_width, []).append(module_name)

    rules: list[BitRule] = []
    for bit_width in sorted(bit_to_targets):
        rules.append(
            BitRule(
                name=f"bit_{bit_width}",
                targets=tuple(sorted(bit_to_targets[bit_width])),
                bit_width=bit_width,
            )
        )
    return tuple(rules)


def _most_common_bit(bit_widths: Any) -> int:
    counts = Counter(int(bit_width) for bit_width in bit_widths)
    return counts.most_common(1)[0][0]


def _bit_counts(bit_widths: Any) -> dict[str, int]:
    counts = Counter(int(bit_width) for bit_width in bit_widths)
    return {str(bit_width): counts[bit_width] for bit_width in sorted(counts)}


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
