from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

from ta_mpq.policy_export import export_candidate_from_group_bits
from ta_mpq.search import build_search_groups, layer_stats_from_report
from ta_mpq.sensitivity import group_sensitivity_overrides_from_profile


@dataclass(frozen=True, slots=True)
class PrecisionAblationGroupStat:
    name: str
    component_type: str
    layer_names: tuple[str, ...]
    parameter_count: int
    prior_sensitivity: float
    selection_score: float
    reference_bit_width: int
    ablated_bit_width: int
    reference_accuracy: float
    ablated_accuracy: float
    accuracy_drop: float
    positive_accuracy_drop: float
    improvement_if_downgraded: float
    normalized_accuracy_drop: float
    combined_sensitivity: float

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["layer_names"] = list(self.layer_names)
        return payload


def build_precision_ablation_manifest(
    report_payload: dict[str, Any],
    reference_candidate_payload: dict[str, Any],
    output_dir: str | Path,
    allowed_bits: tuple[int, ...] = (4, 8, 16),
    floor_bit: int = 4,
    max_groups: int | None = None,
    reference_bit_widths: tuple[int, ...] | None = None,
    ranking_profile_payload: dict[str, Any] | None = None,
    ranking_field: str = "combined_sensitivity",
) -> dict[str, Any]:
    grouping = str(reference_candidate_payload["grouping"])
    reference_group_bits = _load_reference_group_bits(reference_candidate_payload)

    ranking_overrides = None
    if ranking_profile_payload is not None:
        ranking_overrides = group_sensitivity_overrides_from_profile(
            ranking_profile_payload,
            field=ranking_field,
        )

    search_groups = build_search_groups(
        layer_stats_from_report(report_payload),
        grouping=grouping,
    )
    sorted_specs = _select_ablation_groups(
        search_groups=search_groups,
        reference_group_bits=reference_group_bits,
        allowed_bits=allowed_bits,
        floor_bit=floor_bit,
        max_groups=max_groups,
        reference_bit_widths=reference_bit_widths,
        ranking_overrides=ranking_overrides,
    )

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    manifest_entries: list[dict[str, Any]] = []
    for index, spec in enumerate(sorted_specs, start=1):
        ablated_group_bits = dict(reference_group_bits)
        ablated_group_bits[spec["group_name"]] = spec["ablated_bit_width"]
        exported_candidate = export_candidate_from_group_bits(
            report_payload=report_payload,
            grouping=grouping,
            group_bits=ablated_group_bits,
            rank=index,
            metadata={
                "ablation_metadata": {
                    "group_name": spec["group_name"],
                    "component_type": spec["component_type"],
                    "layer_names": list(spec["layer_names"]),
                    "parameter_count": spec["parameter_count"],
                    "selection_score": spec["selection_score"],
                    "reference_bit_width": spec["reference_bit_width"],
                    "ablated_bit_width": spec["ablated_bit_width"],
                }
            },
        )
        candidate_path = output_root / f"ablation-{index:03d}.json"
        _save_json(candidate_path, exported_candidate)

        manifest_entries.append(
            {
                **spec,
                "candidate_path": str(candidate_path),
            }
        )

    manifest = {
        "grouping": grouping,
        "allowed_bits": list(sorted(int(bit) for bit in set(allowed_bits))),
        "floor_bit": floor_bit,
        "reference_bit_widths": (
            list(sorted(int(bit) for bit in set(reference_bit_widths)))
            if reference_bit_widths is not None
            else None
        ),
        "num_ablations": len(manifest_entries),
        "reference_bit_counts": _bit_counts(reference_group_bits.values()),
        "ablations": manifest_entries,
    }
    _save_json(output_root / "manifest.json", manifest)
    return manifest


def build_precision_ablation_profile(
    report_payload: dict[str, Any],
    reference_candidate_payload: dict[str, Any],
    reference_summary_payload: dict[str, Any],
    ablation_manifest_payload: dict[str, Any],
    ablation_evaluation_payloads: dict[str, dict[str, Any]],
    prior_weight: float = 0.25,
) -> dict[str, Any]:
    if not 0.0 <= prior_weight <= 1.0:
        raise ValueError("prior_weight must be between 0 and 1")

    grouping = str(reference_candidate_payload["grouping"])
    reference_accuracy = float(reference_summary_payload.get("accuracy", 0.0))
    search_groups = build_search_groups(
        layer_stats_from_report(report_payload),
        grouping=grouping,
    )
    group_lookup = {group.name: group for group in search_groups}

    raw_positive_drops: dict[str, float] = {}
    raw_stats: list[PrecisionAblationGroupStat] = []
    for entry in ablation_manifest_payload.get("ablations", []):
        group_name = str(entry["group_name"])
        ablated_summary = ablation_evaluation_payloads.get(group_name)
        if ablated_summary is None:
            continue

        group = group_lookup[group_name]
        ablated_accuracy = float(ablated_summary.get("accuracy", 0.0))
        accuracy_drop = reference_accuracy - ablated_accuracy
        positive_drop = max(accuracy_drop, 0.0)
        raw_positive_drops[group_name] = positive_drop
        raw_stats.append(
            PrecisionAblationGroupStat(
                name=group.name,
                component_type=group.component_type,
                layer_names=group.layer_names,
                parameter_count=group.parameter_count,
                prior_sensitivity=group.sensitivity,
                selection_score=float(entry.get("selection_score", group.sensitivity)),
                reference_bit_width=int(entry["reference_bit_width"]),
                ablated_bit_width=int(entry["ablated_bit_width"]),
                reference_accuracy=reference_accuracy,
                ablated_accuracy=ablated_accuracy,
                accuracy_drop=accuracy_drop,
                positive_accuracy_drop=positive_drop,
                improvement_if_downgraded=max(ablated_accuracy - reference_accuracy, 0.0),
                normalized_accuracy_drop=0.0,
                combined_sensitivity=0.0,
            )
        )

    normalized_drops = _normalize_scores(raw_positive_drops)
    groups_payload: list[dict[str, Any]] = []
    for stat in raw_stats:
        normalized_drop = normalized_drops.get(stat.name, 0.0)
        combined_sensitivity = (
            prior_weight * stat.prior_sensitivity
            + (1.0 - prior_weight) * normalized_drop
        )
        groups_payload.append(
            PrecisionAblationGroupStat(
                name=stat.name,
                component_type=stat.component_type,
                layer_names=stat.layer_names,
                parameter_count=stat.parameter_count,
                prior_sensitivity=stat.prior_sensitivity,
                selection_score=stat.selection_score,
                reference_bit_width=stat.reference_bit_width,
                ablated_bit_width=stat.ablated_bit_width,
                reference_accuracy=stat.reference_accuracy,
                ablated_accuracy=stat.ablated_accuracy,
                accuracy_drop=stat.accuracy_drop,
                positive_accuracy_drop=stat.positive_accuracy_drop,
                improvement_if_downgraded=stat.improvement_if_downgraded,
                normalized_accuracy_drop=normalized_drop,
                combined_sensitivity=combined_sensitivity,
            ).to_dict()
        )

    groups_payload.sort(
        key=lambda item: (
            float(item["combined_sensitivity"]),
            float(item["positive_accuracy_drop"]),
            int(item["parameter_count"]),
        ),
        reverse=True,
    )

    return {
        "grouping": grouping,
        "prior_weight": prior_weight,
        "reference_accuracy": reference_accuracy,
        "num_ablations": len(groups_payload),
        "reference_bit_counts": _bit_counts(_load_reference_group_bits(reference_candidate_payload).values()),
        "groups": groups_payload,
    }


def load_evaluation_payloads_from_manifest(
    evaluation_manifest_payload: dict[str, Any],
    base_dir: str | Path | None = None,
) -> dict[str, dict[str, Any]]:
    base_path = Path(base_dir) if base_dir is not None else None
    evaluations: dict[str, dict[str, Any]] = {}
    for record in evaluation_manifest_payload.get("ablations", []):
        evaluation_path = record.get("evaluation_path")
        group_name = record.get("group_name")
        if not evaluation_path or not group_name:
            continue
        resolved_path = Path(evaluation_path)
        if base_path is not None and not resolved_path.is_absolute():
            resolved_path = base_path / resolved_path
        evaluations[str(group_name)] = _load_json(resolved_path)
    return evaluations


def _load_reference_group_bits(reference_candidate_payload: dict[str, Any]) -> dict[str, int]:
    group_bits = reference_candidate_payload.get("group_bit_assignments")
    if not group_bits:
        raise ValueError(
            "Reference candidate payload does not include group_bit_assignments; "
            "use an exported candidate JSON."
        )
    return {str(name): int(bit) for name, bit in group_bits.items()}


def _select_ablation_groups(
    search_groups: list[Any],
    reference_group_bits: dict[str, int],
    allowed_bits: tuple[int, ...],
    floor_bit: int,
    max_groups: int | None,
    reference_bit_widths: tuple[int, ...] | None,
    ranking_overrides: dict[str, float] | None,
) -> list[dict[str, Any]]:
    normalized_allowed_bits = tuple(sorted({int(bit) for bit in allowed_bits}))
    if floor_bit not in normalized_allowed_bits:
        raise ValueError("floor_bit must be included in allowed_bits")
    normalized_reference_bit_widths = (
        tuple(sorted({int(bit) for bit in reference_bit_widths}))
        if reference_bit_widths is not None
        else None
    )

    candidates: list[dict[str, Any]] = []
    for group in search_groups:
        current_bit = reference_group_bits.get(group.name)
        if current_bit is None:
            continue
        if (
            normalized_reference_bit_widths is not None
            and current_bit not in normalized_reference_bit_widths
        ):
            continue
        ablated_bit = _next_lower_bit(current_bit, normalized_allowed_bits, floor_bit)
        if ablated_bit is None:
            continue
        selection_score = (
            float(ranking_overrides[group.name])
            if ranking_overrides and group.name in ranking_overrides
            else float(group.sensitivity)
        )
        candidates.append(
            {
                "group_name": group.name,
                "component_type": group.component_type,
                "layer_names": list(group.layer_names),
                "parameter_count": group.parameter_count,
                "selection_score": selection_score,
                "reference_bit_width": current_bit,
                "ablated_bit_width": ablated_bit,
            }
        )

    candidates.sort(
        key=lambda item: (
            int(item["reference_bit_width"]),
            float(item["selection_score"]),
            int(item["parameter_count"]),
        ),
        reverse=True,
    )
    if max_groups is not None:
        candidates = candidates[:max_groups]
    return candidates


def _next_lower_bit(
    current_bit: int,
    allowed_bits: tuple[int, ...],
    floor_bit: int,
) -> int | None:
    lower_bits = [bit for bit in allowed_bits if bit < current_bit and bit >= floor_bit]
    if not lower_bits:
        return None
    return max(lower_bits)


def _normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    minimum = min(scores.values())
    maximum = max(scores.values())
    if maximum <= minimum:
        return {name: 0.0 for name in scores}
    scale = maximum - minimum
    return {name: (value - minimum) / scale for name, value in scores.items()}


def _bit_counts(bit_widths: Any) -> dict[str, int]:
    counts: dict[int, int] = {}
    for bit_width in bit_widths:
        resolved_bit = int(bit_width)
        counts[resolved_bit] = counts.get(resolved_bit, 0) + 1
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
