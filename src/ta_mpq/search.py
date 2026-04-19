from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
import re
import statistics
from typing import Any, Callable

from ta_mpq.feasibility import LinearLayerStat
from ta_mpq.quantization import HIGH_PRECISION_BIT
from ta_mpq.sensitivity import group_sensitivity_overrides_from_profile


DEFAULT_GROUPING = "per_block_component"
BLOCK_WINDOW_SIZE = 4
EVOLUTION_SEARCH_SUPPORTED_BITS = (2, 4, 8, 16)
DEFAULT_WEIGHT_GROUP_SIZE = 128
DEFAULT_WEIGHT_SYMMETRIC = True
DEFAULT_GROUP_SIZE_OPTIONS = (32, 64, 128)
DEFAULT_SYMMETRIC_OPTIONS = (True, False)
DEFAULT_QUALITY_PRIOR = {
    2: 0.62,
    3: 0.76,
    4: 0.88,
    8: 1.0,
    16: 1.03,
}


@dataclass(frozen=True, slots=True)
class SearchGroup:
    name: str
    component_type: str
    layer_names: tuple[str, ...]
    parameter_count: int
    sensitivity: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SearchCandidate:
    group_bits: tuple[tuple[str, int], ...]
    estimated_average_bit_width: float
    estimated_weight_footprint_gb: float
    proxy_quality_score: float
    fitness: float
    provenance: str
    group_quantization_overrides: tuple[tuple[str, int, bool], ...] = tuple()
    quantization_config_score: float | None = None
    prediction_uncertainty: float | None = None
    conservative_prediction: float | None = None
    budget_alignment_score: float | None = None
    group_value_alignment_score: float | None = None
    reference_accuracy: float | None = None
    reference_advantage_score: float | None = None

    def bits_dict(self) -> dict[str, int]:
        return dict(self.group_bits)

    def quantization_overrides_dict(self) -> dict[str, dict[str, Any]]:
        return {
            group_name: {
                "group_size": int(group_size),
                "symmetric": bool(symmetric),
            }
            for group_name, group_size, symmetric in self.group_quantization_overrides
        }

    def candidate_signature(self) -> tuple[Any, ...]:
        return (
            self.group_bits,
            self.group_quantization_overrides,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["group_bits"] = dict(self.group_bits)
        payload["group_quantization_overrides"] = self.quantization_overrides_dict()
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SearchCandidate":
        group_quantization_overrides = tuple(
            sorted(
                (
                    str(group_name),
                    int(config.get("group_size", DEFAULT_WEIGHT_GROUP_SIZE)),
                    bool(config.get("symmetric", DEFAULT_WEIGHT_SYMMETRIC)),
                )
                for group_name, config in payload.get("group_quantization_overrides", {}).items()
            )
        )
        return cls(
            group_bits=tuple(
                sorted(
                    (str(group_name), int(bit_width))
                    for group_name, bit_width in payload["group_bits"].items()
                )
            ),
            estimated_average_bit_width=float(payload["estimated_average_bit_width"]),
            estimated_weight_footprint_gb=float(payload["estimated_weight_footprint_gb"]),
            proxy_quality_score=float(payload["proxy_quality_score"]),
            fitness=float(payload["fitness"]),
            provenance=str(payload["provenance"]),
            group_quantization_overrides=group_quantization_overrides,
            quantization_config_score=(
                float(payload["quantization_config_score"])
                if payload.get("quantization_config_score") is not None
                else None
            ),
            prediction_uncertainty=(
                float(payload["prediction_uncertainty"])
                if payload.get("prediction_uncertainty") is not None
                else None
            ),
            conservative_prediction=(
                float(payload["conservative_prediction"])
                if payload.get("conservative_prediction") is not None
                else None
            ),
            budget_alignment_score=(
                float(payload["budget_alignment_score"])
                if payload.get("budget_alignment_score") is not None
                else None
            ),
            group_value_alignment_score=(
                float(payload["group_value_alignment_score"])
                if payload.get("group_value_alignment_score") is not None
                else None
            ),
            reference_accuracy=(
                float(payload["reference_accuracy"])
                if payload.get("reference_accuracy") is not None
                else None
            ),
            reference_advantage_score=(
                float(payload["reference_advantage_score"])
                if payload.get("reference_advantage_score") is not None
                else None
            ),
        )


@dataclass(frozen=True, slots=True)
class SearchGenerationSummary:
    generation: int
    best_fitness: float
    mean_fitness: float
    best_proxy_quality_score: float
    best_estimated_weight_footprint_gb: float
    unique_candidate_count: int
    mean_population_diversity: float
    mutation_rate: float
    stagnation_steps: int
    immigrant_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SearchRunResult:
    grouping: str
    target_budget_gb: float
    allowed_bits: tuple[int, ...]
    population_size: int
    generations: int
    num_groups: int
    top_candidates: tuple[SearchCandidate, ...]
    history: tuple[SearchGenerationSummary, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "grouping": self.grouping,
            "target_budget_gb": self.target_budget_gb,
            "allowed_bits": list(self.allowed_bits),
            "population_size": self.population_size,
            "generations": self.generations,
            "num_groups": self.num_groups,
            "top_candidates": [candidate.to_dict() for candidate in self.top_candidates],
            "history": [entry.to_dict() for entry in self.history],
        }


def build_search_groups(
    layer_stats: list[LinearLayerStat],
    grouping: str = DEFAULT_GROUPING,
    sensitivity_overrides: dict[str, float] | None = None,
) -> list[SearchGroup]:
    grouped_layers: dict[str, list[LinearLayerStat]] = {}
    component_types: dict[str, str] = {}

    for layer in layer_stats:
        group_name, component_type = _group_key_for_layer(layer.name, grouping)
        grouped_layers.setdefault(group_name, []).append(layer)
        component_types[group_name] = component_type

    search_groups: list[SearchGroup] = []
    for group_name in sorted(grouped_layers):
        members = grouped_layers[group_name]
        total_params = sum(member.parameter_count for member in members)
        prior_sensitivity = _weighted_mean(
            [
                (_sensitivity_prior(component_types[group_name]), member.parameter_count)
                for member in members
            ]
        )
        sensitivity = prior_sensitivity
        if sensitivity_overrides and group_name in sensitivity_overrides:
            sensitivity = float(sensitivity_overrides[group_name])
        search_groups.append(
            SearchGroup(
                name=group_name,
                component_type=component_types[group_name],
                layer_names=tuple(member.name for member in members),
                parameter_count=total_params,
                sensitivity=sensitivity,
            )
        )
    return search_groups


def layer_stats_from_report(payload: dict[str, Any]) -> list[LinearLayerStat]:
    raw_layer_stats = payload.get("layer_stats")
    if not raw_layer_stats:
        raise ValueError("Feasibility report does not include layer_stats")
    return [
        LinearLayerStat(
            name=str(item["name"]),
            parameter_count=int(item["parameter_count"]),
            in_features=int(item["in_features"]),
            out_features=int(item["out_features"]),
        )
        for item in raw_layer_stats
    ]


def load_layer_stats_from_report(path: str | Path) -> list[LinearLayerStat]:
    report_path = Path(path)
    with report_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return layer_stats_from_report(payload)


def run_proxy_search_from_report(
    report_path: str | Path,
    target_budget_gb: float,
    allowed_bits: tuple[int, ...],
    grouping: str = DEFAULT_GROUPING,
    sensitivity_profile_path: str | Path | None = None,
    sensitivity_field: str = "combined_sensitivity",
    population_size: int = 32,
    generations: int = 15,
    elite_count: int = 4,
    tournament_size: int = 3,
    mutation_rate: float = 0.1,
    top_k: int = 5,
    seed: int = 0,
) -> SearchRunResult:
    layer_stats = load_layer_stats_from_report(report_path)
    sensitivity_overrides = None
    if sensitivity_profile_path is not None:
        sensitivity_profile = _load_json(sensitivity_profile_path)
        sensitivity_overrides = resolve_sensitivity_overrides(
            layer_stats=layer_stats,
            grouping=grouping,
            sensitivity_profile_payload=sensitivity_profile,
            field=sensitivity_field,
        )
    groups = build_search_groups(
        layer_stats,
        grouping=grouping,
        sensitivity_overrides=sensitivity_overrides,
    )
    return run_proxy_evolution_search(
        groups=groups,
        target_budget_gb=target_budget_gb,
        allowed_bits=allowed_bits,
        grouping=grouping,
        population_size=population_size,
        generations=generations,
        elite_count=elite_count,
        tournament_size=tournament_size,
        mutation_rate=mutation_rate,
        top_k=top_k,
        seed=seed,
    )


def run_surrogate_search_from_report(
    report_path: str | Path,
    surrogate_summary_path: str | Path,
    surrogate_model_path: str | Path,
    target_budget_gb: float,
    allowed_bits: tuple[int, ...],
    grouping: str = DEFAULT_GROUPING,
    group_value_prior_path: str | Path | None = None,
    sensitivity_profile_path: str | Path | None = None,
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
) -> SearchRunResult:
    report_payload = _load_json(report_path)
    layer_stats = layer_stats_from_report(report_payload)
    sensitivity_overrides = None
    if sensitivity_profile_path is not None:
        sensitivity_profile = _load_json(sensitivity_profile_path)
        sensitivity_overrides = resolve_sensitivity_overrides(
            layer_stats=layer_stats,
            grouping=grouping,
            sensitivity_profile_payload=sensitivity_profile,
            field=sensitivity_field,
        )
    groups = build_search_groups(
        layer_stats,
        grouping=grouping,
        sensitivity_overrides=sensitivity_overrides,
    )
    group_value_scores = None
    if group_value_prior_path is not None:
        group_value_prior_payload = _load_json(group_value_prior_path)
        group_value_scores = resolve_group_value_scores(
            groups,
            group_value_prior_payload,
            layer_stats=layer_stats,
            target_grouping=grouping,
        )
    surrogate_summary_payload = _load_json(surrogate_summary_path)
    surrogate_model_json = Path(surrogate_model_path).read_text(encoding="utf-8")
    return run_surrogate_evolution_search(
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
        reference_accuracy=reference_accuracy,
        top_k=top_k,
        seed=seed,
    )


def default_seed_assignments(
    groups: list[SearchGroup],
    default_bit_width: int = 4,
    high_precision_bit: int = 8,
) -> dict[str, int]:
    assignments: dict[str, int] = {}
    for group in groups:
        if group.component_type.endswith("mlp.down_proj"):
            assignments[group.name] = high_precision_bit
        else:
            assignments[group.name] = default_bit_width
    return assignments


def exploratory_seed_assignments(
    groups: list[SearchGroup],
    allowed_bits: tuple[int, ...],
) -> dict[str, int]:
    sorted_bits = _normalize_allowed_bits(allowed_bits)
    min_bit = sorted_bits[0]
    mid_low_bit = 3 if 3 in sorted_bits else min_bit
    default_bit = 4 if 4 in sorted_bits else sorted_bits[min(1, len(sorted_bits) - 1)]
    high_bit = sorted_bits[-1]

    assignments: dict[str, int] = {}
    for group in groups:
        if group.component_type.endswith("mlp.down_proj"):
            assignments[group.name] = high_bit
        elif group.component_type.endswith("linear_attn.out_proj"):
            assignments[group.name] = high_bit
        elif group.component_type.endswith("linear_attn.in_proj_qkv"):
            assignments[group.name] = high_bit
        elif group.sensitivity <= 0.62:
            assignments[group.name] = min_bit
        elif group.sensitivity <= 0.78:
            assignments[group.name] = mid_low_bit
        else:
            assignments[group.name] = default_bit
    return assignments


def value_guided_seed_assignments(
    groups: list[SearchGroup],
    allowed_bits: tuple[int, ...],
    group_value_scores: dict[str, float] | None,
) -> dict[str, int] | None:
    if not group_value_scores:
        return None
    normalized_bits = _normalize_allowed_bits(allowed_bits)
    if 4 not in normalized_bits or 8 not in normalized_bits:
        return None

    assignments = {group.name: 4 for group in groups}
    positive_groups = [
        group
        for group in groups
        if float(group_value_scores.get(group.name, 0.0)) > 0
    ]
    for group in positive_groups:
        assignments[group.name] = 8
    if 16 in normalized_bits and positive_groups:
        ranked_positive_groups = sorted(
            positive_groups,
            key=lambda group: float(group_value_scores.get(group.name, 0.0)),
            reverse=True,
        )
        promoted_count = max(1, len(ranked_positive_groups) // 5)
        for group in ranked_positive_groups[:promoted_count]:
            assignments[group.name] = 16
    if not positive_groups:
        return None
    return assignments


def estimate_candidate_average_bit_width(
    groups: list[SearchGroup],
    assignments: dict[str, int],
) -> float:
    total_params = sum(group.parameter_count for group in groups)
    if total_params == 0:
        return 0.0
    weighted_bits = sum(group.parameter_count * assignments[group.name] for group in groups)
    return weighted_bits / total_params


def estimate_candidate_weight_footprint_gb(
    groups: list[SearchGroup],
    assignments: dict[str, int],
) -> float:
    total_bits = sum(group.parameter_count * assignments[group.name] for group in groups)
    return total_bits / 8 / (1024**3)


def estimate_proxy_quality_score(
    groups: list[SearchGroup],
    assignments: dict[str, int],
    quality_prior: dict[int, float] | None = None,
) -> float:
    active_prior = quality_prior or DEFAULT_QUALITY_PRIOR
    numerator = 0.0
    denominator = 0.0
    for group in groups:
        bit_width = assignments[group.name]
        quality = active_prior.get(bit_width, bit_width / max(active_prior))
        weight = group.parameter_count * group.sensitivity
        numerator += weight * quality
        denominator += weight
    if denominator == 0:
        return 0.0
    return numerator / denominator


def repair_assignments_to_budget(
    groups: list[SearchGroup],
    assignments: dict[str, int],
    target_budget_gb: float,
    allowed_bits: tuple[int, ...],
    fill_budget: bool = True,
    min_budget_utilization: float = 0.97,
    quality_prior: dict[int, float] | None = None,
) -> dict[str, int]:
    normalized_bits = _normalize_allowed_bits(allowed_bits)
    active_prior = quality_prior or DEFAULT_QUALITY_PRIOR
    repaired = dict(assignments)

    while estimate_candidate_weight_footprint_gb(groups, repaired) > target_budget_gb:
        candidate_steps: list[tuple[float, str, int]] = []
        for group in groups:
            current_bit = repaired[group.name]
            current_index = normalized_bits.index(current_bit)
            if current_index == 0:
                continue
            next_bit = normalized_bits[current_index - 1]
            score_drop = _quality_drop(group, current_bit, next_bit, active_prior)
            bit_savings = group.parameter_count * (current_bit - next_bit)
            candidate_steps.append((score_drop / bit_savings, group.name, next_bit))
        if not candidate_steps:
            break
        _, group_name, next_bit = min(candidate_steps)
        repaired[group_name] = next_bit

    if not fill_budget:
        return repaired

    while True:
        current_footprint = estimate_candidate_weight_footprint_gb(groups, repaired)
        if target_budget_gb > 0 and (current_footprint / target_budget_gb) >= min_budget_utilization:
            break
        upgrade_steps: list[tuple[float, str, int, float]] = []
        for group in groups:
            current_bit = repaired[group.name]
            current_index = normalized_bits.index(current_bit)
            if current_index == len(normalized_bits) - 1:
                continue
            next_bit = normalized_bits[current_index + 1]
            added_footprint = _added_footprint_gb(group, current_bit, next_bit)
            if current_footprint + added_footprint > target_budget_gb:
                continue
            score_gain = _quality_drop(group, next_bit, current_bit, active_prior)
            bit_cost = group.parameter_count * (next_bit - current_bit)
            upgrade_steps.append((score_gain / bit_cost, group.name, next_bit, added_footprint))
        if not upgrade_steps:
            break
        _, group_name, next_bit, _ = max(upgrade_steps)
        repaired[group_name] = next_bit
    return repaired


def repair_assignments_with_fixed_groups(
    groups: list[SearchGroup],
    assignments: dict[str, int],
    target_budget_gb: float,
    allowed_bits: tuple[int, ...],
    fixed_assignments: dict[str, int] | None = None,
    fill_budget: bool = True,
    min_budget_utilization: float = 0.97,
    quality_prior: dict[int, float] | None = None,
) -> dict[str, int]:
    fixed_group_bits = {str(name): int(bit) for name, bit in (fixed_assignments or {}).items()}
    if not fixed_group_bits:
        return repair_assignments_to_budget(
            groups=groups,
            assignments=assignments,
            target_budget_gb=target_budget_gb,
            allowed_bits=allowed_bits,
            fill_budget=fill_budget,
            min_budget_utilization=min_budget_utilization,
            quality_prior=quality_prior,
        )

    group_lookup = {group.name: group for group in groups}
    unknown_fixed = sorted(set(fixed_group_bits) - set(group_lookup))
    if unknown_fixed:
        raise ValueError(f"Fixed assignments reference unknown groups: {unknown_fixed[:5]}")

    active_groups = [group for group in groups if group.name not in fixed_group_bits]
    remaining_budget_gb = target_budget_gb - _estimate_partial_weight_footprint_gb(
        groups,
        fixed_group_bits,
    )
    if remaining_budget_gb <= 0:
        raise ValueError("Fixed assignments consume the full search budget")

    active_assignments = {
        group.name: int(assignments.get(group.name, min(allowed_bits)))
        for group in active_groups
    }
    repaired_active = repair_assignments_to_budget(
        groups=active_groups,
        assignments=active_assignments,
        target_budget_gb=remaining_budget_gb,
        allowed_bits=allowed_bits,
        fill_budget=fill_budget,
        min_budget_utilization=min_budget_utilization,
        quality_prior=quality_prior,
    )
    merged_assignments = dict(fixed_group_bits)
    merged_assignments.update(repaired_active)
    return merged_assignments


def resolve_surrogate_free_priority_scores(
    groups: list[SearchGroup],
    group_value_scores: dict[str, float] | None = None,
) -> dict[str, float]:
    if group_value_scores and any(abs(float(score)) > 1e-12 for score in group_value_scores.values()):
        return {
            group.name: float(group_value_scores.get(group.name, 0.0))
            for group in groups
        }
    return {group.name: float(group.sensitivity) for group in groups}


def resolve_surrogate_free_priority_lists(
    groups: list[SearchGroup],
    group_priority_scores: dict[str, float],
    promotable_count: int = 30,
    demotable_count: int = 60,
    excluded_group_names: set[str] | None = None,
) -> dict[str, list[str]]:
    excluded = set(excluded_group_names or set())
    eligible_groups = [group for group in groups if group.name not in excluded]
    promotable_names = [
        group.name
        for group in sorted(
            eligible_groups,
            key=lambda item: (
                float(group_priority_scores.get(item.name, 0.0)),
                item.parameter_count,
            ),
            reverse=True,
        )[:promotable_count]
    ]
    demotable_names = [
        group.name
        for group in sorted(
            eligible_groups,
            key=lambda item: (
                float(group_priority_scores.get(item.name, 0.0)),
                item.parameter_count,
            ),
        )[:demotable_count]
    ]
    return {
        "promotable_group_names": promotable_names,
        "demotable_group_names": demotable_names,
    }


def build_surrogate_free_seed_assignments(
    groups: list[SearchGroup],
    target_budget_gb: float,
    allowed_bits: tuple[int, ...],
    group_priority_scores: dict[str, float],
    promotable_group_names: list[str],
    demotable_group_names: list[str],
    fixed_assignments: dict[str, int] | None = None,
    min_budget_utilization: float = 0.99,
    max_seed_count: int = 0,
    selected_seed_provenances: tuple[str, ...] | None = None,
) -> list[tuple[str, dict[str, int]]]:
    normalized_bits = _normalize_allowed_bits(allowed_bits)
    if normalized_bits == (2, 4, 8):
        return _build_surrogate_free_low_bit_seed_assignments(
            groups=groups,
            target_budget_gb=target_budget_gb,
            allowed_bits=normalized_bits,
            fixed_assignments=fixed_assignments,
            min_budget_utilization=min_budget_utilization,
            max_seed_count=max_seed_count,
            selected_seed_provenances=selected_seed_provenances,
        )
    if 8 not in normalized_bits or 4 not in normalized_bits or 16 not in normalized_bits:
        raise ValueError(
            "Surrogate-free sprint seeds require allowed_bits to be either 2,4,8 or include 4, 8, and 16"
        )

    uniform_assignments = {
        group.name: 8
        for group in groups
    }
    uniform_assignments.update({str(name): int(bit) for name, bit in (fixed_assignments or {}).items()})
    normalized_uniform = repair_assignments_with_fixed_groups(
        groups=groups,
        assignments=uniform_assignments,
        target_budget_gb=target_budget_gb,
        allowed_bits=allowed_bits,
        fixed_assignments=fixed_assignments,
        min_budget_utilization=min_budget_utilization,
    )

    seeds: list[tuple[str, dict[str, int]]] = [("uniform_int8_seed", normalized_uniform)]
    if promotable_group_names:
        top_one = promotable_group_names[:1]
        top_two = promotable_group_names[:2]
        late_groups = _late_promotable_group_names(
            groups=groups,
            promotable_group_names=promotable_group_names,
        )
        late_rescues = late_groups[:2] if late_groups else top_two
        seeds.append(
            (
                "single_priority_rescue_seed",
                _build_rescue_seed_assignments(
                    groups=groups,
                    base_assignments=normalized_uniform,
                    rescue_group_names=top_one,
                    demotable_group_names=demotable_group_names,
                    target_budget_gb=target_budget_gb,
                    allowed_bits=allowed_bits,
                    fixed_assignments=fixed_assignments,
                    min_budget_utilization=min_budget_utilization,
                ),
            )
        )
        seeds.append(
            (
                "two_rescue_seed",
                _build_rescue_seed_assignments(
                    groups=groups,
                    base_assignments=normalized_uniform,
                    rescue_group_names=top_two,
                    demotable_group_names=demotable_group_names,
                    target_budget_gb=target_budget_gb,
                    allowed_bits=allowed_bits,
                    fixed_assignments=fixed_assignments,
                    min_budget_utilization=min_budget_utilization,
                ),
            )
        )
        seeds.append(
            (
                "late_layer_rescue_seed",
                _build_rescue_seed_assignments(
                    groups=groups,
                    base_assignments=normalized_uniform,
                    rescue_group_names=late_rescues,
                    demotable_group_names=demotable_group_names,
                    target_budget_gb=target_budget_gb,
                    allowed_bits=allowed_bits,
                    fixed_assignments=fixed_assignments,
                    min_budget_utilization=min_budget_utilization,
                ),
            )
        )
        seeds.append(
            (
                "compression_first_seed",
                _build_compression_first_seed_assignments(
                    groups=groups,
                    base_assignments=normalized_uniform,
                    rescue_group_names=top_two,
                    demotable_group_names=demotable_group_names,
                    target_budget_gb=target_budget_gb,
                    allowed_bits=allowed_bits,
                    fixed_assignments=fixed_assignments,
                    min_budget_utilization=min_budget_utilization,
                ),
            )
        )
    deduped_seeds = dedupe_assignment_candidates(seeds)
    target_seed_count = max_seed_count if max_seed_count > 0 else 5
    if selected_seed_provenances:
        seed_lookup = {provenance: assignments for provenance, assignments in deduped_seeds}
        missing = [
            provenance for provenance in selected_seed_provenances if provenance not in seed_lookup
        ]
        if missing:
            raise ValueError(
                "Unknown surrogate-free seed provenances requested: " + ", ".join(sorted(missing))
            )
        return [
            (provenance, seed_lookup[provenance])
            for provenance in selected_seed_provenances[:target_seed_count]
        ]

    fallback_rescue_sizes = range(3, min(len(promotable_group_names), 5) + 1)
    for rescue_size in fallback_rescue_sizes:
        if len(deduped_seeds) >= target_seed_count:
            break
        deduped_seeds = dedupe_assignment_candidates(
            deduped_seeds
            + [
                (
                    f"fallback_{rescue_size}_rescue_seed",
                    _build_rescue_seed_assignments(
                        groups=groups,
                        base_assignments=normalized_uniform,
                        rescue_group_names=promotable_group_names[:rescue_size],
                        demotable_group_names=demotable_group_names,
                        target_budget_gb=target_budget_gb,
                        allowed_bits=allowed_bits,
                        fixed_assignments=fixed_assignments,
                        min_budget_utilization=min_budget_utilization,
                    ),
                )
            ]
        )

    if len(deduped_seeds) < target_seed_count:
        aggressive_compression = dict(normalized_uniform)
        for group_name in demotable_group_names:
            aggressive_compression[group_name] = 4
        deduped_seeds = dedupe_assignment_candidates(
            deduped_seeds
            + [
                (
                    "aggressive_compression_seed",
                    repair_assignments_with_fixed_groups(
                        groups=groups,
                        assignments=aggressive_compression,
                        target_budget_gb=target_budget_gb,
                        allowed_bits=allowed_bits,
                        fixed_assignments=fixed_assignments,
                        min_budget_utilization=min_budget_utilization,
                    ),
                )
            ]
        )
    return deduped_seeds[:target_seed_count]


def generate_surrogate_free_neighbor_assignments(
    groups: list[SearchGroup],
    base_assignments: dict[str, int],
    target_budget_gb: float,
    allowed_bits: tuple[int, ...],
    group_priority_scores: dict[str, float],
    promotable_group_names: list[str],
    demotable_group_names: list[str],
    fixed_assignments: dict[str, int] | None = None,
    min_budget_utilization: float = 0.99,
) -> list[tuple[str, dict[str, int]]]:
    normalized_bits = _normalize_allowed_bits(allowed_bits)
    if normalized_bits == (2, 4, 8):
        return _generate_surrogate_free_low_bit_neighbor_assignments(
            groups=groups,
            base_assignments=base_assignments,
            target_budget_gb=target_budget_gb,
            allowed_bits=normalized_bits,
            fixed_assignments=fixed_assignments,
            min_budget_utilization=min_budget_utilization,
        )

    fixed_group_names = set((fixed_assignments or {}).keys())
    group_lookup = {group.name: group for group in groups}
    current_16 = [
        group_name
        for group_name in promotable_group_names
        if int(base_assignments.get(group_name, 8)) >= 16 and group_name not in fixed_group_names
    ]
    current_4 = [
        group_name
        for group_name in demotable_group_names
        if int(base_assignments.get(group_name, 8)) <= 4 and group_name not in fixed_group_names
    ]
    unused_promotions = [
        group_name
        for group_name in promotable_group_names
        if int(base_assignments.get(group_name, 8)) < 16 and group_name not in fixed_group_names
    ]
    unused_demotions = [
        group_name
        for group_name in demotable_group_names
        if int(base_assignments.get(group_name, 8)) > 4 and group_name not in fixed_group_names
    ]

    neighbors: list[tuple[str, dict[str, int]]] = []
    if unused_promotions:
        promoted = dict(base_assignments)
        promoted[unused_promotions[0]] = 16
        neighbors.append(
            (
                "add_rescue_neighbor",
                repair_assignments_with_fixed_groups(
                    groups=groups,
                    assignments=promoted,
                    target_budget_gb=target_budget_gb,
                    allowed_bits=allowed_bits,
                    fixed_assignments=fixed_assignments,
                    min_budget_utilization=min_budget_utilization,
                ),
            )
        )

    if current_16 and unused_promotions:
        weakest_16 = min(
            current_16,
            key=lambda group_name: float(group_priority_scores.get(group_name, 0.0)),
        )
        stronger_unused = next(
            (
                group_name
                for group_name in unused_promotions
                if float(group_priority_scores.get(group_name, 0.0))
                > float(group_priority_scores.get(weakest_16, 0.0))
            ),
            unused_promotions[0],
        )
        swapped = dict(base_assignments)
        swapped[weakest_16] = 8
        swapped[stronger_unused] = 16
        neighbors.append(
            (
                "swap_rescue_neighbor",
                repair_assignments_with_fixed_groups(
                    groups=groups,
                    assignments=swapped,
                    target_budget_gb=target_budget_gb,
                    allowed_bits=allowed_bits,
                    fixed_assignments=fixed_assignments,
                    min_budget_utilization=min_budget_utilization,
                ),
            )
        )

    if current_4 and unused_demotions:
        worst_current_demotion = max(
            current_4,
            key=lambda group_name: float(group_priority_scores.get(group_name, 0.0)),
        )
        alternative_demotion = _choose_similar_size_group(
            reference_group_name=worst_current_demotion,
            candidate_group_names=unused_demotions,
            group_lookup=group_lookup,
            group_priority_scores=group_priority_scores,
        )
        if alternative_demotion is not None:
            rebalanced = dict(base_assignments)
            rebalanced[worst_current_demotion] = 8
            rebalanced[alternative_demotion] = 4
            neighbors.append(
                (
                    "swap_demotion_neighbor",
                    repair_assignments_with_fixed_groups(
                        groups=groups,
                        assignments=rebalanced,
                        target_budget_gb=target_budget_gb,
                        allowed_bits=allowed_bits,
                        fixed_assignments=fixed_assignments,
                        min_budget_utilization=min_budget_utilization,
                    ),
                )
            )

    if current_16:
        weakest_16 = min(
            current_16,
            key=lambda group_name: float(group_priority_scores.get(group_name, 0.0)),
        )
        reallocated = dict(base_assignments)
        reallocated[weakest_16] = 8
        reallocate_target = next(
            (
                group_name
                for group_name in promotable_group_names
                if group_name != weakest_16
                and int(base_assignments.get(group_name, 8)) < 16
                and group_name not in fixed_group_names
            ),
            None,
        )
        if reallocate_target is not None:
            reallocated[reallocate_target] = 16
        neighbors.append(
            (
                "rollback_and_respend_neighbor",
                repair_assignments_with_fixed_groups(
                    groups=groups,
                    assignments=reallocated,
                    target_budget_gb=target_budget_gb,
                    allowed_bits=allowed_bits,
                    fixed_assignments=fixed_assignments,
                    min_budget_utilization=min_budget_utilization,
                ),
            )
        )

    return dedupe_assignment_candidates(neighbors)


def _build_surrogate_free_low_bit_seed_assignments(
    groups: list[SearchGroup],
    target_budget_gb: float,
    allowed_bits: tuple[int, ...],
    fixed_assignments: dict[str, int] | None,
    min_budget_utilization: float,
    max_seed_count: int,
    selected_seed_provenances: tuple[str, ...] | None,
) -> list[tuple[str, dict[str, int]]]:
    fixed_group_bits = {str(name): int(bit) for name, bit in (fixed_assignments or {}).items()}
    active_group_names = [group.name for group in groups if group.name not in fixed_group_bits]

    uniform_int4 = {
        group.name: 4
        for group in groups
    }
    uniform_int4.update(fixed_group_bits)
    repaired_uniform = repair_assignments_with_fixed_groups(
        groups=groups,
        assignments=uniform_int4,
        target_budget_gb=target_budget_gb,
        allowed_bits=allowed_bits,
        fixed_assignments=fixed_assignments,
        min_budget_utilization=min_budget_utilization,
    )

    compression_first = dict(repaired_uniform)
    ranked_by_size = sorted(
        (group for group in groups if group.name in active_group_names),
        key=lambda group: (group.parameter_count, group.name),
        reverse=True,
    )
    compression_target = target_budget_gb * 0.84
    for group in ranked_by_size:
        if estimate_candidate_weight_footprint_gb(groups, compression_first) <= compression_target:
            break
        compression_first[group.name] = 2
    for group in sorted(
        (group for group in groups if group.name in active_group_names),
        key=lambda group: (group.parameter_count, group.name),
    )[:2]:
        compression_first[group.name] = 8
    compression_first = repair_assignments_with_fixed_groups(
        groups=groups,
        assignments=compression_first,
        target_budget_gb=target_budget_gb,
        allowed_bits=allowed_bits,
        fixed_assignments=fixed_assignments,
        min_budget_utilization=min_budget_utilization,
    )
    if tuple(sorted(compression_first.items())) == tuple(sorted(repaired_uniform.items())):
        aggressive = dict(repaired_uniform)
        for group in ranked_by_size[:2]:
            aggressive[group.name] = 2
        for group in sorted(
            (group for group in groups if group.name in active_group_names),
            key=lambda group: (group.parameter_count, group.name),
        )[:1]:
            aggressive[group.name] = 8
        compression_first = repair_assignments_with_fixed_groups(
            groups=groups,
            assignments=aggressive,
            target_budget_gb=target_budget_gb,
            allowed_bits=allowed_bits,
            fixed_assignments=fixed_assignments,
            min_budget_utilization=min_budget_utilization,
        )

    deduped_seeds = dedupe_assignment_candidates(
        [
            ("uniform_int4_seed", repaired_uniform),
            ("compression_first_seed", compression_first),
        ]
    )
    target_seed_count = max_seed_count if max_seed_count > 0 else 2
    if selected_seed_provenances:
        seed_lookup = {provenance: assignments for provenance, assignments in deduped_seeds}
        missing = [
            provenance for provenance in selected_seed_provenances if provenance not in seed_lookup
        ]
        if missing:
            raise ValueError(
                "Unknown surrogate-free seed provenances requested: " + ", ".join(sorted(missing))
            )
        return [
            (provenance, seed_lookup[provenance])
            for provenance in selected_seed_provenances[:target_seed_count]
        ]
    return deduped_seeds[:target_seed_count]


def _generate_surrogate_free_low_bit_neighbor_assignments(
    groups: list[SearchGroup],
    base_assignments: dict[str, int],
    target_budget_gb: float,
    allowed_bits: tuple[int, ...],
    fixed_assignments: dict[str, int] | None,
    min_budget_utilization: float,
) -> list[tuple[str, dict[str, int]]]:
    fixed_group_names = set((fixed_assignments or {}).keys())
    active_groups = [group for group in groups if group.name not in fixed_group_names]
    group_lookup = {group.name: group for group in groups}
    normalized_bits = _normalize_allowed_bits(allowed_bits)
    min_bit = normalized_bits[0]
    max_bit = normalized_bits[-1]

    def _adjacent_step(group_name: str, direction: int) -> int | None:
        current_bit = int(base_assignments.get(group_name, normalized_bits[len(normalized_bits) // 2]))
        current_index = normalized_bits.index(current_bit)
        next_index = current_index + direction
        if next_index < 0 or next_index >= len(normalized_bits):
            return None
        return normalized_bits[next_index]

    promoted_candidates = sorted(
        (
            group
            for group in active_groups
            if int(base_assignments.get(group.name, 4)) < max_bit
        ),
        key=lambda group: (
            int(base_assignments.get(group.name, 4)),
            -group.parameter_count,
            group.name,
        ),
    )
    demoted_candidates = sorted(
        (
            group
            for group in active_groups
            if int(base_assignments.get(group.name, 4)) > min_bit
        ),
        key=lambda group: (
            -int(base_assignments.get(group.name, 4)),
            group.parameter_count,
            group.name,
        ),
    )
    eight_bit_groups = [
        group
        for group in active_groups
        if int(base_assignments.get(group.name, 4)) >= 8
    ]
    low_bit_groups = [
        group
        for group in active_groups
        if int(base_assignments.get(group.name, 4)) <= 4
    ]
    groups_by_block: dict[int, list[SearchGroup]] = {}
    for group in active_groups:
        block_indices = _group_block_indices(group)
        block_key = block_indices[0] if block_indices else -1
        groups_by_block.setdefault(block_key, []).append(group)

    neighbors: list[tuple[str, dict[str, int]]] = []

    for group in promoted_candidates[:6]:
        next_bit = _adjacent_step(group.name, +1)
        if next_bit is None:
            continue
        candidate = dict(base_assignments)
        candidate[group.name] = next_bit
        neighbors.append(
            (
                f"single_flip_up:{group.name}",
                repair_assignments_with_fixed_groups(
                    groups=groups,
                    assignments=candidate,
                    target_budget_gb=target_budget_gb,
                    allowed_bits=allowed_bits,
                    fixed_assignments=fixed_assignments,
                    min_budget_utilization=min_budget_utilization,
                ),
            )
        )

    for group in demoted_candidates[:6]:
        next_bit = _adjacent_step(group.name, -1)
        if next_bit is None:
            continue
        candidate = dict(base_assignments)
        candidate[group.name] = next_bit
        neighbors.append(
            (
                f"single_flip_down:{group.name}",
                repair_assignments_with_fixed_groups(
                    groups=groups,
                    assignments=candidate,
                    target_budget_gb=target_budget_gb,
                    allowed_bits=allowed_bits,
                    fixed_assignments=fixed_assignments,
                    min_budget_utilization=min_budget_utilization,
                ),
            )
        )

    for promote_group in promoted_candidates[:4]:
        promote_bit = _adjacent_step(promote_group.name, +1)
        if promote_bit is None:
            continue
        for demote_group in demoted_candidates[:4]:
            if demote_group.name == promote_group.name:
                continue
            demote_bit = _adjacent_step(demote_group.name, -1)
            if demote_bit is None:
                continue
            candidate = dict(base_assignments)
            candidate[promote_group.name] = promote_bit
            candidate[demote_group.name] = demote_bit
            neighbors.append(
                (
                    f"double_flip_budget_balanced:{promote_group.name}:{demote_group.name}",
                    repair_assignments_with_fixed_groups(
                        groups=groups,
                        assignments=candidate,
                        target_budget_gb=target_budget_gb,
                        allowed_bits=allowed_bits,
                        fixed_assignments=fixed_assignments,
                        min_budget_utilization=min_budget_utilization,
                    ),
                )
            )

    for high_group in eight_bit_groups[:4]:
        high_down = _adjacent_step(high_group.name, -1)
        if high_down is None:
            continue
        for low_group in low_bit_groups[:4]:
            if low_group.name == high_group.name:
                continue
            low_up = _adjacent_step(low_group.name, +1)
            if low_up is None:
                continue
            candidate = dict(base_assignments)
            candidate[high_group.name] = high_down
            candidate[low_group.name] = low_up
            neighbors.append(
                (
                    f"band_swap:{high_group.name}:{low_group.name}",
                    repair_assignments_with_fixed_groups(
                        groups=groups,
                        assignments=candidate,
                        target_budget_gb=target_budget_gb,
                        allowed_bits=allowed_bits,
                        fixed_assignments=fixed_assignments,
                        min_budget_utilization=min_budget_utilization,
                    ),
                )
            )

    ranked_blocks = sorted(
        (
            (block_key, members)
            for block_key, members in groups_by_block.items()
            if len(members) >= 2
        ),
        key=lambda item: (
            sum(member.parameter_count for member in item[1]),
            item[0],
        ),
        reverse=True,
    )
    for block_key, members in ranked_blocks[:4]:
        candidate = dict(base_assignments)
        changed = 0
        for member in sorted(members, key=lambda group: group.parameter_count, reverse=True)[:3]:
            current_bit = int(candidate.get(member.name, 4))
            direction = -1 if current_bit > 4 else +1
            next_bit = _adjacent_step(member.name, direction)
            if next_bit is None:
                continue
            candidate[member.name] = next_bit
            changed += 1
        if changed >= 2:
            neighbors.append(
                (
                    f"cluster_flip:block_{block_key}",
                    repair_assignments_with_fixed_groups(
                        groups=groups,
                        assignments=candidate,
                        target_budget_gb=target_budget_gb,
                        allowed_bits=allowed_bits,
                        fixed_assignments=fixed_assignments,
                        min_budget_utilization=min_budget_utilization,
                    ),
                )
            )

    if eight_bit_groups and low_bit_groups:
        for largest_high in sorted(
            eight_bit_groups,
            key=lambda group: group.parameter_count,
            reverse=True,
        )[:3]:
            largest_high_down = _adjacent_step(largest_high.name, -1)
            if largest_high_down is None:
                continue
            for upgrade_count in (2, 3):
                candidate = dict(base_assignments)
                candidate[largest_high.name] = largest_high_down
                upgraded = 0
                for low_group in sorted(low_bit_groups, key=lambda group: group.parameter_count):
                    if low_group.name == largest_high.name:
                        continue
                    low_up = _adjacent_step(low_group.name, +1)
                    if low_up is None:
                        continue
                    candidate[low_group.name] = low_up
                    upgraded += 1
                    if upgraded == upgrade_count:
                        break
                if upgraded:
                    neighbors.append(
                        (
                            f"large_group_reallocation:{largest_high.name}:x{upgraded}",
                            repair_assignments_with_fixed_groups(
                                groups=groups,
                                assignments=candidate,
                                target_budget_gb=target_budget_gb,
                                allowed_bits=allowed_bits,
                                fixed_assignments=fixed_assignments,
                                min_budget_utilization=min_budget_utilization,
                            ),
                        )
                    )

    deduped_neighbors = dedupe_assignment_candidates(neighbors)
    base_signature = tuple(sorted((str(name), int(bit)) for name, bit in base_assignments.items()))
    return [
        (provenance, assignments)
        for provenance, assignments in deduped_neighbors
        if tuple(sorted((str(name), int(bit)) for name, bit in assignments.items())) != base_signature
    ]


def dedupe_assignment_candidates(
    candidate_assignments: list[tuple[str, dict[str, int]]],
) -> list[tuple[str, dict[str, int]]]:
    deduped: list[tuple[str, dict[str, int]]] = []
    seen_signatures: set[tuple[tuple[str, int], ...]] = set()
    for provenance, assignments in candidate_assignments:
        signature = tuple(sorted((str(group_name), int(bit_width)) for group_name, bit_width in assignments.items()))
        if signature in seen_signatures:
            continue
        deduped.append(
            (
                str(provenance),
                {str(group_name): int(bit_width) for group_name, bit_width in signature},
            )
        )
        seen_signatures.add(signature)
    return deduped


def estimate_assignment_prior_delta(
    groups: list[SearchGroup],
    assignments: dict[str, int],
    group_priority_scores: dict[str, float],
) -> float:
    numerator = 0.0
    denominator = 0.0
    for group in groups:
        priority_score = float(group_priority_scores.get(group.name, 0.0))
        if abs(priority_score) <= 1e-12:
            continue
        bit_width = int(assignments.get(group.name, 8))
        if bit_width >= 16:
            bit_shift = 1.0
        elif bit_width <= 4:
            bit_shift = -1.0
        else:
            bit_shift = 0.0
        weight = group.parameter_count * abs(priority_score)
        numerator += weight * bit_shift * (1.0 if priority_score >= 0 else -1.0)
        denominator += weight
    if denominator == 0:
        return 0.0
    return numerator / denominator


def estimate_assignment_search_score(
    groups: list[SearchGroup],
    assignments: dict[str, int],
    target_budget_gb: float,
    group_priority_scores: dict[str, float],
) -> float:
    prior_delta = estimate_assignment_prior_delta(
        groups=groups,
        assignments=assignments,
        group_priority_scores=group_priority_scores,
    )
    budget_alignment = estimate_budget_alignment_score(
        footprint_gb=estimate_candidate_weight_footprint_gb(groups, assignments),
        target_budget_gb=target_budget_gb,
    )
    entropy_bonus = _estimate_assignment_bit_entropy(assignments)
    low_bit_penalty = _estimate_large_group_low_bit_penalty(groups, assignments)
    return (
        prior_delta
        + 0.12 * budget_alignment
        + 0.06 * entropy_bonus
        - 0.10 * low_bit_penalty
    )


def _estimate_assignment_bit_entropy(assignments: dict[str, int]) -> float:
    counts: dict[int, int] = {}
    for bit_width in assignments.values():
        counts[int(bit_width)] = counts.get(int(bit_width), 0) + 1
    total = sum(counts.values())
    if total <= 1 or len(counts) <= 1:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        probability = count / total
        entropy -= probability * math.log(probability + 1e-12)
    return entropy / math.log(len(counts))


def _estimate_large_group_low_bit_penalty(
    groups: list[SearchGroup],
    assignments: dict[str, int],
) -> float:
    if not groups:
        return 0.0
    min_bit = min(int(assignments.get(group.name, 0)) for group in groups)
    total_params = sum(group.parameter_count for group in groups)
    if total_params <= 0:
        return 0.0
    low_bit_mass = sum(
        group.parameter_count
        for group in groups
        if int(assignments.get(group.name, min_bit)) == min_bit
    )
    return low_bit_mass / total_params


def _build_rescue_seed_assignments(
    groups: list[SearchGroup],
    base_assignments: dict[str, int],
    rescue_group_names: list[str],
    demotable_group_names: list[str],
    target_budget_gb: float,
    allowed_bits: tuple[int, ...],
    fixed_assignments: dict[str, int] | None,
    min_budget_utilization: float,
) -> dict[str, int]:
    candidate = dict(base_assignments)
    for group_name in rescue_group_names:
        if group_name in candidate:
            candidate[group_name] = 16
    if estimate_candidate_weight_footprint_gb(groups, candidate) <= target_budget_gb:
        return repair_assignments_with_fixed_groups(
            groups=groups,
            assignments=candidate,
            target_budget_gb=target_budget_gb,
            allowed_bits=allowed_bits,
            fixed_assignments=fixed_assignments,
            min_budget_utilization=min_budget_utilization,
        )

    for group_name in demotable_group_names:
        if group_name in rescue_group_names:
            continue
        if int(candidate.get(group_name, 8)) <= 4:
            continue
        candidate[group_name] = 4
        if estimate_candidate_weight_footprint_gb(groups, candidate) <= target_budget_gb:
            break
    return repair_assignments_with_fixed_groups(
        groups=groups,
        assignments=candidate,
        target_budget_gb=target_budget_gb,
        allowed_bits=allowed_bits,
        fixed_assignments=fixed_assignments,
        min_budget_utilization=min_budget_utilization,
    )


def _build_compression_first_seed_assignments(
    groups: list[SearchGroup],
    base_assignments: dict[str, int],
    rescue_group_names: list[str],
    demotable_group_names: list[str],
    target_budget_gb: float,
    allowed_bits: tuple[int, ...],
    fixed_assignments: dict[str, int] | None,
    min_budget_utilization: float,
) -> dict[str, int]:
    candidate = dict(base_assignments)
    rescue_set = set(rescue_group_names)
    for group_name in demotable_group_names:
        if group_name in rescue_set:
            continue
        candidate[group_name] = 4
        if estimate_candidate_weight_footprint_gb(groups, candidate) < target_budget_gb * 0.94:
            break
    for group_name in rescue_group_names:
        if group_name in candidate:
            candidate[group_name] = 16
    return repair_assignments_with_fixed_groups(
        groups=groups,
        assignments=candidate,
        target_budget_gb=target_budget_gb,
        allowed_bits=allowed_bits,
        fixed_assignments=fixed_assignments,
        min_budget_utilization=min_budget_utilization,
    )


def _late_promotable_group_names(
    groups: list[SearchGroup],
    promotable_group_names: list[str],
) -> list[str]:
    group_lookup = {group.name: group for group in groups}
    block_indices_by_group = {
        group.name: _group_block_indices(group)
        for group in groups
    }
    max_block_index = max(
        (
            max(indices)
            for indices in block_indices_by_group.values()
            if indices
        ),
        default=0,
    )
    late_threshold = max(0, max_block_index - 8)
    return [
        group_name
        for group_name in promotable_group_names
        if max(block_indices_by_group.get(group_name, [0])) >= late_threshold
    ]


def _group_block_indices(group: SearchGroup) -> list[int]:
    indices: list[int] = []
    for layer_name in group.layer_names:
        match = re.match(r"model\.layers\.(\d+)\.", layer_name)
        if match:
            indices.append(int(match.group(1)))
    return indices


def _choose_similar_size_group(
    reference_group_name: str,
    candidate_group_names: list[str],
    group_lookup: dict[str, SearchGroup],
    group_priority_scores: dict[str, float],
) -> str | None:
    reference_group = group_lookup.get(reference_group_name)
    if reference_group is None:
        return None
    ranked_candidates = sorted(
        candidate_group_names,
        key=lambda group_name: (
            float(group_priority_scores.get(group_name, 0.0)),
            abs(group_lookup[group_name].parameter_count - reference_group.parameter_count),
        ),
    )
    for group_name in ranked_candidates:
        candidate_group = group_lookup.get(group_name)
        if candidate_group is None:
            continue
        smaller = min(candidate_group.parameter_count, reference_group.parameter_count)
        larger = max(candidate_group.parameter_count, reference_group.parameter_count)
        if larger == 0 or (smaller / larger) >= 0.5:
            return group_name
    return ranked_candidates[0] if ranked_candidates else None


def crossover_assignments(
    left: dict[str, int],
    right: dict[str, int],
    rng: random.Random,
) -> dict[str, int]:
    child: dict[str, int] = {}
    for group_name in left:
        child[group_name] = left[group_name] if rng.random() < 0.5 else right[group_name]
    return child


def mutate_assignments(
    assignments: dict[str, int],
    allowed_bits: tuple[int, ...],
    rng: random.Random,
    mutation_rate: float,
) -> dict[str, int]:
    normalized_bits = _normalize_allowed_bits(allowed_bits)
    mutated = dict(assignments)
    for group_name, current_bit in list(mutated.items()):
        if rng.random() >= mutation_rate:
            continue
        current_index = normalized_bits.index(current_bit)
        neighbor_indices = [current_index]
        if current_index > 0:
            neighbor_indices.append(current_index - 1)
        if current_index < len(normalized_bits) - 1:
            neighbor_indices.append(current_index + 1)
        if len(normalized_bits) > 2 and rng.random() < 0.2:
            neighbor_indices.extend(
                index for index in range(len(normalized_bits)) if index not in neighbor_indices
            )
        chosen_index = rng.choice(neighbor_indices)
        mutated[group_name] = normalized_bits[chosen_index]
    return mutated


def build_candidate(
    groups: list[SearchGroup],
    assignments: dict[str, int],
    target_budget_gb: float,
    allowed_bits: tuple[int, ...],
    provenance: str,
    quality_prior: dict[int, float] | None = None,
) -> SearchCandidate:
    proxy_quality = estimate_proxy_quality_score(groups, assignments, quality_prior)
    footprint = estimate_candidate_weight_footprint_gb(groups, assignments)
    average_bit_width = estimate_candidate_average_bit_width(groups, assignments)
    budget_utilization = 0.0 if target_budget_gb == 0 else min(footprint / target_budget_gb, 1.0)
    compression_bonus = estimate_compression_bonus(groups, assignments, allowed_bits)
    low_bit_bonus = estimate_low_bit_bonus(groups, assignments)
    fitness = (
        proxy_quality
        + 0.04 * budget_utilization
        + 0.10 * compression_bonus
        + 0.08 * low_bit_bonus
    )
    return SearchCandidate(
        group_bits=tuple(sorted(assignments.items())),
        estimated_average_bit_width=average_bit_width,
        estimated_weight_footprint_gb=footprint,
        proxy_quality_score=proxy_quality,
        fitness=fitness,
        provenance=provenance,
    )


def build_surrogate_candidate(
    groups: list[SearchGroup],
    assignments: dict[str, int],
    surrogate_predictor: Callable[[dict[str, float]], tuple[float, float]],
    target_budget_gb: float,
    allowed_bits: tuple[int, ...],
    provenance: str,
    uncertainty_penalty: float = 0.5,
    group_value_scores: dict[str, float] | None = None,
    reference_accuracy: float | None = None,
) -> SearchCandidate:
    from ta_mpq.surrogate import extract_surrogate_features

    average_bit_width = estimate_candidate_average_bit_width(groups, assignments)
    footprint = estimate_candidate_weight_footprint_gb(groups, assignments)
    synthetic_report = {
        "estimated_average_bit_width": average_bit_width,
        "estimated_weight_footprint_gb": footprint,
        "rule_hits": [],
        "policy": {"rules": []},
    }
    feature_values = extract_surrogate_features(
        groups=groups,
        group_bits=assignments,
        report_payload=synthetic_report,
    )
    surrogate_prediction, prediction_uncertainty = surrogate_predictor(feature_values)
    conservative_prediction = surrogate_prediction - uncertainty_penalty * prediction_uncertainty
    budget_utilization = 0.0 if target_budget_gb == 0 else min(footprint / target_budget_gb, 1.0)
    budget_alignment = estimate_budget_alignment_score(
        footprint_gb=footprint,
        target_budget_gb=target_budget_gb,
    )
    group_value_alignment = estimate_group_value_alignment_score(
        groups=groups,
        assignments=assignments,
        group_value_scores=group_value_scores,
    )
    reference_advantage = estimate_reference_advantage_score(
        conservative_prediction=conservative_prediction,
        reference_accuracy=reference_accuracy,
    )
    fitness = (
        conservative_prediction
        + 0.06 * budget_alignment
        + 0.08 * group_value_alignment
        + 0.02 * budget_utilization
        + 0.12 * reference_advantage
    )
    return SearchCandidate(
        group_bits=tuple(sorted(assignments.items())),
        estimated_average_bit_width=average_bit_width,
        estimated_weight_footprint_gb=footprint,
        proxy_quality_score=surrogate_prediction,
        fitness=fitness,
        provenance=provenance,
        prediction_uncertainty=prediction_uncertainty,
        conservative_prediction=conservative_prediction,
        budget_alignment_score=budget_alignment,
        group_value_alignment_score=group_value_alignment,
        reference_accuracy=reference_accuracy,
        reference_advantage_score=reference_advantage,
    )


def run_proxy_evolution_search(
    groups: list[SearchGroup],
    target_budget_gb: float,
    allowed_bits: tuple[int, ...],
    grouping: str = DEFAULT_GROUPING,
    population_size: int = 32,
    generations: int = 15,
    elite_count: int = 4,
    tournament_size: int = 3,
    mutation_rate: float = 0.1,
    top_k: int = 5,
    seed: int = 0,
    search_groups: list[SearchGroup] | None = None,
    fixed_assignments: dict[str, int] | None = None,
    extra_seed_assignments: list[tuple[str, dict[str, int]]] | None = None,
) -> SearchRunResult:
    normalized_bits = _normalize_allowed_bits(allowed_bits)
    rng = random.Random(seed)
    full_groups = groups
    active_groups = search_groups or groups
    full_group_names = {group.name for group in full_groups}
    active_group_names = {group.name for group in active_groups}
    fixed_group_bits = {str(name): int(bit) for name, bit in (fixed_assignments or {}).items()}
    unknown_fixed = sorted(set(fixed_group_bits) - full_group_names)
    if unknown_fixed:
        raise ValueError(f"Fixed assignments reference unknown groups: {unknown_fixed[:5]}")
    overlapping_assignments = sorted(set(fixed_group_bits) & active_group_names)
    if search_groups is not None and overlapping_assignments:
        raise ValueError(
            "Fixed assignments must not overlap with active search groups; "
            f"found {overlapping_assignments[:5]}"
        )
    remaining_budget_gb = target_budget_gb - _estimate_partial_weight_footprint_gb(
        full_groups,
        fixed_group_bits,
    )
    if remaining_budget_gb <= 0:
        raise ValueError("Fixed assignments consume the full search budget")

    normalized_extra_seeds: list[tuple[str, dict[str, int]]] = []
    if extra_seed_assignments:
        for provenance, assignments in extra_seed_assignments:
            filtered = _filter_assignments_to_groups(assignments, active_group_names)
            if filtered:
                normalized_extra_seeds.append((provenance, filtered))

    def build_full_candidate(
        active_assignments: dict[str, int],
        provenance: str,
    ) -> SearchCandidate:
        merged_assignments = dict(fixed_group_bits)
        merged_assignments.update(active_assignments)
        return build_candidate(
            groups=full_groups,
            assignments=merged_assignments,
            target_budget_gb=target_budget_gb,
            allowed_bits=normalized_bits,
            provenance=provenance,
        )

    def candidate_builder(
        active_assignments: dict[str, int],
        provenance: str,
    ) -> SearchCandidate:
        full_candidate = build_full_candidate(active_assignments, provenance)
        if active_groups is full_groups:
            return full_candidate
        return SearchCandidate(
            group_bits=tuple(sorted(active_assignments.items())),
            estimated_average_bit_width=full_candidate.estimated_average_bit_width,
            estimated_weight_footprint_gb=full_candidate.estimated_weight_footprint_gb,
            proxy_quality_score=full_candidate.proxy_quality_score,
            fitness=full_candidate.fitness,
            provenance=full_candidate.provenance,
        )

    result = _run_evolution_search(
        groups=active_groups,
        target_budget_gb=remaining_budget_gb,
        allowed_bits=normalized_bits,
        grouping=grouping,
        population_size=population_size,
        generations=generations,
        elite_count=elite_count,
        tournament_size=tournament_size,
        mutation_rate=mutation_rate,
        top_k=top_k,
        rng=rng,
        candidate_builder=candidate_builder,
        extra_seed_assignments=normalized_extra_seeds,
    )
    if active_groups is full_groups:
        return result
    top_candidates = tuple(
        build_full_candidate(dict(candidate.group_bits), candidate.provenance)
        for candidate in result.top_candidates
    )
    return SearchRunResult(
        grouping=result.grouping,
        target_budget_gb=target_budget_gb,
        allowed_bits=result.allowed_bits,
        population_size=result.population_size,
        generations=result.generations,
        num_groups=len(full_groups),
        top_candidates=top_candidates,
        history=result.history,
    )


def run_surrogate_evolution_search(
    groups: list[SearchGroup],
    report_payload: dict[str, Any],
    surrogate_summary_payload: dict[str, Any],
    surrogate_model_json: str,
    target_budget_gb: float,
    allowed_bits: tuple[int, ...],
    grouping: str = DEFAULT_GROUPING,
    population_size: int = 32,
    generations: int = 15,
    elite_count: int = 4,
    tournament_size: int = 3,
    mutation_rate: float = 0.1,
    uncertainty_penalty: float = 0.5,
    group_value_scores: dict[str, float] | None = None,
    reference_accuracy: float | None = None,
    top_k: int = 5,
    seed: int = 0,
    search_groups: list[SearchGroup] | None = None,
    fixed_assignments: dict[str, int] | None = None,
    extra_seed_assignments: list[tuple[str, dict[str, int]]] | None = None,
) -> SearchRunResult:
    from ta_mpq.surrogate import build_surrogate_distribution_predictor

    normalized_bits = _normalize_allowed_bits(allowed_bits)
    rng = random.Random(seed)
    surrogate_predictor = build_surrogate_distribution_predictor(
        surrogate_summary_payload=surrogate_summary_payload,
        model_json=surrogate_model_json,
    )
    reference_target_value = resolve_reference_target_value(
        surrogate_summary_payload=surrogate_summary_payload,
        reference_accuracy=reference_accuracy,
    )
    full_groups = groups
    active_groups = search_groups or groups
    full_group_names = {group.name for group in full_groups}
    active_group_names = {group.name for group in active_groups}
    fixed_group_bits = {str(name): int(bit) for name, bit in (fixed_assignments or {}).items()}
    unknown_fixed = sorted(set(fixed_group_bits) - full_group_names)
    if unknown_fixed:
        raise ValueError(f"Fixed assignments reference unknown groups: {unknown_fixed[:5]}")
    overlapping_assignments = sorted(set(fixed_group_bits) & active_group_names)
    if search_groups is not None and overlapping_assignments:
        raise ValueError(
            "Fixed assignments must not overlap with active search groups; "
            f"found {overlapping_assignments[:5]}"
        )
    remaining_budget_gb = target_budget_gb - _estimate_partial_weight_footprint_gb(
        full_groups,
        fixed_group_bits,
    )
    if remaining_budget_gb <= 0:
        raise ValueError("Fixed assignments consume the full search budget")

    normalized_extra_seeds: list[tuple[str, dict[str, int]]] = []
    if extra_seed_assignments:
        for provenance, assignments in extra_seed_assignments:
            filtered = _filter_assignments_to_groups(assignments, active_group_names)
            if filtered:
                normalized_extra_seeds.append((provenance, filtered))
    value_guided_seed = value_guided_seed_assignments(
        groups=active_groups,
        allowed_bits=normalized_bits,
        group_value_scores=_filter_score_map(group_value_scores, active_group_names),
    )
    if value_guided_seed is not None:
        normalized_extra_seeds.append(("value_guided_seed", value_guided_seed))

    def build_full_candidate(
        active_assignments: dict[str, int],
        provenance: str,
    ) -> SearchCandidate:
        merged_assignments = dict(fixed_group_bits)
        merged_assignments.update(active_assignments)
        return build_surrogate_candidate(
            groups=full_groups,
            assignments=merged_assignments,
            surrogate_predictor=surrogate_predictor,
            target_budget_gb=target_budget_gb,
            allowed_bits=normalized_bits,
            provenance=provenance,
            uncertainty_penalty=uncertainty_penalty,
            group_value_scores=group_value_scores,
            reference_accuracy=reference_target_value,
        )

    def candidate_builder(
        active_assignments: dict[str, int],
        provenance: str,
    ) -> SearchCandidate:
        full_candidate = build_full_candidate(active_assignments, provenance)
        if active_groups is full_groups:
            return full_candidate
        return SearchCandidate(
            group_bits=tuple(sorted(active_assignments.items())),
            estimated_average_bit_width=full_candidate.estimated_average_bit_width,
            estimated_weight_footprint_gb=full_candidate.estimated_weight_footprint_gb,
            proxy_quality_score=full_candidate.proxy_quality_score,
            fitness=full_candidate.fitness,
            provenance=full_candidate.provenance,
            prediction_uncertainty=full_candidate.prediction_uncertainty,
            conservative_prediction=full_candidate.conservative_prediction,
            budget_alignment_score=full_candidate.budget_alignment_score,
            group_value_alignment_score=full_candidate.group_value_alignment_score,
            reference_accuracy=full_candidate.reference_accuracy,
            reference_advantage_score=full_candidate.reference_advantage_score,
        )

    result = _run_evolution_search(
        groups=active_groups,
        target_budget_gb=remaining_budget_gb,
        allowed_bits=normalized_bits,
        grouping=grouping,
        population_size=population_size,
        generations=generations,
        elite_count=elite_count,
        tournament_size=tournament_size,
        mutation_rate=mutation_rate,
        top_k=top_k,
        rng=rng,
        extra_seed_assignments=normalized_extra_seeds,
        candidate_builder=candidate_builder,
    )
    if active_groups is full_groups:
        return result
    top_candidates = tuple(
        build_full_candidate(dict(candidate.group_bits), candidate.provenance)
        for candidate in result.top_candidates
    )
    return SearchRunResult(
        grouping=result.grouping,
        target_budget_gb=target_budget_gb,
        allowed_bits=result.allowed_bits,
        population_size=result.population_size,
        generations=result.generations,
        num_groups=len(full_groups),
        top_candidates=top_candidates,
        history=result.history,
    )


def run_budgeted_bf16_allocator(
    groups: list[SearchGroup],
    target_budget_gb: float,
    allowed_bits: tuple[int, ...] = (4, 8, 16),
    grouping: str = DEFAULT_GROUPING,
    group_value_scores: dict[str, float] | None = None,
    fixed_assignments: dict[str, int] | None = None,
    bf16_candidate_fraction: float = 0.15,
    bf16_candidate_names: set[str] | None = None,
    bf16_rescue_scale: float = 0.5,
    bf16_sensitivity_weight: float = 0.35,
    lambda_iterations: int = 48,
    min_budget_utilization: float = 0.99,
) -> tuple[SearchRunResult, dict[str, Any]]:
    normalized_bits = _normalize_allowed_bits(allowed_bits)
    if normalized_bits != (4, 8, 16):
        raise ValueError("Budgeted BF16 allocator currently expects allowed_bits to be exactly (4, 8, 16)")
    if target_budget_gb <= 0:
        raise ValueError("target_budget_gb must be positive")

    fixed_group_bits = {str(name): int(bit) for name, bit in (fixed_assignments or {}).items()}
    group_lookup = {group.name: group for group in groups}
    unknown_fixed = sorted(set(fixed_group_bits) - set(group_lookup))
    if unknown_fixed:
        raise ValueError(f"Fixed assignments reference unknown groups: {unknown_fixed[:5]}")

    bf16_candidates = _resolve_bf16_candidate_names(
        groups=groups,
        group_value_scores=group_value_scores,
        fixed_assignments=fixed_group_bits,
        bf16_candidate_fraction=bf16_candidate_fraction,
        explicit_candidate_names=bf16_candidate_names,
    )
    state_values, score_manifest = _resolve_allocator_state_values(
        groups=groups,
        group_value_scores=group_value_scores,
        bf16_candidate_names=bf16_candidates,
        bf16_rescue_scale=bf16_rescue_scale,
        bf16_sensitivity_weight=bf16_sensitivity_weight,
    )

    target_budget_bytes = target_budget_gb * (1024**3)
    fixed_cost_bytes = sum(
        group_lookup[group_name].parameter_count * bit_width / 8
        for group_name, bit_width in fixed_group_bits.items()
    )
    if fixed_cost_bytes > target_budget_bytes + 1e-9:
        raise ValueError("Fixed assignments consume the full search budget")

    lo = 0.0
    hi = 1.0
    feasible_solution: dict[str, Any] | None = None
    infeasible_solution: dict[str, Any] | None = None
    while True:
        candidate = _solve_budgeted_allocator_for_lambda(
            groups=groups,
            lambda_value=hi,
            state_values=state_values,
            bf16_candidate_names=bf16_candidates,
            fixed_assignments=fixed_group_bits,
        )
        if candidate["total_cost_bytes"] <= target_budget_bytes:
            feasible_solution = candidate
            break
        hi *= 2.0
        if hi > 1e12:
            raise RuntimeError("Failed to bracket a feasible lambda for the BF16 allocator")

    for _ in range(max(1, lambda_iterations)):
        mid = (lo + hi) / 2.0
        candidate = _solve_budgeted_allocator_for_lambda(
            groups=groups,
            lambda_value=mid,
            state_values=state_values,
            bf16_candidate_names=bf16_candidates,
            fixed_assignments=fixed_group_bits,
        )
        if candidate["total_cost_bytes"] > target_budget_bytes:
            lo = mid
            infeasible_solution = candidate
        else:
            hi = mid
            feasible_solution = candidate

    repaired_candidates: list[dict[str, Any]] = []
    for label, candidate in (
        ("feasible_lambda", feasible_solution),
        ("infeasible_lambda", infeasible_solution),
    ):
        if candidate is None:
            continue
        repaired = _repair_budgeted_allocator_assignments(
            groups=groups,
            assignments=dict(candidate["assignments"]),
            state_values=state_values,
            bf16_candidate_names=bf16_candidates,
            fixed_assignments=fixed_group_bits,
            target_budget_bytes=target_budget_bytes,
            min_budget_utilization=min_budget_utilization,
        )
        repaired["source"] = label
        repaired_candidates.append(repaired)

    if not repaired_candidates:
        raise RuntimeError("Budgeted BF16 allocator did not produce any candidate assignments")

    best_candidate = max(
        repaired_candidates,
        key=lambda item: (
            float(item["total_value"]),
            float(item["total_cost_bytes"]),
        ),
    )
    best_assignments = dict(best_candidate["assignments"])
    total_params = sum(group.parameter_count for group in groups)
    normalized_value = float(best_candidate["total_value"]) / max(total_params, 1)
    candidate = SearchCandidate(
        group_bits=tuple(sorted(best_assignments.items())),
        estimated_average_bit_width=estimate_candidate_average_bit_width(groups, best_assignments),
        estimated_weight_footprint_gb=estimate_candidate_weight_footprint_gb(groups, best_assignments),
        proxy_quality_score=normalized_value,
        fitness=normalized_value,
        provenance="budgeted_bf16_allocator",
    )
    result = SearchRunResult(
        grouping=grouping,
        target_budget_gb=target_budget_gb,
        allowed_bits=normalized_bits,
        population_size=0,
        generations=0,
        num_groups=len(groups),
        top_candidates=(candidate,),
        history=tuple(),
    )
    manifest = {
        "strategy": "budgeted_bf16_allocator",
        "grouping": grouping,
        "target_budget_gb": target_budget_gb,
        "target_budget_bytes": target_budget_bytes,
        "fixed_assignments": fixed_group_bits,
        "bf16_candidate_fraction": bf16_candidate_fraction,
        "bf16_candidate_names": sorted(bf16_candidates),
        "bf16_group_names": sorted(group_name for group_name, bit_width in best_assignments.items() if bit_width >= 16),
        "int8_group_names": sorted(group_name for group_name, bit_width in best_assignments.items() if bit_width == 8),
        "int4_group_names": sorted(group_name for group_name, bit_width in best_assignments.items() if bit_width == 4),
        "chosen_lambda": hi,
        "candidate_budget_utilization": (
            0.0 if target_budget_bytes <= 0 else float(best_candidate["total_cost_bytes"]) / target_budget_bytes
        ),
        "candidate_total_value": float(best_candidate["total_value"]),
        "candidate_total_cost_bytes": float(best_candidate["total_cost_bytes"]),
        "repair_source": str(best_candidate.get("source", "unknown")),
        "score_manifest": score_manifest,
    }
    return result, manifest


def aggregate_group_score_overrides(
    layer_stats: list[LinearLayerStat],
    score_overrides: dict[str, float],
    target_grouping: str,
    source_grouping: str = DEFAULT_GROUPING,
) -> dict[str, float]:
    if not score_overrides:
        return {}

    weighted_totals: dict[str, float] = {}
    weight_sums: dict[str, int] = {}
    for layer in layer_stats:
        source_group_name, _ = _group_key_for_layer(layer.name, source_grouping)
        if source_group_name not in score_overrides:
            continue
        target_group_name, _ = _group_key_for_layer(layer.name, target_grouping)
        weighted_totals[target_group_name] = weighted_totals.get(target_group_name, 0.0) + (
            float(score_overrides[source_group_name]) * layer.parameter_count
        )
        weight_sums[target_group_name] = weight_sums.get(target_group_name, 0) + layer.parameter_count

    return {
        group_name: weighted_totals[group_name] / weight_sums[group_name]
        for group_name in weighted_totals
        if weight_sums[group_name] > 0
    }


def resolve_sensitivity_overrides(
    layer_stats: list[LinearLayerStat],
    grouping: str,
    sensitivity_profile_payload: dict[str, Any] | None,
    field: str = "combined_sensitivity",
) -> dict[str, float] | None:
    if not sensitivity_profile_payload:
        return None
    source_grouping = str(sensitivity_profile_payload.get("grouping", DEFAULT_GROUPING))
    source_overrides = group_sensitivity_overrides_from_profile(
        sensitivity_profile_payload,
        field=field,
    )
    if grouping == source_grouping:
        return source_overrides
    return aggregate_group_score_overrides(
        layer_stats=layer_stats,
        score_overrides=source_overrides,
        target_grouping=grouping,
        source_grouping=source_grouping,
    )


def build_group_expansion_mapping(
    target_groups: list[SearchGroup],
    source_grouping: str,
) -> dict[str, tuple[str, ...]]:
    mapping: dict[str, list[str]] = {}
    for group in target_groups:
        if not group.layer_names:
            continue
        source_group_name, _ = _group_key_for_layer(group.layer_names[0], source_grouping)
        mapping.setdefault(source_group_name, []).append(group.name)
    return {
        group_name: tuple(sorted(expanded_group_names))
        for group_name, expanded_group_names in mapping.items()
    }


def expand_group_assignments(
    source_assignments: dict[str, int],
    target_groups: list[SearchGroup],
    source_grouping: str,
) -> dict[str, int]:
    expanded_assignments: dict[str, int] = {}
    for group in target_groups:
        if not group.layer_names:
            continue
        source_group_name, _ = _group_key_for_layer(group.layer_names[0], source_grouping)
        if source_group_name not in source_assignments:
            raise KeyError(f"Missing source assignment for {source_group_name}")
        expanded_assignments[group.name] = int(source_assignments[source_group_name])
    return expanded_assignments


def build_hierarchical_promotion_manifest(
    coarse_groups: list[SearchGroup],
    coarse_candidates: list[SearchCandidate],
    fine_groups: list[SearchGroup],
    coarse_group_value_scores: dict[str, float] | None,
    source_grouping: str,
    max_promoted_fine_groups: int = 160,
) -> dict[str, Any]:
    if not coarse_candidates:
        raise ValueError("At least one coarse candidate is required")

    top_candidates = coarse_candidates[:8]
    best_candidate_bits = top_candidates[0].bits_dict()
    coarse_to_fine = build_group_expansion_mapping(
        target_groups=fine_groups,
        source_grouping=source_grouping,
    )
    coarse_value_scores = coarse_group_value_scores or {}
    top_value_count = max(1, math.ceil(len(coarse_groups) * 0.2))
    top_value_group_names = {
        group.name
        for group in sorted(
            coarse_groups,
            key=lambda item: float(coarse_value_scores.get(item.name, 0.0)),
            reverse=True,
        )[:top_value_count]
    }

    candidate_records: list[dict[str, Any]] = []
    for group in coarse_groups:
        assigned_bits = [
            candidate.bits_dict().get(group.name, 0)
            for candidate in top_candidates
        ]
        fine_group_names = coarse_to_fine.get(group.name, ())
        high_bit_vote_count = sum(1 for bit_width in assigned_bits if bit_width >= 8)
        best_candidate_uses_16 = best_candidate_bits.get(group.name, 0) >= 16
        promoted_by_votes = high_bit_vote_count >= 4
        promoted_by_value = group.name in top_value_group_names
        if not (best_candidate_uses_16 or promoted_by_votes or promoted_by_value):
            continue
        bit_pressure = statistics.fmean(
            1.0 if bit_width >= 16 else 0.5 if bit_width >= 8 else 0.0
            for bit_width in assigned_bits
        )
        promotion_reasons = []
        if best_candidate_uses_16:
            promotion_reasons.append("best_candidate_16bit")
        if promoted_by_votes:
            promotion_reasons.append("high_bit_votes")
        if promoted_by_value:
            promotion_reasons.append("top20_group_value")
        candidate_records.append(
            {
                "name": group.name,
                "component_type": group.component_type,
                "parameter_count": group.parameter_count,
                "assigned_bits": assigned_bits,
                "best_candidate_bit_width": int(best_candidate_bits.get(group.name, 0)),
                "high_bit_vote_count": high_bit_vote_count,
                "bit_pressure_score": bit_pressure,
                "group_value_score": float(coarse_value_scores.get(group.name, 0.0)),
                "fine_groups": list(fine_group_names),
                "fine_group_count": len(fine_group_names),
                "promotion_reasons": promotion_reasons,
            }
        )

    if not candidate_records:
        fallback_group = max(
            coarse_groups,
            key=lambda item: (
                best_candidate_bits.get(item.name, 0),
                float(coarse_value_scores.get(item.name, 0.0)),
            ),
        )
        candidate_records.append(
            {
                "name": fallback_group.name,
                "component_type": fallback_group.component_type,
                "parameter_count": fallback_group.parameter_count,
                "assigned_bits": [candidate.bits_dict().get(fallback_group.name, 0) for candidate in top_candidates],
                "best_candidate_bit_width": int(best_candidate_bits.get(fallback_group.name, 0)),
                "high_bit_vote_count": 0,
                "bit_pressure_score": 0.0,
                "group_value_score": float(coarse_value_scores.get(fallback_group.name, 0.0)),
                "fine_groups": list(coarse_to_fine.get(fallback_group.name, ())),
                "fine_group_count": len(coarse_to_fine.get(fallback_group.name, ())),
                "promotion_reasons": ["fallback_best_coarse_group"],
            }
        )

    ranked_candidates = sorted(
        candidate_records,
        key=lambda item: (
            "best_candidate_16bit" in item["promotion_reasons"],
            "high_bit_votes" in item["promotion_reasons"],
            "top20_group_value" in item["promotion_reasons"],
            float(item["bit_pressure_score"]),
            float(item["group_value_score"]),
            int(item["parameter_count"]),
        ),
        reverse=True,
    )

    promoted_groups: list[dict[str, Any]] = []
    promoted_fine_group_names: set[str] = set()
    for record in ranked_candidates:
        fine_group_names = list(record["fine_groups"])
        if not fine_group_names:
            continue
        if len(promoted_fine_group_names) + len(fine_group_names) > max_promoted_fine_groups:
            continue
        promoted_groups.append(record)
        promoted_fine_group_names.update(fine_group_names)

    if not promoted_groups and ranked_candidates:
        promoted_groups.append(ranked_candidates[0])
        promoted_fine_group_names.update(ranked_candidates[0]["fine_groups"])

    frozen_fine_group_names = sorted(
        group.name for group in fine_groups if group.name not in promoted_fine_group_names
    )
    return {
        "source_grouping": source_grouping,
        "max_promoted_fine_groups": max_promoted_fine_groups,
        "coarse_to_fine_map": {
            group_name: list(expanded_groups)
            for group_name, expanded_groups in coarse_to_fine.items()
        },
        "promotion_candidates": ranked_candidates,
        "promoted_groups": promoted_groups,
        "promoted_group_names": [item["name"] for item in promoted_groups],
        "promoted_fine_group_names": sorted(promoted_fine_group_names),
        "frozen_fine_group_names": frozen_fine_group_names,
    }


def refine_candidate_quantization_configs(
    groups: list[SearchGroup],
    layer_stats: list[LinearLayerStat],
    base_candidates: list[SearchCandidate],
    group_value_scores: dict[str, float] | None = None,
    allowed_group_names: set[str] | None = None,
    group_size_options: tuple[int, ...] = DEFAULT_GROUP_SIZE_OPTIONS,
    symmetric_options: tuple[bool, ...] = DEFAULT_SYMMETRIC_OPTIONS,
    max_tunable_groups: int = 48,
    population_size: int = 40,
    generations: int = 16,
    mutation_rate: float = 0.18,
    config_gain_weight: float = 0.18,
    top_k: int = 5,
    seed: int = 0,
    seed_candidate_count: int = 3,
) -> dict[str, Any]:
    normalized_group_sizes = _normalize_group_size_options(group_size_options)
    normalized_symmetric_options = _normalize_symmetric_options(symmetric_options)
    rng = random.Random(seed)
    allowed_names = set(allowed_group_names or {group.name for group in groups})
    valid_group_sizes = _resolve_valid_group_size_options(groups, layer_stats, normalized_group_sizes)

    refined_candidates: list[SearchCandidate] = []
    seed_runs: list[dict[str, Any]] = []
    for base_index, base_candidate in enumerate(base_candidates[: max(1, seed_candidate_count)], start=1):
        bit_assignments = base_candidate.bits_dict()
        tunable_groups = _rank_tunable_groups_for_quantization_config_search(
            groups=groups,
            bit_assignments=bit_assignments,
            group_value_scores=group_value_scores,
            allowed_group_names=allowed_names,
            valid_group_sizes=valid_group_sizes,
            symmetric_options=normalized_symmetric_options,
            max_tunable_groups=max_tunable_groups,
        )
        tunable_group_names = [group.name for group in tunable_groups]
        default_config_score = _estimate_quantization_config_quality_score(
            groups=tunable_groups,
            bit_assignments=bit_assignments,
            overrides={},
            group_value_scores=group_value_scores,
        )
        if not tunable_groups:
            refined_candidates.append(base_candidate)
            seed_runs.append(
                {
                    "base_candidate_rank": base_index,
                    "base_candidate_provenance": base_candidate.provenance,
                    "tunable_group_count": 0,
                    "tunable_group_names": [],
                    "history": [],
                    "top_candidates": [base_candidate.to_dict()],
                }
            )
            continue

        candidate_builder = lambda overrides, provenance: _build_quantization_config_candidate(  # noqa: E731
            base_candidate=base_candidate,
            tunable_groups=tunable_groups,
            bit_assignments=bit_assignments,
            overrides=overrides,
            group_value_scores=group_value_scores,
            default_config_score=default_config_score,
            config_gain_weight=config_gain_weight,
            provenance=provenance,
        )
        population = _initialize_quantization_config_population(
            tunable_groups=tunable_groups,
            valid_group_sizes=valid_group_sizes,
            symmetric_options=normalized_symmetric_options,
            population_size=population_size,
            rng=rng,
            candidate_builder=candidate_builder,
        )
        history: list[dict[str, Any]] = []
        current_mutation_rate = mutation_rate
        max_mutation_rate = max(0.35, mutation_rate)

        for generation in range(generations + 1):
            population = sorted(population, key=lambda candidate: candidate.fitness, reverse=True)
            history.append(
                {
                    "generation": generation,
                    "best_fitness": population[0].fitness,
                    "mean_fitness": statistics.fmean(candidate.fitness for candidate in population),
                    "best_quantization_config_score": (
                        float(population[0].quantization_config_score or 0.0)
                    ),
                    "unique_candidate_count": len(
                        {candidate.candidate_signature() for candidate in population}
                    ),
                    "mutation_rate": current_mutation_rate,
                }
            )
            if generation == generations:
                break

            next_population = population[: max(2, min(6, population_size // 6))]
            seen_signatures = {candidate.candidate_signature() for candidate in next_population}
            while len(next_population) < population_size:
                left = _tournament_select(population, 3, rng)
                right = _tournament_select_distinct(population, 3, rng, left)
                child_overrides = _crossover_quantization_overrides(
                    left.quantization_overrides_dict(),
                    right.quantization_overrides_dict(),
                    tunable_group_names=tunable_group_names,
                    rng=rng,
                )
                child_overrides = _mutate_quantization_overrides(
                    child_overrides,
                    tunable_group_names=tunable_group_names,
                    valid_group_sizes=valid_group_sizes,
                    symmetric_options=normalized_symmetric_options,
                    rng=rng,
                    mutation_rate=current_mutation_rate,
                )
                child = candidate_builder(child_overrides, f"{base_candidate.provenance}_config_gen_{generation + 1}")
                if child.candidate_signature() in seen_signatures:
                    current_mutation_rate = min(max_mutation_rate, current_mutation_rate * 1.05)
                    continue
                next_population.append(child)
                seen_signatures.add(child.candidate_signature())
            population = next_population
            current_mutation_rate = min(max_mutation_rate, max(mutation_rate, current_mutation_rate * 1.02))

        population = sorted(population, key=lambda candidate: candidate.fitness, reverse=True)
        unique_candidates: list[SearchCandidate] = []
        seen_signatures: set[tuple[Any, ...]] = set()
        for candidate in population:
            if candidate.candidate_signature() in seen_signatures:
                continue
            unique_candidates.append(candidate)
            seen_signatures.add(candidate.candidate_signature())
            if len(unique_candidates) == top_k:
                break
        refined_candidates.extend(unique_candidates)
        seed_runs.append(
            {
                "base_candidate_rank": base_index,
                "base_candidate_provenance": base_candidate.provenance,
                "tunable_group_count": len(tunable_group_names),
                "tunable_group_names": tunable_group_names,
                "tunable_group_options": {
                    group.name: {
                        "group_size_options": list(valid_group_sizes[group.name]),
                        "symmetric_options": list(normalized_symmetric_options),
                        "bit_width": int(bit_assignments[group.name]),
                    }
                    for group in tunable_groups
                },
                "history": history,
                "top_candidates": [candidate.to_dict() for candidate in unique_candidates],
            }
        )

    deduped_candidates: list[SearchCandidate] = []
    seen_signatures: set[tuple[Any, ...]] = set()
    for candidate in sorted(refined_candidates, key=lambda item: item.fitness, reverse=True):
        if candidate.candidate_signature() in seen_signatures:
            continue
        deduped_candidates.append(candidate)
        seen_signatures.add(candidate.candidate_signature())
        if len(deduped_candidates) == top_k:
            break

    return {
        "group_size_options": list(normalized_group_sizes),
        "symmetric_options": list(normalized_symmetric_options),
        "max_tunable_groups": max_tunable_groups,
        "population_size": population_size,
        "generations": generations,
        "seed_candidate_count": max(1, seed_candidate_count),
        "seed_runs": seed_runs,
        "top_candidates": [candidate.to_dict() for candidate in deduped_candidates],
    }


def save_search_result(path: str | Path, result: SearchRunResult) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result.to_dict(), handle, indent=2, sort_keys=True)
        handle.write("\n")


def _initialize_population(
    groups: list[SearchGroup],
    allowed_bits: tuple[int, ...],
    target_budget_gb: float,
    population_size: int,
    rng: random.Random,
    candidate_builder: Callable[[dict[str, int], str], SearchCandidate],
    extra_seed_assignments: list[tuple[str, dict[str, int]]] | None = None,
) -> list[SearchCandidate]:
    seed_assignments: list[tuple[str, dict[str, int]]] = []
    default_seed = repair_assignments_to_budget(
        groups=groups,
        assignments=default_seed_assignments(groups),
        target_budget_gb=target_budget_gb,
        allowed_bits=allowed_bits,
    )
    seed_assignments.append(("default_seed", default_seed))

    exploratory_seed = repair_assignments_to_budget(
        groups=groups,
        assignments=exploratory_seed_assignments(groups, allowed_bits),
        target_budget_gb=target_budget_gb,
        allowed_bits=allowed_bits,
        min_budget_utilization=0.92,
    )
    seed_assignments.append(("exploratory_seed", exploratory_seed))

    min_seed = repair_assignments_to_budget(
        groups=groups,
        assignments={group.name: allowed_bits[0] for group in groups},
        target_budget_gb=target_budget_gb,
        allowed_bits=allowed_bits,
        min_budget_utilization=0.9,
    )
    seed_assignments.append(("min_seed", min_seed))

    max_seed = repair_assignments_to_budget(
        groups=groups,
        assignments={group.name: allowed_bits[-1] for group in groups},
        target_budget_gb=target_budget_gb,
        allowed_bits=allowed_bits,
    )
    seed_assignments.append(("max_seed_repaired", max_seed))
    if extra_seed_assignments:
        for provenance, assignments in extra_seed_assignments:
            repaired = repair_assignments_to_budget(
                groups=groups,
                assignments=assignments,
                target_budget_gb=target_budget_gb,
                allowed_bits=allowed_bits,
                min_budget_utilization=0.92,
            )
            seed_assignments.append((provenance, repaired))

    population: list[SearchCandidate] = []
    seen_assignments: set[tuple[tuple[str, int], ...]] = set()
    for provenance, assignments in seed_assignments:
        candidate = candidate_builder(assignments, provenance)
        if candidate.group_bits in seen_assignments:
            continue
        population.append(candidate)
        seen_assignments.add(candidate.group_bits)

    attempts = 0
    max_attempts = max(population_size * 20, 100)
    while len(population) < population_size and attempts < max_attempts:
        attempts += 1
        random_assignments = {
            group.name: rng.choice(allowed_bits)
            for group in groups
        }
        random_assignments = repair_assignments_to_budget(
            groups=groups,
            assignments=random_assignments,
            target_budget_gb=target_budget_gb,
            allowed_bits=allowed_bits,
            min_budget_utilization=0.9,
        )
        candidate = candidate_builder(random_assignments, "random_seed")
        if candidate.group_bits in seen_assignments:
            continue
        population.append(candidate)
        seen_assignments.add(candidate.group_bits)

    while len(population) < population_size:
        random_assignments = {
            group.name: rng.choice(allowed_bits)
            for group in groups
        }
        random_assignments = repair_assignments_to_budget(
            groups=groups,
            assignments=random_assignments,
            target_budget_gb=target_budget_gb,
            allowed_bits=allowed_bits,
            min_budget_utilization=0.9,
        )
        population.append(candidate_builder(random_assignments, "random_seed_fallback"))
    return population


def _run_evolution_search(
    groups: list[SearchGroup],
    target_budget_gb: float,
    allowed_bits: tuple[int, ...],
    grouping: str,
    population_size: int,
    generations: int,
    elite_count: int,
    tournament_size: int,
    mutation_rate: float,
    top_k: int,
    rng: random.Random,
    candidate_builder: Callable[[dict[str, int], str], SearchCandidate],
    extra_seed_assignments: list[tuple[str, dict[str, int]]] | None = None,
) -> SearchRunResult:
    max_mutation_rate = max(0.3, mutation_rate)
    immigrant_fraction = 0.12
    stagnation_patience = 2
    population = _initialize_population(
        groups=groups,
        allowed_bits=allowed_bits,
        target_budget_gb=target_budget_gb,
        population_size=population_size,
        rng=rng,
        candidate_builder=candidate_builder,
        extra_seed_assignments=extra_seed_assignments,
    )
    history: list[SearchGenerationSummary] = []
    best_fitness_so_far = float("-inf")
    stagnation_steps = 0
    current_mutation_rate = mutation_rate

    for generation in range(generations + 1):
        population = sorted(population, key=lambda candidate: candidate.fitness, reverse=True)
        best_fitness = population[0].fitness
        if best_fitness > best_fitness_so_far + 1e-9:
            best_fitness_so_far = best_fitness
            stagnation_steps = 0
        elif generation > 0:
            stagnation_steps += 1

        current_mutation_rate = min(
            max_mutation_rate,
            mutation_rate * (1.0 + 0.35 * stagnation_steps),
        )
        unique_candidate_count = len({candidate.group_bits for candidate in population})
        immigrant_count = (
            max(1, round(population_size * immigrant_fraction))
            if stagnation_steps >= stagnation_patience
            else 0
        )
        history.append(
            SearchGenerationSummary(
                generation=generation,
                best_fitness=best_fitness,
                mean_fitness=statistics.fmean(candidate.fitness for candidate in population),
                best_proxy_quality_score=population[0].proxy_quality_score,
                best_estimated_weight_footprint_gb=population[0].estimated_weight_footprint_gb,
                unique_candidate_count=unique_candidate_count,
                mean_population_diversity=_estimate_population_diversity(population),
                mutation_rate=current_mutation_rate,
                stagnation_steps=stagnation_steps,
                immigrant_count=immigrant_count,
            )
        )
        if generation == generations:
            break

        next_population = population[:elite_count]
        seen_assignments = {candidate.group_bits for candidate in next_population}
        target_offspring_count = max(elite_count, population_size - immigrant_count)
        attempts = 0
        max_attempts = max(population_size * 40, 200)
        while len(next_population) < target_offspring_count and attempts < max_attempts:
            attempts += 1
            left = _tournament_select(population, tournament_size, rng)
            right = _tournament_select_distinct(population, tournament_size, rng, left)
            child_bits = crossover_assignments(left.bits_dict(), right.bits_dict(), rng)
            child_bits = mutate_assignments(child_bits, allowed_bits, rng, current_mutation_rate)
            child_bits = repair_assignments_to_budget(
                groups=groups,
                assignments=child_bits,
                target_budget_gb=target_budget_gb,
                allowed_bits=allowed_bits,
            )
            child = candidate_builder(child_bits, f"generation_{generation + 1}")
            if child.group_bits in seen_assignments:
                retry_bits = mutate_assignments(
                    child_bits,
                    allowed_bits,
                    rng,
                    min(max_mutation_rate, current_mutation_rate * 1.5),
                )
                retry_bits = repair_assignments_to_budget(
                    groups=groups,
                    assignments=retry_bits,
                    target_budget_gb=target_budget_gb,
                    allowed_bits=allowed_bits,
                )
                child = candidate_builder(retry_bits, f"generation_{generation + 1}_retry")
            if child.group_bits in seen_assignments:
                continue
            next_population.append(child)
            seen_assignments.add(child.group_bits)

        while len(next_population) < population_size:
            immigrant = _build_random_candidate(
                groups=groups,
                allowed_bits=allowed_bits,
                target_budget_gb=target_budget_gb,
                rng=rng,
                candidate_builder=candidate_builder,
                provenance=f"immigrant_generation_{generation + 1}",
            )
            if immigrant.group_bits in seen_assignments:
                fallback_bits = mutate_assignments(
                    immigrant.bits_dict(),
                    allowed_bits,
                    rng,
                    min(max_mutation_rate, current_mutation_rate * 1.5),
                )
                fallback_bits = repair_assignments_to_budget(
                    groups=groups,
                    assignments=fallback_bits,
                    target_budget_gb=target_budget_gb,
                    allowed_bits=allowed_bits,
                )
                immigrant = candidate_builder(
                    fallback_bits,
                    f"immigrant_generation_{generation + 1}_retry",
                )
            next_population.append(immigrant)
            seen_assignments.add(immigrant.group_bits)
        population = next_population

    unique_candidates: list[SearchCandidate] = []
    seen_assignments: set[tuple[tuple[str, int], ...]] = set()
    for candidate in population:
        if candidate.group_bits in seen_assignments:
            continue
        unique_candidates.append(candidate)
        seen_assignments.add(candidate.group_bits)
        if len(unique_candidates) == top_k:
            break

    return SearchRunResult(
        grouping=grouping,
        target_budget_gb=target_budget_gb,
        allowed_bits=allowed_bits,
        population_size=population_size,
        generations=generations,
        num_groups=len(groups),
        top_candidates=tuple(unique_candidates),
        history=tuple(history),
    )


def _normalize_allowed_bits(allowed_bits: tuple[int, ...]) -> tuple[int, ...]:
    normalized = tuple(sorted({int(bit) for bit in allowed_bits}))
    unsupported = [bit for bit in normalized if bit not in EVOLUTION_SEARCH_SUPPORTED_BITS]
    if unsupported:
        raise ValueError(
            "Evolution search currently only supports runtime-executable bit-widths "
            f"{EVOLUTION_SEARCH_SUPPORTED_BITS}; received {unsupported}"
        )
    return normalized


def _normalize_group_size_options(group_size_options: tuple[int, ...]) -> tuple[int, ...]:
    normalized = tuple(sorted({int(option) for option in group_size_options if int(option) > 0}))
    if not normalized:
        raise ValueError("group_size_options must include at least one positive value")
    return normalized


def _normalize_symmetric_options(symmetric_options: tuple[bool, ...]) -> tuple[bool, ...]:
    normalized = tuple(dict.fromkeys(bool(option) for option in symmetric_options))
    if not normalized:
        raise ValueError("symmetric_options must include at least one boolean value")
    return normalized


def _load_json(path: str | Path) -> dict[str, Any]:
    json_path = Path(path)
    with json_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _tournament_select(
    population: list[SearchCandidate],
    tournament_size: int,
    rng: random.Random,
) -> SearchCandidate:
    tournament = rng.sample(population, k=min(tournament_size, len(population)))
    return max(tournament, key=lambda candidate: candidate.fitness)


def _tournament_select_distinct(
    population: list[SearchCandidate],
    tournament_size: int,
    rng: random.Random,
    reference_candidate: SearchCandidate,
) -> SearchCandidate:
    distinct_population = [
        candidate
        for candidate in population
        if candidate.group_bits != reference_candidate.group_bits
    ]
    if not distinct_population:
        return reference_candidate
    return _tournament_select(distinct_population, tournament_size, rng)


def _build_random_candidate(
    groups: list[SearchGroup],
    allowed_bits: tuple[int, ...],
    target_budget_gb: float,
    rng: random.Random,
    candidate_builder: Callable[[dict[str, int], str], SearchCandidate],
    provenance: str,
) -> SearchCandidate:
    random_assignments = {
        group.name: rng.choice(allowed_bits)
        for group in groups
    }
    random_assignments = repair_assignments_to_budget(
        groups=groups,
        assignments=random_assignments,
        target_budget_gb=target_budget_gb,
        allowed_bits=allowed_bits,
        min_budget_utilization=0.9,
    )
    return candidate_builder(random_assignments, provenance)


def _estimate_population_diversity(population: list[SearchCandidate]) -> float:
    if len(population) < 2:
        return 0.0
    distances: list[float] = []
    for left_index in range(len(population)):
        left_bits = dict(population[left_index].group_bits)
        for right_index in range(left_index + 1, len(population)):
            right_bits = dict(population[right_index].group_bits)
            total_groups = len(left_bits)
            mismatches = sum(
                1
                for group_name, left_bit in left_bits.items()
                if right_bits.get(group_name) != left_bit
            )
            distances.append(mismatches / total_groups if total_groups else 0.0)
    if not distances:
        return 0.0
    return statistics.fmean(distances)


def _weighted_mean(values: list[tuple[float, int]]) -> float:
    numerator = sum(value * weight for value, weight in values)
    denominator = sum(weight for _, weight in values)
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _quality_drop(
    group: SearchGroup,
    from_bit: int,
    to_bit: int,
    quality_prior: dict[int, float],
) -> float:
    from_quality = quality_prior.get(from_bit, 0.0)
    to_quality = quality_prior.get(to_bit, 0.0)
    return group.parameter_count * group.sensitivity * abs(from_quality - to_quality)


def _added_footprint_gb(group: SearchGroup, from_bit: int, to_bit: int) -> float:
    bit_delta = abs(to_bit - from_bit)
    return (group.parameter_count * bit_delta) / 8 / (1024**3)


def _estimate_partial_weight_footprint_gb(
    groups: list[SearchGroup],
    assignments: dict[str, int],
) -> float:
    if not assignments:
        return 0.0
    group_lookup = {group.name: group for group in groups}
    total_bits = 0
    for group_name, bit_width in assignments.items():
        group = group_lookup.get(group_name)
        if group is None:
            continue
        total_bits += group.parameter_count * int(bit_width)
    return total_bits / 8 / (1024**3)


def _filter_assignments_to_groups(
    assignments: dict[str, int],
    allowed_group_names: set[str],
) -> dict[str, int]:
    return {
        str(group_name): int(bit_width)
        for group_name, bit_width in assignments.items()
        if group_name in allowed_group_names
    }


def _filter_score_map(
    score_map: dict[str, float] | None,
    allowed_group_names: set[str],
) -> dict[str, float] | None:
    if score_map is None:
        return None
    return {
        str(group_name): float(score)
        for group_name, score in score_map.items()
        if group_name in allowed_group_names
    }


def _resolve_valid_group_size_options(
    groups: list[SearchGroup],
    layer_stats: list[LinearLayerStat],
    group_size_options: tuple[int, ...],
) -> dict[str, tuple[int, ...]]:
    layer_lookup = {layer.name: layer for layer in layer_stats}
    valid_options: dict[str, tuple[int, ...]] = {}
    for group in groups:
        valid_group_sizes: list[int] = []
        member_layers = [layer_lookup[layer_name] for layer_name in group.layer_names if layer_name in layer_lookup]
        for group_size in group_size_options:
            if all(layer.in_features < group_size or layer.in_features % group_size == 0 for layer in member_layers):
                valid_group_sizes.append(group_size)
        valid_options[group.name] = tuple(valid_group_sizes or [DEFAULT_WEIGHT_GROUP_SIZE])
    return valid_options


def _rank_tunable_groups_for_quantization_config_search(
    groups: list[SearchGroup],
    bit_assignments: dict[str, int],
    group_value_scores: dict[str, float] | None,
    allowed_group_names: set[str],
    valid_group_sizes: dict[str, tuple[int, ...]],
    symmetric_options: tuple[bool, ...],
    max_tunable_groups: int,
) -> list[SearchGroup]:
    ranked_groups: list[tuple[float, SearchGroup]] = []
    for group in groups:
        if group.name not in allowed_group_names:
            continue
        if int(bit_assignments.get(group.name, HIGH_PRECISION_BIT)) == HIGH_PRECISION_BIT:
            continue
        has_group_size_choice = len(valid_group_sizes.get(group.name, (DEFAULT_WEIGHT_GROUP_SIZE,))) > 1
        has_symmetry_choice = len(symmetric_options) > 1
        if not has_group_size_choice and not has_symmetry_choice:
            continue
        value_score = float(group_value_scores.get(group.name, 0.0)) if group_value_scores else 0.0
        importance = group.parameter_count * (
            0.75
            + group.sensitivity
            + 4.0 * max(value_score, 0.0)
        )
        if int(bit_assignments[group.name]) <= 4:
            importance *= 1.35
        ranked_groups.append((importance, group))
    ranked_groups.sort(key=lambda item: item[0], reverse=True)
    return [group for _, group in ranked_groups[:max_tunable_groups]]


def _estimate_quantization_config_quality_score(
    groups: list[SearchGroup],
    bit_assignments: dict[str, int],
    overrides: dict[str, dict[str, Any]],
    group_value_scores: dict[str, float] | None,
) -> float:
    if not groups:
        return 0.0
    numerator = 0.0
    denominator = 0.0
    for group in groups:
        bit_width = int(bit_assignments[group.name])
        override = overrides.get(group.name, {})
        group_size = int(override.get("group_size", DEFAULT_WEIGHT_GROUP_SIZE))
        symmetric = bool(override.get("symmetric", DEFAULT_WEIGHT_SYMMETRIC))
        value_score = float(group_value_scores.get(group.name, 0.0)) if group_value_scores else 0.0
        importance = group.parameter_count * (
            0.75
            + group.sensitivity
            + 4.0 * max(value_score, 0.0)
        )
        if bit_width <= 4:
            importance *= 1.35
        group_size_score = {
            32: 1.0,
            64: 0.9,
            128: 0.75,
            256: 0.55,
        }.get(group_size, max(0.2, 128.0 / float(max(group_size, 128))))
        symmetric_score = (
            0.9 if symmetric else 1.0
            if bit_width <= 4
            else 0.97 if symmetric else 1.0
        )
        numerator += importance * (0.7 * group_size_score + 0.3 * symmetric_score)
        denominator += importance
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _initialize_quantization_config_population(
    tunable_groups: list[SearchGroup],
    valid_group_sizes: dict[str, tuple[int, ...]],
    symmetric_options: tuple[bool, ...],
    population_size: int,
    rng: random.Random,
    candidate_builder: Callable[[dict[str, dict[str, Any]], str], SearchCandidate],
) -> list[SearchCandidate]:
    smallest_group_sizes = {
        group.name: min(valid_group_sizes[group.name])
        for group in tunable_groups
    }
    preferred_asymmetric = False if False in symmetric_options else symmetric_options[0]

    seed_states: list[tuple[str, dict[str, dict[str, Any]]]] = [
        ("default_quant_config_seed", {}),
        (
            "aggressive_quant_config_seed",
            {
                group.name: {
                    "group_size": smallest_group_sizes[group.name],
                    "symmetric": preferred_asymmetric,
                }
                for group in tunable_groups
            },
        ),
        (
            "focused_quant_config_seed",
            {
                group.name: {
                    "group_size": smallest_group_sizes[group.name],
                    "symmetric": preferred_asymmetric,
                }
                for group in tunable_groups[: max(1, len(tunable_groups) // 3)]
            },
        ),
    ]

    population: list[SearchCandidate] = []
    seen_signatures: set[tuple[Any, ...]] = set()
    for provenance, overrides in seed_states:
        candidate = candidate_builder(overrides, provenance)
        if candidate.candidate_signature() in seen_signatures:
            continue
        population.append(candidate)
        seen_signatures.add(candidate.candidate_signature())

    while len(population) < population_size:
        random_overrides: dict[str, dict[str, Any]] = {}
        for group in tunable_groups:
            if rng.random() < 0.45:
                random_overrides[group.name] = {
                    "group_size": rng.choice(valid_group_sizes[group.name]),
                    "symmetric": rng.choice(symmetric_options),
                }
        candidate = candidate_builder(random_overrides, "random_quant_config_seed")
        if candidate.candidate_signature() in seen_signatures:
            continue
        population.append(candidate)
        seen_signatures.add(candidate.candidate_signature())
    return population


def _crossover_quantization_overrides(
    left: dict[str, dict[str, Any]],
    right: dict[str, dict[str, Any]],
    tunable_group_names: list[str],
    rng: random.Random,
) -> dict[str, dict[str, Any]]:
    child: dict[str, dict[str, Any]] = {}
    for group_name in tunable_group_names:
        source = left if rng.random() < 0.5 else right
        if group_name in source:
            child[group_name] = dict(source[group_name])
    return child


def _mutate_quantization_overrides(
    overrides: dict[str, dict[str, Any]],
    tunable_group_names: list[str],
    valid_group_sizes: dict[str, tuple[int, ...]],
    symmetric_options: tuple[bool, ...],
    rng: random.Random,
    mutation_rate: float,
) -> dict[str, dict[str, Any]]:
    mutated = {
        group_name: dict(config)
        for group_name, config in overrides.items()
    }
    for group_name in tunable_group_names:
        if rng.random() >= mutation_rate:
            continue
        if rng.random() < 0.15:
            mutated.pop(group_name, None)
            continue
        next_group_size = rng.choice(valid_group_sizes[group_name])
        next_symmetric = rng.choice(symmetric_options)
        if next_group_size == DEFAULT_WEIGHT_GROUP_SIZE and next_symmetric == DEFAULT_WEIGHT_SYMMETRIC:
            mutated.pop(group_name, None)
            continue
        mutated[group_name] = {
            "group_size": next_group_size,
            "symmetric": next_symmetric,
        }
    return mutated


def _build_quantization_config_candidate(
    base_candidate: SearchCandidate,
    tunable_groups: list[SearchGroup],
    bit_assignments: dict[str, int],
    overrides: dict[str, dict[str, Any]],
    group_value_scores: dict[str, float] | None,
    default_config_score: float,
    config_gain_weight: float,
    provenance: str,
) -> SearchCandidate:
    normalized_overrides = {
        str(group_name): {
            "group_size": int(config.get("group_size", DEFAULT_WEIGHT_GROUP_SIZE)),
            "symmetric": bool(config.get("symmetric", DEFAULT_WEIGHT_SYMMETRIC)),
        }
        for group_name, config in overrides.items()
        if (
            int(config.get("group_size", DEFAULT_WEIGHT_GROUP_SIZE)) != DEFAULT_WEIGHT_GROUP_SIZE
            or bool(config.get("symmetric", DEFAULT_WEIGHT_SYMMETRIC)) != DEFAULT_WEIGHT_SYMMETRIC
        )
    }
    config_score = _estimate_quantization_config_quality_score(
        groups=tunable_groups,
        bit_assignments=bit_assignments,
        overrides=normalized_overrides,
        group_value_scores=group_value_scores,
    ) - default_config_score
    return SearchCandidate(
        group_bits=base_candidate.group_bits,
        estimated_average_bit_width=base_candidate.estimated_average_bit_width,
        estimated_weight_footprint_gb=base_candidate.estimated_weight_footprint_gb,
        proxy_quality_score=base_candidate.proxy_quality_score,
        fitness=base_candidate.fitness + config_gain_weight * config_score,
        provenance=provenance,
        group_quantization_overrides=tuple(
            sorted(
                (
                    group_name,
                    int(config["group_size"]),
                    bool(config["symmetric"]),
                )
                for group_name, config in normalized_overrides.items()
            )
        ),
        quantization_config_score=config_score,
        prediction_uncertainty=base_candidate.prediction_uncertainty,
        conservative_prediction=base_candidate.conservative_prediction,
        budget_alignment_score=base_candidate.budget_alignment_score,
        group_value_alignment_score=base_candidate.group_value_alignment_score,
        reference_accuracy=base_candidate.reference_accuracy,
        reference_advantage_score=base_candidate.reference_advantage_score,
    )


def estimate_compression_bonus(
    groups: list[SearchGroup],
    assignments: dict[str, int],
    allowed_bits: tuple[int, ...],
) -> float:
    min_bit = min(allowed_bits)
    max_bit = max(allowed_bits)
    if min_bit == max_bit:
        return 0.0
    numerator = 0.0
    denominator = 0.0
    for group in groups:
        bit_width = assignments[group.name]
        compression_fraction = (max_bit - bit_width) / (max_bit - min_bit)
        weight = group.parameter_count * max(1.15 - group.sensitivity, 0.1)
        numerator += weight * compression_fraction
        denominator += weight
    if denominator == 0:
        return 0.0
    return numerator / denominator


def estimate_budget_alignment_score(
    footprint_gb: float,
    target_budget_gb: float,
) -> float:
    if target_budget_gb <= 0:
        return 0.0
    normalized_gap = abs(target_budget_gb - footprint_gb) / target_budget_gb
    return max(0.0, 1.0 - normalized_gap)


def resolve_group_value_scores(
    groups: list[SearchGroup],
    group_value_prior_payload: dict[str, Any] | None,
    layer_stats: list[LinearLayerStat] | None = None,
    target_grouping: str | None = None,
) -> dict[str, float]:
    if not group_value_prior_payload:
        return {}

    group_scores_payload = group_value_prior_payload.get("group_scores", {})
    component_scores_payload = group_value_prior_payload.get("component_scores", {})
    source_grouping = str(group_value_prior_payload.get("grouping", DEFAULT_GROUPING))

    if layer_stats is not None and target_grouping and target_grouping != source_grouping:
        source_groups = build_search_groups(layer_stats, grouping=source_grouping)
        source_scores = resolve_group_value_scores(
            source_groups,
            group_value_prior_payload,
        )
        aggregated_scores = aggregate_group_score_overrides(
            layer_stats=layer_stats,
            score_overrides=source_scores,
            target_grouping=target_grouping,
            source_grouping=source_grouping,
        )
        return {
            group.name: float(aggregated_scores.get(group.name, 0.0))
            for group in groups
        }

    resolved_scores: dict[str, float] = {}
    for group in groups:
        group_score_payload = group_scores_payload.get(group.name)
        if group_score_payload is not None:
            resolved_scores[group.name] = float(group_score_payload.get("score", 0.0))
            continue
        component_score_payload = component_scores_payload.get(group.component_type)
        if component_score_payload is not None:
            resolved_scores[group.name] = float(component_score_payload.get("score", 0.0))
            continue
        resolved_scores[group.name] = 0.0
    return resolved_scores


def _resolve_bf16_candidate_names(
    groups: list[SearchGroup],
    group_value_scores: dict[str, float] | None,
    fixed_assignments: dict[str, int],
    bf16_candidate_fraction: float,
    explicit_candidate_names: set[str] | None,
) -> set[str]:
    if explicit_candidate_names is not None:
        return {
            str(group_name)
            for group_name in explicit_candidate_names
            if group_name in {group.name for group in groups} and group_name not in fixed_assignments
        }
    if bf16_candidate_fraction <= 0:
        return set()

    has_group_value_signal = bool(group_value_scores) and any(
        abs(float(score)) > 1e-12 for score in group_value_scores.values()
    )
    eligible_groups = [group for group in groups if group.name not in fixed_assignments]
    if not eligible_groups:
        return set()
    candidate_count = max(1, math.ceil(len(eligible_groups) * bf16_candidate_fraction))
    ranked_groups = sorted(
        eligible_groups,
        key=lambda group: (
            group.sensitivity,
            float(group_value_scores.get(group.name, 0.0)) if has_group_value_signal and group_value_scores else 0.0,
            group.parameter_count,
        ),
        reverse=True,
    )
    return {group.name for group in ranked_groups[:candidate_count]}


def _resolve_allocator_state_values(
    groups: list[SearchGroup],
    group_value_scores: dict[str, float] | None,
    bf16_candidate_names: set[str],
    bf16_rescue_scale: float,
    bf16_sensitivity_weight: float,
) -> tuple[dict[str, dict[int, float]], list[dict[str, Any]]]:
    has_group_value_signal = bool(group_value_scores) and any(
        abs(float(score)) > 1e-12 for score in group_value_scores.values()
    )
    state_values: dict[str, dict[int, float]] = {}
    score_manifest: list[dict[str, Any]] = []
    for group in groups:
        raw_group_value = float(group_value_scores.get(group.name, 0.0)) if group_value_scores else 0.0
        positive_group_value = max(raw_group_value, 0.0)
        base_signal = (
            positive_group_value + 0.10 * group.sensitivity
            if has_group_value_signal
            else group.sensitivity
        )
        rescue_signal = (
            positive_group_value + bf16_sensitivity_weight * group.sensitivity
            if has_group_value_signal
            else group.sensitivity
        )
        value_4_to_8 = group.parameter_count * base_signal
        value_8_to_16 = (
            group.parameter_count
            * bf16_rescue_scale
            * rescue_signal
            * _bf16_component_bonus(group.component_type)
            if group.name in bf16_candidate_names
            else 0.0
        )
        state_values[group.name] = {
            4: 0.0,
            8: value_4_to_8,
            16: value_4_to_8 + value_8_to_16,
        }
        score_manifest.append(
            {
                "name": group.name,
                "component_type": group.component_type,
                "parameter_count": group.parameter_count,
                "sensitivity": float(group.sensitivity),
                "raw_group_value_score": raw_group_value,
                "value_4_to_8": value_4_to_8,
                "value_8_to_16": value_8_to_16,
                "bf16_eligible": group.name in bf16_candidate_names,
            }
        )
    score_manifest.sort(
        key=lambda item: (
            float(item["bf16_eligible"]),
            float(item["value_8_to_16"]),
            float(item["value_4_to_8"]),
        ),
        reverse=True,
    )
    return state_values, score_manifest


def _bf16_component_bonus(component_type: str) -> float:
    if component_type.endswith("mlp.up_proj"):
        return 1.2
    if component_type.endswith("mlp.gate_proj"):
        return 1.15
    if component_type.endswith("self_attn.k_proj"):
        return 1.1
    if component_type.endswith("linear_attn.in_proj_b"):
        return 1.08
    if component_type.endswith("linear_attn.in_proj_a"):
        return 1.05
    if component_type.endswith("mlp.down_proj"):
        return 0.95
    return 1.0


def _solve_budgeted_allocator_for_lambda(
    groups: list[SearchGroup],
    lambda_value: float,
    state_values: dict[str, dict[int, float]],
    bf16_candidate_names: set[str],
    fixed_assignments: dict[str, int],
) -> dict[str, Any]:
    assignments: dict[str, int] = {}
    total_value = 0.0
    total_cost_bytes = 0.0
    for group in groups:
        if group.name in fixed_assignments:
            chosen_bit = int(fixed_assignments[group.name])
        else:
            candidate_bits = [4, 8]
            if group.name in bf16_candidate_names:
                candidate_bits.append(16)
            chosen_bit = max(
                candidate_bits,
                key=lambda bit_width: (
                    float(state_values[group.name][bit_width]) - lambda_value * (group.parameter_count * bit_width / 8),
                    -(group.parameter_count * bit_width / 8),
                ),
            )
        assignments[group.name] = chosen_bit
        total_value += float(state_values[group.name][chosen_bit])
        total_cost_bytes += group.parameter_count * chosen_bit / 8
    return {
        "assignments": assignments,
        "total_value": total_value,
        "total_cost_bytes": total_cost_bytes,
    }


def _repair_budgeted_allocator_assignments(
    groups: list[SearchGroup],
    assignments: dict[str, int],
    state_values: dict[str, dict[int, float]],
    bf16_candidate_names: set[str],
    fixed_assignments: dict[str, int],
    target_budget_bytes: float,
    min_budget_utilization: float,
) -> dict[str, Any]:
    repaired = dict(assignments)
    current_cost_bytes = sum(group.parameter_count * repaired[group.name] / 8 for group in groups)

    while current_cost_bytes > target_budget_bytes + 1e-9:
        downgrade_steps: list[tuple[float, float, str, int]] = []
        for group in groups:
            if group.name in fixed_assignments:
                continue
            current_bit = int(repaired[group.name])
            next_bit = 8 if current_bit == 16 else 4 if current_bit == 8 else None
            if next_bit is None:
                continue
            saved_bytes = group.parameter_count * (current_bit - next_bit) / 8
            value_loss = float(state_values[group.name][current_bit]) - float(state_values[group.name][next_bit])
            downgrade_steps.append((value_loss / saved_bytes, saved_bytes, group.name, next_bit))
        if not downgrade_steps:
            break
        _, saved_bytes, group_name, next_bit = min(downgrade_steps)
        current_bit = repaired[group_name]
        repaired[group_name] = next_bit
        current_cost_bytes -= saved_bytes

    while True:
        current_utilization = 0.0 if target_budget_bytes <= 0 else current_cost_bytes / target_budget_bytes
        if current_utilization >= min_budget_utilization:
            break
        upgrade_steps: list[tuple[float, float, str, int]] = []
        for group in groups:
            if group.name in fixed_assignments:
                continue
            current_bit = int(repaired[group.name])
            next_bit = 8 if current_bit == 4 else 16 if current_bit == 8 and group.name in bf16_candidate_names else None
            if next_bit is None:
                continue
            added_bytes = group.parameter_count * (next_bit - current_bit) / 8
            if current_cost_bytes + added_bytes > target_budget_bytes + 1e-9:
                continue
            value_gain = float(state_values[group.name][next_bit]) - float(state_values[group.name][current_bit])
            upgrade_steps.append((value_gain / added_bytes, added_bytes, group.name, next_bit))
        if not upgrade_steps:
            break
        _, added_bytes, group_name, next_bit = max(upgrade_steps)
        repaired[group_name] = next_bit
        current_cost_bytes += added_bytes

    total_value = sum(float(state_values[group.name][repaired[group.name]]) for group in groups)
    return {
        "assignments": repaired,
        "total_value": total_value,
        "total_cost_bytes": current_cost_bytes,
    }


def estimate_group_value_alignment_score(
    groups: list[SearchGroup],
    assignments: dict[str, int],
    group_value_scores: dict[str, float] | None,
) -> float:
    if not group_value_scores:
        return 0.0
    numerator = 0.0
    denominator = 0.0
    for group in groups:
        value_score = float(group_value_scores.get(group.name, 0.0))
        if abs(value_score) <= 1e-12:
            continue
        weight = group.parameter_count * abs(value_score)
        assigned_bit = assignments[group.name]
        if value_score > 0:
            if assigned_bit >= 16:
                alignment = 1.0
            elif assigned_bit >= 8:
                alignment = 0.75
            else:
                alignment = 0.0
        else:
            if assigned_bit <= 4:
                alignment = 1.0
            elif assigned_bit <= 8:
                alignment = 0.25
            else:
                alignment = 0.0
        numerator += weight * alignment
        denominator += weight
    if denominator == 0:
        return 0.0
    return numerator / denominator


def estimate_reference_advantage_score(
    conservative_prediction: float,
    reference_accuracy: float | None,
) -> float:
    if reference_accuracy is None:
        return 0.0
    gap = conservative_prediction - reference_accuracy
    if gap >= 0:
        return gap
    return 0.5 * gap


def resolve_reference_target_value(
    surrogate_summary_payload: dict[str, Any],
    reference_accuracy: float | None,
) -> float | None:
    if reference_accuracy is None:
        return None
    target_metric = str(surrogate_summary_payload.get("target_metric", "accuracy"))
    if "advantage_over" in target_metric:
        return 0.0
    return reference_accuracy


def estimate_low_bit_bonus(
    groups: list[SearchGroup],
    assignments: dict[str, int],
) -> float:
    numerator = 0.0
    denominator = 0.0
    for group in groups:
        bit_width = assignments[group.name]
        low_bit_indicator = 1.0 if bit_width <= 3 else 0.0
        weight = group.parameter_count * max(1.1 - group.sensitivity, 0.1)
        numerator += weight * low_bit_indicator
        denominator += weight
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _group_key_for_layer(layer_name: str, grouping: str) -> tuple[str, str]:
    match = re.match(r"model\.layers\.(\d+)\.(.+)", layer_name)
    if match is None:
        if grouping == "per_module":
            return layer_name, layer_name
        return f"global::{layer_name}", layer_name

    block_number = int(match.group(1))
    block_index = str(block_number)
    component_type = match.group(2)

    if grouping == "per_module":
        return layer_name, component_type
    if grouping == "per_block_component":
        return f"block:{block_index}:{component_type}", component_type
    if grouping == "per_block_window_component":
        window_start = (block_number // BLOCK_WINDOW_SIZE) * BLOCK_WINDOW_SIZE
        window_end = window_start + BLOCK_WINDOW_SIZE - 1
        return f"window:{window_start}-{window_end}:{component_type}", component_type
    if grouping == "per_component_family":
        return f"component:{component_type}", component_type
    if grouping == "per_block":
        return f"block:{block_index}", f"block_{block_index}"
    raise ValueError(f"Unknown grouping strategy: {grouping}")


def _sensitivity_prior(component_type: str) -> float:
    if component_type.endswith("mlp.down_proj"):
        return 1.0
    if component_type.endswith("linear_attn.out_proj"):
        return 0.95
    if component_type.endswith("linear_attn.in_proj_qkv"):
        return 0.9
    if component_type.endswith("mlp.up_proj"):
        return 0.85
    if component_type.endswith("mlp.gate_proj"):
        return 0.8
    if component_type.endswith("linear_attn.in_proj_z"):
        return 0.75
    if component_type.endswith("linear_attn.in_proj_a"):
        return 0.65
    if component_type.endswith("linear_attn.in_proj_b"):
        return 0.6
    return 0.5
