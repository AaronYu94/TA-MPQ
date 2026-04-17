from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import re
import statistics
from typing import Any, Callable

from ta_mpq.feasibility import LinearLayerStat
from ta_mpq.sensitivity import group_sensitivity_overrides_from_profile


DEFAULT_GROUPING = "per_block_component"
EVOLUTION_SEARCH_SUPPORTED_BITS = (4, 8, 16)
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
    prediction_uncertainty: float | None = None
    conservative_prediction: float | None = None
    budget_alignment_score: float | None = None
    group_value_alignment_score: float | None = None
    reference_accuracy: float | None = None
    reference_advantage_score: float | None = None

    def bits_dict(self) -> dict[str, int]:
        return dict(self.group_bits)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["group_bits"] = dict(self.group_bits)
        return payload


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
        sensitivity_overrides = group_sensitivity_overrides_from_profile(
            sensitivity_profile,
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
        sensitivity_overrides = group_sensitivity_overrides_from_profile(
            sensitivity_profile,
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
        group_value_scores = resolve_group_value_scores(groups, group_value_prior_payload)
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
) -> SearchRunResult:
    normalized_bits = _normalize_allowed_bits(allowed_bits)
    rng = random.Random(seed)
    return _run_evolution_search(
        groups=groups,
        target_budget_gb=target_budget_gb,
        allowed_bits=normalized_bits,
        grouping=grouping,
        population_size=population_size,
        generations=generations,
        elite_count=elite_count,
        tournament_size=tournament_size,
        mutation_rate=mutation_rate,
        top_k=top_k,
        rng=rng,
        candidate_builder=lambda assignments, provenance: build_candidate(
            groups=groups,
            assignments=assignments,
            target_budget_gb=target_budget_gb,
            allowed_bits=normalized_bits,
            provenance=provenance,
        ),
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
    extra_seed_assignments: list[tuple[str, dict[str, int]]] = []
    value_guided_seed = value_guided_seed_assignments(
        groups=groups,
        allowed_bits=normalized_bits,
        group_value_scores=group_value_scores,
    )
    if value_guided_seed is not None:
        extra_seed_assignments.append(("value_guided_seed", value_guided_seed))
    return _run_evolution_search(
        groups=groups,
        target_budget_gb=target_budget_gb,
        allowed_bits=normalized_bits,
        grouping=grouping,
        population_size=population_size,
        generations=generations,
        elite_count=elite_count,
        tournament_size=tournament_size,
        mutation_rate=mutation_rate,
        top_k=top_k,
        rng=rng,
        extra_seed_assignments=extra_seed_assignments,
        candidate_builder=lambda assignments, provenance: build_surrogate_candidate(
            groups=groups,
            assignments=assignments,
            surrogate_predictor=surrogate_predictor,
            target_budget_gb=target_budget_gb,
            allowed_bits=normalized_bits,
            provenance=provenance,
            uncertainty_penalty=uncertainty_penalty,
            group_value_scores=group_value_scores,
            reference_accuracy=reference_target_value,
        ),
    )


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
) -> dict[str, float]:
    if not group_value_prior_payload:
        return {}
    group_scores_payload = group_value_prior_payload.get("group_scores", {})
    component_scores_payload = group_value_prior_payload.get("component_scores", {})
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

    block_index = match.group(1)
    component_type = match.group(2)

    if grouping == "per_module":
        return layer_name, component_type
    if grouping == "per_block_component":
        return f"block:{block_index}:{component_type}", component_type
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
