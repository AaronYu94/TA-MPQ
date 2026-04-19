from __future__ import annotations

from dataclasses import dataclass, replace
import random
from typing import Any

from ta_mpq.quant_search.budget import PolicyBudgetStats, compute_policy_budget_stats, policy_bits, target_int4_bits
from ta_mpq.quant_search.group_registry import GroupInfo, group_lookup, total_param_count
from ta_mpq.quant_search.policy_hash import canonical_assignment_hash, canonical_registry_hash


ALLOWED_POLICY_BITS = (2, 4, 8)
METHOD_NAME = "task_sensitivity_exact_budget"


@dataclass(frozen=True, slots=True)
class BuiltPolicy:
    policy_id: str
    method: str
    policy_hash: str
    bitwidths: tuple[tuple[str, int], ...]
    stats: PolicyBudgetStats
    proxy_score: float
    source: dict[str, Any]

    def bitwidth_dict(self) -> dict[str, int]:
        return dict(self.bitwidths)

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "method": self.method,
            "policy_hash": self.policy_hash,
            "bitwidths": dict(self.bitwidths),
            "stats": self.stats.to_dict(),
            "proxy_score": self.proxy_score,
            "source": self.source,
        }


def build_uniform_int4_policy(
    groups: list[GroupInfo],
    policy_id: str = "taqeb_uniform_int4",
) -> BuiltPolicy:
    bitwidths = {group.group_id: 4 for group in groups}
    return _build_policy(
        policy_id=policy_id,
        bitwidths=bitwidths,
        groups=groups,
        scores={},
        source={
            "builder": "uniform_int4_baseline",
            "k_or_mass_target": 0.0,
        },
    )


def relabel_policy(
    policy: BuiltPolicy,
    *,
    policy_id: str,
    builder: str,
    source_updates: dict[str, Any] | None = None,
) -> BuiltPolicy:
    updated_source = dict(policy.source)
    updated_source["builder"] = str(builder)
    if source_updates:
        updated_source.update(source_updates)
    return replace(policy, policy_id=str(policy_id), source=updated_source)


def build_policy_from_bitwidths(
    groups: list[GroupInfo],
    scores: dict[str, dict[str, Any]],
    bitwidths: dict[str, int],
    *,
    policy_id: str,
    source: dict[str, Any],
    method_name: str = METHOD_NAME,
    path_index: int | None = None,
) -> BuiltPolicy:
    return _build_policy(
        policy_id=policy_id,
        bitwidths=bitwidths,
        groups=groups,
        source=source,
        scores=scores,
        method_name=method_name,
        path_index=path_index,
    )


def built_policy_from_payload(
    payload: dict[str, Any],
    groups: list[GroupInfo],
    *,
    method_name: str | None = None,
) -> BuiltPolicy:
    normalized_bitwidths = {
        str(group_id): int(bitwidth)
        for group_id, bitwidth in sorted(dict(payload.get("bitwidths", {})).items())
    }
    raw_stats = dict(payload.get("stats", {}))
    path_index = raw_stats.get("path_index")
    stats = compute_policy_budget_stats(
        normalized_bitwidths,
        groups,
        actual_bytes=raw_stats.get("actual_bytes"),
        path_index=int(path_index) if path_index is not None else None,
    )
    return BuiltPolicy(
        policy_id=str(payload["policy_id"]),
        method=str(method_name or payload.get("method") or METHOD_NAME),
        policy_hash=str(
            payload.get("policy_hash")
            or canonical_assignment_hash(
                normalized_bitwidths,
                grouping="per_block_component",
                budget_rule="raw_weight_int4",
                registry_hash=canonical_registry_hash(groups),
            )
        ),
        bitwidths=tuple(sorted(normalized_bitwidths.items())),
        stats=stats,
        proxy_score=float(payload.get("proxy_score", payload.get("proxy_quality_score", 0.0))),
        source=dict(payload.get("source", {})),
    )


def build_equal_count_threshold_policy(
    groups: list[GroupInfo],
    scores: dict[str, dict[str, Any]],
    k: int,
    policy_id: str,
    source: dict[str, Any] | None = None,
) -> BuiltPolicy:
    if k < 0:
        raise ValueError("k must be non-negative")
    assignments = {group.group_id: 4 for group in groups}
    ranked_desc = sorted(
        groups,
        key=lambda group: (
            -_score_value(group.group_id, scores, field="score"),
            group.group_id,
        ),
    )
    ranked_asc = sorted(
        groups,
        key=lambda group: (
            _score_value(group.group_id, scores, field="score"),
            group.group_id,
        ),
    )
    promoted_ids = {group.group_id for group in ranked_desc[:k]}
    for group_id in promoted_ids:
        assignments[group_id] = 8
    demoted = 0
    for group in ranked_asc:
        if group.group_id in promoted_ids:
            continue
        if demoted >= 2 * k:
            break
        assignments[group.group_id] = 2
        demoted += 1
    return _build_policy(
        policy_id=policy_id,
        bitwidths=assignments,
        groups=groups,
        scores=scores,
        source=source
        or {
            "builder": "equal_count_threshold",
            "k_or_mass_target": k,
        },
    )


def build_size_weighted_threshold_policy(
    groups: list[GroupInfo],
    scores: dict[str, dict[str, Any]],
    promotion_mass_fraction: float,
    policy_id: str,
    source: dict[str, Any] | None = None,
    fill_remaining_slack: bool = True,
) -> BuiltPolicy:
    if not 0.0 <= promotion_mass_fraction <= 1.0:
        raise ValueError("promotion_mass_fraction must be between 0 and 1")
    assignments = {group.group_id: 4 for group in groups}
    total_params = total_param_count(groups)
    target_promotion_mass = promotion_mass_fraction * total_params
    promoted_param_count = 0

    ranked_promotions = sorted(
        groups,
        key=lambda group: (
            -_benefit_density(group, scores),
            group.group_id,
        ),
    )
    for group in ranked_promotions:
        if target_promotion_mass <= 0:
            break
        if promoted_param_count >= target_promotion_mass:
            break
        assignments[group.group_id] = 8
        promoted_param_count += group.param_count

    repaired = repair_to_budget(
        bitwidths=assignments,
        groups=groups,
        scores=scores,
        fill_remaining_slack=fill_remaining_slack,
    )
    return _build_policy(
        policy_id=policy_id,
        bitwidths=repaired,
        groups=groups,
        scores=scores,
        source=source
        or {
            "builder": "size_weighted_threshold",
            "k_or_mass_target": promotion_mass_fraction,
        },
    )


def build_random_exact_budget_policy(
    groups: list[GroupInfo],
    scores: dict[str, dict[str, Any]],
    promotion_mass_fraction: float,
    seed: int,
    policy_id: str,
    source: dict[str, Any] | None = None,
) -> BuiltPolicy:
    rng = random.Random(seed)
    assignments = {group.group_id: 4 for group in groups}
    shuffled_groups = list(groups)
    rng.shuffle(shuffled_groups)

    target_promotion_mass = promotion_mass_fraction * total_param_count(groups)
    promoted_mass = 0
    promoted_ids: set[str] = set()
    for group in shuffled_groups:
        if target_promotion_mass <= 0 or promoted_mass >= target_promotion_mass:
            break
        assignments[group.group_id] = 8
        promoted_ids.add(group.group_id)
        promoted_mass += group.param_count

    demotion_pool = [group for group in shuffled_groups if group.group_id not in promoted_ids]
    rng.shuffle(demotion_pool)
    target_bits = target_int4_bits(groups)
    for group in demotion_pool:
        if policy_bits(assignments, groups) <= target_bits:
            break
        assignments[group.group_id] = 2

    repaired = repair_to_budget(bitwidths=assignments, groups=groups, scores=scores)
    return _build_policy(
        policy_id=policy_id,
        bitwidths=repaired,
        groups=groups,
        scores=scores,
        source=source
        or {
            "builder": "random_exact_budget_baseline",
            "k_or_mass_target": promotion_mass_fraction,
            "seed": seed,
        },
    )


def build_inverse_sensitivity_policy(
    groups: list[GroupInfo],
    scores: dict[str, dict[str, Any]],
    promotion_mass_fraction: float,
    policy_id: str,
    source: dict[str, Any] | None = None,
) -> BuiltPolicy:
    assignments = {group.group_id: 4 for group in groups}
    total_params = total_param_count(groups)
    target_promotion_mass = promotion_mass_fraction * total_params

    ranked_promotions = sorted(
        groups,
        key=lambda group: (
            _benefit_density(group, scores),
            group.group_id,
        ),
    )
    promoted_mass = 0
    promoted_ids: set[str] = set()
    for group in ranked_promotions:
        if target_promotion_mass <= 0 or promoted_mass >= target_promotion_mass:
            break
        assignments[group.group_id] = 8
        promoted_ids.add(group.group_id)
        promoted_mass += group.param_count

    target_bits = target_int4_bits(groups)
    ranked_demotions = sorted(
        [group for group in groups if group.group_id not in promoted_ids],
        key=lambda group: (
            _demotion_loss_density(group, scores),
            group.group_id,
        ),
        reverse=True,
    )
    for group in ranked_demotions:
        if policy_bits(assignments, groups) <= target_bits:
            break
        assignments[group.group_id] = 2

    return _build_policy(
        policy_id=policy_id,
        bitwidths=assignments,
        groups=groups,
        scores=scores,
        source=source
        or {
            "builder": "inverse_sensitivity_baseline",
            "k_or_mass_target": promotion_mass_fraction,
        },
    )


def repair_to_budget(
    bitwidths: dict[str, int],
    groups: list[GroupInfo],
    scores: dict[str, dict[str, Any]],
    target_bits: int | None = None,
    fill_remaining_slack: bool = True,
) -> dict[str, int]:
    lookup = group_lookup(groups)
    repaired = {
        group.group_id: int(bitwidths.get(group.group_id, 4))
        for group in groups
    }
    unsupported = sorted(
        {
            int(bitwidth)
            for bitwidth in repaired.values()
            if int(bitwidth) not in ALLOWED_POLICY_BITS
        }
    )
    if unsupported:
        raise ValueError(f"Unsupported policy bit-widths: {unsupported}")

    resolved_target_bits = int(target_bits) if target_bits is not None else target_int4_bits(groups)
    while policy_bits(repaired, groups) > resolved_target_bits:
        candidates: list[tuple[float, str, int]] = []
        for group in groups:
            current_bit = repaired[group.group_id]
            if current_bit == 8:
                saved_bits = 4 * lookup[group.group_id].param_count
                loss = _benefit_value(group.group_id, scores)
                candidates.append((loss / max(saved_bits, 1), group.group_id, 4))
            elif current_bit == 4:
                saved_bits = 2 * lookup[group.group_id].param_count
                loss = _demotion_cost_value(group.group_id, scores)
                candidates.append((loss / max(saved_bits, 1), group.group_id, 2))
        if not candidates:
            break
        _, group_id, next_bit = min(candidates, key=lambda item: (item[0], item[1], item[2]))
        repaired[group_id] = next_bit

    if not fill_remaining_slack:
        return repaired

    while True:
        current_bits = policy_bits(repaired, groups)
        remaining_slack = resolved_target_bits - current_bits
        if remaining_slack <= 0:
            break
        upgrades: list[tuple[float, float, str, int]] = []
        for group in groups:
            current_bit = repaired[group.group_id]
            if current_bit == 2:
                cost = 2 * lookup[group.group_id].param_count
                value = _demotion_cost_value(group.group_id, scores)
                next_bit = 4
            elif current_bit == 4:
                cost = 4 * lookup[group.group_id].param_count
                value = _benefit_value(group.group_id, scores)
                next_bit = 8
            else:
                continue
            if cost > remaining_slack:
                continue
            upgrades.append((value / max(cost, 1), value, group.group_id, next_bit))
        if not upgrades:
            break
        _, _, group_id, next_bit = sorted(
            upgrades,
            key=lambda item: (
                -item[0],
                -item[1],
                item[2],
                item[3],
            ),
        )[0]
        repaired[group_id] = next_bit
    return repaired


def _build_policy(
    policy_id: str,
    bitwidths: dict[str, int],
    groups: list[GroupInfo],
    source: dict[str, Any],
    scores: dict[str, dict[str, Any]] | None = None,
    method_name: str = METHOD_NAME,
    path_index: int | None = None,
) -> BuiltPolicy:
    normalized = {
        group.group_id: int(bitwidths[group.group_id])
        for group in groups
    }
    stats = compute_policy_budget_stats(normalized, groups, path_index=path_index)
    policy_hash = canonical_assignment_hash(
        normalized,
        grouping="per_block_component",
        budget_rule="raw_weight_int4",
        registry_hash=canonical_registry_hash(groups),
    )
    return BuiltPolicy(
        policy_id=policy_id,
        method=method_name,
        policy_hash=policy_hash,
        bitwidths=tuple(sorted(normalized.items())),
        stats=stats,
        proxy_score=estimate_policy_proxy_score(normalized, scores or {}),
        source=source,
    )


def estimate_policy_proxy_score(
    bitwidths: dict[str, int],
    scores: dict[str, dict[str, Any]],
) -> float:
    proxy = 0.0
    for group_id, bitwidth in sorted(bitwidths.items()):
        if int(bitwidth) == 8:
            proxy += _benefit_value(group_id, scores)
        elif int(bitwidth) == 2:
            proxy -= _demotion_cost_value(group_id, scores)
    return float(proxy)


def fraction_policy_id(
    prefix: str,
    promotion_mass_fraction: float,
    suffix: str,
) -> str:
    milli = int(round(float(promotion_mass_fraction) * 1000.0))
    return f"{prefix}_m{milli:03d}_{suffix}"


def _benefit_density(group: GroupInfo, scores: dict[str, dict[str, Any]]) -> float:
    return _benefit_value(group.group_id, scores) / max(4 * group.param_count, 1)


def _demotion_loss_density(group: GroupInfo, scores: dict[str, dict[str, Any]]) -> float:
    return _demotion_cost_value(group.group_id, scores) / max(2 * group.param_count, 1)


def _benefit_value(group_id: str, scores: dict[str, dict[str, Any]]) -> float:
    payload = dict(scores.get(group_id, {}))
    if payload.get("benefit_8_over_4") is not None:
        return float(payload["benefit_8_over_4"])
    return _score_value(group_id, scores, field="score")


def _demotion_cost_value(group_id: str, scores: dict[str, dict[str, Any]]) -> float:
    payload = dict(scores.get(group_id, {}))
    if payload.get("demotion_cost_4_to_2") is not None:
        return float(payload["demotion_cost_4_to_2"])
    return _score_value(group_id, scores, field="score")


def _score_value(
    group_id: str,
    scores: dict[str, dict[str, Any]],
    field: str,
) -> float:
    payload = dict(scores.get(group_id, {}))
    value = payload.get(field)
    return float(value) if value is not None else 0.0
