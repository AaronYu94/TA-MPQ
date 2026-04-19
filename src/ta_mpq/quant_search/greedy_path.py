from __future__ import annotations

from dataclasses import replace
from typing import Any

from ta_mpq.quant_search.budget import policy_bits, target_int4_bits
from ta_mpq.quant_search.group_registry import GroupInfo
from ta_mpq.quant_search.policy_builder import BuiltPolicy, build_policy_from_bitwidths


METHOD_NAME = "greedy_path_frontier"


def build_greedy_max8_path(
    groups: list[GroupInfo],
    scores: dict[str, dict[str, Any]],
    *,
    target_bits: int | None = None,
    min_bitwidth_by_group: dict[str, int] | None = None,
    max_bitwidth_by_group: dict[str, int] | None = None,
    demotion_candidate_pool_size: int = 64,
    demotion_beam_width: int = 16,
    overshoot_penalty: float = 10.0,
    allow_under_budget: bool = True,
) -> list[BuiltPolicy]:
    resolved_target_bits = int(target_bits) if target_bits is not None else target_int4_bits(groups)
    min_bits = {str(group_id): int(bitwidth) for group_id, bitwidth in (min_bitwidth_by_group or {}).items()}
    max_bits = {str(group_id): int(bitwidth) for group_id, bitwidth in (max_bitwidth_by_group or {}).items()}

    policy = {group.group_id: 4 for group in groups if group.is_quantizable}
    path = [
        build_policy_from_bitwidths(
            groups=groups,
            scores=scores,
            bitwidths=policy,
            policy_id="uniform_int4_anchor",
            method_name=METHOD_NAME,
            path_index=0,
            source={
                "method_name": METHOD_NAME,
                "builder": "greedy_path_state",
                "path_builder": "build_greedy_max8_path",
                "endpoint_kind": "uniform_int4_anchor",
                "profile_kind": _profile_kind(scores),
                "promoted_group": None,
                "demoted_groups": [],
                "constraints_applied": {
                    "min_bitwidth_by_group": min_bits,
                    "max_bitwidth_by_group": max_bits,
                },
            },
        )
    ]

    promotion_candidates = sorted(
        [group for group in groups if group.is_quantizable],
        key=lambda group: (
            _promotion_priority(group, scores),
            _benefit_8_over_4(group.group_id, scores),
            -group.param_count,
            group.group_id,
        ),
        reverse=True,
    )

    while True:
        accepted_any = False
        for promoted_group in promotion_candidates:
            group_id = promoted_group.group_id
            if policy[group_id] != 4:
                continue
            if max_bits.get(group_id, 8) < 8:
                continue

            trial = dict(policy)
            trial[group_id] = 8
            deficit_bits = policy_bits(trial, groups) - resolved_target_bits
            demoted_groups: list[str] = []

            if deficit_bits > 0:
                chosen_demotions = choose_demotions_for_deficit(
                    trial,
                    groups,
                    scores,
                    required_savings_bits=deficit_bits,
                    exclude_groups={group_id},
                    min_bitwidth_by_group=min_bits,
                    candidate_pool_size=demotion_candidate_pool_size,
                    beam_width=demotion_beam_width,
                    overshoot_penalty=overshoot_penalty,
                )
                if chosen_demotions is None:
                    continue
                for demoted_group in chosen_demotions:
                    trial[demoted_group] = 2
                demoted_groups = list(chosen_demotions)

            if not _is_feasible(trial, groups, resolved_target_bits, allow_under_budget):
                continue

            policy = trial
            accepted_any = True
            path.append(
                build_policy_from_bitwidths(
                    groups=groups,
                    scores=scores,
                    bitwidths=policy,
                    policy_id=f"path_{len(path):04d}",
                    method_name=METHOD_NAME,
                    path_index=len(path),
                    source={
                        "method_name": METHOD_NAME,
                        "builder": "greedy_path_state",
                        "path_builder": "build_greedy_max8_path",
                        "endpoint_kind": None,
                        "profile_kind": _profile_kind(scores),
                        "promoted_group": group_id,
                        "demoted_groups": demoted_groups,
                        "constraints_applied": {
                            "min_bitwidth_by_group": min_bits,
                            "max_bitwidth_by_group": max_bits,
                        },
                    },
                )
            )

        if not accepted_any:
            break

    if path:
        path[-1] = replace(
            path[-1],
            source={
                **path[-1].source,
                "endpoint_kind": "greedy_max8_endpoint",
            },
        )
    return dedupe_policies(path)


def choose_demotions_for_deficit(
    trial_policy: dict[str, int],
    groups: list[GroupInfo],
    scores: dict[str, dict[str, Any]],
    *,
    required_savings_bits: int,
    exclude_groups: set[str],
    min_bitwidth_by_group: dict[str, int] | None,
    candidate_pool_size: int,
    beam_width: int,
    overshoot_penalty: float,
) -> list[str] | None:
    candidates: list[dict[str, Any]] = []
    min_bits = min_bitwidth_by_group or {}
    for group in groups:
        group_id = group.group_id
        if group_id in exclude_groups:
            continue
        if trial_policy.get(group_id) != 4:
            continue
        if int(min_bits.get(group_id, 2)) > 2:
            continue
        saved_bits = 2 * group.param_count
        demotion_cost = _demotion_cost_4_to_2(group_id, scores)
        candidates.append(
            {
                "group_id": group_id,
                "saved_bits": saved_bits,
                "demotion_cost": demotion_cost,
                "score_per_saved_bit": demotion_cost / max(1, saved_bits),
                "param_count": group.param_count,
            }
        )

    candidates.sort(
        key=lambda item: (
            item["score_per_saved_bit"],
            item["demotion_cost"],
            item["param_count"],
            item["group_id"],
        )
    )
    return local_beam_subset_cover(
        candidates[:candidate_pool_size],
        required_savings_bits=required_savings_bits,
        beam_width=beam_width,
        overshoot_penalty=overshoot_penalty,
    )


def local_beam_subset_cover(
    candidates: list[dict[str, Any]],
    *,
    required_savings_bits: int,
    beam_width: int,
    overshoot_penalty: float,
) -> list[str] | None:
    beam: list[tuple[int, float, list[str]]] = [(0, 0.0, [])]
    for candidate in candidates:
        next_beam = list(beam)
        for saved_bits, demotion_cost, selected_group_ids in beam:
            next_beam.append(
                (
                    saved_bits + int(candidate["saved_bits"]),
                    demotion_cost + float(candidate["demotion_cost"]),
                    selected_group_ids + [str(candidate["group_id"])],
                )
            )

        next_beam.sort(
            key=lambda state: _beam_state_key(
                state,
                required_savings_bits=required_savings_bits,
                overshoot_penalty=overshoot_penalty,
            )
        )
        beam = next_beam[: max(1, int(beam_width))]

    valid_states = [state for state in beam if state[0] >= required_savings_bits]
    if not valid_states:
        return None
    valid_states.sort(
        key=lambda state: (
            state[1] + (
                overshoot_penalty
                * max(0, state[0] - required_savings_bits)
                / max(1, required_savings_bits)
            ),
            max(0, state[0] - required_savings_bits),
            len(state[2]),
            tuple(state[2]),
        )
    )
    return list(valid_states[0][2])


def select_coarse_from_greedy_path(
    path: list[BuiltPolicy],
    *,
    count: int = 8,
    exclude_anchor: bool = True,
    exclude_endpoint: bool = True,
) -> list[BuiltPolicy]:
    unique_path = dedupe_policies(path)
    candidates = list(unique_path)
    if exclude_anchor:
        candidates = [
            policy
            for policy in candidates
            if policy.source.get("endpoint_kind") != "uniform_int4_anchor"
        ]
    if exclude_endpoint:
        candidates = [
            policy
            for policy in candidates
            if policy.source.get("endpoint_kind") != "greedy_max8_endpoint"
        ]
    if len(candidates) <= count:
        return sorted(candidates, key=_path_index)

    x_values = [policy.stats.realized_8bit_param_mass_fraction for policy in candidates]
    x_min = min(x_values)
    x_max = max(x_values)
    targets = [
        x_min + ((index + 0.5) / max(1, count)) * (x_max - x_min)
        for index in range(count)
    ]

    selected: list[BuiltPolicy] = []
    used_hashes: set[str] = set()
    for target in targets:
        unused = [policy for policy in candidates if policy.policy_hash not in used_hashes]
        if not unused:
            break
        chosen = min(
            unused,
            key=lambda policy: (
                abs(policy.stats.realized_8bit_param_mass_fraction - target),
                policy.stats.realized_2bit_param_mass_fraction,
                _path_index(policy),
                policy.policy_hash,
            ),
        )
        selected.append(chosen)
        used_hashes.add(chosen.policy_hash)
    return sorted(selected, key=_path_index)


def select_refine_from_greedy_path(
    path: list[BuiltPolicy],
    *,
    seed_path_index: int,
    count: int = 4,
    radius: int | None = None,
) -> list[BuiltPolicy]:
    unique_path = sorted(dedupe_policies(path), key=lambda policy: _path_index(policy))
    if not unique_path:
        return []
    position_by_index = {
        _path_index(policy): position
        for position, policy in enumerate(unique_path)
    }
    seed_position = position_by_index[int(seed_path_index)]
    resolved_radius = int(radius) if radius is not None else max(4, len(unique_path) // 16)
    left = max(0, seed_position - resolved_radius)
    right = min(len(unique_path) - 1, seed_position + resolved_radius)
    local = unique_path[left : right + 1]
    if len(local) <= count:
        return local

    seed_policy = unique_path[seed_position]
    selected: list[BuiltPolicy] = [seed_policy]
    used_hashes = {seed_policy.policy_hash}

    x_min = local[0].stats.realized_8bit_param_mass_fraction
    x_max = local[-1].stats.realized_8bit_param_mass_fraction
    targets = [
        x_min + ((index + 0.5) / max(1, count)) * (x_max - x_min)
        for index in range(count)
    ]

    for target in targets:
        if len(selected) >= count:
            break
        unused = [policy for policy in local if policy.policy_hash not in used_hashes]
        if not unused:
            break
        chosen = min(
            unused,
            key=lambda policy: (
                abs(policy.stats.realized_8bit_param_mass_fraction - target),
                abs(_path_index(policy) - int(seed_path_index)),
                policy.stats.realized_2bit_param_mass_fraction,
                policy.policy_hash,
            ),
        )
        selected.append(chosen)
        used_hashes.add(chosen.policy_hash)
    return sorted(selected, key=_path_index)


def select_refine_candidates_from_coarse(
    path: list[BuiltPolicy],
    coarse_rows: list[dict[str, Any]],
    *,
    count: int = 4,
    tie_band_correct_answers: int = 1,
    tie_break: list[str] | tuple[str, ...] = (),
    allow_top2_for_candidate_generation: bool = True,
    second_seed_max_gap_correct_answers: int = 1,
    second_seed_min_realized_fraction_distance: float = 0.04,
) -> list[BuiltPolicy]:
    from ta_mpq.quant_search.frontier_search import select_top_rows

    if not coarse_rows:
        return []
    selected_rows = select_top_rows(
        coarse_rows,
        top_k=2 if allow_top2_for_candidate_generation else 1,
        tie_band_correct_answers=tie_band_correct_answers,
        tie_break=tie_break,
    )
    unique_path = {policy.policy_hash: policy for policy in dedupe_policies(path)}
    seed_rows = [selected_rows[0]]
    if len(selected_rows) > 1:
        candidate_row = selected_rows[1]
        gap = int(selected_rows[0].get("correct", 0)) - int(candidate_row.get("correct", 0))
        distance = abs(
            float(selected_rows[0].get("realized_8bit_param_mass_fraction", selected_rows[0].get("promotion_mass_fraction", 0.0)))
            - float(candidate_row.get("realized_8bit_param_mass_fraction", candidate_row.get("promotion_mass_fraction", 0.0)))
        )
        if (
            gap <= int(second_seed_max_gap_correct_answers)
            and distance >= float(second_seed_min_realized_fraction_distance)
            and str(candidate_row.get("policy_hash")) != str(selected_rows[0].get("policy_hash"))
        ):
            seed_rows.append(candidate_row)

    candidate_pool: list[BuiltPolicy] = []
    seed_path_indices: set[int] = set()
    for seed_row in seed_rows:
        seed_policy = unique_path.get(str(seed_row["policy_hash"]))
        if seed_policy is None:
            continue
        seed_path_indices.add(_path_index(seed_policy))
        candidate_pool.extend(
            select_refine_from_greedy_path(
                path,
                seed_path_index=_path_index(seed_policy),
                count=max(1, count),
            )
        )
    deduped_pool = sorted(dedupe_policies(candidate_pool), key=_path_index)
    return _select_evenly_spaced(
        deduped_pool,
        count=count,
        include_seed_indices=seed_path_indices,
    )


def dedupe_policies(policies: list[BuiltPolicy]) -> list[BuiltPolicy]:
    deduped: list[BuiltPolicy] = []
    seen: set[str] = set()
    for policy in policies:
        if policy.policy_hash in seen:
            continue
        deduped.append(policy)
        seen.add(policy.policy_hash)
    return deduped


def _select_evenly_spaced(
    policies: list[BuiltPolicy],
    *,
    count: int,
    include_seed_indices: set[int] | None = None,
) -> list[BuiltPolicy]:
    ordered = sorted(dedupe_policies(policies), key=_path_index)
    if len(ordered) <= count:
        return ordered

    selected: list[BuiltPolicy] = []
    used_hashes: set[str] = set()
    seed_indices = set(include_seed_indices or set())
    for policy in ordered:
        if _path_index(policy) in seed_indices and policy.policy_hash not in used_hashes:
            selected.append(policy)
            used_hashes.add(policy.policy_hash)
    if len(selected) >= count:
        return sorted(selected[:count], key=_path_index)

    x_min = ordered[0].stats.realized_8bit_param_mass_fraction
    x_max = ordered[-1].stats.realized_8bit_param_mass_fraction
    targets = [
        x_min + ((index + 0.5) / max(1, count)) * (x_max - x_min)
        for index in range(count)
    ]
    for target in targets:
        if len(selected) >= count:
            break
        unused = [policy for policy in ordered if policy.policy_hash not in used_hashes]
        if not unused:
            break
        chosen = min(
            unused,
            key=lambda policy: (
                abs(policy.stats.realized_8bit_param_mass_fraction - target),
                policy.stats.realized_2bit_param_mass_fraction,
                policy.stats.budget_slack_fraction,
                _path_index(policy),
                policy.policy_hash,
            ),
        )
        selected.append(chosen)
        used_hashes.add(chosen.policy_hash)
    return sorted(selected, key=_path_index)


def _promotion_priority(group: GroupInfo, scores: dict[str, dict[str, Any]]) -> float:
    return _benefit_8_over_4(group.group_id, scores) / max(1, 4 * group.param_count)


def _benefit_8_over_4(group_id: str, scores: dict[str, dict[str, Any]]) -> float:
    payload = dict(scores.get(group_id, {}))
    if payload.get("benefit_8_over_4") is not None:
        return float(payload["benefit_8_over_4"])
    return float(payload.get("score") or 0.0)


def _demotion_cost_4_to_2(group_id: str, scores: dict[str, dict[str, Any]]) -> float:
    payload = dict(scores.get(group_id, {}))
    if payload.get("demotion_cost_4_to_2") is not None:
        return float(payload["demotion_cost_4_to_2"])
    return float(payload.get("score") or 0.0)


def _is_feasible(
    bitwidths: dict[str, int],
    groups: list[GroupInfo],
    target_bits: int,
    allow_under_budget: bool,
) -> bool:
    current_bits = policy_bits(bitwidths, groups)
    if allow_under_budget:
        return current_bits <= target_bits
    return current_bits == target_bits


def _beam_state_key(
    state: tuple[int, float, list[str]],
    *,
    required_savings_bits: int,
    overshoot_penalty: float,
) -> tuple[Any, ...]:
    saved_bits, demotion_cost, selected_group_ids = state
    shortfall = max(0, required_savings_bits - saved_bits)
    overshoot = max(0, saved_bits - required_savings_bits)
    return (
        shortfall > 0,
        demotion_cost + (overshoot_penalty * overshoot / max(1, required_savings_bits)),
        shortfall,
        overshoot,
        len(selected_group_ids),
        tuple(selected_group_ids),
    )


def _profile_kind(scores: dict[str, dict[str, Any]]) -> str:
    if not scores:
        return "unknown"
    sample_payload = next(iter(scores.values()))
    if sample_payload.get("risk_2") is not None and sample_payload.get("benefit_8_over_4") is not None:
        return "taq_kl_groupwise"
    return "bootstrap_legacy_scalar"


def _path_index(policy: BuiltPolicy) -> int:
    raw_index = policy.stats.path_index
    return int(raw_index) if raw_index is not None else 0
