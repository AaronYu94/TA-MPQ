#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ta_mpq.quant_search.budget import compute_policy_budget_stats
from ta_mpq.quant_search.config import load_config, parse_float_list
from ta_mpq.quant_search.frontier_search import write_duplicate_policy_report
from ta_mpq.quant_search.greedy_path import (
    METHOD_NAME as GREEDY_PATH_METHOD_NAME,
    build_greedy_max8_path,
    select_coarse_from_greedy_path,
)
from ta_mpq.quant_search.group_registry import load_group_registry
from ta_mpq.quant_search.policy_builder import (
    fraction_policy_id,
    build_size_weighted_threshold_policy,
    relabel_policy,
)
from ta_mpq.quant_search.policy_hash import duplicate_policy_hashes
from ta_mpq.quant_search.policy_io import save_policy_payload, write_built_policy
from ta_mpq.quant_search.sensitivity import load_sensitivity_profile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--group-registry", type=str, default="")
    parser.add_argument("--sensitivity", type=str, default="")
    parser.add_argument("--builder", type=str, default="size_weighted")
    parser.add_argument("--builder-label", type=str, default="")
    parser.add_argument("--promotion-mass-grid", type=str, default="")
    parser.add_argument("--grid", type=str, default="")
    parser.add_argument("--target-budget", type=str, default="int4")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--no-baselines", action="store_true")
    parser.add_argument("--existing-ea-baseline", type=str, default="")
    args = parser.parse_args()

    config = load_config(args.config) if args.config else {}
    if args.target_budget != "int4":
        raise ValueError("Only int4 target budget is currently supported")

    grouping = str(config.get("grouping") or "per_block_component")
    model_id = str(config.get("model_id") or "")
    policy_config = dict(config.get("policy_builder", {}))
    frontier_config = dict(config.get("frontier_search", {}))
    frontier_mode = str(frontier_config.get("mode") or "sector_coarse_to_fine_budgeted")
    coarse_config = dict(frontier_config.get("coarse", {}))
    hard_limits = dict(frontier_config.get("hard_limits", {}))
    group_registry_path = Path(
        args.group_registry
        or policy_config.get("group_registry_path")
        or PROJECT_ROOT / "artifacts" / "group_registry" / f"{grouping}.jsonl"
    )
    sensitivity_path = Path(
        args.sensitivity
        or policy_config.get("sensitivity_path")
        or PROJECT_ROOT / "artifacts" / "sensitivity" / f"{config.get('task', 'task')}_taqkl_lite.json"
    )
    output_dir = Path(
        args.output_dir
        or policy_config.get("output_dir")
        or PROJECT_ROOT / "artifacts" / "policies" / "task_sensitivity_exact_budget"
    )

    groups = load_group_registry(group_registry_path)
    sensitivity_profile = load_sensitivity_profile(sensitivity_path)
    scores = dict(sensitivity_profile.get("groups", {}))
    output_dir.mkdir(parents=True, exist_ok=True)

    if frontier_mode == GREEDY_PATH_METHOD_NAME:
        _build_greedy_path_frontier(
            config=config,
            groups=groups,
            scores=scores,
            output_dir=output_dir,
            model_id=model_id or None,
            hard_limits=hard_limits,
            coarse_config=coarse_config,
            sensitivity_profile=sensitivity_profile,
        )
        print(output_dir)
        return

    builder_label = args.builder_label or str(policy_config.get("builder_label") or "coarse_grid")
    if args.grid:
        promotion_mass_grid = _load_grid(Path(args.grid))
        if builder_label == "coarse_grid":
            builder_label = "refined_grid"
    elif args.promotion_mass_grid:
        promotion_mass_grid = parse_float_list(args.promotion_mass_grid)
    else:
        promotion_mass_grid = parse_float_list(
            coarse_config.get("promotion_mass_fractions")
            or policy_config.get("promotion_mass_grid")
            or [0.00, 0.03, 0.06, 0.10, 0.14, 0.18, 0.23, 0.29]
        )

    payloads: list[dict[str, Any]] = []
    for promotion_mass_fraction in promotion_mass_grid:
        if args.builder != "size_weighted":
            raise ValueError(f"Unsupported builder: {args.builder}")
        policy = build_size_weighted_threshold_policy(
            groups=groups,
            scores=scores,
            promotion_mass_fraction=float(promotion_mass_fraction),
            policy_id=fraction_policy_id("taqeb", promotion_mass_fraction, builder_label),
            source={
                "builder": builder_label,
                "strategy": "size_weighted_threshold",
                "sensitivity_file": str(sensitivity_path),
                "k_or_mass_target": float(promotion_mass_fraction),
            },
        )
        payloads.append(
            write_built_policy(
                output_dir / f"{policy.policy_id}.json",
                policy=policy,
                groups=groups,
                model_id=model_id or None,
            )
        )

    if args.existing_ea_baseline:
        payload = _import_existing_candidate(
            candidate_path=Path(args.existing_ea_baseline),
            groups=groups,
        )
        save_policy_payload(output_dir / f"{payload['policy_id']}.json", payload)
        payloads.append(payload)

    manifest = {
        "num_policies": len(payloads),
        "policy_ids": [payload["policy_id"] for payload in payloads],
    }
    max_unique_policies = hard_limits.get("max_unique_quantized_policies")
    if max_unique_policies is not None and len(payloads) > int(max_unique_policies):
        raise ValueError(
            f"Generated {len(payloads)} policies, exceeding max_unique_quantized_policies={max_unique_policies}"
        )
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output_dir)


def _build_greedy_path_frontier(
    *,
    config: dict[str, Any],
    groups: list[Any],
    scores: dict[str, dict[str, Any]],
    output_dir: Path,
    model_id: str | None,
    hard_limits: dict[str, Any],
    coarse_config: dict[str, Any],
    sensitivity_profile: dict[str, Any],
) -> None:
    greedy_config = dict(dict(config.get("frontier_search", {})).get("greedy_path", {}))
    constraints_config = dict(greedy_config.get("constraints", {}))
    demotion_config = dict(greedy_config.get("demotion_repair", {}))
    slack_config = dict(greedy_config.get("slack_policy", {}))
    policy_selection = dict(coarse_config.get("policy_selection", {}))

    min_bitwidth_by_group = {
        str(group_id): int(bitwidth)
        for group_id, bitwidth in dict(constraints_config.get("min_bitwidth", {})).items()
    }
    max_bitwidth_by_group = {
        str(group_id): int(bitwidth)
        for group_id, bitwidth in dict(constraints_config.get("max_bitwidth", {})).items()
    }
    allow_under_budget = bool(slack_config.get("allow_under_budget", True))

    path = build_greedy_max8_path(
        groups=groups,
        scores=scores,
        min_bitwidth_by_group=min_bitwidth_by_group or None,
        max_bitwidth_by_group=max_bitwidth_by_group or None,
        demotion_candidate_pool_size=int(demotion_config.get("candidate_pool_size") or 64),
        demotion_beam_width=int(demotion_config.get("beam_width") or 16),
        overshoot_penalty=float(demotion_config.get("overshoot_penalty") or 10.0),
        allow_under_budget=allow_under_budget,
    )
    coarse_policies = select_coarse_from_greedy_path(
        path,
        count=int(policy_selection.get("count") or 8),
        exclude_anchor=bool(policy_selection.get("exclude_anchor", True)),
        exclude_endpoint=bool(policy_selection.get("exclude_endpoint", True)),
    )
    if len(coarse_policies) < int(policy_selection.get("count") or 8):
        print(
            f"[greedy_path_frontier] warning: requested {int(policy_selection.get('count') or 8)} coarse policies "
            f"but only {len(coarse_policies)} unique path states were available.",
            flush=True,
        )

    profile_kind = _profile_kind(scores)
    if profile_kind == "bootstrap_legacy_scalar":
        print(
            "[greedy_path_frontier] warning: bootstrap_legacy_scalar sensitivity profile in use.",
            flush=True,
        )

    path_payloads: list[dict[str, Any]] = []
    for policy in path:
        path_index = int(policy.stats.path_index or 0)
        endpoint_kind = str(policy.source.get("endpoint_kind") or "")
        if endpoint_kind == "uniform_int4_anchor":
            policy_id = "gpf_uniform_int4_anchor"
        elif endpoint_kind == "greedy_max8_endpoint":
            policy_id = f"gpf_path_{path_index:04d}_endpoint"
        else:
            policy_id = f"gpf_path_{path_index:04d}"
        relabeled = relabel_policy(
            policy,
            policy_id=policy_id,
            builder="greedy_path_state",
            source_updates={
                "method_name": GREEDY_PATH_METHOD_NAME,
                "profile_kind": profile_kind,
            },
        )
        payload = write_built_policy(
            output_dir / f"{relabeled.policy_id}.json",
            policy=relabeled,
            groups=groups,
            model_id=model_id,
        )
        path_payloads.append(payload)

    coarse_payloads: list[dict[str, Any]] = []
    for index, policy in enumerate(sorted(coarse_policies, key=lambda item: int(item.stats.path_index or 0))):
        relabeled = relabel_policy(
            policy,
            policy_id=f"gpf_coarse_{index:02d}_i{int(policy.stats.path_index or 0):04d}",
            builder="coarse_path",
            source_updates={
                "selection_method": "evenly_spaced_path_snapshots",
                "selection_coordinate": "realized_8bit_param_mass_fraction",
                "selected_from_path_index": int(policy.stats.path_index or 0),
            },
        )
        coarse_payloads.append(
            write_built_policy(
                output_dir / f"{relabeled.policy_id}.json",
                policy=relabeled,
                groups=groups,
                model_id=model_id,
            )
        )

    endpoint_policy = next(
        (policy for policy in path if str(policy.source.get("endpoint_kind") or "") == "greedy_max8_endpoint"),
        None,
    )
    endpoint_payload = None
    if endpoint_policy is not None:
        relabeled_endpoint = relabel_policy(
            endpoint_policy,
            policy_id="gpf_endpoint_diagnostic",
            builder="endpoint_diagnostic",
            source_updates={
                "report_label": "greedy_max8_endpoint_diagnostic",
            },
        )
        endpoint_payload = write_built_policy(
            output_dir / f"{relabeled_endpoint.policy_id}.json",
            policy=relabeled_endpoint,
            groups=groups,
            model_id=model_id,
        )

    duplicates = duplicate_policy_hashes(path_payloads)
    if duplicates:
        write_duplicate_policy_report(output_dir / "duplicate_policy_hashes.json", duplicates)

    max_unique_policies = hard_limits.get("max_unique_quantized_policies")
    if max_unique_policies is not None and len(coarse_payloads) > int(max_unique_policies):
        raise ValueError(
            f"Generated {len(coarse_payloads)} coarse policies, exceeding max_unique_quantized_policies="
            f"{max_unique_policies}"
        )

    endpoint_stats = endpoint_policy.stats.to_dict() if endpoint_policy is not None else {}
    print(
        (
            "[greedy_path_frontier] path_states="
            f"{len(path_payloads)} unique_path_states={len({payload['policy_hash'] for payload in path_payloads})} "
            f"coarse_selected={len(coarse_payloads)} endpoint_path_index={endpoint_stats.get('path_index', -1)}"
        ),
        flush=True,
    )

    manifest = {
        "mode": GREEDY_PATH_METHOD_NAME,
        "profile_kind": profile_kind,
        "sensitivity_metadata": dict(sensitivity_profile.get("metadata", {})),
        "num_path_states": len(path_payloads),
        "num_unique_path_states": len({payload["policy_hash"] for payload in path_payloads}),
        "coarse_selected_path_indices": [
            int(payload.get("stats", {}).get("path_index") or 0)
            for payload in coarse_payloads
        ],
        "coarse_policy_ids": [payload["policy_id"] for payload in coarse_payloads],
        "endpoint_policy_id": endpoint_payload["policy_id"] if endpoint_payload is not None else None,
        "endpoint_stats": endpoint_stats,
        "path": [
            {
                "policy_id": payload["policy_id"],
                "policy_hash": payload["policy_hash"],
                "path_index": int(payload.get("stats", {}).get("path_index") or 0),
                "endpoint_kind": str(payload.get("source", {}).get("endpoint_kind") or ""),
                "policy_path": str(output_dir / f"{payload['policy_id']}.json"),
                "realized_8bit_param_mass_fraction": float(
                    payload.get("stats", {}).get("realized_8bit_param_mass_fraction", 0.0)
                ),
                "realized_2bit_param_mass_fraction": float(
                    payload.get("stats", {}).get("realized_2bit_param_mass_fraction", 0.0)
                ),
                "budget_slack_fraction": float(payload.get("stats", {}).get("budget_slack_fraction", 0.0)),
            }
            for payload in path_payloads
        ],
    }
    manifest_path = output_dir / "greedy_path_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_grid(path: Path) -> list[float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        values = payload.get("promotion_mass_grid")
    else:
        values = payload
    return [float(item) for item in values]


def _import_existing_candidate(
    candidate_path: Path,
    groups: list[Any],
) -> dict[str, Any]:
    payload = json.loads(candidate_path.read_text(encoding="utf-8"))
    group_bit_assignments = dict(payload.get("group_bit_assignments", {}))
    stats = compute_policy_budget_stats(group_bit_assignments, groups).to_dict()
    payload["policy_id"] = payload.get("policy_id") or f"taqeb_ea_{candidate_path.stem}"
    payload["method"] = "task_sensitivity_exact_budget"
    payload["bitwidths"] = {
        str(group_id): int(bitwidth)
        for group_id, bitwidth in sorted(group_bit_assignments.items())
    }
    payload["stats"] = stats
    payload["source"] = {
        "builder": "ea_baseline",
        "strategy": "imported_existing_candidate",
        "candidate_path": str(candidate_path),
    }
    return payload


def _profile_kind(scores: dict[str, dict[str, Any]]) -> str:
    if not scores:
        return "unknown"
    sample_payload = next(iter(scores.values()))
    if sample_payload.get("risk_2") is not None and sample_payload.get("benefit_8_over_4") is not None:
        return "taq_kl_groupwise"
    return "bootstrap_legacy_scalar"


if __name__ == "__main__":
    main()
