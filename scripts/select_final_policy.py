#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ta_mpq.quant_search.config import load_config
from ta_mpq.quant_search.frontier_search import choose_refinement_grid, load_frontier_results_csv
from ta_mpq.quant_search.greedy_path import (
    METHOD_NAME as GREEDY_PATH_METHOD_NAME,
    select_refine_candidates_from_coarse,
)
from ta_mpq.quant_search.group_registry import load_group_registry
from ta_mpq.quant_search.policy_builder import built_policy_from_payload, relabel_policy
from ta_mpq.quant_search.policy_io import load_policy_payload, write_built_policy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--coarse-results", type=str, default="")
    parser.add_argument("--refine-top-regions", type=int, default=2)
    parser.add_argument("--refine-step", type=float, default=0.01)
    parser.add_argument("--refine-radius", type=float, default=0.03)
    parser.add_argument("--write-refined-grid", type=str, default="")
    parser.add_argument("--make-refined-grid", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    frontier_config = dict(config.get("frontier_search", {}))
    coarse_config = dict(frontier_config.get("coarse", {}))
    refine_config = dict(frontier_config.get("refine", {}))
    frontier_mode = str(frontier_config.get("mode") or "sector_coarse_to_fine_budgeted")
    results_root = PROJECT_ROOT / "artifacts" / "results" / "task_sensitivity_exact_budget"
    coarse_results_path = Path(args.coarse_results or (results_root / "frontier_coarse.csv"))
    refined_grid_path = Path(
        args.write_refined_grid or PROJECT_ROOT / "artifacts" / "configs" / "refined_grid.json"
    )

    if frontier_mode == GREEDY_PATH_METHOD_NAME:
        refined_grid_path = _select_greedy_path_refine(
            config=config,
            coarse_results_path=coarse_results_path,
            refined_grid_path=refined_grid_path,
            coarse_config=coarse_config,
            refine_config=refine_config,
            requested_top_regions=max(1, int(args.refine_top_regions or 1)),
        )
        print(refined_grid_path)
        return

    coarse_rows = load_frontier_results_csv(coarse_results_path)
    refined_grid = choose_refinement_grid(
        coarse_results=coarse_rows,
        tie_band_correct_answers=int(coarse_config.get("tie_band_correct_answers", 1)),
        tie_break=list(coarse_config.get("tie_break", [])),
        num_threshold_subpoints=int(refine_config.get("num_threshold_subpoints") or 4),
        include_winner_fraction=bool(refine_config.get("include_coarse_winner_fraction", True)),
        top_k_coarse_candidates=int(args.refine_top_regions or 1),
        max_fraction=float(frontier_config.get("search_axis", {}).get("max_fraction", 0.30)),
    )
    payload = {
        "promotion_mass_grid": refined_grid,
        "source_results": str(coarse_results_path),
        "top_k_coarse_candidates": int(args.refine_top_regions or 1),
        "num_threshold_subpoints": int(refine_config.get("num_threshold_subpoints") or 4),
        "include_coarse_winner_fraction": bool(refine_config.get("include_coarse_winner_fraction", True)),
        "subpoint_rule": str(refine_config.get("subpoint_rule") or "local_sector_quartiles"),
    }
    refined_grid_path.parent.mkdir(parents=True, exist_ok=True)
    refined_grid_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(refined_grid_path)


def _select_greedy_path_refine(
    *,
    config: dict[str, object],
    coarse_results_path: Path,
    refined_grid_path: Path,
    coarse_config: dict[str, object],
    refine_config: dict[str, object],
    requested_top_regions: int,
) -> Path:
    grouping = str(config.get("grouping") or "per_block_component")
    policy_dir = Path(
        dict(config.get("policy_builder", {})).get("output_dir")
        or PROJECT_ROOT / "artifacts" / "policies" / "task_sensitivity_exact_budget"
    )
    manifest_path = policy_dir / "greedy_path_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Greedy path manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    registry_path = Path(
        dict(config.get("sensitivity", {})).get("group_registry_path")
        or PROJECT_ROOT / "artifacts" / "group_registry" / f"{grouping}.jsonl"
    )
    groups = load_group_registry(registry_path)
    path = [
        built_policy_from_payload(load_policy_payload(entry["policy_path"]), groups)
        for entry in manifest.get("path", [])
    ]
    coarse_rows = [
        row
        for row in load_frontier_results_csv(coarse_results_path)
        if bool(row.get("was_evaluated", True))
    ]

    seed_config = dict(refine_config.get("seed_selection", {}))
    policy_selection = dict(refine_config.get("policy_selection", {}))
    count = int(policy_selection.get("count") or refine_config.get("default_count") or 4)
    refined_policies = select_refine_candidates_from_coarse(
        path,
        coarse_rows,
        count=count,
        tie_band_correct_answers=int(coarse_config.get("tie_band_correct_answers") or 1),
        tie_break=list(coarse_config.get("tie_break", [])),
        allow_top2_for_candidate_generation=bool(
            seed_config.get("allow_top2_for_candidate_generation", requested_top_regions > 1)
        ),
        second_seed_max_gap_correct_answers=int(
            dict(seed_config.get("second_seed_conditions", {})).get("max_score_gap_correct_answers") or 1
        ),
        second_seed_min_realized_fraction_distance=float(
            dict(seed_config.get("second_seed_conditions", {})).get("min_realized_fraction_distance") or 0.04
        ),
    )

    refined_payloads: list[dict[str, object]] = []
    for index, policy in enumerate(refined_policies):
        relabeled = relabel_policy(
            policy,
            policy_id=f"gpf_refine_{index:02d}_i{int(policy.stats.path_index or 0):04d}",
            builder="refined_path",
            source_updates={
                "selection_method": "local_evenly_spaced_path_snapshots",
                "selection_coordinate": "realized_8bit_param_mass_fraction",
                "selected_from_path_index": int(policy.stats.path_index or 0),
            },
        )
        refined_payloads.append(
            write_built_policy(
                policy_dir / f"{relabeled.policy_id}.json",
                policy=relabeled,
                groups=groups,
                model_id=str(config.get("model_id") or ""),
            )
        )

    refined_grid_payload = {
        "mode": GREEDY_PATH_METHOD_NAME,
        "source_results": str(coarse_results_path),
        "path_manifest": str(manifest_path),
        "refined_policy_ids": [payload["policy_id"] for payload in refined_payloads],
        "refined_policy_hashes": [payload["policy_hash"] for payload in refined_payloads],
        "selected_path_indices": [
            int(payload.get("stats", {}).get("path_index") or 0)
            for payload in refined_payloads
        ],
        "count": len(refined_payloads),
        "seed_selection": seed_config,
        "policy_selection": policy_selection,
    }
    refined_grid_path.parent.mkdir(parents=True, exist_ok=True)
    refined_grid_path.write_text(
        json.dumps(refined_grid_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "[greedy_path_frontier] refined_selected_path_indices="
        + ",".join(str(index) for index in refined_grid_payload["selected_path_indices"]),
        flush=True,
    )
    return refined_grid_path


if __name__ == "__main__":
    main()
