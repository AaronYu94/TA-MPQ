from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ta_mpq.quant_search.budget import policy_bits, target_int4_bits
from ta_mpq.quant_search.frontier_search import (
    choose_refinement_grid,
    resolve_policy_parallelism,
    select_best_policy_id,
)
from ta_mpq.quant_search.greedy_path import (
    build_greedy_max8_path,
    select_coarse_from_greedy_path,
    select_refine_candidates_from_coarse,
)
from ta_mpq.quant_search.group_registry import GroupInfo
from ta_mpq.quant_search.policy_builder import (
    build_equal_count_threshold_policy,
    build_size_weighted_threshold_policy,
    build_uniform_int4_policy,
    repair_to_budget,
)
from ta_mpq.quant_search.policy_hash import canonical_assignment_hash
from ta_mpq.quant_search.policy_io import canonical_policy_hash, load_policy_payload, write_built_policy
from ta_mpq.quant_search.sensitivity import load_sensitivity_profile, save_sensitivity_profile


class QuantSearchTests(unittest.TestCase):
    def test_equal_size_budget_identity(self) -> None:
        groups = [
            GroupInfo(
                group_id=f"block:{index}:linear_attn.in_proj_a",
                module_path=f"model.layers.{index}.linear_attn.in_proj_a",
                block_idx=index,
                component="linear_attn.in_proj_a",
                param_count=1,
            )
            for index in range(30)
        ]
        scores = {
            group.group_id: {
                "param_count": group.param_count,
                "score": float(30 - index),
            }
            for index, group in enumerate(groups)
        }

        for k in [0, 1, 2, 5, 10]:
            policy = build_equal_count_threshold_policy(
                groups=groups,
                scores=scores,
                k=k,
                policy_id=f"eq-{k}",
            )
            bitwidths = policy.bitwidth_dict()
            self.assertEqual(sum(1 for bit in bitwidths.values() if bit == 8), k)
            self.assertEqual(sum(1 for bit in bitwidths.values() if bit == 2), 2 * k)
            self.assertEqual(sum(1 for bit in bitwidths.values() if bit == 4), 30 - (3 * k))
            self.assertEqual(policy_bits(bitwidths, groups), target_int4_bits(groups))

    def test_size_weighted_repair_never_exceeds_budget(self) -> None:
        groups = [
            GroupInfo(
                group_id=f"block:{index}:mlp.down_proj",
                module_path=f"model.layers.{index}.mlp.down_proj",
                block_idx=index,
                component="mlp.down_proj",
                param_count=param_count,
            )
            for index, param_count in enumerate([11, 7, 5, 3, 2, 1])
        ]
        scores = {
            group.group_id: {
                "param_count": group.param_count,
                "score": float(index + 1),
                "benefit_8_over_4": float(index + 1),
                "demotion_cost_4_to_2": float(index + 1),
            }
            for index, group in enumerate(groups)
        }

        for fraction in [0.0, 0.05, 0.10, 0.20, 0.30]:
            policy = build_size_weighted_threshold_policy(
                groups=groups,
                scores=scores,
                promotion_mass_fraction=fraction,
                policy_id=f"weighted-{fraction}",
            )
            bitwidths = policy.bitwidth_dict()
            self.assertLessEqual(policy.stats.raw_weight_bits, policy.stats.target_int4_bits)
            self.assertTrue(all(bit in {2, 4, 8} for bit in bitwidths.values()))
            self.assertEqual(set(bitwidths), {group.group_id for group in groups})

    def test_repair_fills_slack_when_possible(self) -> None:
        groups = [
            GroupInfo(
                group_id=f"block:{index}:linear_attn.out_proj",
                module_path=f"model.layers.{index}.linear_attn.out_proj",
                block_idx=index,
                component="linear_attn.out_proj",
                param_count=param_count,
            )
            for index, param_count in enumerate([9, 7, 5, 3])
        ]
        scores = {
            group.group_id: {
                "param_count": group.param_count,
                "score": 1.0 + index,
                "benefit_8_over_4": 1.0 + index,
                "demotion_cost_4_to_2": 1.0 + index,
            }
            for index, group in enumerate(groups)
        }
        repaired = repair_to_budget(
            bitwidths={group.group_id: 2 for group in groups},
            groups=groups,
            scores=scores,
            fill_remaining_slack=True,
        )
        remaining_slack = target_int4_bits(groups) - policy_bits(repaired, groups)
        possible_upgrade_costs: list[int] = []
        for group in groups:
            current_bit = repaired[group.group_id]
            if current_bit == 2:
                possible_upgrade_costs.append(2 * group.param_count)
            elif current_bit == 4:
                possible_upgrade_costs.append(4 * group.param_count)
        self.assertTrue(all(cost > remaining_slack for cost in possible_upgrade_costs))

    def test_sensitivity_json_round_trip(self) -> None:
        payload = {
            "metadata": {
                "method": "taq_kl_lite",
                "model_id": "unit-test-model",
                "task": "math500",
            },
            "groups": {
                "block:1:mlp.down_proj": {
                    "param_count": 11,
                    "score": 0.5,
                    "risk_4": 0.5,
                    "combined_sensitivity": 0.5,
                },
                "block:0:linear_attn.in_proj_a": {
                    "param_count": 7,
                    "score": 0.25,
                    "benefit_8_over_4": 0.25,
                },
            },
            "module_activation_stats": [
                {"name": "model.layers.0.mlp.down_proj", "mean_abs_input": 1.5}
            ],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "sensitivity.json"
            save_sensitivity_profile(output_path, payload)
            reloaded = load_sensitivity_profile(output_path)
        self.assertEqual(list(reloaded["groups"]), sorted(payload["groups"]))
        self.assertEqual(reloaded["groups"]["block:1:mlp.down_proj"]["param_count"], 11)
        self.assertAlmostEqual(reloaded["groups"]["block:0:linear_attn.in_proj_a"]["score"], 0.25)
        self.assertAlmostEqual(
            reloaded["groups"]["block:1:mlp.down_proj"]["combined_sensitivity"],
            0.5,
        )
        self.assertEqual(len(reloaded["module_activation_stats"]), 1)

    def test_policy_hash_determinism(self) -> None:
        groups = [
            GroupInfo(
                group_id="block:0:linear_attn.in_proj_a",
                module_path="model.layers.0.linear_attn.in_proj_a",
                block_idx=0,
                component="linear_attn.in_proj_a",
                param_count=13,
            ),
            GroupInfo(
                group_id="block:1:linear_attn.in_proj_a",
                module_path="model.layers.1.linear_attn.in_proj_a",
                block_idx=1,
                component="linear_attn.in_proj_a",
                param_count=17,
            ),
        ]
        policy = build_uniform_int4_policy(groups)
        with tempfile.TemporaryDirectory() as temp_dir:
            path_a = Path(temp_dir) / "policy-a.json"
            path_b = Path(temp_dir) / "policy-b.json"
            payload_a = write_built_policy(path_a, policy=policy, groups=groups, model_id="unit-test-model")
            payload_b = write_built_policy(path_b, policy=policy, groups=groups, model_id="unit-test-model")
            reloaded_a = load_policy_payload(path_a)
            reloaded_b = load_policy_payload(path_b)
        self.assertEqual(payload_a["policy_id"], payload_b["policy_id"])
        self.assertEqual(json.dumps(reloaded_a, sort_keys=True), json.dumps(reloaded_b, sort_keys=True))
        self.assertEqual(canonical_policy_hash(reloaded_a), canonical_policy_hash(reloaded_b))
        self.assertEqual(canonical_policy_hash(reloaded_a), str(reloaded_a["policy_hash"]))

    def test_policy_hash_same_for_identical_assignments(self) -> None:
        hash_a = canonical_assignment_hash(
            {"b": 4, "a": 8},
            grouping="per_block_component",
            budget_rule="raw_weight_int4",
        )
        hash_b = canonical_assignment_hash(
            {"a": 8, "b": 4},
            grouping="per_block_component",
            budget_rule="raw_weight_int4",
        )
        self.assertEqual(hash_a, hash_b)

    def test_policy_hash_differs_for_different_assignments(self) -> None:
        hash_a = canonical_assignment_hash({"a": 8, "b": 4})
        hash_b = canonical_assignment_hash({"a": 8, "b": 2})
        self.assertNotEqual(hash_a, hash_b)

    def test_greedy_path_budget_feasible(self) -> None:
        groups, scores = _build_greedy_fixture()
        path = build_greedy_max8_path(
            groups,
            scores,
            min_bitwidth_by_group={"global::lm_head": 4},
        )
        self.assertGreaterEqual(len(path), 2)
        for policy in path:
            self.assertLessEqual(policy.stats.raw_weight_bits, policy.stats.target_int4_bits)
            self.assertTrue(all(bit in {2, 4, 8} for _, bit in policy.bitwidths))

    def test_greedy_path_endpoint_marked(self) -> None:
        groups, scores = _build_greedy_fixture()
        path = build_greedy_max8_path(
            groups,
            scores,
            min_bitwidth_by_group={"global::lm_head": 4},
        )
        self.assertEqual(path[-1].source.get("endpoint_kind"), "greedy_max8_endpoint")

    def test_coarse_selection_unique(self) -> None:
        groups, scores = _build_greedy_fixture()
        path = build_greedy_max8_path(
            groups,
            scores,
            min_bitwidth_by_group={"global::lm_head": 4},
        )
        selected = select_coarse_from_greedy_path(path, count=4)
        self.assertEqual(len(selected), len({policy.policy_hash for policy in selected}))

    def test_refine_selection_unique(self) -> None:
        groups, scores = _build_greedy_fixture()
        path = build_greedy_max8_path(
            groups,
            scores,
            min_bitwidth_by_group={"global::lm_head": 4},
        )
        coarse = select_coarse_from_greedy_path(path, count=4)
        coarse_rows = [
            {
                "policy_id": policy.policy_id,
                "policy_hash": policy.policy_hash,
                "builder": "coarse_path",
                "path_index": int(policy.stats.path_index or 0),
                "correct": 20 - index,
                "score": (20 - index) / 25.0,
                "proxy_score": policy.proxy_score,
                "promotion_mass_fraction": policy.stats.promotion_mass_fraction,
                "realized_8bit_param_mass_fraction": policy.stats.realized_8bit_param_mass_fraction,
                "realized_2bit_param_mass_fraction": policy.stats.realized_2bit_param_mass_fraction,
                "budget_slack_fraction": policy.stats.budget_slack_fraction,
            }
            for index, policy in enumerate(coarse)
        ]
        refined = select_refine_candidates_from_coarse(
            path,
            coarse_rows,
            count=4,
            tie_band_correct_answers=1,
            tie_break=[
                "higher_proxy_score",
                "lower_promotion_mass_fraction",
                "lower_twobit_mass",
                "smaller_budget_slack",
            ],
        )
        self.assertEqual(len(refined), len({policy.policy_hash for policy in refined}))

    def test_parallelism_auto_equals_num_policies(self) -> None:
        parallelism = resolve_policy_parallelism(
            stage="refine",
            num_policies=4,
            cfg={
                "execution": {
                    "policy_parallelism": {
                        "mode": "auto",
                        "max_parallel_policies": None,
                        "stage_overrides": {
                            "refine": {
                                "max_parallel_policies": None,
                            }
                        },
                    }
                }
            },
        )
        self.assertEqual(parallelism, 4)

    def test_parallelism_cap_applied(self) -> None:
        parallelism = resolve_policy_parallelism(
            stage="coarse",
            num_policies=8,
            cfg={
                "execution": {
                    "policy_parallelism": {
                        "mode": "auto",
                        "max_parallel_policies": 3,
                    }
                }
            },
        )
        self.assertEqual(parallelism, 3)

    def test_choose_refinement_grid_uses_local_sector_quartiles(self) -> None:
        coarse_rows = [
            {
                "policy_id": f"coarse-{fraction}",
                "builder": "coarse_grid",
                "correct": correct,
                "score": correct / 25.0,
                "proxy_score": proxy,
                "promotion_mass_fraction": fraction,
                "twobit_mass_fraction": twobit_mass,
                "budget_slack_fraction": slack,
            }
            for fraction, correct, proxy, twobit_mass, slack in [
                (0.00, 14, 0.5, 0.00, 0.00),
                (0.03, 15, 0.6, 0.02, 0.01),
                (0.06, 16, 0.7, 0.03, 0.02),
                (0.10, 16, 0.9, 0.05, 0.02),
                (0.14, 15, 0.8, 0.06, 0.03),
            ]
        ]
        refined = choose_refinement_grid(coarse_rows)
        self.assertEqual(refined, [0.07, 0.09, 0.1, 0.11, 0.13])

    def test_select_best_policy_id_respects_tie_band_and_tie_break(self) -> None:
        rows = [
            {
                "policy_id": "candidate-a",
                "stage": "coarse",
                "correct": 16,
                "score": 16 / 25.0,
                "proxy_score": 0.7,
                "promotion_mass_fraction": 0.06,
                "twobit_mass_fraction": 0.03,
                "budget_slack_fraction": 0.02,
            },
            {
                "policy_id": "candidate-b",
                "stage": "coarse",
                "correct": 15,
                "score": 15 / 25.0,
                "proxy_score": 0.9,
                "promotion_mass_fraction": 0.10,
                "twobit_mass_fraction": 0.05,
                "budget_slack_fraction": 0.02,
            },
        ]
        best_policy_id = select_best_policy_id(
            rows,
            preferred_stage="coarse",
            tie_band_correct_answers=1,
            tie_break=[
                "higher_proxy_score",
                "lower_promotion_mass_fraction",
                "lower_twobit_mass",
                "smaller_budget_slack",
            ],
        )
        self.assertEqual(best_policy_id, "candidate-b")


def _build_greedy_fixture() -> tuple[list[GroupInfo], dict[str, dict[str, float]]]:
    groups = [
        GroupInfo(
            group_id="global::lm_head",
            module_path="lm_head",
            block_idx=-1,
            component="lm_head",
            param_count=20,
        ),
        GroupInfo(
            group_id="block:0:self_attn.k_proj",
            module_path="model.layers.0.self_attn.k_proj",
            block_idx=0,
            component="self_attn.k_proj",
            param_count=9,
        ),
        GroupInfo(
            group_id="block:0:self_attn.v_proj",
            module_path="model.layers.0.self_attn.v_proj",
            block_idx=0,
            component="self_attn.v_proj",
            param_count=8,
        ),
        GroupInfo(
            group_id="block:0:linear_attn.out_proj",
            module_path="model.layers.0.linear_attn.out_proj",
            block_idx=0,
            component="linear_attn.out_proj",
            param_count=7,
        ),
        GroupInfo(
            group_id="block:1:self_attn.k_proj",
            module_path="model.layers.1.self_attn.k_proj",
            block_idx=1,
            component="self_attn.k_proj",
            param_count=6,
        ),
        GroupInfo(
            group_id="block:1:self_attn.v_proj",
            module_path="model.layers.1.self_attn.v_proj",
            block_idx=1,
            component="self_attn.v_proj",
            param_count=5,
        ),
        GroupInfo(
            group_id="block:0:mlp.up_proj",
            module_path="model.layers.0.mlp.up_proj",
            block_idx=0,
            component="mlp.up_proj",
            param_count=4,
        ),
        GroupInfo(
            group_id="block:1:mlp.up_proj",
            module_path="model.layers.1.mlp.up_proj",
            block_idx=1,
            component="mlp.up_proj",
            param_count=3,
        ),
    ]
    scores = {
        group.group_id: {
            "score": float(10 - index),
            "benefit_8_over_4": float(18 - (index * 2)),
            "demotion_cost_4_to_2": float(1 + index),
            "risk_2": float(3 + index),
            "risk_4": float(2 + index),
            "risk_8": float(1 + index),
        }
        for index, group in enumerate(groups)
    }
    return groups, scores


if __name__ == "__main__":
    unittest.main()
