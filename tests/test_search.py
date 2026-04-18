from __future__ import annotations

from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ta_mpq.feasibility import LinearLayerStat
from ta_mpq.search import (
    build_surrogate_candidate,
    build_search_groups,
    default_seed_assignments,
    estimate_candidate_weight_footprint_gb,
    _estimate_population_diversity,
    estimate_budget_alignment_score,
    estimate_group_value_alignment_score,
    estimate_reference_advantage_score,
    repair_assignments_to_budget,
    resolve_reference_target_value,
    resolve_group_value_scores,
    run_proxy_evolution_search,
    run_surrogate_evolution_search,
    value_guided_seed_assignments,
)


class SearchEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.layer_stats = [
            LinearLayerStat(
                name="model.layers.0.mlp.down_proj",
                parameter_count=1_000_000,
                in_features=1024,
                out_features=1024,
            ),
            LinearLayerStat(
                name="model.layers.0.mlp.up_proj",
                parameter_count=1_000_000,
                in_features=1024,
                out_features=1024,
            ),
            LinearLayerStat(
                name="model.layers.0.linear_attn.out_proj",
                parameter_count=500_000,
                in_features=1024,
                out_features=512,
            ),
            LinearLayerStat(
                name="model.layers.0.linear_attn.in_proj_b",
                parameter_count=500_000,
                in_features=1024,
                out_features=512,
            ),
            LinearLayerStat(
                name="model.layers.1.mlp.down_proj",
                parameter_count=1_000_000,
                in_features=1024,
                out_features=1024,
            ),
            LinearLayerStat(
                name="model.layers.1.mlp.up_proj",
                parameter_count=1_000_000,
                in_features=1024,
                out_features=1024,
            ),
        ]

    def test_grouping_can_collapse_by_component_family(self) -> None:
        groups = build_search_groups(self.layer_stats, grouping="per_component_family")
        group_names = {group.name for group in groups}
        self.assertIn("component:mlp.down_proj", group_names)
        self.assertIn("component:mlp.up_proj", group_names)
        self.assertEqual(len(groups), 4)

        down_proj_group = next(group for group in groups if group.name == "component:mlp.down_proj")
        self.assertEqual(down_proj_group.parameter_count, 2_000_000)

    def test_grouping_can_take_task_specific_sensitivity_overrides(self) -> None:
        groups = build_search_groups(
            self.layer_stats,
            grouping="per_block_component",
            sensitivity_overrides={"block:0:linear_attn.in_proj_b": 0.98},
        )

        in_proj_b = next(group for group in groups if group.name == "block:0:linear_attn.in_proj_b")
        down_proj = next(group for group in groups if group.name == "block:0:mlp.down_proj")
        self.assertEqual(in_proj_b.sensitivity, 0.98)
        self.assertEqual(down_proj.sensitivity, 1.0)

    def test_budget_repair_preserves_sensitive_layers_longer(self) -> None:
        groups = build_search_groups(self.layer_stats, grouping="per_block_component")
        assignments = {group.name: 8 for group in groups}

        target_assignments = default_seed_assignments(groups)
        target_budget = estimate_candidate_weight_footprint_gb(groups, target_assignments)

        repaired = repair_assignments_to_budget(
            groups=groups,
            assignments=assignments,
            target_budget_gb=target_budget,
            allowed_bits=(4, 8),
        )

        self.assertLessEqual(
            estimate_candidate_weight_footprint_gb(groups, repaired),
            target_budget,
        )
        self.assertEqual(repaired["block:0:mlp.down_proj"], 8)
        self.assertEqual(repaired["block:1:mlp.down_proj"], 8)
        self.assertEqual(repaired["block:0:linear_attn.in_proj_b"], 4)

    def test_proxy_search_returns_budget_feasible_candidates(self) -> None:
        groups = build_search_groups(self.layer_stats, grouping="per_block_component")
        target_budget = estimate_candidate_weight_footprint_gb(groups, default_seed_assignments(groups))

        result = run_proxy_evolution_search(
            groups=groups,
            target_budget_gb=target_budget,
            allowed_bits=(4, 8),
            grouping="per_block_component",
            population_size=12,
            generations=4,
            elite_count=2,
            seed=7,
        )

        self.assertEqual(result.num_groups, len(groups))
        self.assertEqual(len(result.history), 5)
        self.assertGreaterEqual(len(result.top_candidates), 1)
        self.assertIn("mutation_rate", result.history[0].to_dict())
        self.assertIn("mean_population_diversity", result.history[0].to_dict())

        for candidate in result.top_candidates:
            self.assertLessEqual(candidate.estimated_weight_footprint_gb, target_budget)

        top_candidate = result.top_candidates[0]
        self.assertIn(("block:0:mlp.down_proj", 8), top_candidate.group_bits)

    def test_surrogate_search_returns_budget_feasible_candidates(self) -> None:
        groups = build_search_groups(self.layer_stats, grouping="per_block_component")
        target_budget = estimate_candidate_weight_footprint_gb(groups, default_seed_assignments(groups))

        result = run_surrogate_evolution_search(
            groups=groups,
            report_payload={},
            surrogate_summary_payload={
                "backend": "mean_baseline",
                "feature_names": ["estimated_average_bit_width"],
                "predictions": [
                    {"policy_id": "bootstrap-a", "prediction": 0.33},
                    {"policy_id": "bootstrap-b", "prediction": 0.39},
                ],
            },
            surrogate_model_json="",
            target_budget_gb=target_budget,
            allowed_bits=(4, 8),
            grouping="per_block_component",
            population_size=10,
            generations=3,
            elite_count=2,
            seed=5,
        )

        self.assertGreaterEqual(len(result.top_candidates), 1)
        for candidate in result.top_candidates:
            self.assertLessEqual(candidate.estimated_weight_footprint_gb, target_budget)
            self.assertEqual(candidate.prediction_uncertainty, 0.0)

    def test_budget_alignment_score_prefers_budget_matched_candidates(self) -> None:
        self.assertGreater(
            estimate_budget_alignment_score(footprint_gb=16.5, target_budget_gb=16.7),
            estimate_budget_alignment_score(footprint_gb=12.0, target_budget_gb=16.7),
        )

    def test_reference_advantage_score_rewards_clearing_baseline(self) -> None:
        self.assertGreater(
            estimate_reference_advantage_score(0.41, 0.39),
            estimate_reference_advantage_score(0.38, 0.39),
        )

    def test_reference_target_value_uses_zero_for_advantage_targets(self) -> None:
        self.assertEqual(
            resolve_reference_target_value(
                {"target_metric": "accuracy_advantage_over_best_baseline"},
                0.39,
            ),
            0.0,
        )
        self.assertEqual(
            resolve_reference_target_value({"target_metric": "accuracy"}, 0.39),
            0.39,
        )

    def test_build_surrogate_candidate_tracks_alignment_metadata(self) -> None:
        groups = build_search_groups(self.layer_stats, grouping="per_block_component")
        assignments = default_seed_assignments(groups)
        target_budget = estimate_candidate_weight_footprint_gb(groups, assignments)
        candidate = build_surrogate_candidate(
            groups=groups,
            assignments=assignments,
            surrogate_predictor=lambda _features: (0.41, 0.02),
            target_budget_gb=target_budget,
            allowed_bits=(4, 8),
            provenance="unit_test",
            uncertainty_penalty=0.5,
            group_value_scores={"block:0:mlp.down_proj": 0.04},
            reference_accuracy=0.39,
        )
        self.assertIsNotNone(candidate.conservative_prediction)
        self.assertIsNotNone(candidate.budget_alignment_score)
        self.assertIsNotNone(candidate.group_value_alignment_score)
        self.assertEqual(candidate.reference_accuracy, 0.39)
        self.assertGreater(candidate.reference_advantage_score, 0.0)

    def test_population_diversity_is_zero_for_identical_candidates(self) -> None:
        groups = build_search_groups(self.layer_stats, grouping="per_block_component")
        assignments = default_seed_assignments(groups)
        target_budget = estimate_candidate_weight_footprint_gb(groups, assignments)
        candidate_a = build_surrogate_candidate(
            groups=groups,
            assignments=assignments,
            surrogate_predictor=lambda _features: (0.40, 0.02),
            target_budget_gb=target_budget,
            allowed_bits=(4, 8),
            provenance="unit_test_a",
            uncertainty_penalty=0.5,
            reference_accuracy=0.39,
        )
        candidate_b = build_surrogate_candidate(
            groups=groups,
            assignments=dict(assignments),
            surrogate_predictor=lambda _features: (0.41, 0.02),
            target_budget_gb=target_budget,
            allowed_bits=(4, 8),
            provenance="unit_test_b",
            uncertainty_penalty=0.5,
            reference_accuracy=0.39,
        )
        self.assertEqual(_estimate_population_diversity([candidate_a, candidate_b]), 0.0)

    def test_group_value_alignment_prefers_putting_8bit_on_positive_groups(self) -> None:
        groups = build_search_groups(self.layer_stats, grouping="per_block_component")
        aligned = default_seed_assignments(groups)
        misaligned = dict(aligned)
        aligned["block:0:mlp.down_proj"] = 8
        misaligned["block:0:mlp.down_proj"] = 4
        score_map = {
            "block:0:mlp.down_proj": {
                "score": 0.05,
                "uplift_8_over_4": 0.05,
                "uplift_16_over_8": 0.0,
                "preferred_bit": 8,
                "confidence": 1.0,
            }
        }
        self.assertGreater(
            estimate_group_value_alignment_score(groups, aligned, score_map),
            estimate_group_value_alignment_score(groups, misaligned, score_map),
        )

    def test_resolve_group_value_scores_falls_back_to_component_scores(self) -> None:
        groups = build_search_groups(self.layer_stats, grouping="per_block_component")
        resolved = resolve_group_value_scores(
            groups,
            {
                "group_scores": {},
                "component_scores": {
                    "mlp.down_proj": {"score": 0.03},
                    "linear_attn.out_proj": {"score": -0.02},
                },
            },
        )
        self.assertAlmostEqual(resolved["block:0:mlp.down_proj"]["score"], 0.03)
        self.assertAlmostEqual(resolved["block:0:linear_attn.out_proj"]["score"], -0.02)

    def test_value_guided_seed_assignments_uses_positive_group_scores(self) -> None:
        groups = build_search_groups(self.layer_stats, grouping="per_block_component")
        assignments = value_guided_seed_assignments(
            groups=groups,
            allowed_bits=(4, 8),
            group_value_scores={
                "block:0:mlp.down_proj": {"score": 0.04, "uplift_8_over_4": 0.04, "preferred_bit": 8},
                "block:1:mlp.down_proj": {"score": -0.01, "uplift_8_over_4": -0.01, "preferred_bit": 4},
            },
        )
        assert assignments is not None
        self.assertEqual(assignments["block:0:mlp.down_proj"], 8)
        self.assertEqual(assignments["block:1:mlp.down_proj"], 4)

    def test_value_guided_seed_assignments_can_promote_top_groups_to_16bit(self) -> None:
        groups = build_search_groups(self.layer_stats, grouping="per_block_component")
        assignments = value_guided_seed_assignments(
            groups=groups,
            allowed_bits=(4, 8, 16),
            group_value_scores={
                "block:0:mlp.down_proj": {
                    "score": 0.12,
                    "uplift_8_over_4": 0.08,
                    "uplift_16_over_8": 0.04,
                    "preferred_bit": 16,
                },
                "block:1:mlp.down_proj": {
                    "score": 0.03,
                    "uplift_8_over_4": 0.03,
                    "uplift_16_over_8": 0.0,
                    "preferred_bit": 8,
                },
                "block:0:linear_attn.out_proj": {
                    "score": 0.01,
                    "uplift_8_over_4": 0.01,
                    "uplift_16_over_8": 0.0,
                    "preferred_bit": 8,
                },
            },
        )
        assert assignments is not None
        self.assertEqual(assignments["block:0:mlp.down_proj"], 16)
        self.assertEqual(assignments["block:1:mlp.down_proj"], 8)

    def test_surrogate_search_accepts_16bit_in_search_space(self) -> None:
        groups = build_search_groups(self.layer_stats, grouping="per_block_component")
        target_budget = estimate_candidate_weight_footprint_gb(groups, default_seed_assignments(groups))

        result = run_surrogate_evolution_search(
            groups=groups,
            report_payload={},
            surrogate_summary_payload={
                "backend": "mean_baseline",
                "target_metric": "accuracy",
                "feature_names": ["estimated_average_bit_width"],
                "predictions": [
                    {"policy_id": "bootstrap-a", "prediction": 0.33},
                    {"policy_id": "bootstrap-b", "prediction": 0.39},
                ],
            },
            surrogate_model_json="",
            target_budget_gb=target_budget,
            allowed_bits=(4, 8, 16),
            grouping="per_block_component",
            population_size=10,
            generations=2,
            elite_count=2,
            seed=5,
        )

        self.assertGreaterEqual(len(result.top_candidates), 1)

    def test_proxy_search_history_tracks_diversity_and_stagnation(self) -> None:
        groups = build_search_groups(self.layer_stats, grouping="per_block_component")
        target_budget = estimate_candidate_weight_footprint_gb(groups, default_seed_assignments(groups))
        result = run_proxy_evolution_search(
            groups=groups,
            target_budget_gb=target_budget,
            allowed_bits=(4, 8),
            grouping="per_block_component",
            population_size=10,
            generations=3,
            elite_count=2,
            seed=3,
        )
        history_payload = [entry.to_dict() for entry in result.history]
        self.assertEqual(len(history_payload), 4)
        self.assertTrue(all(entry["unique_candidate_count"] >= 1 for entry in history_payload))
        self.assertTrue(all(entry["mutation_rate"] >= 0.1 for entry in history_payload))
        self.assertTrue(all(entry["stagnation_steps"] >= 0 for entry in history_payload))


if __name__ == "__main__":
    unittest.main()
