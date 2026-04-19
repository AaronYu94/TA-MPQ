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
    aggregate_group_score_overrides,
    build_surrogate_candidate,
    run_budgeted_bf16_allocator,
    build_surrogate_free_seed_assignments,
    build_hierarchical_promotion_manifest,
    build_search_groups,
    dedupe_assignment_candidates,
    default_seed_assignments,
    estimate_assignment_search_score,
    estimate_candidate_weight_footprint_gb,
    _estimate_population_diversity,
    estimate_budget_alignment_score,
    expand_group_assignments,
    generate_surrogate_free_neighbor_assignments,
    estimate_group_value_alignment_score,
    estimate_reference_advantage_score,
    refine_candidate_quantization_configs,
    repair_assignments_to_budget,
    repair_assignments_with_fixed_groups,
    resolve_reference_target_value,
    resolve_group_value_scores,
    resolve_surrogate_free_priority_lists,
    resolve_surrogate_free_priority_scores,
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

    def test_grouping_can_collapse_nearby_layers_into_block_windows(self) -> None:
        groups = build_search_groups(self.layer_stats, grouping="per_block_window_component")
        group_names = {group.name for group in groups}
        self.assertIn("window:0-3:mlp.down_proj", group_names)
        self.assertIn("window:0-3:mlp.up_proj", group_names)

        down_proj_group = next(group for group in groups if group.name == "window:0-3:mlp.down_proj")
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

    def test_budget_repair_can_hold_fixed_groups_constant(self) -> None:
        groups = build_search_groups(self.layer_stats, grouping="per_block_component")
        uniform_budget = estimate_candidate_weight_footprint_gb(
            groups,
            {group.name: 8 for group in groups},
        )

        repaired = repair_assignments_with_fixed_groups(
            groups=groups,
            assignments={group.name: 16 for group in groups},
            target_budget_gb=uniform_budget,
            allowed_bits=(4, 8, 16),
            fixed_assignments={"block:0:mlp.down_proj": 8},
            min_budget_utilization=0.99,
        )

        self.assertEqual(repaired["block:0:mlp.down_proj"], 8)
        self.assertLessEqual(
            estimate_candidate_weight_footprint_gb(groups, repaired),
            uniform_budget,
        )

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
        score_map = {"block:0:mlp.down_proj": 0.05}
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
        self.assertAlmostEqual(resolved["block:0:mlp.down_proj"], 0.03)
        self.assertAlmostEqual(resolved["block:0:linear_attn.out_proj"], -0.02)

    def test_group_value_scores_can_aggregate_from_fine_groups_to_windows(self) -> None:
        coarse_groups = build_search_groups(self.layer_stats, grouping="per_block_window_component")
        resolved = resolve_group_value_scores(
            coarse_groups,
            {
                "grouping": "per_block_component",
                "group_scores": {
                    "block:0:mlp.down_proj": {"score": 0.2},
                    "block:1:mlp.down_proj": {"score": 0.6},
                },
                "component_scores": {},
            },
            layer_stats=self.layer_stats,
            target_grouping="per_block_window_component",
        )
        self.assertAlmostEqual(resolved["window:0-3:mlp.down_proj"], 0.4)

    def test_aggregate_group_score_overrides_uses_parameter_weighting(self) -> None:
        aggregated = aggregate_group_score_overrides(
            layer_stats=self.layer_stats,
            score_overrides={
                "block:0:mlp.down_proj": 0.2,
                "block:1:mlp.down_proj": 0.6,
                "block:0:linear_attn.out_proj": -0.2,
            },
            target_grouping="per_block_window_component",
            source_grouping="per_block_component",
        )
        self.assertAlmostEqual(aggregated["window:0-3:mlp.down_proj"], 0.4)
        self.assertAlmostEqual(aggregated["window:0-3:linear_attn.out_proj"], -0.2)

    def test_surrogate_free_seed_assignments_stay_near_budget(self) -> None:
        groups = build_search_groups(self.layer_stats, grouping="per_block_component")
        uniform_budget = estimate_candidate_weight_footprint_gb(
            groups,
            {group.name: 8 for group in groups},
        )
        priority_scores = resolve_surrogate_free_priority_scores(groups)
        priority_lists = resolve_surrogate_free_priority_lists(
            groups=groups,
            group_priority_scores=priority_scores,
            promotable_count=3,
            demotable_count=3,
            excluded_group_names=set(),
        )

        seeds = build_surrogate_free_seed_assignments(
            groups=groups,
            target_budget_gb=uniform_budget,
            allowed_bits=(4, 8, 16),
            group_priority_scores=priority_scores,
            promotable_group_names=priority_lists["promotable_group_names"],
            demotable_group_names=priority_lists["demotable_group_names"],
            min_budget_utilization=0.99,
        )

        self.assertGreaterEqual(len(seeds), 3)
        self.assertLessEqual(len(seeds), 5)
        for _provenance, assignments in seeds:
            footprint = estimate_candidate_weight_footprint_gb(groups, assignments)
            self.assertLessEqual(footprint, uniform_budget)
            self.assertGreaterEqual(footprint / uniform_budget, 0.99)

    def test_surrogate_free_seed_assignments_can_select_explicit_subset(self) -> None:
        groups = build_search_groups(self.layer_stats, grouping="per_block_component")
        uniform_budget = estimate_candidate_weight_footprint_gb(
            groups,
            {group.name: 8 for group in groups},
        )
        priority_scores = resolve_surrogate_free_priority_scores(groups)
        priority_lists = resolve_surrogate_free_priority_lists(
            groups=groups,
            group_priority_scores=priority_scores,
            promotable_count=3,
            demotable_count=3,
            excluded_group_names=set(),
        )

        seeds = build_surrogate_free_seed_assignments(
            groups=groups,
            target_budget_gb=uniform_budget,
            allowed_bits=(4, 8, 16),
            group_priority_scores=priority_scores,
            promotable_group_names=priority_lists["promotable_group_names"],
            demotable_group_names=priority_lists["demotable_group_names"],
            min_budget_utilization=0.99,
            selected_seed_provenances=(
                "uniform_int8_seed",
                "single_priority_rescue_seed",
            ),
        )

        self.assertEqual(
            [provenance for provenance, _assignments in seeds],
            ["uniform_int8_seed", "single_priority_rescue_seed"],
        )

    def test_surrogate_free_seed_assignments_can_limit_seed_count(self) -> None:
        groups = build_search_groups(self.layer_stats, grouping="per_block_component")
        uniform_budget = estimate_candidate_weight_footprint_gb(
            groups,
            {group.name: 8 for group in groups},
        )
        priority_scores = resolve_surrogate_free_priority_scores(groups)
        priority_lists = resolve_surrogate_free_priority_lists(
            groups=groups,
            group_priority_scores=priority_scores,
            promotable_count=3,
            demotable_count=3,
            excluded_group_names=set(),
        )

        seeds = build_surrogate_free_seed_assignments(
            groups=groups,
            target_budget_gb=uniform_budget,
            allowed_bits=(4, 8, 16),
            group_priority_scores=priority_scores,
            promotable_group_names=priority_lists["promotable_group_names"],
            demotable_group_names=priority_lists["demotable_group_names"],
            min_budget_utilization=0.99,
            max_seed_count=3,
        )

        self.assertEqual(len(seeds), 3)

    def test_surrogate_free_neighbors_are_deduped_and_scored(self) -> None:
        groups = build_search_groups(self.layer_stats, grouping="per_block_component")
        uniform_assignments = {group.name: 8 for group in groups}
        uniform_budget = estimate_candidate_weight_footprint_gb(groups, uniform_assignments)
        priority_scores = resolve_surrogate_free_priority_scores(groups)
        priority_lists = resolve_surrogate_free_priority_lists(
            groups=groups,
            group_priority_scores=priority_scores,
            promotable_count=3,
            demotable_count=3,
            excluded_group_names=set(),
        )

        neighbors = generate_surrogate_free_neighbor_assignments(
            groups=groups,
            base_assignments=uniform_assignments,
            target_budget_gb=uniform_budget,
            allowed_bits=(4, 8, 16),
            group_priority_scores=priority_scores,
            promotable_group_names=priority_lists["promotable_group_names"],
            demotable_group_names=priority_lists["demotable_group_names"],
            min_budget_utilization=0.99,
        )
        deduped = dedupe_assignment_candidates(neighbors + neighbors[:1])

        self.assertEqual(len(deduped), len(neighbors))
        self.assertGreaterEqual(len(deduped), 1)
        for _provenance, assignments in deduped:
            self.assertIsInstance(
                estimate_assignment_search_score(
                    groups=groups,
                    assignments=assignments,
                    target_budget_gb=uniform_budget,
                    group_priority_scores=priority_scores,
                ),
                float,
            )

    def test_surrogate_free_low_bit_seed_assignments_use_two_seed_policy(self) -> None:
        groups = build_search_groups(self.layer_stats, grouping="per_block_component")
        int4_budget = estimate_candidate_weight_footprint_gb(
            groups,
            {group.name: 4 for group in groups},
        )

        seeds = build_surrogate_free_seed_assignments(
            groups=groups,
            target_budget_gb=int4_budget,
            allowed_bits=(2, 4, 8),
            group_priority_scores={group.name: 0.0 for group in groups},
            promotable_group_names=[group.name for group in groups],
            demotable_group_names=[group.name for group in groups],
            min_budget_utilization=0.99,
            max_seed_count=2,
        )

        self.assertEqual(
            [provenance for provenance, _assignments in seeds],
            ["uniform_int4_seed", "compression_first_seed"],
        )
        for _provenance, assignments in seeds:
            footprint = estimate_candidate_weight_footprint_gb(groups, assignments)
            self.assertLessEqual(footprint, int4_budget)
            self.assertGreaterEqual(footprint / int4_budget, 0.99)

    def test_surrogate_free_low_bit_neighbors_include_multiple_operator_families(self) -> None:
        groups = build_search_groups(self.layer_stats, grouping="per_block_component")
        base_assignments = {group.name: 4 for group in groups}
        base_assignments["block:0:mlp.down_proj"] = 8
        base_assignments["block:0:mlp.up_proj"] = 8
        base_assignments["block:1:mlp.down_proj"] = 2
        target_budget_gb = estimate_candidate_weight_footprint_gb(groups, base_assignments)

        neighbors = generate_surrogate_free_neighbor_assignments(
            groups=groups,
            base_assignments=base_assignments,
            target_budget_gb=target_budget_gb,
            allowed_bits=(2, 4, 8),
            group_priority_scores={group.name: 0.0 for group in groups},
            promotable_group_names=[group.name for group in groups],
            demotable_group_names=[group.name for group in groups],
            min_budget_utilization=0.99,
        )

        provenances = [provenance for provenance, _assignments in neighbors]
        self.assertTrue(any(provenance.startswith("single_flip_") for provenance in provenances))
        self.assertGreaterEqual(len(provenances), 6)
        self.assertTrue(any(provenance.startswith("double_flip_budget_balanced:") for provenance in provenances))
        self.assertTrue(
            any(
                provenance.startswith(("band_swap:", "cluster_flip:", "large_group_reallocation:"))
                for provenance in provenances
            )
        )
        for _provenance, assignments in neighbors:
            self.assertTrue(all(bit in {2, 4, 8} for bit in assignments.values()))
            self.assertNotEqual(assignments, base_assignments)
            self.assertLessEqual(
                estimate_candidate_weight_footprint_gb(groups, assignments),
                target_budget_gb,
            )

    def test_value_guided_seed_assignments_uses_positive_group_scores(self) -> None:
        groups = build_search_groups(self.layer_stats, grouping="per_block_component")
        assignments = value_guided_seed_assignments(
            groups=groups,
            allowed_bits=(4, 8),
            group_value_scores={
                "block:0:mlp.down_proj": 0.04,
                "block:1:mlp.down_proj": -0.01,
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
                "block:0:mlp.down_proj": 0.08,
                "block:1:mlp.down_proj": 0.03,
                "block:0:linear_attn.out_proj": 0.01,
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

    def test_budgeted_bf16_allocator_can_trade_two_int4_groups_for_one_bf16_group(self) -> None:
        custom_layer_stats = [
            LinearLayerStat(
                name="model.layers.0.mlp.up_proj",
                parameter_count=1_000_000,
                in_features=1024,
                out_features=1024,
            ),
            LinearLayerStat(
                name="model.layers.1.mlp.gate_proj",
                parameter_count=1_000_000,
                in_features=1024,
                out_features=1024,
            ),
            LinearLayerStat(
                name="model.layers.2.linear_attn.in_proj_b",
                parameter_count=1_000_000,
                in_features=1024,
                out_features=1024,
            ),
            LinearLayerStat(
                name="model.layers.3.linear_attn.in_proj_a",
                parameter_count=1_000_000,
                in_features=1024,
                out_features=1024,
            ),
        ]
        groups = build_search_groups(custom_layer_stats, grouping="per_block_component")
        uniform_budget = estimate_candidate_weight_footprint_gb(
            groups,
            {group.name: 8 for group in groups},
        )

        result, manifest = run_budgeted_bf16_allocator(
            groups=groups,
            target_budget_gb=uniform_budget,
            grouping="per_block_component",
            group_value_scores={
                "block:0:mlp.up_proj": 1.0,
                "block:1:mlp.gate_proj": 0.8,
                "block:2:linear_attn.in_proj_b": 0.05,
                "block:3:linear_attn.in_proj_a": 0.05,
            },
            bf16_candidate_fraction=0.5,
            bf16_rescue_scale=3.0,
        )

        self.assertEqual(len(result.top_candidates), 1)
        candidate_bits = result.top_candidates[0].bits_dict()
        self.assertLessEqual(
            result.top_candidates[0].estimated_weight_footprint_gb,
            uniform_budget,
        )
        self.assertEqual(candidate_bits["block:0:mlp.up_proj"], 16)
        self.assertEqual(
            sum(1 for group_name in ("block:2:linear_attn.in_proj_b", "block:3:linear_attn.in_proj_a") if candidate_bits[group_name] == 4),
            2,
        )
        self.assertIn("block:0:mlp.up_proj", manifest["bf16_group_names"])

    def test_budgeted_bf16_allocator_reduces_bf16_when_budget_gets_tighter(self) -> None:
        custom_layer_stats = [
            LinearLayerStat(
                name="model.layers.0.mlp.up_proj",
                parameter_count=1_000_000,
                in_features=1024,
                out_features=1024,
            ),
            LinearLayerStat(
                name="model.layers.1.mlp.gate_proj",
                parameter_count=1_000_000,
                in_features=1024,
                out_features=1024,
            ),
            LinearLayerStat(
                name="model.layers.2.linear_attn.in_proj_b",
                parameter_count=1_000_000,
                in_features=1024,
                out_features=1024,
            ),
            LinearLayerStat(
                name="model.layers.3.linear_attn.in_proj_a",
                parameter_count=1_000_000,
                in_features=1024,
                out_features=1024,
            ),
        ]
        groups = build_search_groups(custom_layer_stats, grouping="per_block_component")
        loose_budget = estimate_candidate_weight_footprint_gb(
            groups,
            {group.name: 8 for group in groups},
        )
        tight_budget = estimate_candidate_weight_footprint_gb(
            groups,
            {
                "block:0:mlp.up_proj": 8,
                "block:1:mlp.gate_proj": 8,
                "block:2:linear_attn.in_proj_b": 4,
                "block:3:linear_attn.in_proj_a": 4,
            },
        )
        allocator_kwargs = {
            "groups": groups,
            "group_value_scores": {
                "block:0:mlp.up_proj": 1.0,
                "block:1:mlp.gate_proj": 0.8,
                "block:2:linear_attn.in_proj_b": 0.05,
                "block:3:linear_attn.in_proj_a": 0.05,
            },
            "grouping": "per_block_component",
            "bf16_candidate_fraction": 0.5,
            "bf16_rescue_scale": 3.0,
        }

        loose_result, loose_manifest = run_budgeted_bf16_allocator(
            target_budget_gb=loose_budget,
            **allocator_kwargs,
        )
        tight_result, tight_manifest = run_budgeted_bf16_allocator(
            target_budget_gb=tight_budget,
            **allocator_kwargs,
        )

        self.assertGreaterEqual(
            len(loose_manifest["bf16_group_names"]),
            len(tight_manifest["bf16_group_names"]),
        )
        self.assertLessEqual(
            tight_result.top_candidates[0].estimated_weight_footprint_gb,
            tight_budget,
        )

    def test_expand_group_assignments_projects_coarse_bits_to_fine_groups(self) -> None:
        fine_groups = build_search_groups(self.layer_stats, grouping="per_block_component")
        expanded = expand_group_assignments(
            {
                "window:0-3:mlp.down_proj": 16,
                "window:0-3:mlp.up_proj": 8,
                "window:0-3:linear_attn.out_proj": 8,
                "window:0-3:linear_attn.in_proj_b": 4,
            },
            target_groups=fine_groups,
            source_grouping="per_block_window_component",
        )
        self.assertEqual(expanded["block:0:mlp.down_proj"], 16)
        self.assertEqual(expanded["block:1:mlp.down_proj"], 16)
        self.assertEqual(expanded["block:0:linear_attn.in_proj_b"], 4)

    def test_constrained_surrogate_search_keeps_frozen_groups_fixed(self) -> None:
        groups = build_search_groups(self.layer_stats, grouping="per_block_component")
        target_budget = estimate_candidate_weight_footprint_gb(
            groups,
            {group.name: 8 for group in groups},
        )
        fixed_assignments = {
            group.name: 4
            for group in groups
            if group.name != "block:0:mlp.down_proj"
        }
        active_groups = [group for group in groups if group.name == "block:0:mlp.down_proj"]

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
            population_size=8,
            generations=2,
            elite_count=2,
            seed=11,
            search_groups=active_groups,
            fixed_assignments=fixed_assignments,
            extra_seed_assignments=[
                (
                    "coarse_projection",
                    {
                        "block:0:mlp.down_proj": 8,
                        "block:0:mlp.up_proj": 4,
                        "block:0:linear_attn.out_proj": 4,
                        "block:0:linear_attn.in_proj_b": 4,
                        "block:1:mlp.down_proj": 4,
                        "block:1:mlp.up_proj": 4,
                    },
                )
            ],
        )
        self.assertGreaterEqual(len(result.top_candidates), 1)
        frozen_names = set(fixed_assignments)
        for candidate in result.top_candidates:
            bits = dict(candidate.group_bits)
            for group_name in frozen_names:
                self.assertEqual(bits[group_name], 4)

    def test_constrained_proxy_search_keeps_frozen_groups_fixed(self) -> None:
        groups = build_search_groups(self.layer_stats, grouping="per_block_component")
        target_budget = estimate_candidate_weight_footprint_gb(
            groups,
            {group.name: 8 for group in groups},
        )
        fixed_assignments = {
            group.name: 4
            for group in groups
            if group.name != "block:0:mlp.down_proj"
        }
        active_groups = [group for group in groups if group.name == "block:0:mlp.down_proj"]

        result = run_proxy_evolution_search(
            groups=groups,
            target_budget_gb=target_budget,
            allowed_bits=(4, 8),
            grouping="per_block_component",
            population_size=8,
            generations=2,
            elite_count=2,
            seed=13,
            search_groups=active_groups,
            fixed_assignments=fixed_assignments,
            extra_seed_assignments=[
                (
                    "coarse_projection",
                    {
                        "block:0:mlp.down_proj": 8,
                        "block:0:mlp.up_proj": 4,
                        "block:0:linear_attn.out_proj": 4,
                        "block:0:linear_attn.in_proj_b": 4,
                        "block:1:mlp.down_proj": 4,
                        "block:1:mlp.up_proj": 4,
                    },
                )
            ],
        )

        self.assertGreaterEqual(len(result.top_candidates), 1)
        for candidate in result.top_candidates:
            bits = dict(candidate.group_bits)
            for group_name in fixed_assignments:
                self.assertEqual(bits[group_name], 4)

    def test_hierarchical_candidates_stay_within_budget(self) -> None:
        coarse_groups = build_search_groups(self.layer_stats, grouping="per_block_window_component")
        fine_groups = build_search_groups(self.layer_stats, grouping="per_block_component")
        target_budget = estimate_candidate_weight_footprint_gb(
            fine_groups,
            default_seed_assignments(fine_groups),
        )
        coarse_result = run_proxy_evolution_search(
            groups=coarse_groups,
            target_budget_gb=target_budget,
            allowed_bits=(4, 8, 16),
            grouping="per_block_window_component",
            population_size=10,
            generations=2,
            elite_count=2,
            top_k=4,
            seed=17,
        )
        manifest = build_hierarchical_promotion_manifest(
            coarse_groups=coarse_groups,
            coarse_candidates=list(coarse_result.top_candidates),
            fine_groups=fine_groups,
            coarse_group_value_scores={
                "window:0-3:mlp.down_proj": 0.08,
                "window:0-3:mlp.up_proj": 0.03,
            },
            source_grouping="per_block_window_component",
            max_promoted_fine_groups=2,
        )
        expanded_best = expand_group_assignments(
            coarse_result.top_candidates[0].bits_dict(),
            target_groups=fine_groups,
            source_grouping="per_block_window_component",
        )
        active_groups = [
            group
            for group in fine_groups
            if group.name in set(manifest["promoted_fine_group_names"])
        ]
        fixed_assignments = {
            group_name: expanded_best[group_name]
            for group_name in manifest["frozen_fine_group_names"]
        }
        fine_result = run_surrogate_evolution_search(
            groups=fine_groups,
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
            allowed_bits=(4, 8, 16),
            grouping="per_block_component",
            population_size=8,
            generations=2,
            elite_count=2,
            seed=23,
            search_groups=active_groups,
            fixed_assignments=fixed_assignments,
            extra_seed_assignments=[("expanded_best", expanded_best)],
        )
        self.assertGreaterEqual(len(fine_result.top_candidates), 1)
        for candidate in fine_result.top_candidates:
            self.assertLessEqual(candidate.estimated_weight_footprint_gb, target_budget)

    def test_quantization_config_refinement_preserves_bits_and_footprint(self) -> None:
        groups = build_search_groups(self.layer_stats, grouping="per_block_component")
        assignments = default_seed_assignments(groups)
        target_budget = estimate_candidate_weight_footprint_gb(groups, assignments)
        base_candidate = build_surrogate_candidate(
            groups=groups,
            assignments=assignments,
            surrogate_predictor=lambda _features: (0.40, 0.01),
            target_budget_gb=target_budget,
            allowed_bits=(4, 8),
            provenance="unit_test_base",
            uncertainty_penalty=0.5,
            group_value_scores={"block:0:mlp.down_proj": 0.04},
            reference_accuracy=0.39,
        )

        refinement = refine_candidate_quantization_configs(
            groups=groups,
            layer_stats=self.layer_stats,
            base_candidates=[base_candidate],
            group_value_scores={"block:0:mlp.down_proj": 0.04},
            allowed_group_names={group.name for group in groups},
            group_size_options=(64, 128),
            symmetric_options=(True, False),
            max_tunable_groups=2,
            population_size=8,
            generations=2,
            top_k=2,
            seed=7,
            seed_candidate_count=1,
        )

        self.assertGreaterEqual(len(refinement["top_candidates"]), 1)
        top_candidate = refinement["top_candidates"][0]
        self.assertEqual(top_candidate["group_bits"], dict(base_candidate.group_bits))
        self.assertEqual(
            top_candidate["estimated_weight_footprint_gb"],
            base_candidate.estimated_weight_footprint_gb,
        )
        for override in top_candidate["group_quantization_overrides"].values():
            self.assertIn(int(override["group_size"]), (64, 128))
            self.assertIn(bool(override["symmetric"]), (True, False))

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
