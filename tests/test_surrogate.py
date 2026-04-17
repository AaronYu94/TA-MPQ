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

from ta_mpq.surrogate import (
    _apply_surrogate_calibration,
    _pairwise_ranking_accuracy,
    build_ablation_adjusted_group_value_prior,
    build_group_value_prior_from_dataset,
    build_surrogate_dataset_from_manifest,
    predict_surrogate_distribution,
    predict_surrogate_target,
    resolve_manifest_paths,
    train_surrogate_model,
)


class SurrogateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.report_payload = {
            "model_id": "unit-test-model",
            "estimated_average_bit_width": 5.0,
            "estimated_weight_footprint_gb": 1.25,
            "rule_hits": [{"matched": True}, {"matched": True}],
            "layer_stats": [
                {
                    "name": "model.layers.0.linear_attn.out_proj",
                    "parameter_count": 200,
                    "in_features": 10,
                    "out_features": 20,
                },
                {
                    "name": "model.layers.0.mlp.down_proj",
                    "parameter_count": 300,
                    "in_features": 15,
                    "out_features": 20,
                },
                {
                    "name": "model.layers.1.linear_attn.out_proj",
                    "parameter_count": 200,
                    "in_features": 10,
                    "out_features": 20,
                },
                {
                    "name": "model.layers.1.mlp.down_proj",
                    "parameter_count": 300,
                    "in_features": 15,
                    "out_features": 20,
                },
            ],
        }
        self.candidate_payload = {
            "group_bit_assignments": {
                "block:0:linear_attn.out_proj": 8,
                "block:0:mlp.down_proj": 8,
                "block:1:linear_attn.out_proj": 4,
                "block:1:mlp.down_proj": 4,
            }
        }
        self.sensitivity_payload = {
            "groups": [
                {
                    "name": "block:0:linear_attn.out_proj",
                    "combined_sensitivity": 0.9,
                },
                {
                    "name": "block:0:mlp.down_proj",
                    "combined_sensitivity": 1.0,
                },
                {
                    "name": "block:1:linear_attn.out_proj",
                    "combined_sensitivity": 0.6,
                },
                {
                    "name": "block:1:mlp.down_proj",
                    "combined_sensitivity": 0.8,
                },
            ]
        }
        self.eval_payload = {
            "task_name": "math500",
            "accuracy": 0.44,
            "mean_latency_sec": 0.83,
            "mean_total_peak_memory_mb": 1234.0,
            "results": [],
        }

    def test_resolve_manifest_paths_makes_relative_paths_absolute(self) -> None:
        payload = {
            "task_name": "math500",
            "grouping": "per_block_component",
            "records": [
                {
                    "policy_id": "candidate-a",
                    "task_name": "math500",
                    "candidate_path": "outputs/policies/a.json",
                    "report_path": "outputs/reports/a.json",
                    "evaluation_path": "outputs/evals/a.json",
                    "sensitivity_profile_path": "outputs/sensitivity/a.json",
                }
            ],
        }

        resolved = resolve_manifest_paths(payload, base_dir=PROJECT_ROOT)
        record = resolved["records"][0]
        self.assertTrue(Path(record["candidate_path"]).is_absolute())
        self.assertTrue(Path(record["report_path"]).is_absolute())
        self.assertTrue(Path(record["evaluation_path"]).is_absolute())
        self.assertTrue(Path(record["sensitivity_profile_path"]).is_absolute())

    def test_build_surrogate_dataset_from_manifest_extracts_features(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            report_path = temp_root / "report.json"
            candidate_path = temp_root / "candidate.json"
            sensitivity_path = temp_root / "sensitivity.json"
            evaluation_path = temp_root / "candidate-a-quantized.json"
            native_evaluation_path = temp_root / "candidate-a-native.json"
            report_path.write_text(json.dumps(self.report_payload), encoding="utf-8")
            candidate_path.write_text(json.dumps(self.candidate_payload), encoding="utf-8")
            sensitivity_path.write_text(json.dumps(self.sensitivity_payload), encoding="utf-8")
            evaluation_path.write_text(json.dumps(self.eval_payload), encoding="utf-8")
            native_evaluation_path.write_text(
                json.dumps(
                    {
                        **self.eval_payload,
                        "accuracy": 0.31,
                    }
                ),
                encoding="utf-8",
            )

            uniform8_evaluation_path = temp_root / "uniform-8-quantized.json"
            uniform8_native_path = temp_root / "uniform-8-native.json"
            uniform8_evaluation_path.write_text(
                json.dumps(
                    {
                        **self.eval_payload,
                        "accuracy": 0.51,
                    }
                ),
                encoding="utf-8",
            )
            uniform8_native_path.write_text(
                json.dumps(
                    {
                        **self.eval_payload,
                        "accuracy": 0.31,
                    }
                ),
                encoding="utf-8",
            )

            manifest = {
                "task_name": "math500",
                "grouping": "per_block_component",
                "records": [
                    {
                        "policy_id": "candidate-a",
                        "task_name": "math500",
                        "candidate_path": str(candidate_path),
                        "report_path": str(report_path),
                        "evaluation_path": str(evaluation_path),
                        "sensitivity_profile_path": str(sensitivity_path),
                    },
                    {
                        "policy_id": "uniform-4bit",
                        "task_name": "math500",
                        "uniform_bit_width": 4,
                        "report_path": str(report_path),
                        "evaluation_path": str(evaluation_path),
                        "sensitivity_profile_path": str(sensitivity_path),
                    },
                    {
                        "policy_id": "uniform-8bit",
                        "task_name": "math500",
                        "uniform_bit_width": 8,
                        "report_path": str(report_path),
                        "evaluation_path": str(uniform8_evaluation_path),
                        "sensitivity_profile_path": str(sensitivity_path),
                    },
                ],
            }

            dataset = build_surrogate_dataset_from_manifest(
                manifest,
                uniform_baseline_bit_width=8,
            )
            self.assertEqual(dataset.task_name, "math500")
            self.assertEqual(len(dataset.examples), 3)
            self.assertEqual(dataset.to_dict()["num_features"], len(dataset.feature_names))
            self.assertIn("estimated_average_bit_width", dataset.feature_names)
            self.assertIn("mlp_down_proj_8bit_fraction", dataset.feature_names)
            self.assertAlmostEqual(dataset.baseline_metrics["uniform_accuracy"], 0.51)
            self.assertAlmostEqual(dataset.baseline_metrics["native_accuracy"], 0.31)
            self.assertAlmostEqual(dataset.baseline_metrics["uniform_baseline_bit_width"], 8.0)
            first = dataset.examples[0]
            self.assertEqual(first.policy_id, "candidate-a")
            self.assertAlmostEqual(first.target_values["accuracy"], 0.44)
            self.assertAlmostEqual(first.target_values["accuracy_advantage_over_native"], 0.13)
            self.assertAlmostEqual(first.target_values["accuracy_advantage_over_uniform"], -0.07)
            self.assertAlmostEqual(
                first.target_values["accuracy_advantage_over_best_baseline"],
                -0.07,
            )
            self.assertIn(
                "block:0:mlp.down_proj",
                first.metadata["group_bit_assignments"],
            )
            self.assertGreater(first.feature_values["estimated_average_bit_width"], 0.0)

    def test_build_ablation_adjusted_group_value_prior_reweights_scores(self) -> None:
        base_prior = {
            "group_scores": {
                "block:0:linear_attn.out_proj": {"score": 0.08},
                "block:0:mlp.down_proj": {"score": 0.04},
                "block:1:linear_attn.out_proj": {"score": 0.03},
            },
            "component_scores": {},
        }
        profiles = [
            {
                "groups": [
                    {
                        "name": "block:0:linear_attn.out_proj",
                        "reference_bit_width": 8,
                        "ablated_bit_width": 4,
                        "accuracy_drop": 0.0,
                        "positive_accuracy_drop": 0.0,
                    },
                    {
                        "name": "block:0:mlp.down_proj",
                        "reference_bit_width": 16,
                        "ablated_bit_width": 8,
                        "accuracy_drop": 0.0,
                        "positive_accuracy_drop": 0.0,
                    },
                    {
                        "name": "block:1:linear_attn.out_proj",
                        "reference_bit_width": 16,
                        "ablated_bit_width": 8,
                        "accuracy_drop": 0.06,
                        "positive_accuracy_drop": 0.06,
                    },
                ]
            }
        ]

        adjusted = build_ablation_adjusted_group_value_prior(
            base_prior_payload=base_prior,
            ablation_profile_payloads=profiles,
        )

        self.assertLess(adjusted["group_scores"]["block:0:linear_attn.out_proj"]["score"], 0.0)
        self.assertLessEqual(
            adjusted["group_scores"]["block:0:mlp.down_proj"]["score"],
            0.01,
        )
        self.assertGreaterEqual(
            adjusted["group_scores"]["block:1:linear_attn.out_proj"]["score"],
            0.06,
        )
        self.assertEqual(len(adjusted["ablation_adjustments"]), 3)

    def test_build_group_value_prior_from_dataset_summarizes_4bit_vs_8bit_uplift(self) -> None:
        dataset_payload = {
            "task_name": "math500",
            "baseline_metrics": {
                "uniform_accuracy": 0.39,
                "native_accuracy": 0.29,
            },
            "examples": [
                {
                    "policy_id": "a",
                    "target_values": {"accuracy_advantage_over_best_baseline": 0.02},
                    "metadata": {
                        "group_bit_assignments": {
                            "block:0:mlp.down_proj": 8,
                            "block:0:linear_attn.out_proj": 4,
                        }
                    },
                },
                {
                    "policy_id": "b",
                    "target_values": {"accuracy_advantage_over_best_baseline": 0.01},
                    "metadata": {
                        "group_bit_assignments": {
                            "block:0:mlp.down_proj": 8,
                            "block:0:linear_attn.out_proj": 4,
                        }
                    },
                },
                {
                    "policy_id": "c",
                    "target_values": {"accuracy_advantage_over_best_baseline": -0.02},
                    "metadata": {
                        "group_bit_assignments": {
                            "block:0:mlp.down_proj": 4,
                            "block:0:linear_attn.out_proj": 8,
                        }
                    },
                },
                {
                    "policy_id": "d",
                    "target_values": {"accuracy_advantage_over_best_baseline": -0.01},
                    "metadata": {
                        "group_bit_assignments": {
                            "block:0:mlp.down_proj": 4,
                            "block:0:linear_attn.out_proj": 8,
                        }
                    },
                },
            ],
        }

        prior = build_group_value_prior_from_dataset(dataset_payload, min_support=2)
        self.assertEqual(prior["task_name"], "math500")
        self.assertGreater(prior["group_scores"]["block:0:mlp.down_proj"]["score"], 0.0)
        self.assertLess(prior["group_scores"]["block:0:linear_attn.out_proj"]["score"], 0.0)
        self.assertIn("mlp.down_proj", prior["component_scores"])

    def test_build_surrogate_dataset_requires_requested_uniform_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            report_path = temp_root / "report.json"
            candidate_path = temp_root / "candidate.json"
            sensitivity_path = temp_root / "sensitivity.json"
            evaluation_path = temp_root / "candidate-a-quantized.json"
            native_evaluation_path = temp_root / "candidate-a-native.json"
            report_path.write_text(json.dumps(self.report_payload), encoding="utf-8")
            candidate_path.write_text(json.dumps(self.candidate_payload), encoding="utf-8")
            sensitivity_path.write_text(json.dumps(self.sensitivity_payload), encoding="utf-8")
            evaluation_path.write_text(json.dumps(self.eval_payload), encoding="utf-8")
            native_evaluation_path.write_text(
                json.dumps({**self.eval_payload, "accuracy": 0.31}),
                encoding="utf-8",
            )

            manifest = {
                "task_name": "math500",
                "grouping": "per_block_component",
                "records": [
                    {
                        "policy_id": "candidate-a",
                        "task_name": "math500",
                        "candidate_path": str(candidate_path),
                        "report_path": str(report_path),
                        "evaluation_path": str(evaluation_path),
                        "sensitivity_profile_path": str(sensitivity_path),
                    }
                ],
            }

            with self.assertRaises(ValueError):
                build_surrogate_dataset_from_manifest(
                    manifest,
                    target_metric="accuracy_advantage_over_uniform",
                    uniform_baseline_bit_width=8,
                )

    def test_train_surrogate_model_falls_back_without_xgboost(self) -> None:
        dataset_payload = {
            "task_name": "math500",
            "feature_names": ["feature_a", "feature_b"],
            "examples": [
                {
                    "policy_id": "candidate-a",
                    "feature_values": {"feature_a": 1.0, "feature_b": 2.0},
                    "target_values": {"accuracy": 0.4},
                },
                {
                    "policy_id": "candidate-b",
                    "feature_values": {"feature_a": 2.0, "feature_b": 3.0},
                    "target_values": {"accuracy": 0.6},
                },
            ],
        }

        result = train_surrogate_model(dataset_payload, target_metric="accuracy")
        self.assertEqual(result["backend"], "mean_baseline")
        self.assertEqual(result["num_examples"], 2)
        self.assertIsNone(result["validation_mae"])
        self.assertIsNone(result["validation_pairwise_accuracy"])
        self.assertEqual(result["calibration_weight"], 0.0)
        self.assertEqual(result["uncertainty_floor"], 0.0)
        self.assertEqual(len(result["predictions"]), 2)

    def test_predict_surrogate_target_supports_mean_baseline_summary(self) -> None:
        summary = {
            "backend": "mean_baseline",
            "feature_names": ["feature_a"],
            "predictions": [
                {"policy_id": "a", "prediction": 0.25},
                {"policy_id": "b", "prediction": 0.35},
            ],
        }
        prediction = predict_surrogate_target(
            {"feature_a": 10.0},
            surrogate_summary_payload=summary,
            model_json="",
        )
        self.assertAlmostEqual(prediction, 0.30)

    def test_predict_surrogate_distribution_supports_mean_baseline_summary(self) -> None:
        summary = {
            "backend": "mean_baseline",
            "feature_names": ["feature_a"],
            "predictions": [
                {"policy_id": "a", "prediction": 0.25},
                {"policy_id": "b", "prediction": 0.35},
            ],
        }
        mean_prediction, prediction_std = predict_surrogate_distribution(
            {"feature_a": 10.0},
            surrogate_summary_payload=summary,
            model_json="",
        )
        self.assertAlmostEqual(mean_prediction, 0.30)
        self.assertEqual(prediction_std, 0.0)

    def test_apply_surrogate_calibration_shrinks_toward_anchor(self) -> None:
        summary = {
            "target_mean": 0.40,
            "calibration_weight": 0.25,
            "uncertainty_floor": 0.03,
        }
        calibrated_mean, calibrated_std = _apply_surrogate_calibration(
            prediction_mean=0.60,
            prediction_std=0.04,
            surrogate_summary_payload=summary,
        )
        self.assertAlmostEqual(calibrated_mean, 0.45)
        self.assertGreater(calibrated_std, 0.04)

    def test_pairwise_ranking_accuracy_handles_ties(self) -> None:
        score = _pairwise_ranking_accuracy(
            targets=[0.8, 0.6, 0.2],
            predictions=[0.4, 0.4, 0.1],
        )
        self.assertGreater(score, 0.5)
        self.assertLess(score, 1.0)


if __name__ == "__main__":
    unittest.main()
