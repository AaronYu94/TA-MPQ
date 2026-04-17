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

from ta_mpq.closed_loop import (
    append_record_if_novel,
    artifact_dir_from_record,
    best_record_by_accuracy,
    build_executed_record,
    estimate_candidate_novelty,
    normalized_policy_distance,
    select_novel_candidates,
)


class ClosedLoopTests(unittest.TestCase):
    def test_select_novel_candidates_filters_existing_policy(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            candidate_a = root / "candidate-a.json"
            candidate_b = root / "candidate-b.json"
            payload_a = {
                "backend_projections": {
                    "llmcompressor": {
                        "projected_module_bit_assignments": {
                            "layer.a": 4,
                            "layer.b": 8,
                        }
                    }
                },
                "module_bit_assignments": {
                    "layer.a": 4,
                    "layer.b": 8,
                },
            }
            payload_b = {
                "backend_projections": {
                    "llmcompressor": {
                        "projected_module_bit_assignments": {
                            "layer.a": 4,
                            "layer.b": 4,
                        }
                    }
                },
                "module_bit_assignments": {
                    "layer.a": 4,
                    "layer.b": 4,
                },
            }
            candidate_a.write_text(json.dumps(payload_a), encoding="utf-8")
            candidate_b.write_text(json.dumps(payload_b), encoding="utf-8")

            executed_manifest = {
                "task_name": "math500",
                "records": [
                    {
                        "policy_id": "existing-a",
                        "task_name": "math500",
                        "candidate_path": str(candidate_a),
                        "report_path": "report.json",
                        "evaluation_path": "eval.json",
                        "sensitivity_profile_path": "sensitivity.json",
                        "sensitivity_field": "combined_sensitivity",
                        "provenance": "surrogate_guided",
                    }
                ],
            }
            candidate_manifest = {
                "candidates": [
                    {"rank": 1, "path": str(candidate_a), "fitness": 0.9, "estimated_average_bit_width": 5.0, "estimated_weight_footprint_gb": 1.0},
                    {"rank": 2, "path": str(candidate_b), "fitness": 0.8, "estimated_average_bit_width": 4.0, "estimated_weight_footprint_gb": 0.8},
                ]
            }

            selected = select_novel_candidates(
                executed_manifest_payload=executed_manifest,
                candidate_manifest_payload=candidate_manifest,
                base_dir=root,
                limit=2,
            )

            self.assertEqual(len(selected), 1)
            self.assertEqual(Path(selected[0]["path"]), candidate_b)

    def test_select_novel_candidates_uses_uncertainty_and_diversity_acquisition(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            candidate_a = root / "candidate-a.json"
            candidate_b = root / "candidate-b.json"
            candidate_c = root / "candidate-c.json"

            candidate_a.write_text(
                json.dumps(
                    {
                        "backend_projections": {
                            "llmcompressor": {
                                "projected_module_bit_assignments": {
                                    "layer.a": 4,
                                    "layer.b": 4,
                                }
                            }
                        },
                        "module_bit_assignments": {
                            "layer.a": 4,
                            "layer.b": 4,
                        },
                    }
                ),
                encoding="utf-8",
            )
            candidate_b.write_text(
                json.dumps(
                    {
                        "backend_projections": {
                            "llmcompressor": {
                                "projected_module_bit_assignments": {
                                    "layer.a": 4,
                                    "layer.b": 8,
                                }
                            }
                        },
                        "module_bit_assignments": {
                            "layer.a": 4,
                            "layer.b": 8,
                        },
                    }
                ),
                encoding="utf-8",
            )
            candidate_c.write_text(
                json.dumps(
                    {
                        "backend_projections": {
                            "llmcompressor": {
                                "projected_module_bit_assignments": {
                                    "layer.a": 8,
                                    "layer.b": 8,
                                }
                            }
                        },
                        "module_bit_assignments": {
                            "layer.a": 8,
                            "layer.b": 8,
                        },
                    }
                ),
                encoding="utf-8",
            )

            executed_manifest = {
                "task_name": "math500",
                "records": [
                    {
                        "policy_id": "existing-a",
                        "task_name": "math500",
                        "candidate_path": str(candidate_a),
                        "report_path": "report.json",
                        "evaluation_path": "eval.json",
                        "sensitivity_profile_path": "sensitivity.json",
                        "sensitivity_field": "combined_sensitivity",
                        "provenance": "surrogate_guided_closed_loop",
                    }
                ],
            }
            candidate_manifest = {
                "candidates": [
                    {
                        "rank": 1,
                        "path": str(candidate_b),
                        "fitness": 0.90,
                        "conservative_prediction": 0.36,
                        "prediction_uncertainty": 0.01,
                        "estimated_average_bit_width": 6.0,
                        "estimated_weight_footprint_gb": 1.0,
                    },
                    {
                        "rank": 2,
                        "path": str(candidate_c),
                        "fitness": 0.88,
                        "conservative_prediction": 0.34,
                        "prediction_uncertainty": 0.08,
                        "estimated_average_bit_width": 7.0,
                        "estimated_weight_footprint_gb": 1.1,
                    },
                ]
            }

            selected = select_novel_candidates(
                executed_manifest_payload=executed_manifest,
                candidate_manifest_payload=candidate_manifest,
                base_dir=root,
                limit=1,
            )

            self.assertEqual(len(selected), 1)
            self.assertEqual(Path(selected[0]["path"]), candidate_c)
            self.assertGreater(selected[0]["selection_acquisition_score"], 0.0)
            self.assertAlmostEqual(selected[0]["selection_novelty_score"], 1.0)

    def test_append_record_if_novel_skips_duplicate_signature(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            candidate = root / "candidate.json"
            candidate.write_text(
                json.dumps(
                    {
                        "backend_projections": {
                            "llmcompressor": {
                                "projected_module_bit_assignments": {"layer.a": 4}
                            }
                        },
                        "module_bit_assignments": {"layer.a": 4},
                    }
                ),
                encoding="utf-8",
            )
            manifest = {"task_name": "math500", "records": []}
            record = build_executed_record(
                policy_id="candidate-1",
                task_name="math500",
                candidate_path=str(candidate),
                report_path="report.json",
                evaluation_path="eval.json",
                sensitivity_profile_path="sensitivity.json",
                sensitivity_field="combined_sensitivity",
                provenance="surrogate_guided_closed_loop",
            )
            manifest = append_record_if_novel(manifest, record, base_dir=root)
            manifest = append_record_if_novel(manifest, record, base_dir=root)
            self.assertEqual(len(manifest["records"]), 1)

    def test_best_record_by_accuracy_prefers_highest_accuracy(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            eval_a = root / "eval-a.json"
            eval_b = root / "eval-b.json"
            eval_a.write_text(json.dumps({"accuracy": 0.31}), encoding="utf-8")
            eval_b.write_text(json.dumps({"accuracy": 0.42}), encoding="utf-8")
            manifest = {
                "task_name": "math500",
                "records": [
                    {
                        "policy_id": "a",
                        "task_name": "math500",
                        "evaluation_path": str(eval_a),
                        "provenance": "surrogate_guided_closed_loop",
                    },
                    {
                        "policy_id": "b",
                        "task_name": "math500",
                        "evaluation_path": str(eval_b),
                        "provenance": "surrogate_guided_closed_loop",
                    },
                ],
            }
            best = best_record_by_accuracy(
                manifest,
                base_dir=root,
                task_name="math500",
                provenance_prefix="surrogate_guided_closed_loop",
            )
            self.assertIsNotNone(best)
            self.assertEqual(best["policy_id"], "b")
            self.assertAlmostEqual(best["accuracy"], 0.42)

    def test_artifact_dir_from_record_reads_report_output_dir(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            report_path = root / "report.json"
            report_path.write_text(
                json.dumps({"output_dir": "/artifacts/policy-a"}),
                encoding="utf-8",
            )
            record = {
                "policy_id": "policy-a",
                "report_path": str(report_path),
            }
            artifact_dir = artifact_dir_from_record(record, base_dir=root)
            self.assertEqual(artifact_dir, "/artifacts/policy-a")

    def test_normalized_policy_distance_and_novelty(self) -> None:
        assignments_a = {"layer.a": 4, "layer.b": 4}
        assignments_b = {"layer.a": 4, "layer.b": 8}
        assignments_c = {"layer.a": 8, "layer.b": 8}

        self.assertAlmostEqual(normalized_policy_distance(assignments_a, assignments_b), 0.5)
        self.assertAlmostEqual(normalized_policy_distance(assignments_a, assignments_c), 1.0)
        novelty = estimate_candidate_novelty(assignments_b, [assignments_a, assignments_c])
        self.assertAlmostEqual(novelty, 0.5)


if __name__ == "__main__":
    unittest.main()
