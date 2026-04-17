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

from ta_mpq.policy_export import (
    build_backend_projection,
    build_project_policy,
    export_top_candidates,
    expand_group_bits_to_module_assignments,
    load_policy_from_candidate,
)


class PolicyExportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.report_payload = {
            "contract_name": "unit-test-contract",
            "model_id": "unit-test-model",
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
                    "name": "model.layers.1.mlp.up_proj",
                    "parameter_count": 300,
                    "in_features": 15,
                    "out_features": 20,
                },
            ],
        }
        self.search_payload = {
            "grouping": "per_block_component",
            "top_candidates": [
                {
                    "group_bits": {
                        "block:0:linear_attn.out_proj": 8,
                        "block:0:mlp.down_proj": 3,
                        "block:1:linear_attn.out_proj": 4,
                        "block:1:mlp.up_proj": 2,
                    },
                    "fitness": 0.91,
                    "proxy_quality_score": 0.87,
                    "conservative_prediction": 0.42,
                    "budget_alignment_score": 0.96,
                    "reference_accuracy": 0.39,
                    "reference_advantage_score": 0.03,
                    "prediction_uncertainty": 0.07,
                    "estimated_average_bit_width": 4.3,
                    "estimated_weight_footprint_gb": 0.000001,
                    "provenance": "unit-test",
                }
            ],
        }

    def test_expand_group_bits_to_module_assignments(self) -> None:
        module_assignments = expand_group_bits_to_module_assignments(
            report_payload=self.report_payload,
            grouping="per_block_component",
            group_bits=self.search_payload["top_candidates"][0]["group_bits"],
        )

        self.assertEqual(module_assignments["model.layers.0.linear_attn.out_proj"], 8)
        self.assertEqual(module_assignments["model.layers.0.mlp.down_proj"], 3)
        self.assertEqual(module_assignments["model.layers.1.mlp.up_proj"], 2)

    def test_build_backend_projection_downgrades_unsupported_bits(self) -> None:
        projection = build_backend_projection(
            module_assignments={
                "model.layers.0.linear_attn.out_proj": 8,
                "model.layers.0.mlp.down_proj": 3,
                "model.layers.1.mlp.up_proj": 2,
            },
            backend="llmcompressor",
        )

        self.assertEqual(projection["projected_bit_counts"], {"4": 2, "8": 1})
        self.assertEqual(projection["downgraded_module_count"], 2)
        self.assertEqual(
            projection["projected_module_bit_assignments"]["model.layers.0.mlp.down_proj"],
            4,
        )

    def test_build_backend_projection_preserves_16bit_as_ignore(self) -> None:
        projection = build_backend_projection(
            module_assignments={
                "model.layers.0.linear_attn.out_proj": 8,
                "model.layers.0.mlp.down_proj": 16,
                "model.layers.1.mlp.up_proj": 4,
            },
            backend="llmcompressor",
        )

        self.assertEqual(
            projection["projected_module_bit_assignments"]["model.layers.0.mlp.down_proj"],
            16,
        )
        self.assertIn("model.layers.0.mlp.down_proj", projection["projected_policy"]["ignore"])

    def test_build_project_policy_uses_most_common_bit_as_default(self) -> None:
        policy = build_project_policy(
            {
                "model.layers.0.linear_attn.out_proj": 4,
                "model.layers.0.mlp.down_proj": 8,
                "model.layers.1.linear_attn.out_proj": 4,
            }
        )

        self.assertEqual(policy.default_bit_width, 4)
        self.assertEqual(len(policy.rules), 1)
        self.assertEqual(policy.rules[0].bit_width, 8)
        self.assertEqual(policy.rules[0].targets, ("model.layers.0.mlp.down_proj",))

    def test_build_project_policy_maps_16bit_modules_to_ignore(self) -> None:
        policy = build_project_policy(
            {
                "model.layers.0.linear_attn.out_proj": 4,
                "model.layers.0.mlp.down_proj": 16,
                "model.layers.1.linear_attn.out_proj": 8,
            }
        )

        self.assertEqual(policy.default_bit_width, 4)
        self.assertIn("model.layers.0.mlp.down_proj", policy.ignore)
        self.assertEqual(len(policy.rules), 1)
        self.assertEqual(policy.rules[0].bit_width, 8)
        self.assertEqual(policy.rules[0].targets, ("model.layers.1.linear_attn.out_proj",))

    def test_export_top_candidates_writes_manifest_and_candidate_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            report_path = temp_root / "report.json"
            search_path = temp_root / "search.json"
            output_dir = temp_root / "exported"
            report_path.write_text(json.dumps(self.report_payload), encoding="utf-8")
            search_path.write_text(json.dumps(self.search_payload), encoding="utf-8")

            manifest = export_top_candidates(
                report_path=report_path,
                search_result_path=search_path,
                output_dir=output_dir,
                top_k=1,
            )

            self.assertEqual(manifest["num_exported_candidates"], 1)
            self.assertTrue((output_dir / "manifest.json").exists())
            self.assertTrue((output_dir / "candidate-01.json").exists())

            candidate_payload = json.loads((output_dir / "candidate-01.json").read_text())
            self.assertEqual(candidate_payload["bit_counts"], {"2": 1, "3": 1, "4": 1, "8": 1})
            self.assertEqual(candidate_payload["conservative_prediction"], 0.42)
            self.assertEqual(candidate_payload["budget_alignment_score"], 0.96)
            self.assertEqual(candidate_payload["reference_accuracy"], 0.39)
            self.assertEqual(candidate_payload["reference_advantage_score"], 0.03)
            self.assertEqual(candidate_payload["prediction_uncertainty"], 0.07)
            self.assertEqual(
                candidate_payload["backend_projections"]["llmcompressor"]["projected_bit_counts"],
                {"4": 3, "8": 1},
            )
            manifest_candidate = manifest["candidates"][0]
            self.assertEqual(manifest_candidate["conservative_prediction"], 0.42)
            self.assertEqual(manifest_candidate["budget_alignment_score"], 0.96)
            self.assertEqual(manifest_candidate["reference_accuracy"], 0.39)
            self.assertEqual(manifest_candidate["reference_advantage_score"], 0.03)
            self.assertEqual(manifest_candidate["prediction_uncertainty"], 0.07)

    def test_load_policy_from_candidate_reads_project_and_backend_versions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            candidate_path = temp_root / "candidate.json"
            candidate_path.write_text(
                json.dumps(
                    {
                        "project_policy": {
                            "default_bit_width": 4,
                            "default_targets": ["Linear"],
                            "ignore": [],
                            "rules": [
                                {
                                    "name": "bit_8",
                                    "targets": ["model.layers.0.linear_attn.out_proj"],
                                    "bit_width": 8,
                                    "group_size": 128,
                                    "symmetric": True,
                                }
                            ],
                        },
                        "backend_projections": {
                            "llmcompressor": {
                                "projected_policy": {
                                    "default_bit_width": 4,
                                    "default_targets": ["Linear"],
                                    "ignore": [],
                                    "rules": [],
                                }
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            project_policy = load_policy_from_candidate(candidate_path, source="project")
            backend_policy = load_policy_from_candidate(candidate_path, source="llmcompressor")

            self.assertEqual(project_policy.default_bit_width, 4)
            self.assertEqual(len(project_policy.rules), 1)
            self.assertEqual(backend_policy.default_bit_width, 4)
            self.assertEqual(len(backend_policy.rules), 0)


if __name__ == "__main__":
    unittest.main()
