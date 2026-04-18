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

from ta_mpq.local_search import (
    build_no_surrogate_local_search_round,
    select_best_candidate_from_evaluation_manifest,
)


class LocalSearchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.report_payload = {
            "contract_name": "unit-test-contract",
            "model_id": "unit-test-model",
            "layer_stats": [
                {
                    "name": "model.layers.0.mlp.down_proj",
                    "parameter_count": 300,
                    "in_features": 15,
                    "out_features": 20,
                },
                {
                    "name": "model.layers.0.mlp.up_proj",
                    "parameter_count": 300,
                    "in_features": 15,
                    "out_features": 20,
                },
                {
                    "name": "model.layers.0.linear_attn.in_proj_z",
                    "parameter_count": 150,
                    "in_features": 15,
                    "out_features": 10,
                },
            ],
        }
        self.base_candidate_payload = {
            "grouping": "per_block_component",
            "group_bit_assignments": {
                "block:0:mlp.down_proj": 8,
                "block:0:mlp.up_proj": 8,
                "block:0:linear_attn.in_proj_z": 8,
            },
        }
        self.group_value_prior_payload = {
            "group_scores": {
                "block:0:mlp.down_proj": {
                    "score": 0.18,
                    "uplift_8_over_4": 0.12,
                    "uplift_16_over_8": 0.16,
                    "preferred_bit": 16,
                    "confidence": 0.9,
                },
                "block:0:mlp.up_proj": {
                    "score": -0.10,
                    "uplift_8_over_4": -0.10,
                    "uplift_16_over_8": -0.05,
                    "preferred_bit": 4,
                    "confidence": 0.8,
                },
            }
        }
        self.ablation_profiles = [
            {
                "groups": [
                    {
                        "name": "block:0:mlp.up_proj",
                        "reference_bit_width": 8,
                        "ablated_bit_width": 4,
                        "positive_accuracy_drop": 0.0,
                        "improvement_if_downgraded": 0.08,
                        "combined_sensitivity": 0.1,
                    },
                    {
                        "name": "block:0:mlp.down_proj",
                        "reference_bit_width": 16,
                        "ablated_bit_width": 8,
                        "positive_accuracy_drop": 0.10,
                        "improvement_if_downgraded": 0.0,
                        "combined_sensitivity": 0.9,
                    },
                ]
            }
        ]

    def test_build_no_surrogate_round_generates_swap_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = build_no_surrogate_local_search_round(
                report_payload=self.report_payload,
                base_candidate_payloads=[self.base_candidate_payload],
                base_candidate_paths=["outputs/policies/base-candidate.json"],
                target_budget_gb=0.00001,
                output_dir=temp_dir,
                allowed_bits=(4, 8, 16),
                beam_size=2,
                max_candidates=4,
                group_value_prior_payload=self.group_value_prior_payload,
                group_value_prior_path="outputs/surrogate/unit-prior.json",
                ablation_profile_payloads=self.ablation_profiles,
                ablation_profile_paths=["outputs/sensitivity/unit-ablation.json"],
            )

            self.assertGreaterEqual(manifest["num_exported_candidates"], 1)
            candidate_paths = [Path(item["path"]) for item in manifest["candidates"]]
            exported_payloads = [
                json.loads(candidate_path.read_text(encoding="utf-8"))
                for candidate_path in candidate_paths
            ]
            self.assertTrue(
                any(
                    payload["group_bit_assignments"]["block:0:mlp.down_proj"] == 16
                    and payload["group_bit_assignments"]["block:0:mlp.up_proj"] == 4
                    for payload in exported_payloads
                )
            )

    def test_select_best_candidate_prefers_accuracy_then_smaller_model(self) -> None:
        evaluation_manifest = {
            "candidates": [
                {
                    "candidate_path": "candidate-01.json",
                    "accuracy": 0.36,
                    "estimated_weight_footprint_gb": 23.8,
                    "proposal_score": 0.12,
                },
                {
                    "candidate_path": "candidate-02.json",
                    "accuracy": 0.36,
                    "estimated_weight_footprint_gb": 23.4,
                    "proposal_score": 0.05,
                },
                {
                    "candidate_path": "candidate-03.json",
                    "accuracy": 0.32,
                    "estimated_weight_footprint_gb": 23.1,
                    "proposal_score": 0.20,
                },
            ]
        }

        best = select_best_candidate_from_evaluation_manifest(evaluation_manifest)

        self.assertIsNotNone(best)
        self.assertEqual(best["candidate_path"], "candidate-02.json")


if __name__ == "__main__":
    unittest.main()
