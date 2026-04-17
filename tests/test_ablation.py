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

from ta_mpq.ablation import (
    build_precision_ablation_manifest,
    build_precision_ablation_profile,
)


class PrecisionAblationTests(unittest.TestCase):
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
                    "name": "model.layers.1.mlp.up_proj",
                    "parameter_count": 300,
                    "in_features": 15,
                    "out_features": 20,
                },
            ],
        }
        self.reference_candidate_payload = {
            "grouping": "per_block_component",
            "group_bit_assignments": {
                "block:0:linear_attn.out_proj": 16,
                "block:0:mlp.down_proj": 8,
                "block:1:mlp.up_proj": 8,
            },
        }
        self.ranking_profile_payload = {
            "groups": [
                {
                    "name": "block:0:linear_attn.out_proj",
                    "combined_sensitivity": 0.70,
                },
                {
                    "name": "block:0:mlp.down_proj",
                    "combined_sensitivity": 0.95,
                },
                {
                    "name": "block:1:mlp.up_proj",
                    "combined_sensitivity": 0.20,
                },
            ]
        }

    def test_build_precision_ablation_manifest_generates_step_down_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = build_precision_ablation_manifest(
                report_payload=self.report_payload,
                reference_candidate_payload=self.reference_candidate_payload,
                output_dir=temp_dir,
                allowed_bits=(4, 8, 16),
                floor_bit=4,
                max_groups=2,
                ranking_profile_payload=self.ranking_profile_payload,
            )

            self.assertEqual(manifest["num_ablations"], 2)
            first_entry = manifest["ablations"][0]
            second_entry = manifest["ablations"][1]

            self.assertEqual(first_entry["group_name"], "block:0:linear_attn.out_proj")
            self.assertEqual(first_entry["reference_bit_width"], 16)
            self.assertEqual(first_entry["ablated_bit_width"], 8)

            self.assertEqual(second_entry["group_name"], "block:0:mlp.down_proj")
            self.assertEqual(second_entry["reference_bit_width"], 8)
            self.assertEqual(second_entry["ablated_bit_width"], 4)

            first_candidate = json.loads(Path(first_entry["candidate_path"]).read_text(encoding="utf-8"))
            self.assertEqual(
                first_candidate["group_bit_assignments"]["block:0:linear_attn.out_proj"],
                8,
            )
            self.assertEqual(
                first_candidate["ablation_metadata"]["reference_bit_width"],
                16,
            )

    def test_build_precision_ablation_manifest_can_filter_reference_bits(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = build_precision_ablation_manifest(
                report_payload=self.report_payload,
                reference_candidate_payload=self.reference_candidate_payload,
                output_dir=temp_dir,
                allowed_bits=(4, 8, 16),
                floor_bit=4,
                max_groups=8,
                reference_bit_widths=(8,),
                ranking_profile_payload=self.ranking_profile_payload,
            )

            self.assertEqual(manifest["reference_bit_widths"], [8])
            self.assertEqual(manifest["num_ablations"], 2)
            for entry in manifest["ablations"]:
                self.assertEqual(entry["reference_bit_width"], 8)
                self.assertEqual(entry["ablated_bit_width"], 4)

    def test_build_precision_ablation_profile_uses_accuracy_drop_as_signal(self) -> None:
        reference_summary = {"accuracy": 0.40}
        ablation_manifest_payload = {
            "ablations": [
                {
                    "group_name": "block:0:linear_attn.out_proj",
                    "reference_bit_width": 16,
                    "ablated_bit_width": 8,
                    "selection_score": 0.70,
                },
                {
                    "group_name": "block:0:mlp.down_proj",
                    "reference_bit_width": 8,
                    "ablated_bit_width": 4,
                    "selection_score": 0.95,
                },
            ]
        }
        ablation_evaluations = {
            "block:0:linear_attn.out_proj": {"accuracy": 0.35},
            "block:0:mlp.down_proj": {"accuracy": 0.20},
        }

        profile = build_precision_ablation_profile(
            report_payload=self.report_payload,
            reference_candidate_payload=self.reference_candidate_payload,
            reference_summary_payload=reference_summary,
            ablation_manifest_payload=ablation_manifest_payload,
            ablation_evaluation_payloads=ablation_evaluations,
            prior_weight=0.25,
        )

        self.assertEqual(profile["reference_accuracy"], 0.40)
        self.assertEqual(profile["num_ablations"], 2)
        self.assertEqual(profile["groups"][0]["name"], "block:0:mlp.down_proj")
        self.assertAlmostEqual(profile["groups"][0]["accuracy_drop"], 0.20, places=6)
        self.assertAlmostEqual(profile["groups"][0]["normalized_accuracy_drop"], 1.0, places=6)
        self.assertGreater(
            profile["groups"][0]["combined_sensitivity"],
            profile["groups"][1]["combined_sensitivity"],
        )


if __name__ == "__main__":
    unittest.main()
