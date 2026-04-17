from __future__ import annotations

import unittest

from ta_mpq.feasibility import inspect_policy_targets_against_named_modules


class FeasibilityTargetMatchingTests(unittest.TestCase):
    def test_inspect_policy_targets_summarizes_exact_and_class_matches(self) -> None:
        named_module_types = {
            "lm_head": "Linear",
            "model.layers.0.linear_attn.in_proj_qkv": "Linear",
            "model.layers.0.mlp.down_proj": "Linear",
            "model.layers.0.self_attn.q_proj": "Linear",
            "model.layers.0.norm": "RMSNorm",
        }
        recipe_config = {
            "ignore": [],
            "config_groups": {
                "default": {
                    "targets": ["Linear"],
                },
                "bit_8": {
                    "targets": [
                        "lm_head",
                        "model.layers.0.linear_attn.in_proj_qkv",
                        "model.layers.99.mlp.down_proj",
                    ]
                },
            },
        }

        summary = inspect_policy_targets_against_named_modules(
            named_module_types=named_module_types,
            recipe_config=recipe_config,
        )

        self.assertEqual(summary["num_named_modules"], 5)
        self.assertEqual(summary["matched_target_count"], 3)
        self.assertEqual(summary["unmatched_target_count"], 1)
        self.assertEqual(
            summary["all_unmatched_targets"],
            ["model.layers.99.mlp.down_proj"],
        )

        default_group = summary["group_summaries"]["default"]
        self.assertEqual(default_group["matched_module_count"], 4)
        self.assertEqual(default_group["target_kind_counts"]["class_name"], 1)

        bit_8_group = summary["group_summaries"]["bit_8"]
        self.assertEqual(bit_8_group["matched_target_count"], 2)
        self.assertEqual(bit_8_group["unmatched_target_count"], 1)
        self.assertIn("lm_head", bit_8_group["matched_targets_sample"])


if __name__ == "__main__":
    unittest.main()
