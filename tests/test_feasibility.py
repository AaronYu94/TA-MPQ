from __future__ import annotations

from pathlib import Path
import sys
import types
import unittest
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ta_mpq.feasibility import (
    _build_oneshot_kwargs,
    LinearLayerStat,
    build_policy_target_integrity_manifest,
    build_feasibility_report,
    inspect_policy_targets_against_named_modules,
)
from ta_mpq.quantization import BitRule, MixedPrecisionPolicy


class FeasibilityTargetMatchingTests(unittest.TestCase):
    def test_build_oneshot_kwargs_uses_tokenizer_without_unsupported_device_arg(self) -> None:
        tokenizer_stub = object()
        transformers_stub = types.SimpleNamespace(
            AutoTokenizer=types.SimpleNamespace(
                from_pretrained=mock.Mock(return_value=tokenizer_stub)
            )
        )
        with mock.patch.dict(sys.modules, {"transformers": transformers_stub}):
            kwargs = _build_oneshot_kwargs(
                model_id="Qwen/Qwen3.5-27B",
                processor_strategy="tokenizer",
                oneshot_device="cuda:0",
            )

        self.assertIs(kwargs["tokenizer"], tokenizer_stub)
        self.assertNotIn("oneshot_device", kwargs)

    def test_build_oneshot_kwargs_accepts_auto_strategy_without_tokenizer(self) -> None:
        kwargs = _build_oneshot_kwargs(
            model_id="Qwen/Qwen3.5-27B",
            processor_strategy="auto",
            oneshot_device=None,
        )

        self.assertEqual(kwargs, {})

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
        self.assertEqual(summary["ignore_matched_target_count"], 0)

    def test_build_feasibility_report_can_include_full_model_residual(self) -> None:
        layer_stats = [
            LinearLayerStat(
                name="lm_head",
                parameter_count=80,
                in_features=8,
                out_features=10,
            ),
        ]

        report = build_feasibility_report(
            layer_stats,
            total_model_parameters=100,
            non_linear_parameter_count=20,
            non_linear_bit_width=16,
        )

        expected_non_linear_footprint = (20 * 16) / 8 / (1024**3)
        self.assertEqual(report["total_linear_parameters"], 80)
        self.assertEqual(report["total_model_parameters"], 100)
        self.assertEqual(report["total_non_linear_parameters"], 20)
        self.assertAlmostEqual(
            report["estimated_non_linear_weight_footprint_gb"],
            expected_non_linear_footprint,
        )
        self.assertAlmostEqual(
            report["estimated_full_model_weight_footprint_gb"],
            report["estimated_weight_footprint_gb"] + expected_non_linear_footprint,
        )
        self.assertEqual(report["budget_accounting_mode"], "matched_linear_weight_budget")
        self.assertAlmostEqual(
            report["matched_linear_weight_footprint_gb"],
            report["estimated_weight_footprint_gb"],
        )

    def test_integrity_manifest_flags_unresolved_reload_target(self) -> None:
        policy = MixedPrecisionPolicy(
            default_bit_width=8,
            ignore=("lm_head",),
            rules=(
                BitRule(
                    name="bit_4_rule",
                    targets=("model.layers.0.linear_attn.in_proj_qkv",),
                    bit_width=4,
                ),
            ),
        )
        source_matching = {
            "matched_targets": ["model.layers.0.linear_attn.in_proj_qkv"],
            "all_unmatched_targets": [],
            "ignore_matched_targets": ["lm_head"],
            "ignore_unmatched_targets": [],
        }
        reload_matching = {
            "matched_targets": [],
            "all_unmatched_targets": ["model.layers.0.linear_attn.in_proj_qkv"],
            "ignore_matched_targets": ["lm_head"],
            "ignore_unmatched_targets": [],
        }

        manifest = build_policy_target_integrity_manifest(
            policy=policy,
            source_target_matching=source_matching,
            reload_target_matching=reload_matching,
        )

        self.assertFalse(manifest["is_clean"])
        self.assertIn(
            r"re:.*model\.layers\.0\.linear_attn\.in_proj_qkv$",
            manifest["unresolved_target_warnings"],
        )
        self.assertIn("lm_head", manifest["reload_matched_targets"])


if __name__ == "__main__":
    unittest.main()
