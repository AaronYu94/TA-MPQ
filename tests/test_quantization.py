from __future__ import annotations

from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ta_mpq.quantization import (
    HIGH_PRECISION_BIT,
    BitRule,
    MixedPrecisionPolicy,
    assign_bits_to_modules,
    default_feasibility_policy,
    estimate_average_bit_width,
    estimate_weight_footprint_gb,
    to_llmcompressor_recipe_config,
    validate_backend_support,
)


class QuantizationPolicyTests(unittest.TestCase):
    def test_regex_assignment(self) -> None:
        policy = MixedPrecisionPolicy(
            default_bit_width=4,
            rules=(BitRule(name="down_proj_int8", targets=("re:.*down_proj$",), bit_width=8),),
        )
        assignments = assign_bits_to_modules(
            [
                "model.layers.0.mlp.down_proj",
                "model.layers.0.mlp.up_proj",
            ],
            policy,
        )
        self.assertEqual(assignments["model.layers.0.mlp.down_proj"], 8)
        self.assertEqual(assignments["model.layers.0.mlp.up_proj"], 4)

    def test_average_bit_width(self) -> None:
        assignments = {"a": 8, "b": 4}
        params = {"a": 10, "b": 10}
        self.assertEqual(estimate_average_bit_width(params, assignments), 6.0)

    def test_weight_footprint_estimate(self) -> None:
        assignments = {"a": 8}
        params = {"a": 8 * 1024 * 1024}
        footprint = estimate_weight_footprint_gb(params, assignments)
        self.assertGreater(footprint, 0.0)

    def test_llmcompressor_recipe_emits_default_and_override(self) -> None:
        recipe = to_llmcompressor_recipe_config(default_feasibility_policy())
        self.assertIn("default", recipe["config_groups"])
        self.assertIn("down_proj_int8", recipe["config_groups"])
        self.assertEqual(recipe["config_groups"]["default"]["format"], "pack-quantized")
        self.assertEqual(recipe["config_groups"]["down_proj_int8"]["format"], "pack-quantized")

    def test_backend_support_rejects_project_only_bits(self) -> None:
        policy = MixedPrecisionPolicy(
            default_bit_width=4,
            rules=(BitRule(name="low_bit", targets=("re:.*q_proj$",), bit_width=3),),
        )
        with self.assertRaises(ValueError):
            validate_backend_support(policy, backend="llmcompressor")

    def test_policy_round_trip_from_dict(self) -> None:
        original = MixedPrecisionPolicy(
            default_bit_width=4,
            ignore=("lm_head",),
            rules=(
                BitRule(
                    name="preserve_down_proj",
                    targets=("re:.*down_proj$",),
                    bit_width=8,
                    group_size=64,
                    symmetric=False,
                ),
            ),
        )

        reconstructed = MixedPrecisionPolicy.from_dict(original.to_dict())
        self.assertEqual(reconstructed, original)

    def test_assign_bits_to_modules_respects_ignore_as_high_precision(self) -> None:
        policy = MixedPrecisionPolicy(
            default_bit_width=4,
            ignore=("model.layers.0.mlp.down_proj",),
            rules=(),
        )
        assignments = assign_bits_to_modules(
            [
                "model.layers.0.mlp.down_proj",
                "model.layers.0.mlp.up_proj",
            ],
            policy,
        )
        self.assertEqual(assignments["model.layers.0.mlp.down_proj"], HIGH_PRECISION_BIT)
        self.assertEqual(assignments["model.layers.0.mlp.up_proj"], 4)


if __name__ == "__main__":
    unittest.main()
