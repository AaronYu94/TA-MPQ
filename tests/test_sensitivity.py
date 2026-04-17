from __future__ import annotations

from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ta_mpq.feasibility import LinearLayerStat
from ta_mpq.sensitivity import (
    ModuleActivationStat,
    build_task_sensitivity_profile,
    group_sensitivity_overrides_from_profile,
)


class TaskSensitivityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.layer_stats = [
            LinearLayerStat(
                name="model.layers.0.mlp.down_proj",
                parameter_count=1_000_000,
                in_features=1024,
                out_features=1024,
            ),
            LinearLayerStat(
                name="model.layers.0.linear_attn.in_proj_b",
                parameter_count=250_000,
                in_features=1024,
                out_features=256,
            ),
            LinearLayerStat(
                name="model.layers.1.mlp.down_proj",
                parameter_count=1_000_000,
                in_features=1024,
                out_features=1024,
            ),
        ]
        self.activation_stats = [
            ModuleActivationStat(
                name="model.layers.0.mlp.down_proj",
                parameter_count=1_000_000,
                mean_abs_input=3.0,
                mean_abs_output=2.0,
                num_observations=4,
            ),
            ModuleActivationStat(
                name="model.layers.0.linear_attn.in_proj_b",
                parameter_count=250_000,
                mean_abs_input=0.4,
                mean_abs_output=0.6,
                num_observations=4,
            ),
            ModuleActivationStat(
                name="model.layers.1.mlp.down_proj",
                parameter_count=1_000_000,
                mean_abs_input=2.8,
                mean_abs_output=2.2,
                num_observations=4,
            ),
        ]

    def test_build_task_sensitivity_profile_blends_prior_and_activation(self) -> None:
        profile = build_task_sensitivity_profile(
            layer_stats=self.layer_stats,
            activation_stats=self.activation_stats,
            grouping="per_component_family",
            activation_weight=0.5,
        )

        groups = {group["name"]: group for group in profile["groups"]}
        self.assertIn("component:mlp.down_proj", groups)
        self.assertIn("component:linear_attn.in_proj_b", groups)

        down_proj = groups["component:mlp.down_proj"]
        in_proj_b = groups["component:linear_attn.in_proj_b"]
        self.assertGreater(down_proj["normalized_activation_score"], in_proj_b["normalized_activation_score"])
        self.assertGreater(down_proj["combined_sensitivity"], in_proj_b["combined_sensitivity"])

    def test_group_sensitivity_overrides_from_profile_uses_combined_field(self) -> None:
        profile = build_task_sensitivity_profile(
            layer_stats=self.layer_stats,
            activation_stats=self.activation_stats,
            grouping="per_block_component",
            activation_weight=0.6,
        )

        overrides = group_sensitivity_overrides_from_profile(profile)
        self.assertIn("block:0:mlp.down_proj", overrides)
        self.assertIn("block:0:linear_attn.in_proj_b", overrides)
        self.assertGreater(overrides["block:0:mlp.down_proj"], overrides["block:0:linear_attn.in_proj_b"])


if __name__ == "__main__":
    unittest.main()
