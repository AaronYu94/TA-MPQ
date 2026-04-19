from __future__ import annotations

from types import SimpleNamespace
import unittest

from ta_mpq.transformers_compat import patch_qwen3_5_modeling_module


class TransformersCompatTests(unittest.TestCase):
    def test_patch_qwen3_5_modeling_module_injects_conv1d_shims(self) -> None:
        module = SimpleNamespace(
            causal_conv1d_fn=None,
            causal_conv1d_update=None,
            torch_causal_conv1d_update=object(),
            chunk_gated_delta_rule=object(),
            fused_recurrent_gated_delta_rule=object(),
            is_fast_path_available=False,
        )

        patched = patch_qwen3_5_modeling_module(module)

        self.assertTrue(patched)
        self.assertIsNotNone(module.causal_conv1d_fn)
        self.assertIs(module.causal_conv1d_update, module.torch_causal_conv1d_update)
        self.assertTrue(module.is_fast_path_available)

    def test_patch_qwen3_5_modeling_module_skips_without_fla_ops(self) -> None:
        module = SimpleNamespace(
            causal_conv1d_fn=None,
            causal_conv1d_update=None,
            torch_causal_conv1d_update=object(),
            chunk_gated_delta_rule=None,
            fused_recurrent_gated_delta_rule=None,
            is_fast_path_available=False,
        )

        patched = patch_qwen3_5_modeling_module(module)

        self.assertFalse(patched)
        self.assertIsNone(module.causal_conv1d_fn)
        self.assertIsNone(module.causal_conv1d_update)
        self.assertFalse(module.is_fast_path_available)

    def test_injected_conv1d_fn_matches_plain_torch_conv(self) -> None:
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("torch is not installed in the local unit-test environment")

        module = SimpleNamespace(
            causal_conv1d_fn=None,
            causal_conv1d_update=None,
            torch_causal_conv1d_update=object(),
            chunk_gated_delta_rule=object(),
            fused_recurrent_gated_delta_rule=object(),
            is_fast_path_available=False,
        )
        patch_qwen3_5_modeling_module(module)

        x = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
        weight = torch.tensor(
            [
                [1.0, 0.0],
                [0.5, 0.5],
                [0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        bias = torch.tensor([0.0, 1.0, -1.0], dtype=torch.float32)

        out = module.causal_conv1d_fn(x=x, weight=weight, bias=bias, activation=None)

        expected = torch.nn.functional.conv1d(
            x,
            weight.unsqueeze(1),
            bias,
            padding=1,
            groups=3,
        )[..., : x.shape[-1]]
        self.assertTrue(torch.allclose(out, expected))


if __name__ == "__main__":
    unittest.main()
