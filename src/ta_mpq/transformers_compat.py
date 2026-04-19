from __future__ import annotations

from typing import Any


def _torch_causal_conv1d_fn(
    x: Any,
    weight: Any,
    bias: Any = None,
    activation: str | None = None,
    seq_idx: Any = None,
) -> Any:
    import torch.nn.functional as F

    del seq_idx

    kernel_width = int(weight.shape[-1])
    hidden_states = x.to(weight.dtype)
    out = F.conv1d(
        hidden_states,
        weight.unsqueeze(1),
        bias,
        padding=kernel_width - 1,
        groups=hidden_states.shape[1],
    )[..., : x.shape[-1]]
    if activation in {"silu", "swish"}:
        out = F.silu(out)
    return out.to(x.dtype)


def patch_qwen3_5_modeling_module(module: Any) -> bool:
    chunk_rule = getattr(module, "chunk_gated_delta_rule", None)
    recurrent_rule = getattr(module, "fused_recurrent_gated_delta_rule", None)
    if not (chunk_rule and recurrent_rule):
        return False

    patched = False
    if getattr(module, "causal_conv1d_fn", None) is None:
        module.causal_conv1d_fn = _torch_causal_conv1d_fn
        patched = True
    if getattr(module, "causal_conv1d_update", None) is None:
        module.causal_conv1d_update = getattr(module, "torch_causal_conv1d_update", None)
        patched = True

    module.is_fast_path_available = all(
        (
            getattr(module, "causal_conv1d_fn", None),
            getattr(module, "causal_conv1d_update", None),
            chunk_rule,
            recurrent_rule,
        )
    )
    return patched


def apply_qwen3_5_fast_path_compat_patch() -> bool:
    try:
        from transformers.models.qwen3_5 import modeling_qwen3_5
    except Exception:
        return False

    if getattr(modeling_qwen3_5, "_ta_mpq_fast_path_compat_applied", False):
        return True

    patched = patch_qwen3_5_modeling_module(modeling_qwen3_5)
    modeling_qwen3_5._ta_mpq_fast_path_compat_applied = patched
    return patched
