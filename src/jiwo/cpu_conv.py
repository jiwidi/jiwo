"""A fast CPU path for the causal depthwise convolution of Qwen3.5 linear attention.

Without the CUDA causal-conv1d kernel, transformers runs F.conv1d with one group per channel. Torch builds without
oneDNN (for example the macOS arm64 wheels) run that convolution as one small loop per channel and per batch row.
Then every forward pass of a Qwen3.5 0.8B model takes at least about 3.5 s on an Apple M4 Pro CPU. This module
computes the same 4-tap causal convolution as shifted multiply-adds in float32. The output is the same: identical in
bfloat16, equal to rounding in float32. Tensors on other devices use the original function.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from transformers.activations import ACT2FN


def cpu_causal_conv1d(
    hidden_states: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None, activation: str | None
) -> torch.Tensor:
    """The causal depthwise convolution of hidden_states (batch, channels, length) with weight (channels, taps)."""
    length = hidden_states.shape[-1]
    taps = weight.shape[-1]
    out = torch.empty(hidden_states.shape, dtype=weight.dtype, device=hidden_states.device)
    w = weight.float()
    for row in range(hidden_states.shape[0]):  # one batch row at a time keeps the float32 copy small
        x = F.pad(hidden_states[row : row + 1].float(), (taps - 1, 0))
        acc = x[:, :, 0:length] * w[:, 0, None]
        for tap in range(1, taps):
            acc.addcmul_(x[:, :, tap : tap + length], w[:, tap, None])
        if bias is not None:
            acc.add_(bias.float()[:, None])
        out[row : row + 1] = acc.to(weight.dtype)
    if activation is not None:
        out = ACT2FN[activation](out)
    return out.to(hidden_states.dtype)


def install() -> None:
    """Route CPU tensors of the Qwen3.5 causal convolution to cpu_causal_conv1d. Do nothing twice."""
    try:
        from transformers.models.qwen3_5 import modeling_qwen3_5
    except ImportError:
        return
    original = modeling_qwen3_5.causal_conv1d_fn
    if getattr(original, "jiwo_cpu_path", False):
        return

    def causal_conv1d_fn(
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        activation: str | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if hidden_states.device.type != "cpu":
            return original(hidden_states, weight, bias, activation=activation, **kwargs)  # type: ignore[no-any-return]
        return cpu_causal_conv1d(hidden_states, weight, bias, activation)

    causal_conv1d_fn.jiwo_cpu_path = True  # type: ignore[attr-defined]
    causal_conv1d_fn.jiwo_original = original  # type: ignore[attr-defined]
    modeling_qwen3_5.causal_conv1d_fn = causal_conv1d_fn


def transformers_causal_conv1d() -> Any:
    """Return the transformers convolution function, also after install() replaced it."""
    from transformers.models.qwen3_5 import modeling_qwen3_5

    current = modeling_qwen3_5.causal_conv1d_fn
    return getattr(current, "jiwo_original", current)
