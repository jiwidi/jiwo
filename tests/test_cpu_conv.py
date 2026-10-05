"""The CPU path of the Qwen3.5 causal convolution gives the output of the transformers fallback."""

import pytest
import torch
from transformers.models.qwen3_5 import modeling_qwen3_5

from jiwo import cpu_conv


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cpu_path_matches_the_transformers_fallback(dtype: torch.dtype) -> None:
    generator = torch.Generator().manual_seed(0)
    hidden = torch.randn(3, 64, 37, generator=generator).to(dtype)
    weight = torch.randn(64, 4, generator=generator).to(dtype)
    # Other tests load CPU models, which installs the CPU path for the whole process. Compare with the original.
    original = cpu_conv.transformers_causal_conv1d()
    assert not getattr(original, "jiwo_cpu_path", False)
    expected = original(hidden, weight, None, activation="silu")
    actual = cpu_conv.cpu_causal_conv1d(hidden, weight, None, "silu")
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_install_routes_cpu_tensors_once(monkeypatch: pytest.MonkeyPatch) -> None:
    original = cpu_conv.transformers_causal_conv1d()
    monkeypatch.setattr(modeling_qwen3_5, "causal_conv1d_fn", original)  # start from the original, restore after
    cpu_conv.install()
    installed = modeling_qwen3_5.causal_conv1d_fn
    cpu_conv.install()
    assert modeling_qwen3_5.causal_conv1d_fn is installed
    assert installed is not original
    assert cpu_conv.transformers_causal_conv1d() is original
