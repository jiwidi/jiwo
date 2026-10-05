"""Shared test fixtures. Model tests use a tiny random Qwen3.5 from the local Hugging Face cache."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests.helpers import write_checkpoint

TINY_MODEL = "trl-internal-testing/tiny-Qwen3_5ForConditionalGeneration"


@pytest.fixture(scope="session")
def tiny_model_name() -> str:
    """The tiny model id. Skip the test when the model is not in the local Hugging Face cache."""
    from huggingface_hub import try_to_load_from_cache

    if not isinstance(try_to_load_from_cache(TINY_MODEL, "config.json"), str):
        pytest.skip(f"{TINY_MODEL} is not in the local cache (run: uv run hf download {TINY_MODEL})")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    return TINY_MODEL


@pytest.fixture(scope="session")
def checkpoint(tiny_model_name: str, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A full checkpoint with 16 answer codes that saw at most 4 options in training."""
    return write_checkpoint(tiny_model_name, tmp_path_factory.mktemp("ck") / "model", max_trained_options=4)
