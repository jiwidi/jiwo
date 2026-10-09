"""Answer one request: validate it, run the model in batches and build the answers.

The server, the Decision Index engine and `DecisionModel.decide` all use `decide`, so they give the same answers.
"""

from __future__ import annotations

import gc
from typing import TYPE_CHECKING

import torch

from jiwo.answers import build_answer
from jiwo.prompting import PromptItem
from jiwo.schema import DecisionRequest, DecisionResponse, SchemaError, option_keys, validate_request

if TYPE_CHECKING:
    from jiwo.model import DecisionModel

MAX_LENGTH = 8192  # the longest prompt in tokens
BATCH_TOKENS = 262_144  # the most padded tokens in one forward pass on a CUDA GPU
HOST_BATCH_TOKENS = 16_384  # on a CPU or an Apple GPU, where the model shares the system memory


class CapacityError(ValueError):
    """The request does not fit in the GPU memory of this machine, even one question at a time."""


def default_batch_tokens(device: str) -> int:
    """The default batch token limit for a device.

    A float32 batch of 262144 tokens needs about 50 GB of activations. On a CPU, an allocation does not fail before
    the system swaps, so the out-of-memory retry cannot help. A CPU or an Apple GPU therefore uses a smaller limit.
    The limit does not change the answers.
    """
    return BATCH_TOKENS if device.startswith("cuda") else HOST_BATCH_TOKENS


def check_option_limit(model: DecisionModel, request: DecisionRequest) -> None:
    """Refuse a question with more options than the model saw in training."""
    limit = model.config.max_trained_options
    for name, question in request["questions"].items():
        count = len(option_keys(question))
        if count > limit:
            raise SchemaError(
                f"Question {name!r} has {count} options. This model supports at most {limit} "
                "options per choice, the most it saw in training."
            )


def predict_with_backoff(
    model: DecisionModel,
    items: list[PromptItem],
    batch_size: int,
    max_length: int,
    ids: list[list[int]] | None = None,
) -> list[list[float]]:
    """Predict. After a CUDA out-of-memory error, free the memory and retry with half the batch size."""
    size = batch_size
    while True:
        out_of_memory = False
        try:
            return model.predict_proba(items, batch_size=size, max_length=max_length, ids=ids)
        except torch.OutOfMemoryError:
            out_of_memory = True
        if out_of_memory:  # outside the except block, so the traceback no longer holds the tensors
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()
            if size == 1:
                raise CapacityError("This request needs more GPU memory than this server has: too many tokens.")
            size = max(1, size // 2)


def decide(
    model: DecisionModel,
    body: object,
    *,
    name: str,
    max_length: int = MAX_LENGTH,
    batch_tokens: int | None = None,
) -> DecisionResponse:
    """Answer one request body (a state and named questions).

    Raise SchemaError for an invalid request, PromptTooLongError for a prompt longer than `max_length` tokens and
    CapacityError when the request does not fit in GPU memory. Without `batch_tokens`, use the device default.
    """
    if batch_tokens is None:
        batch_tokens = default_batch_tokens(model.device_name)
    request = validate_request(body)
    check_option_limit(model, request)
    names = list(request["questions"])
    items = [PromptItem(request["state"], request["questions"][key]) for key in names]
    ids = model.token_ids(items, max_length)  # checks the lengths before any compute
    batch_size = max(1, min(len(items), batch_tokens // max(len(row) for row in ids)))
    probabilities = predict_with_backoff(model, items, batch_size, max_length, ids)
    answers = {
        key: build_answer(request["questions"][key], values) for key, values in zip(names, probabilities, strict=True)
    }
    tokens = sum(len(row) for row in ids)
    return {"model": name, "answers": answers, "usage": {"input_tokens": tokens, "output_tokens": 0}}
