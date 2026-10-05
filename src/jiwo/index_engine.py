"""An in-process engine for the Decision Index kit (install the "index" extra).

Run it with the kit:

    python -m decision_index run --engine jiwo.index_engine:JiwoEngine --model <checkpoint or repo id> ...

Options (--option key=value): revision, device, max_length (default 8192), batch_tokens (default 262144 on CUDA,
16384 on CPU and MPS), cudnn_attention (default true) and model_name (the model name in responses). The kit's
load_engine takes the engine name as its own `name` argument, so this option cannot be called `name`.

The engine calls the same request code as the server (`jiwo.inference.decide`), so it gives the same answers as the
kit's http engine against `jiwo serve` with the same settings. Errors whose message has one of the kit's capacity
markers become `Unsupported`, as in the http engine. Other errors stay errors.
"""

from __future__ import annotations

from typing import Any

import torch
from decision_index.engines import Engine, Unsupported
from decision_index.engines.http import CAPACITY_MARKERS

from jiwo import __version__
from jiwo.inference import MAX_LENGTH, CapacityError, decide, default_batch_tokens
from jiwo.model import DecisionModel, PromptTooLongError
from jiwo.schema import DecisionResponse, SchemaError


class JiwoEngine(Engine):  # type: ignore[misc]  # the kit has no type information
    name = "jiwo"
    latency = "In-process request wall time, including prompt construction and inference. It excludes model loading."

    def __init__(
        self,
        model: str,
        revision: str | None = None,
        device: str | None = None,
        max_length: int = MAX_LENGTH,
        batch_tokens: int | None = None,
        cudnn_attention: bool = True,
        model_name: str | None = None,
        **options: Any,
    ) -> None:
        super().__init__(**options)
        if not cudnn_attention:
            torch.backends.cuda.enable_cudnn_sdp(False)
        self.model = DecisionModel.from_pretrained(model, revision=revision, device=device)
        self.model_name = model_name or self.model.model_name
        self.max_length = int(max_length)
        self.batch_tokens = default_batch_tokens(self.model.device_name) if batch_tokens is None else int(batch_tokens)
        self.provenance = {
            "kind": "jiwo",
            "checkpoint": model,
            "revision": revision,
            "device": self.model.device_name,
            "base_model": self.model.config.base_model,
            "max_length": self.max_length,
            "batch_tokens": self.batch_tokens,
            "cudnn_attention": bool(cudnn_attention),
            "policy": "The jiwo request code of the server, in process. Prompts longer than max_length and choice "
            "questions with more options than the model saw in training are refused as unsupported, never truncated.",
        }

    def __call__(self, state: Any, questions: Any) -> tuple[DecisionResponse, None]:
        body = {"state": state, "questions": questions}
        try:
            response = decide(
                self.model, body, name=self.model_name, max_length=self.max_length, batch_tokens=self.batch_tokens
            )
        except (SchemaError, PromptTooLongError, CapacityError) as problem:
            message = str(problem)
            if any(marker in message for marker in CAPACITY_MARKERS):
                raise Unsupported(message) from problem
            raise
        return response, None

    def runtime(self) -> dict[str, Any]:
        info: dict[str, Any] = {"jiwo": __version__, "torch": torch.__version__, "device": self.model.device_name}
        if self.model.device_name.startswith("cuda"):
            info["gpu"] = torch.cuda.get_device_name()
        return info

    def synchronize(self) -> None:
        if self.model.device_name.startswith("cuda"):
            torch.cuda.synchronize()
        elif self.model.device_name.startswith("mps"):
            torch.mps.synchronize()
