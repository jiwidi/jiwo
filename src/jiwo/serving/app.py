"""A Jev-compatible HTTP API: POST /v1/systemone with a state and typed questions.

Settings (environment):
- JIWO_CHECKPOINT: the checkpoint directory or Hugging Face repository id (required).
- JIWO_DEVICE: cuda, mps or cpu (default: the best available).
- JIWO_API_KEY: if set, requests need "Authorization: Bearer <key>".
- JIWO_MAX_BODY_BYTES: the largest request body (default 2000000 bytes).
- JIWO_MAX_LENGTH: the longest prompt in tokens (default 8192). Longer prompts get a 400 error. Nothing is
  truncated.
- JIWO_BATCH_TOKENS: the most padded tokens in one forward pass (default 262144 on CUDA, 16384 on CPU and MPS).
  A request with many long questions runs in several batches. The answers do not change.
- JIWO_CUDNN_ATTENTION: "0" switches off the cuDNN attention kernel (default on). Some Gemma 4 models need this.
- JIWO_CUDA_GRAPHS: "1" captures CUDA graphs for fixed (rows, length) shapes at start-up and replays them for each
  request (jiwo.graphs). This makes a request faster on a GPU, most of all for small models. Batches outside the captured shapes run
  eagerly. The capture takes a few minutes. If it fails, the server logs the error and runs eagerly. Default off.
- JIWO_MODEL_NAME: the model name in responses, for example jiwo-0.8b (default: the checkpoint's metadata name,
  else the name of the Hugging Face repository, else "jiwo").
- JIWO_HOST and PORT: the address of the server (default 127.0.0.1 and 8765).
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import os
from dataclasses import dataclass
from typing import Any

import torch
from fastapi import FastAPI, Header, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from jiwo import __version__
from jiwo.graphs import GraphRunner
from jiwo.inference import BATCH_TOKENS, MAX_LENGTH, CapacityError, decide, default_batch_tokens
from jiwo.model import DecisionModel, PromptTooLongError
from jiwo.schema import SchemaError

MAX_BODY_BYTES = 2_000_000
log = logging.getLogger(__name__)


@dataclass
class Service:
    model: DecisionModel | None = None
    lock: asyncio.Lock | None = None
    name: str = "jiwo"


service = Service()
# No interactive docs or OpenAPI schema: the server exposes only the endpoints below.
app = FastAPI(title="jiwo", version=__version__, docs_url=None, redoc_url=None, openapi_url=None)


def error(status: int, message: str, kind: str = "invalid_request") -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"message": message, "type": kind}})


def int_setting(name: str, default: int) -> int:
    """Read a non-negative integer setting from the environment."""
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    try:
        number = int(value)
    except ValueError:
        number = -1
    if number < 0:
        raise RuntimeError(f"{name} must be a non-negative integer, not {value!r}. Correct the setting.")
    return number


def load(checkpoint: str | None = None, device: str | None = None) -> None:
    """Load the model. The arguments override JIWO_CHECKPOINT and JIWO_DEVICE."""
    path = checkpoint or os.environ.get("JIWO_CHECKPOINT")
    if not path:
        raise RuntimeError("Set JIWO_CHECKPOINT to a checkpoint directory or a Hugging Face repository id.")
    settings = (
        ("JIWO_MAX_BODY_BYTES", MAX_BODY_BYTES),
        ("JIWO_MAX_LENGTH", MAX_LENGTH),
        ("JIWO_BATCH_TOKENS", BATCH_TOKENS),
    )
    for name, default in settings:
        int_setting(name, default)  # a bad setting stops the start, not the first request
    if os.environ.get("JIWO_CUDNN_ATTENTION", "1") == "0":
        torch.backends.cuda.enable_cudnn_sdp(False)
    service.model = DecisionModel.from_pretrained(path, device=device or os.environ.get("JIWO_DEVICE"))
    if os.environ.get("JIWO_CUDA_GRAPHS") == "1":
        try:
            GraphRunner(service.model).attach().capture()
        except Exception:  # serve eagerly rather than not at all
            service.model.graphs = None
            log.exception("The CUDA graph capture failed. The server runs eagerly.")
    service.lock = asyncio.Lock()
    service.name = os.environ.get("JIWO_MODEL_NAME") or service.model.model_name


def authorized(authorization: str | None) -> bool:
    expected = os.environ.get("JIWO_API_KEY")
    if not expected:
        return True
    return hmac.compare_digest((authorization or "").encode(), f"Bearer {expected}".encode())


async def read_body(request: Request, limit: int) -> bytes | None:
    """The request body, or None when it is larger than `limit` bytes. Stop reading at the limit."""
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > limit:
        return None
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > limit:
            return None
    return bytes(body)


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"status": "ok" if service.model is not None else "no-model", "model": service.name}


@app.get("/v1/models")
async def models() -> dict[str, Any]:
    return {"data": [{"id": service.name, "object": "model"}]}


@app.post("/v1/systemone", response_model=None)
async def systemone(request: Request, authorization: str | None = Header(default=None)) -> Any:
    if not authorized(authorization):
        return error(401, "Invalid or missing API key. Send the header 'Authorization: Bearer <key>'.", "unauthorized")
    if service.model is None or service.lock is None:
        return error(503, "The model is not loaded. Try again when GET /health reports ok.", "unavailable")
    limit = int_setting("JIWO_MAX_BODY_BYTES", MAX_BODY_BYTES)
    raw = await read_body(request, limit)
    if raw is None:
        return error(413, f"The request body is larger than {limit} bytes. Send a smaller request.")
    try:
        body = json.loads(raw)
    except (ValueError, RecursionError):
        return error(400, "The request body is not valid JSON. Send a JSON object with state and questions.")
    max_length = int_setting("JIWO_MAX_LENGTH", MAX_LENGTH)
    batch_tokens = int_setting("JIWO_BATCH_TOKENS", default_batch_tokens(service.model.device_name))
    try:
        async with service.lock:
            return await run_in_threadpool(
                decide, service.model, body, name=service.name, max_length=max_length, batch_tokens=batch_tokens
            )
    except (SchemaError, PromptTooLongError, CapacityError) as problem:
        return error(400, str(problem))


def main() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    load()
    uvicorn.run(app, host=os.environ.get("JIWO_HOST", "127.0.0.1"), port=int_setting("PORT", 8765))


if __name__ == "__main__":
    main()
