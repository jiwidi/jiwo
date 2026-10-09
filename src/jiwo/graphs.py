"""CUDA-graph replay of the decision forward pass, for serving.

An eager forward of a small model is launch-bound: the GPU waits for Python to queue hundreds of small kernels. A CUDA
graph records the kernels of one input shape once and replays them with one launch. A graph needs fixed shapes, so the
runner pads each batch to a (rows, length) bucket and captures every bucket at start-up. After start-up it never
captures: a batch that fits no bucket returns None and the caller runs it eagerly. A batch wider than the widest
graph of its length also runs eagerly: it has enough work to hide the launch cost, and one eager pass pads less than
several padded replays.

The graph forward has no attention mask. Rows are right-padded and every layer is causal (attention, Qwen3.5
gated delta-net, short convolutions), so the padding after a row's last token cannot change that token's hidden state.
The answers equal the masked eager forward up to kernel round-off. With jiwo-4b on 879 Decision Index requests on
one RTX PRO 6000, 99.8% of the fields pick the same option as the eager server, and the others are near ties.

The server turns this on with JIWO_CUDA_GRAPHS=1. The capture takes a few minutes at start-up.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from jiwo.model import DecisionModel

log = logging.getLogger(__name__)

LENGTHS = (64, 128, 192, 256, 320, 384, 512, 640, 768, 1024, 1280, 1536, 2048, 3072, 4096, 6144, 8192)
ROWS = (1, 2, 4, 8, 16, 32)
TOKEN_BUDGET = 32_768  # capture (rows, length) only when rows * length fits, or when rows is 1
WARMUP_RUNS = 3  # eager runs on a side stream before a capture: Triton autotuning and lazy init happen there


class GraphRunner:
    def __init__(
        self,
        model: DecisionModel,
        lengths: Sequence[int] = LENGTHS,
        rows: Sequence[int] = ROWS,
        token_budget: int = TOKEN_BUDGET,
    ) -> None:
        if not str(model.device_name).startswith("cuda"):
            raise ValueError("CUDA graphs need a CUDA device.")
        self.model = model
        self.lengths = sorted(lengths)
        self.shapes = [(b, t) for t in self.lengths for b in sorted(rows) if b == 1 or b * t <= token_budget]
        self.widths: dict[int, list[int]] = {}
        for b, t in self.shapes:
            self.widths.setdefault(t, []).append(b)
        self.forward = self.model.last_logits
        self.graphs: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.cuda.CUDAGraph]] = {}
        self.pool = torch.cuda.graph_pool_handle()
        self.replays = 0

    @torch.inference_mode()
    def capture(self) -> float:
        """Capture every shape, longest first, so the shared memory pool starts at its largest size. Return seconds."""
        started = time.monotonic()
        device = self.model.device_name
        pad = int(self.model.tokenizer.pad_token_id)
        self.model.eval()
        for rows, length in sorted(self.shapes, key=lambda shape: -shape[0] * shape[1]):
            ids = torch.full((rows, length), pad, dtype=torch.long, device=device)
            last = torch.full((rows,), length - 1, dtype=torch.long, device=device)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(WARMUP_RUNS):
                    self.forward(ids, None, last)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self.pool):
                out = self.forward(ids, None, last)
            self.graphs[(rows, length)] = (ids, last, out, graph)
        torch.cuda.synchronize()
        seconds = time.monotonic() - started
        log.info("captured %d CUDA graphs in %.1f s", len(self.graphs), seconds)
        return seconds

    def attach(self) -> GraphRunner:
        self.model.graphs = self
        return self

    def bucket(self, length: int) -> int | None:
        return next((t for t in self.lengths if length <= t), None)

    @torch.inference_mode()
    def logits(self, ids: Sequence[Sequence[int]]) -> torch.Tensor | None:
        """Unmasked readout logits, one row per input row, or None when no captured graph holds all rows."""
        length = self.bucket(max(len(row) for row in ids))
        width = next((width for width in self.widths.get(length or 0, []) if width >= len(ids)), None)
        if length is None or width is None or (width, length) not in self.graphs:
            return None
        host = torch.full((width, length), int(self.model.tokenizer.pad_token_id), dtype=torch.long)
        last = torch.zeros(width, dtype=torch.long)
        for index, row in enumerate(ids):
            host[index, : len(row)] = torch.tensor(row, dtype=torch.long)
            last[index] = len(row) - 1
        static_ids, static_last, static_out, graph = self.graphs[(width, length)]
        static_ids.copy_(host)
        static_last.copy_(last)
        graph.replay()
        self.replays += 1
        return static_out[: len(ids)].clone()  # the next replay reuses the output memory
