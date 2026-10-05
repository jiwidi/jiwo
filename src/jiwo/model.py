"""A decoder backbone with an answer-code readout.

The model reads one prompt per (state, question) pair. It takes the hidden state at the LAST REAL token (right
padding plus a gather, which is exact for both softmax attention and Qwen3.5's recurrent linear-attention layers),
applies a linear readout with one row per answer code, masks the codes that the question does not use, and returns
logits. Softmax(logits / temperature) is the answer distribution in display order.
"""

from __future__ import annotations

import itertools
import json
import math
import re
import string
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self, cast

import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

from jiwo.inference import MAX_LENGTH, decide
from jiwo.prompting import DisplayOrder, PromptItem, decision_messages
from jiwo.schema import Content, DecisionResponse, JSONValue

MAX_CODES = 255
FORMAT_VERSION = 1
MASK_VALUE = -1e9  # finite, so arithmetic on the logits of unused codes stays finite
CONFIG_FILE = "decision_config.json"
READOUT_FILE = "readout.safetensors"
ADAPTER_CONFIG_FILE = "adapter_config.json"  # written by peft for a LoRA checkpoint
REPO_ID = re.compile(r"[A-Za-z0-9][\w.-]*/[A-Za-z0-9][\w.-]*")  # a Hugging Face repository id: owner/name


class PromptTooLongError(ValueError):
    """A prompt is longer than the model's maximum length. Nothing is truncated silently."""


@dataclass(frozen=True)
class Batch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    last_index: torch.Tensor
    counts: tuple[int, ...]
    input_tokens: int


@dataclass(frozen=True)
class DecisionConfig:
    base_model: str
    revision: str | None
    codes: tuple[str, ...]
    token_ids: tuple[int, ...]
    temperature: float = 1.0
    max_trained_options: int = MAX_CODES
    metadata: dict[str, JSONValue] = field(default_factory=dict)
    type_temperatures: dict[str, float] = field(default_factory=dict)

    def temperature_for(self, question_type: str) -> float:
        """The fitted temperature for a question type, else the global one."""
        return self.type_temperatures.get(question_type, self.temperature)

    def to_json(self) -> dict[str, JSONValue]:
        return {
            "format_version": FORMAT_VERSION,
            "base_model": self.base_model,
            "revision": self.revision,
            "codes": list(self.codes),
            "token_ids": list(self.token_ids),
            "temperature": self.temperature,
            "type_temperatures": dict(self.type_temperatures),
            "max_trained_options": self.max_trained_options,
            "metadata": self.metadata,
        }

    @staticmethod
    def from_json(value: dict[str, Any]) -> DecisionConfig:
        if value.get("format_version") != FORMAT_VERSION:
            raise ValueError(
                f"Unsupported checkpoint format {value.get('format_version')!r}. This jiwo version reads format "
                f"{FORMAT_VERSION}. Use a jiwo version that matches the checkpoint."
            )
        return DecisionConfig(
            base_model=str(value["base_model"]),
            revision=value.get("revision"),
            codes=tuple(value["codes"]),
            token_ids=tuple(int(i) for i in value["token_ids"]),
            temperature=float(value["temperature"]),
            type_temperatures={str(k): float(v) for k, v in (value.get("type_temperatures") or {}).items()},
            max_trained_options=int(value.get("max_trained_options", MAX_CODES)),
            metadata=dict(value.get("metadata") or {}),
        )


def pick_device(requested: str | None = None) -> str:
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def default_dtype(device: str) -> torch.dtype:
    return torch.bfloat16 if device.startswith(("cuda", "mps")) else torch.float32


def chat_text(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    return str(
        tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    )


def answer_codes(tokenizer: Any, limit: int = MAX_CODES) -> tuple[tuple[str, ...], tuple[int, ...]]:
    """The first `limit` codes (A..Z, then AA, AB, ...) that are one token, also right after the chat prefix."""
    candidates = list(string.ascii_uppercase) + [
        "".join(pair) for pair in itertools.product(string.ascii_uppercase, repeat=2)
    ]
    codes = [code for code in candidates if len(tokenizer.encode(code, add_special_tokens=False)) == 1][:limit]
    token_ids = [tokenizer.encode(code, add_special_tokens=False)[0] for code in codes]
    if len(codes) < limit or len(set(token_ids)) != len(token_ids):
        raise ValueError(f"The tokenizer does not provide {limit} distinct single-token answer codes.")
    prefix = chat_text(tokenizer, [{"role": "user", "content": "Choose an option."}])
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    for code, token_id in zip(codes, token_ids, strict=True):
        if tokenizer.encode(prefix + code, add_special_tokens=False) != [*prefix_ids, token_id]:
            raise ValueError(f"Answer code {code!r} is not a single token after the chat prefix.")
    return tuple(codes), tuple(token_ids)


def load_text_decoder(base_model: str, revision: str | None, dtype: torch.dtype) -> tuple[Any, torch.Tensor]:
    """Load the text decoder (no vision tower, no LM head) and the LM head weight matrix."""
    architectures = AutoConfig.from_pretrained(base_model, revision=revision).architectures or []
    multimodal = any("ConditionalGeneration" in name for name in architectures)
    loader = AutoModelForImageTextToText if multimodal else AutoModelForCausalLM
    full = loader.from_pretrained(base_model, revision=revision, dtype=dtype, attn_implementation="sdpa")
    head = full.get_output_embeddings().weight.detach().clone()
    decoder = getattr(full.model, "language_model", full.model)
    del full
    return decoder, head


def load_lora_decoder(base_model: str, revision: str | None, adapter: Path, dtype: torch.dtype) -> Any:
    """Load the pinned base text decoder and merge the LoRA adapter of a checkpoint into it."""
    try:
        from peft import PeftModel
    except ImportError as error:  # peft is in the "lora" extra
        raise ImportError(
            "A LoRA checkpoint needs peft. Install the lora extra, for example: uv sync --extra lora."
        ) from error
    decoder, _ = load_text_decoder(base_model, revision, dtype)
    return PeftModel.from_pretrained(decoder, str(adapter)).merge_and_unload()


def checkpoint_directory(name_or_path: str | Path, revision: str | None = None) -> Path:
    """The local directory of a checkpoint. A Hugging Face repository id (owner/name) is downloaded first.

    `revision` (a branch, tag or commit) applies only to a repository id.
    """
    path = Path(name_or_path).expanduser()
    if path.is_dir():
        return path
    if not REPO_ID.fullmatch(str(name_or_path)):
        raise FileNotFoundError(
            f"{name_or_path} is not a checkpoint directory or a Hugging Face repository id. "
            "Give a local directory or an id such as eljiwo/jiwo-0.8b."
        )
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(str(name_or_path), revision=revision))


class DecisionModel(torch.nn.Module):
    """A loaded checkpoint: the backbone, the readout, the tokenizer and the decision config."""

    def __init__(
        self, backbone: Any, readout: torch.nn.Linear, tokenizer: Any, config: DecisionConfig, device: str
    ) -> None:
        super().__init__()
        self.backbone: Any = backbone
        self.readout = readout
        self.tokenizer = tokenizer
        self.config = config
        self.device_name = device
        self.repository_name: str | None = None  # set by from_pretrained for a Hugging Face repository id
        self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token_id is None:
            raise ValueError("The tokenizer has no padding token. Use the tokenizer files of a jiwo checkpoint.")

    # ---- loading -------------------------------------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        name_or_path: str | Path,
        *,
        revision: str | None = None,
        device: str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Self:
        """Load a checkpoint from a local directory or from a Hugging Face repository id (owner/name)."""
        model = cls.from_checkpoint(checkpoint_directory(name_or_path, revision), device=device, dtype=dtype)
        if not Path(name_or_path).expanduser().is_dir():
            model.repository_name = str(name_or_path).rsplit("/", 1)[-1]
        return model

    @classmethod
    def from_checkpoint(cls, path: str | Path, *, device: str | None = None, dtype: torch.dtype | None = None) -> Self:
        """Load a checkpoint directory for inference. A LoRA checkpoint is merged into its pinned base model."""
        directory = Path(path)
        config = DecisionConfig.from_json(json.loads((directory / CONFIG_FILE).read_text()))
        device = pick_device(device)
        if device.startswith("cpu"):
            from jiwo import cpu_conv

            cpu_conv.install()  # Qwen3.5 linear attention: avoid the per-channel CPU convolution loop
        dtype = dtype or default_dtype(device)
        tokenizer = AutoTokenizer.from_pretrained(str(directory))
        codes, token_ids = answer_codes(tokenizer, len(config.codes))
        if codes != config.codes or token_ids != config.token_ids:
            raise ValueError(
                "The answer codes in decision_config.json do not match the tokenizer of the checkpoint. "
                "Keep the tokenizer files that were saved with the checkpoint."
            )
        if (directory / ADAPTER_CONFIG_FILE).is_file():
            backbone = load_lora_decoder(config.base_model, config.revision, directory, dtype)
        else:
            backbone = AutoModel.from_pretrained(str(directory), dtype=dtype, attn_implementation="sdpa")
        readout = torch.nn.Linear(backbone.config.hidden_size, len(token_ids), bias=False, dtype=dtype)
        readout.load_state_dict(load_file(str(directory / READOUT_FILE)))
        model = cls(backbone, readout, tokenizer, config, device)
        model.requires_grad_(False)
        model.to(device)
        model.eval()
        return model

    # ---- properties --------------------------------------------------------------------------------------------

    @property
    def codes(self) -> tuple[str, ...]:
        return self.config.codes

    @property
    def temperature(self) -> float:
        return self.config.temperature

    @property
    def model_name(self) -> str:
        """The name in responses: the "name" in the checkpoint metadata, else the repository name, else "jiwo"."""
        return str(self.config.metadata.get("name") or self.repository_name or "jiwo")

    # ---- encoding and forward ----------------------------------------------------------------------------------

    def prompt_text(self, item: PromptItem) -> str:
        return chat_text(self.tokenizer, decision_messages(item.state, item.question, self.codes, item.order))

    def encode(self, items: Sequence[PromptItem], max_length: int = MAX_LENGTH) -> Batch:
        """Tokenize the prompts with right padding. Raise PromptTooLongError if a prompt is too long."""
        if not items:
            raise ValueError("A batch needs at least one item.")
        counts = tuple(item.count for item in items)
        if max(counts) > len(self.codes):
            raise ValueError(f"A question has {max(counts)} options but the model has {len(self.codes)} codes.")
        encoded = self.tokenizer(
            [self.prompt_text(item) for item in items], return_tensors="pt", padding=True, add_special_tokens=False
        )
        lengths = encoded["attention_mask"].sum(dim=1)
        if int(lengths.max()) > max_length:
            raise PromptTooLongError(
                f"A prompt has {int(lengths.max())} tokens, more than the maximum context length of {max_length} tokens."
            )
        return Batch(
            input_ids=encoded["input_ids"].to(self.device_name),
            attention_mask=encoded["attention_mask"].to(self.device_name),
            last_index=(lengths - 1).to(self.device_name),
            counts=counts,
            input_tokens=int(lengths.sum()),
        )

    def readout_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """The fp32 logits of all answer codes for the hidden states of the last real tokens."""
        return cast(torch.Tensor, self.readout(hidden)).float()

    def forward(self, batch: Batch) -> torch.Tensor:
        hidden = self.backbone(
            input_ids=batch.input_ids, attention_mask=batch.attention_mask, use_cache=False
        ).last_hidden_state
        rows = torch.arange(hidden.shape[0], device=hidden.device)
        logits = self.readout_logits(hidden[rows, batch.last_index])
        positions = torch.arange(logits.shape[1], device=logits.device)[None, :]
        limits = torch.tensor(batch.counts, device=logits.device)[:, None]
        return logits.masked_fill(positions >= limits, MASK_VALUE)

    @torch.inference_mode()
    def predict_logits(
        self, items: Sequence[PromptItem], batch_size: int = 16, max_length: int = MAX_LENGTH
    ) -> list[list[float]]:
        """Raw logits per item in DISPLAY order, only the item's own codes. The result keeps the input order.

        Batches take the items longest first, so each batch pads to rows of a similar length. Right padding and
        the gather at the last real token make a row's logits independent of its batch.
        """
        was_training = self.training
        self.eval()
        order = sorted(range(len(items)), key=lambda index: len(self.prompt_text(items[index])), reverse=True)
        result: list[list[float]] = [[] for _ in items]
        try:
            for start in range(0, len(order), batch_size):
                indices = order[start : start + batch_size]
                chunk = [items[index] for index in indices]
                logits = self(self.encode(chunk, max_length)).cpu()
                for index, row, item in zip(indices, logits, chunk, strict=True):
                    result[index] = cast(list[float], row[: item.count].tolist())
        finally:
            self.train(was_training)
        return result

    def predict_proba(
        self,
        items: Sequence[PromptItem],
        batch_size: int = 16,
        temperature: float | None = None,
        max_length: int = MAX_LENGTH,
    ) -> list[list[float]]:
        """Probabilities per item in CANONICAL option order (option_keys), after the temperature.

        Without an explicit temperature, each item uses the fitted temperature of its question type.
        """
        if temperature is not None and (not math.isfinite(temperature) or temperature <= 0):
            raise ValueError("The temperature must be positive and finite.")
        result: list[list[float]] = []
        for item, logits in zip(items, self.predict_logits(items, batch_size, max_length), strict=True):
            scale = temperature if temperature is not None else self.config.temperature_for(item.question["type"])
            probabilities = torch.softmax(torch.tensor(logits, dtype=torch.float64) / scale, dim=0).tolist()
            order = item.order or DisplayOrder.identity(item.count)
            result.append(order.to_canonical(probabilities))
        return result

    # ---- requests ----------------------------------------------------------------------------------------------

    def decide(
        self,
        state: Content,
        questions: Mapping[str, object],
        *,
        max_length: int = MAX_LENGTH,
        batch_tokens: int | None = None,
    ) -> DecisionResponse:
        """Answer the named questions about one state, as the server does. Return the response object.

        Raise SchemaError for an invalid question and PromptTooLongError for a prompt longer than `max_length`.
        """
        body = {"state": state, "questions": questions}
        return decide(self, body, name=self.model_name, max_length=max_length, batch_tokens=batch_tokens)
