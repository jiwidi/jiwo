"""Test helpers: write small checkpoints in the jiwo format."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoTokenizer

from jiwo.model import CONFIG_FILE, READOUT_FILE, DecisionConfig, answer_codes, load_text_decoder
from jiwo.schema import JSONValue


def write_checkpoint(
    base_model: str,
    directory: Path,
    *,
    codes: int = 16,
    max_trained_options: int | None = None,
    metadata: dict[str, JSONValue] | None = None,
    lora_rank: int | None = None,
) -> Path:
    """Write a checkpoint in the jiwo format with a random readout (seeded). Tests check mechanics, not quality.

    With `lora_rank`, the checkpoint holds a random LoRA adapter instead of the backbone weights.
    """
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    names, token_ids = answer_codes(tokenizer, codes)
    decoder, _ = load_text_decoder(base_model, None, torch.float32)
    generator = torch.Generator().manual_seed(0)
    readout = torch.randn(len(token_ids), decoder.config.hidden_size, generator=generator)
    directory.mkdir(parents=True)
    if lora_rank:
        from peft import LoraConfig, get_peft_model

        torch.manual_seed(0)
        settings = LoraConfig(r=lora_rank, lora_alpha=2 * lora_rank, target_modules="all-linear", bias="none")
        adapted = get_peft_model(decoder, settings)
        with torch.no_grad():
            for name, parameter in adapted.named_parameters():
                if "lora_B" in name:
                    parameter.normal_(0.0, 0.05)  # a nonzero adapter changes the logits
        adapted.save_pretrained(str(directory), save_embedding_layers=False)
    else:
        decoder.save_pretrained(str(directory))
    tokenizer.save_pretrained(str(directory))
    save_file({"weight": readout.contiguous()}, str(directory / READOUT_FILE))
    config = DecisionConfig(
        base_model=base_model,
        revision=None,
        codes=names,
        token_ids=token_ids,
        max_trained_options=max_trained_options or codes,
        metadata=metadata or {},
    )
    (directory / CONFIG_FILE).write_text(json.dumps(config.to_json(), indent=2) + "\n")
    return directory
