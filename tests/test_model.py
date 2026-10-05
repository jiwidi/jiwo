"""Model tests with a tiny random Qwen3.5. They check mechanics, not quality."""

from pathlib import Path
from typing import Any

import pytest
import torch

from jiwo import model as model_module
from jiwo.model import DecisionModel, PromptTooLongError, checkpoint_directory
from jiwo.prompting import DisplayOrder, PromptItem
from tests.helpers import write_checkpoint

QUESTION: Any = {
    "type": "choice",
    "instructions": "Which team?",
    "criteria": {"a": "Payments", "b": "Parcels", "c": None},
}


@pytest.fixture(scope="module")
def model(tiny_model_name: str, tmp_path_factory: pytest.TempPathFactory) -> DecisionModel:
    directory = write_checkpoint(tiny_model_name, tmp_path_factory.mktemp("model") / "ck", codes=64)
    return DecisionModel.from_checkpoint(directory, device="cpu", dtype=torch.float32)


def test_right_padding_batch_matches_single_items(model: DecisionModel) -> None:
    items = [PromptItem("short", QUESTION), PromptItem("a much longer state " * 20, QUESTION)]
    batched = model.predict_logits(items, batch_size=2)
    single = [model.predict_logits([item], batch_size=1)[0] for item in items]
    for left, right in zip(batched, single, strict=True):
        assert left == pytest.approx(right, abs=1e-5)


def test_length_sorted_batches_return_results_in_input_order(model: DecisionModel) -> None:
    states = ["a much longer state " * 12, "s", "medium state " * 5, "tiny", "the longest state of all " * 20]
    items = [PromptItem(state, QUESTION) for state in states]
    batched = model.predict_logits(items, batch_size=2)
    single = [model.predict_logits([item], batch_size=1)[0] for item in items]
    for left, right in zip(batched, single, strict=True):
        assert left == pytest.approx(right, abs=1e-5)


def test_lora_checkpoint_loads_merged_into_its_base_model(tiny_model_name: str, tmp_path: Path) -> None:
    items = [PromptItem("state", QUESTION), PromptItem("a longer state " * 10, QUESTION)]
    base = DecisionModel.from_checkpoint(
        write_checkpoint(tiny_model_name, tmp_path / "base", codes=64), device="cpu", dtype=torch.float32
    )
    adapter = write_checkpoint(tiny_model_name, tmp_path / "lora", codes=64, lora_rank=4)
    assert (adapter / "adapter_config.json").is_file()
    assert not list(adapter.glob("model*.safetensors"))
    lora = DecisionModel.from_checkpoint(adapter, device="cpu", dtype=torch.float32)
    assert not hasattr(lora.backbone, "peft_config")  # merged, not wrapped
    assert not any(parameter.requires_grad for parameter in lora.parameters())
    base_logits, lora_logits = base.predict_logits(items), lora.predict_logits(items)
    assert max(abs(a - b) for a, b in zip(base_logits[0], lora_logits[0], strict=True)) > 1e-3


def test_permuted_display_order_maps_back_to_canonical(model: DecisionModel) -> None:
    base = PromptItem("state", QUESTION)
    shuffled = PromptItem("state", QUESTION, DisplayOrder((2, 0, 1)))
    probabilities = model.predict_proba([base, shuffled])
    assert all(len(row) == 3 and sum(row) == pytest.approx(1.0) for row in probabilities)


def test_masked_codes_get_no_probability(model: DecisionModel) -> None:
    batch = model.encode([PromptItem("s", QUESTION)])
    logits = model(batch)
    assert logits.shape[1] == 64
    assert torch.softmax(logits, dim=-1)[0, 3:].sum().item() < 1e-12


def test_long_prompt_is_rejected_not_truncated(model: DecisionModel) -> None:
    with pytest.raises(PromptTooLongError):
        model.encode([PromptItem("word " * 500, QUESTION)], max_length=64)


def test_explicit_temperature_must_be_positive(model: DecisionModel) -> None:
    with pytest.raises(ValueError, match="temperature"):
        model.predict_proba([PromptItem("s", QUESTION)], temperature=0.0)


def test_checkpoint_metadata_and_model_name(tiny_model_name: str, tmp_path: Path) -> None:
    directory = write_checkpoint(tiny_model_name, tmp_path / "named", metadata={"name": "jiwo-test", "step": 3})
    loaded = DecisionModel.from_checkpoint(directory, device="cpu", dtype=torch.float32)
    assert loaded.config.metadata["step"] == 3
    assert loaded.model_name == "jiwo-test"
    assert loaded.config.max_trained_options == 16


def test_from_pretrained_reads_a_local_directory(checkpoint: Path) -> None:
    loaded = DecisionModel.from_pretrained(checkpoint, device="cpu", dtype=torch.float32)
    assert loaded.model_name == "jiwo" and loaded.config.max_trained_options == 4


def test_from_pretrained_downloads_a_repository_id(checkpoint: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str | None]] = []

    def download(repo_id: str, revision: str | None = None) -> str:
        calls.append((repo_id, revision))
        return str(checkpoint)

    monkeypatch.setattr("huggingface_hub.snapshot_download", download)
    assert checkpoint_directory("owner/jiwo-test", revision="v1") == checkpoint
    assert calls == [("owner/jiwo-test", "v1")]


def test_a_repository_id_names_the_model(checkpoint: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda repo_id, revision=None: str(checkpoint))
    loaded = DecisionModel.from_pretrained("owner/jiwo-test", device="cpu", dtype=torch.float32)
    assert loaded.model_name == "jiwo-test"


def test_checkpoint_directory_refuses_a_missing_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="not a checkpoint directory"):
        checkpoint_directory(tmp_path / "missing" / "deeper")
    with pytest.raises(FileNotFoundError, match=r"eljiwo/jiwo-0\.8b"):
        checkpoint_directory("no-slash-here")


def test_unsupported_checkpoint_format_is_refused() -> None:
    with pytest.raises(ValueError, match="Unsupported checkpoint format"):
        model_module.DecisionConfig.from_json({"format_version": 99})
