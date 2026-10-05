"""The Decision Index engine gives the server's answers and refuses capacity limits as unsupported."""

from pathlib import Path
from typing import Any

import pytest
import torch
from fastapi.testclient import TestClient

pytest.importorskip("decision_index")

from decision_index.engines import Unsupported, load_engine, validate

from jiwo.index_engine import JiwoEngine
from jiwo.schema import SchemaError
from jiwo.serving import app as server

STATE = {"ticket": "The parcel arrived crushed.", "order": 1234}
QUESTIONS: dict[str, Any] = {
    "team": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "Payments", "shipping": None}},
    "angry": {"type": "noul", "instructions": "The customer is angry."},
}


@pytest.fixture(scope="module")
def engine(checkpoint: Path) -> JiwoEngine:
    loaded = load_engine("jiwo.index_engine:JiwoEngine", model=str(checkpoint), device="cpu")
    assert isinstance(loaded, JiwoEngine)
    return loaded


def test_engine_answers_like_the_http_server(engine: JiwoEngine, checkpoint: Path) -> None:
    response, raw = engine(state=STATE, questions=QUESTIONS)
    validate(QUESTIONS, response)
    assert raw is None
    server.load(str(checkpoint), device="cpu")
    served = TestClient(server.app).post("/v1/systemone", json={"state": STATE, "questions": QUESTIONS}).json()
    assert response["model"] == served["model"]
    assert response["answers"] == served["answers"]  # the same code, and JSON keeps every float exact


def test_capacity_limits_are_unsupported(engine: JiwoEngine, monkeypatch: pytest.MonkeyPatch) -> None:
    many = {"many": {"type": "choice", "criteria": {str(i): None for i in range(5)}}}
    with pytest.raises(Unsupported, match="options per choice"):
        engine("s", many)
    monkeypatch.setattr(engine, "max_length", 8)
    with pytest.raises(Unsupported, match="maximum context length"):
        engine("s", QUESTIONS)


def test_out_of_memory_at_batch_one_is_unsupported(engine: JiwoEngine, monkeypatch: pytest.MonkeyPatch) -> None:
    def always(items: Any, batch_size: int = 16, **options: Any) -> Any:
        raise torch.OutOfMemoryError("CUDA out of memory (test)")

    monkeypatch.setattr(engine.model, "predict_proba", always)
    with pytest.raises(Unsupported, match="too many tokens"):
        engine("s", QUESTIONS)


def test_other_request_errors_stay_errors(engine: JiwoEngine) -> None:
    with pytest.raises(SchemaError, match="'type' must be one of"):
        engine("s", {"q": {"type": "rank"}})


def test_engine_reports_its_provenance_and_runtime(engine: JiwoEngine, checkpoint: Path) -> None:
    assert engine.provenance["checkpoint"] == str(checkpoint)
    assert engine.provenance["max_length"] == 8192 and engine.provenance["batch_tokens"] == 16_384  # the CPU default
    assert engine.runtime()["device"] == "cpu"
    engine.synchronize()
    engine.warmup()  # the kit's two-option warmup question


def test_engine_options_set_the_name_and_the_attention_kernel(checkpoint: Path) -> None:
    before = torch.backends.cuda.cudnn_sdp_enabled()
    try:
        named = load_engine(  # through the kit: its load_engine takes the engine name as its own `name` argument
            "jiwo.index_engine:JiwoEngine",
            model=str(checkpoint),
            device="cpu",
            model_name="jiwo-test",
            cudnn_attention=False,
        )
        assert torch.backends.cuda.cudnn_sdp_enabled() is False
    finally:
        torch.backends.cuda.enable_cudnn_sdp(before)
    assert named("s", QUESTIONS)[0]["model"] == "jiwo-test"
