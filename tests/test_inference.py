"""Request-level tests: the Python API answers like the server."""

from pathlib import Path
from typing import Any

import pytest
import torch
from fastapi.testclient import TestClient

from jiwo.inference import check_option_limit, decide
from jiwo.model import DecisionModel, PromptTooLongError
from jiwo.schema import SchemaError, validate_request
from jiwo.serving import app as server

STATE = "The parcel arrived crushed."
QUESTIONS: dict[str, Any] = {
    "team": {"type": "choice", "criteria": {"billing": "Payments", "shipping": "Parcels", "account": None}},
    "angry": {"type": "noul", "instructions": "The customer is angry."},
    "urgency": {"type": "score", "criteria": ["low", "mid", "high"]},
}


@pytest.fixture(scope="module")
def model(checkpoint: Path) -> DecisionModel:
    return DecisionModel.from_pretrained(checkpoint, device="cpu", dtype=torch.float32)


def test_decide_gives_the_same_answers_as_the_server(model: DecisionModel, checkpoint: Path) -> None:
    response = model.decide(STATE, QUESTIONS)
    server.load(str(checkpoint), device="cpu")
    served = TestClient(server.app).post("/v1/systemone", json={"state": STATE, "questions": QUESTIONS}).json()
    assert response["model"] == served["model"] == "jiwo"
    assert response["usage"] == served["usage"]
    assert response["answers"] == served["answers"]  # the same code, and JSON keeps every float exact


def test_decide_answers_every_question_type(model: DecisionModel) -> None:
    answers = model.decide(STATE, QUESTIONS)["answers"]
    assert answers["team"]["type"] == "choice" and sum(answers["team"]["probabilities"].values()) == pytest.approx(1.0)
    assert answers["angry"]["type"] == "noul" and 0.0 <= answers["angry"]["noul"] <= 1.0
    assert answers["urgency"]["type"] == "score" and 0.0 <= answers["urgency"]["score"] <= 2.0


def test_decide_validates_the_request(model: DecisionModel) -> None:
    with pytest.raises(SchemaError, match="'questions'"):
        model.decide(STATE, {})
    with pytest.raises(SchemaError, match="'state'"):
        decide(model, {"questions": QUESTIONS}, name="jiwo")
    with pytest.raises(PromptTooLongError, match="maximum context length of 8 tokens"):
        model.decide(STATE, QUESTIONS, max_length=8)


def test_option_limit_names_the_question_and_the_limit(model: DecisionModel) -> None:
    request = validate_request(
        {"state": "s", "questions": {"many": {"type": "choice", "criteria": {str(i): None for i in range(5)}}}}
    )
    with pytest.raises(SchemaError, match=r"'many' has 5 options\. This model supports at most 4 options per choice"):
        check_option_limit(model, request)


def test_batch_tokens_set_the_batch_size(model: DecisionModel, monkeypatch: pytest.MonkeyPatch) -> None:
    sizes: list[int] = []
    original = model.predict_proba

    def spy(items: Any, batch_size: int = 16, **options: Any) -> Any:
        sizes.append(batch_size)
        return original(items, batch_size=batch_size, **options)

    monkeypatch.setattr(model, "predict_proba", spy)
    model.decide(STATE, QUESTIONS, batch_tokens=1)
    model.decide(STATE, QUESTIONS)
    assert sizes == [1, 3]


def test_default_batch_tokens_depend_on_the_device() -> None:
    from jiwo.inference import BATCH_TOKENS, HOST_BATCH_TOKENS, default_batch_tokens

    assert default_batch_tokens("cuda") == default_batch_tokens("cuda:1") == BATCH_TOKENS == 262_144
    assert default_batch_tokens("cpu") == default_batch_tokens("mps") == HOST_BATCH_TOKENS == 16_384
