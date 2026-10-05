"""Server tests with the tiny random Qwen3.5 checkpoint."""

from pathlib import Path
from typing import Any

import pytest
import torch
from fastapi.testclient import TestClient

from jiwo.serving import app as server


@pytest.fixture(scope="module")
def client(checkpoint: Path) -> TestClient:
    server.load(str(checkpoint), device="cpu")
    return TestClient(server.app)


BODY = {
    "state": "The parcel arrived crushed.",
    "questions": {
        "team": {"type": "choice", "criteria": {"billing": "Payments", "shipping": "Parcels", "account": None}},
        "angry": {"type": "noul", "instructions": "The customer is angry."},
        "urgency": {"type": "score", "criteria": ["low", "mid", "high"]},
    },
}


def test_systemone_returns_jev_shaped_answers(client: TestClient) -> None:
    response = client.post("/v1/systemone", json=BODY)
    assert response.status_code == 200, response.text
    answers = response.json()["answers"]
    assert answers["team"]["choice"] in {"billing", "shipping", "account"}
    assert set(answers["team"]["probabilities"]) == {"billing", "shipping", "account"}
    assert 0.0 <= answers["angry"]["noul"] <= 1.0
    assert set(answers["urgency"]) == {"type", "score", "confidence", "legend", "probabilities"}
    assert response.json()["usage"]["input_tokens"] > 0


def test_schema_errors_are_400(client: TestClient) -> None:
    response = client.post(
        "/v1/systemone", json={"state": "s", "questions": {"q": {"type": "score", "criteria": ["one"]}}}
    )
    assert response.status_code == 400 and "2 to 10 levels" in response.json()["error"]["message"]


def test_too_many_options_for_the_trained_model(client: TestClient) -> None:
    body = {"state": "s", "questions": {"q": {"type": "choice", "criteria": {str(i): None for i in range(6)}}}}
    response = client.post("/v1/systemone", json=body)
    message = response.json()["error"]["message"]
    assert response.status_code == 400 and "at most 4 options" in message
    assert "options per choice" in message  # clients can match this text as a capacity error


def test_long_prompt_names_the_maximum_context_length(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JIWO_MAX_LENGTH", "16")
    response = client.post("/v1/systemone", json=BODY)
    assert response.status_code == 400
    assert "maximum context length of 16 tokens" in response.json()["error"]["message"]
    monkeypatch.setenv("JIWO_MAX_LENGTH", "100000")
    assert client.post("/v1/systemone", json=BODY).status_code == 200


def test_api_key(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JIWO_API_KEY", "secret")
    assert client.post("/v1/systemone", json=BODY).status_code == 401
    assert client.post("/v1/systemone", json=BODY, headers={"Authorization": "Bearer secret"}).status_code == 200


def test_model_name_comes_from_the_environment(
    checkpoint: Path, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert client.get("/v1/models").json()["data"][0]["id"] == "jiwo"
    monkeypatch.setenv("JIWO_MODEL_NAME", "jiwo-0.8b")
    server.load(str(checkpoint), device="cpu")
    try:
        assert client.post("/v1/systemone", json=BODY).json()["model"] == "jiwo-0.8b"
        assert client.get("/health").json()["model"] == "jiwo-0.8b"
    finally:
        monkeypatch.delenv("JIWO_MODEL_NAME")
        server.load(str(checkpoint), device="cpu")


def test_a_small_token_budget_splits_batches_without_changing_answers(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def numbers(answers: dict[str, Any]) -> dict[str, Any]:
        return {name: answer.get("probabilities", answer.get("noul")) for name, answer in answers.items()}

    model = server.service.model
    assert model is not None
    sizes: list[int] = []
    original = model.predict_proba

    def spy(items: Any, batch_size: int = 16, **options: Any) -> Any:
        sizes.append(batch_size)
        return original(items, batch_size=batch_size, **options)

    monkeypatch.setattr(model, "predict_proba", spy)
    whole = numbers(client.post("/v1/systemone", json=BODY).json()["answers"])
    monkeypatch.setenv("JIWO_BATCH_TOKENS", "1")  # one prompt per batch
    split = numbers(client.post("/v1/systemone", json=BODY).json()["answers"])
    assert sizes == [3, 1]
    for name, values in whole.items():
        assert split[name] == pytest.approx(values, abs=1e-4)


def test_out_of_memory_retries_with_smaller_batches(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    model = server.service.model
    assert model is not None
    sizes: list[int] = []
    original = model.predict_proba

    def flaky(items: Any, batch_size: int = 16, **options: Any) -> Any:
        sizes.append(batch_size)
        if batch_size > 1:
            raise torch.OutOfMemoryError("CUDA out of memory (test)")
        return original(items, batch_size=batch_size, **options)

    monkeypatch.setattr(model, "predict_proba", flaky)
    response = client.post("/v1/systemone", json=BODY)
    assert response.status_code == 200, response.text
    assert sizes == [3, 1]


def test_out_of_memory_at_batch_one_is_a_capacity_error(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    model = server.service.model
    assert model is not None

    def always(items: Any, batch_size: int = 16, **options: Any) -> Any:
        raise torch.OutOfMemoryError("CUDA out of memory (test)")

    monkeypatch.setattr(model, "predict_proba", always)
    response = client.post("/v1/systemone", json=BODY)
    assert response.status_code == 400
    assert "too many tokens" in response.json()["error"]["message"]  # clients can match this text as a capacity error


def test_cudnn_attention_can_be_switched_off(checkpoint: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    before = torch.backends.cuda.cudnn_sdp_enabled()
    monkeypatch.setenv("JIWO_CUDNN_ATTENTION", "0")
    try:
        server.load(str(checkpoint), device="cpu")
        assert torch.backends.cuda.cudnn_sdp_enabled() is False
    finally:
        torch.backends.cuda.enable_cudnn_sdp(before)
        monkeypatch.delenv("JIWO_CUDNN_ATTENTION")
        server.load(str(checkpoint), device="cpu")


def test_api_key_errors_use_the_error_envelope(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JIWO_API_KEY", "secret")
    response = client.post("/v1/systemone", json=BODY, headers={"Authorization": "Bearer wrong"})
    assert response.status_code == 401
    assert response.json()["error"]["type"] == "unauthorized"
    assert client.get("/health").status_code == 200  # health and models need no key


def test_invalid_json_is_400(client: TestClient) -> None:
    response = client.post("/v1/systemone", content=b"{not json", headers={"content-type": "application/json"})
    assert response.status_code == 400
    assert response.json()["error"]["message"].startswith("The request body is not valid JSON.")


def test_deeply_nested_json_is_400_not_500(client: TestClient) -> None:
    response = client.post("/v1/systemone", content=b"[" * 100_000 + b"]" * 100_000)
    assert response.status_code == 400


def test_large_bodies_are_refused_before_parsing(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JIWO_MAX_BODY_BYTES", "100")
    response = client.post("/v1/systemone", json={"state": "x" * 200, "questions": {}})
    assert response.status_code == 413
    assert "larger than 100 bytes" in response.json()["error"]["message"]

    def chunks() -> Any:  # no content-length header: the server counts the streamed bytes
        yield b'{"state": "'
        yield b"x" * 200
        yield b'"}'

    assert client.post("/v1/systemone", content=chunks()).status_code == 413


def test_a_request_body_that_is_not_an_object_is_400(client: TestClient) -> None:
    response = client.post("/v1/systemone", json=["state", "questions"])
    assert response.status_code == 400
    assert "JSON object" in response.json()["error"]["message"]


def test_no_model_is_503(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server.service, "model", None)
    assert client.post("/v1/systemone", json=BODY).status_code == 503
    assert client.get("/health").json()["status"] == "no-model"


def test_interactive_docs_are_not_served(client: TestClient) -> None:
    for path in ("/docs", "/redoc", "/openapi.json"):
        assert client.get(path).status_code == 404


def test_bad_settings_stop_the_start(checkpoint: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JIWO_MAX_LENGTH", "many")
    with pytest.raises(RuntimeError, match="JIWO_MAX_LENGTH must be a non-negative integer"):
        server.load(str(checkpoint), device="cpu")
    monkeypatch.delenv("JIWO_MAX_LENGTH")
    monkeypatch.delenv("JIWO_CHECKPOINT", raising=False)
    with pytest.raises(RuntimeError, match="Set JIWO_CHECKPOINT"):
        server.load(None, device="cpu")


def test_main_loads_the_model_and_runs_uvicorn(checkpoint: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import uvicorn

    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **options: calls.append(options))
    monkeypatch.setenv("JIWO_CHECKPOINT", str(checkpoint))
    monkeypatch.setenv("JIWO_DEVICE", "cpu")
    monkeypatch.setenv("PORT", "9001")
    server.main()
    assert calls == [{"host": "127.0.0.1", "port": 9001}]
    assert server.service.model is not None
