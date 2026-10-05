"""Command-line tests: help, the decide command and the serve command."""

import io
import json
from pathlib import Path
from typing import Any

import pytest

from jiwo import cli

REQUEST = {"state": "The parcel arrived crushed.", "questions": {"angry": {"type": "noul", "instructions": "Angry?"}}}


def test_help_lists_the_commands(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as stop:
        cli.main(["--help"])
    assert stop.value.code == 0
    out = capsys.readouterr().out
    assert "serve" in out and "decide" in out


def test_a_command_is_required(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as stop:
        cli.main([])
    assert stop.value.code == 2
    assert "COMMAND" in capsys.readouterr().err


def test_decide_reads_a_file_and_prints_the_response(
    checkpoint: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    request = tmp_path / "request.json"
    request.write_text(json.dumps(REQUEST))
    with pytest.raises(SystemExit) as stop:
        cli.main(["decide", str(checkpoint), str(request), "--device", "cpu"])
    assert stop.value.code == 0
    response = json.loads(capsys.readouterr().out)
    assert set(response) == {"model", "answers", "usage"}
    assert 0.0 <= response["answers"]["angry"]["noul"] <= 1.0


def test_decide_reads_standard_input(
    checkpoint: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(REQUEST)))
    with pytest.raises(SystemExit) as stop:
        cli.main(["decide", str(checkpoint), "--device", "cpu"])
    assert stop.value.code == 0
    assert json.loads(capsys.readouterr().out)["answers"]["angry"]["type"] == "noul"


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("{not json", "The request is not valid JSON"),
        ('["state"]', "The request body must be a JSON object."),
        ('{"state": "s", "questions": {"q": {"type": "rank"}}}', "'type' must be one of"),
    ],
)
def test_decide_refuses_a_bad_request_before_it_loads_the_model(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], content: str, message: str
) -> None:
    request = tmp_path / "request.json"
    request.write_text(content)
    with pytest.raises(SystemExit) as stop:
        cli.main(["decide", str(tmp_path / "no-model-here" / "x"), str(request)])  # the model is never loaded
    assert stop.value.code == 2
    assert message in capsys.readouterr().err


def test_decide_reports_a_missing_request_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as stop:
        cli.main(["decide", "owner/model", str(tmp_path / "missing.json")])
    assert stop.value.code == 2
    assert "Cannot read the request file" in capsys.readouterr().err


def test_decide_reports_a_missing_model(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    request = tmp_path / "request.json"
    request.write_text(json.dumps(REQUEST))
    with pytest.raises(SystemExit) as stop:
        cli.main(["decide", str(tmp_path / "missing" / "model"), str(request)])
    assert stop.value.code == 2
    assert "not a checkpoint directory" in capsys.readouterr().err


def test_decide_reports_a_prompt_that_is_too_long(
    checkpoint: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    request = tmp_path / "request.json"
    request.write_text(json.dumps({**REQUEST, "state": "word " * 9000}))
    with pytest.raises(SystemExit) as stop:
        cli.main(["decide", str(checkpoint), str(request), "--device", "cpu"])
    assert stop.value.code == 2
    assert "maximum context length" in capsys.readouterr().err


def test_serve_runs_the_server(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[Any] = []
    monkeypatch.setattr("jiwo.serving.app.main", lambda: calls.append("serve"))
    cli.main(["serve"])
    assert calls == ["serve"]
