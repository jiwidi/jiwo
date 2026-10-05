"""The jiwo command line.

jiwo serve                     serve a model over HTTP (settings come from JIWO_* environment variables)
jiwo decide MODEL [REQUEST]    answer one request (a JSON file, or standard input) and print the response
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DESCRIPTION = "Run jiwo decision models."
SERVE_HELP = "Serve a model over HTTP (POST /v1/systemone). The settings come from the JIWO_* environment variables."
DECIDE_HELP = "Answer one request and print the response as JSON."


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="jiwo", description=DESCRIPTION)
    commands = root.add_subparsers(dest="command", required=True, metavar="COMMAND")
    commands.add_parser("serve", help=SERVE_HELP, description=SERVE_HELP)
    decide = commands.add_parser("decide", help=DECIDE_HELP, description=DECIDE_HELP)
    decide.add_argument(
        "model", help="a checkpoint directory or a Hugging Face repository id, for example eljiwo/jiwo-0.8b"
    )
    decide.add_argument(
        "request", nargs="?", type=Path, help='a JSON file with "state" and "questions" (default: standard input)'
    )
    decide.add_argument("--revision", help="the branch, tag or commit of a Hugging Face repository")
    decide.add_argument("--device", help="cuda, mps or cpu (default: the best available)")
    return root


def read_request(path: Path | None) -> object:
    """The request object from a file or from standard input. Raise ValueError for a bad file."""
    try:
        text = path.read_text() if path is not None else sys.stdin.read()
    except OSError as error:
        raise ValueError(f"Cannot read the request file {path}: {error.strerror}.") from error
    try:
        return json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"The request is not valid JSON: {error.msg} at line {error.lineno}.") from error


def run_decide(args: argparse.Namespace) -> int:
    """Answer one request. Return the exit code: 0 for an answer, 2 for a bad request."""
    from jiwo.inference import CapacityError, decide
    from jiwo.model import DecisionModel, PromptTooLongError
    from jiwo.schema import SchemaError, validate_request

    try:
        body = read_request(args.request)
        validate_request(body)  # before the slow model load
    except ValueError as error:  # SchemaError is a ValueError
        print(error, file=sys.stderr)
        return 2
    try:
        model = DecisionModel.from_pretrained(args.model, revision=args.revision, device=args.device)
    except FileNotFoundError as error:
        print(error, file=sys.stderr)
        return 2
    try:
        response = decide(model, body, name=model.model_name)
    except (SchemaError, PromptTooLongError, CapacityError) as error:
        print(error, file=sys.stderr)
        return 2
    print(json.dumps(response, indent=2, ensure_ascii=False))
    return 0


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    if args.command == "serve":
        from jiwo.serving.app import main as serve

        serve()
        return
    raise SystemExit(run_decide(args))


if __name__ == "__main__":
    main()
