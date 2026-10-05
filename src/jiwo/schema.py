"""The Jev request and response schema, and validation at the system boundary.

A request has a state and named questions. A question is one of three types:
- choice: pick one of N named options (criteria maps option keys to descriptions or null).
- score: rate the state on 2 to 10 ordered levels (criteria lists the level descriptions, lowest first).
- noul: a yes/no statement (criteria may describe "true" and "false").
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, NotRequired, TypedDict, cast

type JSONValue = str | int | float | bool | None | list[JSONValue] | dict[str, JSONValue]
type Content = str | dict[str, JSONValue] | list[JSONValue]
type QuestionType = Literal["choice", "score", "noul"]

QUESTION_TYPES: tuple[QuestionType, ...] = ("choice", "score", "noul")
MAX_CHOICE_OPTIONS = 255
MIN_SCORE_LEVELS = 2
MAX_SCORE_LEVELS = 10
NOUL_KEYS = ("false", "true")
MAX_QUESTIONS = 64
MAX_KEY_LENGTH = 200


class SchemaError(ValueError):
    """The request or question does not follow the Jev schema."""


class ChoiceQuestion(TypedDict):
    type: Literal["choice"]
    instructions: NotRequired[Content | None]
    criteria: dict[str, Content | None]


class ScoreQuestion(TypedDict):
    type: Literal["score"]
    instructions: NotRequired[Content | None]
    criteria: list[Content]


class NoulQuestion(TypedDict):
    type: Literal["noul"]
    instructions: NotRequired[Content | None]
    criteria: NotRequired[dict[str, Content | None] | None]


type Question = ChoiceQuestion | ScoreQuestion | NoulQuestion


class ChoiceAnswer(TypedDict):
    type: Literal["choice"]
    choice: str
    confidence: float
    probabilities: dict[str, float]


class ScoreAnswer(TypedDict):
    type: Literal["score"]
    score: float
    confidence: float
    legend: dict[str, Content | None]
    probabilities: dict[str, float]


class NoulAnswer(TypedDict):
    type: Literal["noul"]
    noul: float


type Answer = ChoiceAnswer | ScoreAnswer | NoulAnswer


class Usage(TypedDict):
    input_tokens: int
    output_tokens: int


class DecisionRequest(TypedDict):
    state: Content
    questions: dict[str, Question]
    model: NotRequired[str]


class DecisionResponse(TypedDict):
    model: str
    answers: dict[str, Answer]
    usage: Usage


def _is_content(value: object) -> bool:
    return isinstance(value, (str, dict, list))


def _check_instructions(question: Mapping[str, object]) -> None:
    instructions = question.get("instructions")
    if instructions is not None and not _is_content(instructions):
        raise SchemaError("'instructions' must be a string, an object, an array, or null.")


def validate_question(value: object) -> Question:
    """Return the question if it follows the schema. Raise SchemaError with the first problem found."""
    if not isinstance(value, Mapping):
        raise SchemaError("A question must be an object.")
    question_type = value.get("type")
    if question_type not in QUESTION_TYPES:
        raise SchemaError(f"'type' must be one of {list(QUESTION_TYPES)}, not {question_type!r}.")
    _check_instructions(value)
    criteria = value.get("criteria")
    if question_type == "choice":
        if not isinstance(criteria, Mapping) or not criteria:
            raise SchemaError("A choice question needs 'criteria': an object that maps option keys to descriptions.")
        if len(criteria) > MAX_CHOICE_OPTIONS:
            raise SchemaError(f"A choice question can have at most {MAX_CHOICE_OPTIONS} options, not {len(criteria)}.")
        for key, description in criteria.items():
            if not isinstance(key, str) or not key.strip() or len(key) > MAX_KEY_LENGTH:
                raise SchemaError(f"Option keys must be non-empty strings of at most {MAX_KEY_LENGTH} characters.")
            if description is not None and not _is_content(description):
                raise SchemaError(f"The description of option {key!r} must be a string, an object, an array, or null.")
    elif question_type == "score":
        if not isinstance(criteria, list):
            raise SchemaError("A score question needs 'criteria': a list of level descriptions, lowest level first.")
        if not MIN_SCORE_LEVELS <= len(criteria) <= MAX_SCORE_LEVELS:
            raise SchemaError(
                f"A score question needs {MIN_SCORE_LEVELS} to {MAX_SCORE_LEVELS} levels, not {len(criteria)}."
            )
        if not all(_is_content(level) for level in criteria):
            raise SchemaError("Every score level must be a string, an object, or an array.")
    else:
        if criteria is not None:
            if not isinstance(criteria, Mapping) or not set(criteria) <= set(NOUL_KEYS):
                raise SchemaError("Noul 'criteria' may only describe the keys 'true' and 'false'.")
            if not all(description is None or _is_content(description) for description in criteria.values()):
                raise SchemaError("Noul descriptions must be strings, objects, arrays, or null.")
    return cast(Question, dict(value))


def validate_request(value: object) -> DecisionRequest:
    """Return the request if it follows the schema. Raise SchemaError with the first problem found."""
    if not isinstance(value, Mapping):
        raise SchemaError("The request body must be a JSON object.")
    state = value.get("state")
    if not _is_content(state):
        raise SchemaError("'state' must be a string, an object, or an array.")
    questions = value.get("questions")
    if not isinstance(questions, Mapping) or not questions:
        raise SchemaError("'questions' must be a non-empty object that maps names to questions.")
    if len(questions) > MAX_QUESTIONS:
        raise SchemaError(f"A request can have at most {MAX_QUESTIONS} questions, not {len(questions)}.")
    checked: dict[str, Question] = {}
    for name, question in questions.items():
        if not isinstance(name, str) or not name.strip():
            raise SchemaError("Question names must be non-empty strings.")
        try:
            checked[name] = validate_question(question)
        except SchemaError as error:
            raise SchemaError(f"Question {name!r}: {error}") from error
    model = value.get("model")
    if model is not None and not isinstance(model, str):
        raise SchemaError("'model' must be a string.")
    request: DecisionRequest = {"state": cast(Content, state), "questions": checked}
    if model is not None:
        request["model"] = model
    return request


def option_keys(question: Question) -> list[str]:
    """The answer keys in canonical order: choice keys as given, score levels "0".."n-1", noul "false", "true"."""
    if question["type"] == "choice":
        return list(question["criteria"])
    if question["type"] == "score":
        return [str(index) for index in range(len(question["criteria"]))]
    return list(NOUL_KEYS)
