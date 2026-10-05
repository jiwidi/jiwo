import pytest

from jiwo.schema import SchemaError, option_keys, validate_question, validate_request


def test_valid_request_passes_and_keeps_questions() -> None:
    request = validate_request(
        {
            "state": {"ticket": "x"},
            "model": "jiwo",
            "questions": {
                "team": {"type": "choice", "criteria": {"a": "A team", "b": None}},
                "urgent": {"type": "noul", "instructions": "It is urgent."},
                "level": {"type": "score", "criteria": ["low", "high"]},
            },
        }
    )
    assert set(request["questions"]) == {"team", "urgent", "level"}
    assert request["model"] == "jiwo"


@pytest.mark.parametrize(
    "question, message",
    [
        ({"type": "rank"}, "'type' must be one of"),
        ({"type": "choice", "criteria": {}}, "needs 'criteria'"),
        ({"type": "choice", "criteria": {f"k{i}": None for i in range(256)}}, "at most 255"),
        ({"type": "score", "criteria": ["only one"]}, "2 to 10 levels"),
        ({"type": "score", "criteria": [str(i) for i in range(11)]}, "2 to 10 levels"),
        ({"type": "noul", "criteria": {"maybe": "x"}}, "'true' and 'false'"),
        ({"type": "choice", "criteria": {"": "x"}}, "non-empty strings"),
    ],
)
def test_invalid_questions_raise_clear_errors(question: dict[str, object], message: str) -> None:
    with pytest.raises(SchemaError, match=message):
        validate_question(question)


def test_request_errors_name_the_question() -> None:
    with pytest.raises(SchemaError, match="Question 'bad'"):
        validate_request({"state": "s", "questions": {"bad": {"type": "score", "criteria": []}}})


def test_request_needs_state_and_questions() -> None:
    with pytest.raises(SchemaError, match="'state'"):
        validate_request({"questions": {"q": {"type": "noul"}}})
    with pytest.raises(SchemaError, match="'questions'"):
        validate_request({"state": "s", "questions": {}})


def test_option_keys_canonical_order() -> None:
    assert option_keys({"type": "choice", "criteria": {"x": None, "y": None}}) == ["x", "y"]
    assert option_keys({"type": "score", "criteria": ["a", "b", "c"]}) == ["0", "1", "2"]
    assert option_keys({"type": "noul"}) == ["false", "true"]
