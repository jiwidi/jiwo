import math

import pytest

from jiwo.answers import answer_probabilities, build_answer, choice_confidence, score_confidence
from jiwo.schema import Question

TEAM: Question = {"type": "choice", "criteria": {"account": None, "shipping": None, "billing": None}}
URGENCY: Question = {"type": "score", "criteria": ["Not urgent", "Somewhat urgent", "Very urgent"]}


def test_confidence_formulas_on_a_worked_example() -> None:
    # Synthetic numbers. Choice: (0.6 - 1/3) / (1 - 1/3) = 0.4.
    choice = build_answer(TEAM, [0.1, 0.6, 0.3])
    assert choice["type"] == "choice" and choice["choice"] == "shipping"
    assert math.isclose(choice["confidence"], 0.4, abs_tol=1e-9)
    # Score: E[level] = 0.5 + 2 * 0.3 = 1.1. E|level - 1| = 0.5. Baseline (1 + 0 + 1) / 3. Confidence 1 - 0.75.
    score = build_answer(URGENCY, [0.2, 0.5, 0.3])
    assert score["type"] == "score"
    assert math.isclose(score["score"], 1.1, abs_tol=1e-9)
    assert math.isclose(score["confidence"], 0.25, abs_tol=1e-9)
    assert score["legend"] == {"0": "Not urgent", "1": "Somewhat urgent", "2": "Very urgent"}


def test_noul_answer_is_probability_of_true() -> None:
    answer = build_answer({"type": "noul"}, [0.3, 0.7])
    assert answer == {"type": "noul", "noul": 0.7}


def test_uniform_and_certain_confidence() -> None:
    assert choice_confidence([1 / 3] * 3) == pytest.approx(0.0)
    assert choice_confidence([0.0, 1.0, 0.0]) == pytest.approx(1.0)
    assert score_confidence([0.0, 0.0, 1.0]) == pytest.approx(1.0)


def test_probabilities_are_normalized_and_validated() -> None:
    answer = build_answer(TEAM, [1.0, 1.0, 2.0])
    assert answer["type"] == "choice"
    assert answer["probabilities"]["billing"] == pytest.approx(0.5)
    with pytest.raises(ValueError, match="Expected 3"):
        build_answer(TEAM, [0.5, 0.5])
    with pytest.raises(ValueError, match="non-negative"):
        build_answer(TEAM, [0.5, -0.1, 0.6])


def test_answer_probabilities_round_trip() -> None:
    for question, values in ((TEAM, [0.2, 0.5, 0.3]), (URGENCY, [0.1, 0.1, 0.8]), ({"type": "noul"}, [0.4, 0.6])):
        recovered = answer_probabilities(question, build_answer(question, values))
        assert recovered == pytest.approx(values)
