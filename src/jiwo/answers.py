"""Turn a probability vector over a question's options into a Jev-style answer.

The confidence formulas:
- choice: (p_best - 1/n) / (1 - 1/n), so a uniform distribution gives 0 and a certain one gives 1.
- score: 1 - E|level - best| / baseline, where baseline is the mean distance to the scale midpoint.
- noul: no confidence field. The probability of "true" is the answer.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from jiwo.schema import Answer, ChoiceAnswer, Content, NoulAnswer, Question, ScoreAnswer, option_keys


def normalize(probabilities: Sequence[float]) -> list[float]:
    """Return the probabilities divided by their sum. Raise ValueError for bad input."""
    values = [float(value) for value in probabilities]
    if not values:
        raise ValueError("The probability vector is empty.")
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("Probabilities must be finite and non-negative.")
    total = sum(values)
    if total <= 0:
        raise ValueError("Probabilities must have a positive sum.")
    return [value / total for value in values]


def choice_confidence(values: Sequence[float]) -> float:
    count = len(values)
    if count == 1:
        return 1.0
    best = max(values)
    return min(1.0, max(0.0, (best - 1 / count) / (1 - 1 / count)))


def score_confidence(values: Sequence[float]) -> float:
    count = len(values)
    best = max(range(count), key=values.__getitem__)
    distance = sum(probability * abs(index - best) for index, probability in enumerate(values))
    midpoint = (count - 1) / 2
    baseline = sum(abs(index - midpoint) for index in range(count)) / count
    return min(1.0, max(0.0, 1.0 - distance / baseline))


def expected_score(values: Sequence[float]) -> float:
    return sum(index * probability for index, probability in enumerate(values))


def build_answer(question: Question, probabilities: Sequence[float]) -> Answer:
    """Build the answer for one question from probabilities in option_keys(question) order."""
    keys = option_keys(question)
    if len(probabilities) != len(keys):
        raise ValueError(f"Expected {len(keys)} probabilities, got {len(probabilities)}.")
    values = normalize(probabilities)
    if question["type"] == "noul":
        noul: NoulAnswer = {"type": "noul", "noul": values[keys.index("true")]}
        return noul
    distribution = dict(zip(keys, values, strict=True))
    if question["type"] == "choice":
        best = max(range(len(values)), key=values.__getitem__)
        choice: ChoiceAnswer = {
            "type": "choice",
            "choice": keys[best],
            "confidence": choice_confidence(values),
            "probabilities": distribution,
        }
        return choice
    legend: dict[str, Content | None] = dict(zip(keys, question["criteria"], strict=True))
    score: ScoreAnswer = {
        "type": "score",
        "score": expected_score(values),
        "confidence": score_confidence(values),
        "legend": legend,
        "probabilities": distribution,
    }
    return score


def answer_probabilities(question: Question, answer: Answer) -> list[float]:
    """Read the probability vector (option_keys order) back out of an answer, for example an answer of the server."""
    keys = option_keys(question)
    if answer["type"] != question["type"]:
        raise ValueError(f"Answer type {answer['type']!r} does not match question type {question['type']!r}.")
    if answer["type"] == "noul":
        positive = float(answer["noul"])
        if not 0.0 <= positive <= 1.0:
            raise ValueError("A noul answer must be between 0 and 1.")
        return [1.0 - positive, positive]
    distribution = answer["probabilities"]
    missing = [key for key in keys if key not in distribution]
    if missing:
        raise ValueError(f"The answer has no probability for options {missing}.")
    return normalize([float(distribution[key]) for key in keys])
