import pytest

from jiwo.prompting import DisplayOrder, decision_messages, option_texts

CODES = tuple("ABCDEFGHIJ")


def test_choice_prompt_lists_keys_and_descriptions_in_display_order() -> None:
    question = {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "Payments", "other": None}}
    messages = decision_messages({"ticket": "refund"}, question, CODES, DisplayOrder((1, 0)))  # type: ignore[arg-type]
    user = messages[1]["content"]
    assert 'State:\n{"ticket": "refund"}' in user
    assert "A: other\nB: billing: Payments" in user
    assert user.endswith("Return only the letter code of the best option.")


def test_noul_and_score_texts() -> None:
    assert option_texts({"type": "noul"}) == ["No, the statement is false.", "Yes, the statement is true."]
    assert option_texts({"type": "noul", "criteria": {"true": "Spam"}})[1] == "Spam"
    assert option_texts({"type": "score", "criteria": ["low", {"level": "high"}]}) == ["low", '{"level": "high"}']


def test_display_order_maps_both_ways() -> None:
    order = DisplayOrder((2, 0, 1))
    canonical = [0.1, 0.2, 0.7]
    display = order.to_display(canonical)
    assert display == [0.7, 0.1, 0.2]
    assert order.to_canonical(display) == canonical
    with pytest.raises(ValueError):
        DisplayOrder((0, 0, 1))


def test_too_many_options_for_codes() -> None:
    question = {"type": "choice", "criteria": {str(i): None for i in range(12)}}
    with pytest.raises(ValueError, match="only 10 answer codes"):
        decision_messages("s", question, CODES)  # type: ignore[arg-type]


def test_code_like_keys_are_quoted_so_codes_stay_distinct() -> None:
    question = {"type": "choice", "criteria": {"A": "Response A is better.", "B": "Response B is better.", "tie": None}}
    user = decision_messages("s", question, CODES, DisplayOrder((1, 0, 2)))[1]["content"]  # type: ignore[arg-type]
    assert 'A: option "B": Response B is better.' in user
    assert 'B: option "A": Response A is better.' in user
    assert "C: tie" in user
