"""Tests for the Pydantic models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from phone_menu_transcriber.models import MenuOption, MenuResult


def test_menu_option_str() -> None:
    assert str(MenuOption(key="1", action="Residential sales")) == "[1] Residential sales"


def test_menu_result_defaults() -> None:
    result = MenuResult()
    assert result.options == []
    assert result.raw_transcript == ""


def test_menu_result_round_trips_json() -> None:
    result = MenuResult(
        options=[MenuOption(key="0", action="Operator")],
        raw_transcript="press 0 for the operator",
    )
    restored = MenuResult.model_validate_json(result.model_dump_json())
    assert restored == result


def test_menu_option_requires_fields() -> None:
    with pytest.raises(ValidationError):
        MenuOption(key="1")  # type: ignore[call-arg]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1", "1"),
        ("press 1", "1"),
        ("Press One", "1"),
        ("  two. ", "2"),
        ("dial nine", "9"),
        ("pound", "#"),
        ("Hash", "#"),
        ("star 9", "*9"),
        ("press one two", "12"),
        ("oh", "0"),
    ],
)
def test_key_is_normalized(raw: str, expected: str) -> None:
    """LLM output drifts; normalizing beats rejecting an otherwise good menu."""
    assert MenuOption(key=raw, action="Sales").key == expected


@pytest.mark.parametrize("raw", ["", "   ", "1 or 2", "press one or two", "abc", "1234567"])
def test_unusable_keys_are_rejected(raw: str) -> None:
    """Normalization is a safety net, not a licence to accept anything."""
    with pytest.raises(ValidationError):
        MenuOption(key=raw, action="Sales")


def test_blank_action_is_rejected() -> None:
    with pytest.raises(ValidationError):
        MenuOption(key="1", action="   ")


def test_action_is_stripped() -> None:
    assert MenuOption(key="1", action="  Residential sales  ").action == "Residential sales"


@pytest.mark.parametrize(("raw", "expected"), [(1, "1"), (0, "0"), (12, "12")])
def test_integer_keys_are_coerced(raw: int, expected: str) -> None:
    """Models emit `{"key": 1}` even under a string schema; don't lose the menu."""
    assert MenuOption(key=raw, action="Sales").key == expected


def test_non_string_keys_are_still_rejected() -> None:
    with pytest.raises(ValidationError):
        MenuOption(key=["1"], action="Sales")
