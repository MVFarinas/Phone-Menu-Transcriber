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
