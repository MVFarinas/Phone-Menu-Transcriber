"""Shared fixtures and a fake extractor for offline, deterministic tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from phone_menu_transcriber.models import MenuOption, MenuResult

EXAMPLE_TRANSCRIPT = (
    " For residential sales press 1. For installer and integrator sales press 2. "
    "For product questions or technical support press 3. All other calls press 4."
)

EXPECTED_OPTIONS = [
    MenuOption(key="1", action="Residential sales"),
    MenuOption(key="2", action="Installer and integrator sales"),
    MenuOption(key="3", action="Product questions or technical support"),
    MenuOption(key="4", action="All other calls"),
]


class FakeExtractor:
    """An ``Extractor`` that returns canned options without any network call."""

    def __init__(self, options: list[MenuOption] | None = None) -> None:
        self._options = EXPECTED_OPTIONS if options is None else options

    def extract(self, transcript: str) -> MenuResult:
        return MenuResult(options=list(self._options), raw_transcript=transcript)


@pytest.fixture
def fake_extractor() -> FakeExtractor:
    return FakeExtractor()


@pytest.fixture
def example_transcript() -> str:
    return EXAMPLE_TRANSCRIPT


@pytest.fixture
def sample_audio_path() -> Path:
    return Path(__file__).resolve().parent.parent / "examples" / "audio_prompt.wav"
