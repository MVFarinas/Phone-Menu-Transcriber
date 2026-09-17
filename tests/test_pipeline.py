"""Tests for the shared transcribe-then-extract pipeline (Whisper mocked)."""

from __future__ import annotations

import pytest

from phone_menu_transcriber import pipeline
from phone_menu_transcriber.models import MenuOption, MenuResult
from phone_menu_transcriber.pipeline import WHISPER_MODELS, WhisperModel
from tests.conftest import EXPECTED_OPTIONS, FakeExtractor


def test_transcribe_and_extract_happy_path(
    monkeypatch: pytest.MonkeyPatch, example_transcript: str
) -> None:
    monkeypatch.setattr(pipeline, "transcribe_audio", lambda *_a, **_k: example_transcript)

    result = pipeline.transcribe_and_extract("ignored.wav", extractor=FakeExtractor())

    assert result.options == EXPECTED_OPTIONS
    assert result.raw_transcript == example_transcript


def test_transcribe_and_extract_skips_llm_on_silence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline, "transcribe_audio", lambda *_a, **_k: "   ")

    def _should_not_run(_transcript: str) -> None:
        raise AssertionError("extractor must not be called on empty transcript")

    extractor = FakeExtractor()
    monkeypatch.setattr(extractor, "extract", _should_not_run)

    result = pipeline.transcribe_and_extract("ignored.wav", extractor=extractor)
    assert result.options == []


def test_transcribe_and_extract_stamps_the_transcript(
    monkeypatch: pytest.MonkeyPatch, example_transcript: str
) -> None:
    """The transcript must survive an extractor that does not echo it back.

    ``Extractor`` only promises options, so a backend returning a bare
    ``MenuResult`` used to make a real menu look like silence to the API.
    """

    class SilentExtractor:
        def extract(self, transcript: str) -> MenuResult:
            return MenuResult(options=[MenuOption(key="1", action="Sales")])

    monkeypatch.setattr(pipeline, "transcribe_audio", lambda *_a, **_k: example_transcript)

    result = pipeline.transcribe_and_extract("ignored.wav", extractor=SilentExtractor())

    assert result.raw_transcript == example_transcript
    assert len(result.options) == 1


def test_default_model_is_accepted() -> None:
    assert pipeline.DEFAULT_WHISPER_MODEL.value in WHISPER_MODELS


def test_model_list_matches_the_installed_whisper() -> None:
    """Guard against the accepted model list falling behind the library.

    Skipped when Whisper is not importable, so the suite stays offline and
    torch-free by default.
    """
    whisper = pytest.importorskip("whisper", reason="Whisper is not installed here")
    assert WHISPER_MODELS == whisper.available_models()


def test_english_only_variants_are_offered() -> None:
    """The .en builds outperform the multilingual ones on English menus."""
    assert WhisperModel.BASE_EN.value in WHISPER_MODELS
    assert WhisperModel.TURBO.value in WHISPER_MODELS
