"""Tests for the Whisper transcription wrapper."""

from __future__ import annotations

import sys

import pytest

from phone_menu_transcriber.transcription import TranscriptionError, transcribe_audio


def test_missing_whisper_raises_a_helpful_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Whisper is an optional extra, so a missing install is a realistic path.

    It should explain how to fix itself rather than dumping an ImportError
    traceback from somewhere deep in numba.
    """
    monkeypatch.setitem(sys.modules, "whisper", None)

    with pytest.raises(TranscriptionError) as excinfo:
        transcribe_audio("clip.wav")

    message = str(excinfo.value)
    assert "phone-menu-transcriber[whisper]" in message
    assert "ffmpeg" in message


def test_transcription_error_is_a_runtime_error() -> None:
    """The API relies on ordering these two apart; keep the hierarchy honest."""
    assert issubclass(TranscriptionError, RuntimeError)


def test_whisper_is_usable_when_the_extra_is_installed() -> None:
    whisper = pytest.importorskip("whisper", reason="Whisper extra is not installed here")
    assert callable(whisper.load_model)
