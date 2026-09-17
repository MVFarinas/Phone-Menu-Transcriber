"""Tests for the FastAPI service (Whisper + extractor mocked)."""

from __future__ import annotations

import inspect
import tempfile
from pathlib import Path
from typing import cast

import pytest
from fastapi import UploadFile
from fastapi.testclient import TestClient

from phone_menu_transcriber import api
from phone_menu_transcriber.extraction import ExtractionError
from phone_menu_transcriber.models import MenuResult
from phone_menu_transcriber.transcription import TranscriptionError
from tests.conftest import EXPECTED_OPTIONS

client = TestClient(api.app)


class _ExplodingUpload:
    """An upload whose second read fails, to exercise partial-write cleanup."""

    filename = "clip.wav"

    def __init__(self) -> None:
        self.file = self
        self._reads = 0

    def read(self, _size: int) -> bytes:
        self._reads += 1
        if self._reads == 1:
            return b"partial data"
        raise OSError("disk full")


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_transcribe_returns_options(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        api,
        "transcribe_and_extract",
        lambda *_a, **_k: MenuResult(options=EXPECTED_OPTIONS, raw_transcript="press 1 ..."),
    )

    response = client.post("/transcribe", files={"file": ("clip.wav", b"RIFF....")})

    assert response.status_code == 200
    body = response.json()
    assert body["options"][0] == {"key": "1", "action": "Residential sales"}


def test_transcribe_rejects_unknown_model() -> None:
    response = client.post(
        "/transcribe",
        params={"model": "enormous"},
        files={"file": ("clip.wav", b"RIFF....")},
    )
    assert response.status_code == 422


def test_transcribe_is_not_a_coroutine() -> None:
    """A coroutine handler would run Whisper on the event loop and stall /health."""
    assert not inspect.iscoroutinefunction(api.transcribe)


def test_transcribe_200_on_silence(monkeypatch: pytest.MonkeyPatch) -> None:
    """Silence is a valid result, not a malformed request."""
    monkeypatch.setattr(
        api,
        "transcribe_and_extract",
        lambda *_a, **_k: MenuResult(options=[], raw_transcript="   "),
    )
    response = client.post("/transcribe", files={"file": ("clip.wav", b"RIFF....")})
    assert response.status_code == 200
    assert response.json()["options"] == []


def test_transcribe_502_on_extraction_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*_a: object, **_k: object) -> MenuResult:
        raise ExtractionError("Could not reach Ollama")

    monkeypatch.setattr(api, "transcribe_and_extract", _boom)
    response = client.post("/transcribe", files={"file": ("clip.wav", b"RIFF....")})
    assert response.status_code == 502
    assert "Ollama" in response.json()["detail"]


def test_transcribe_400_on_undecodable_audio(monkeypatch: pytest.MonkeyPatch) -> None:
    """Whisper's ffmpeg failure is a bad upload, not an internal server error."""

    def _boom(*_a: object, **_k: object) -> MenuResult:
        raise RuntimeError("Failed to load audio: ffmpeg: Invalid data found")

    monkeypatch.setattr(api, "transcribe_and_extract", _boom)
    response = client.post("/transcribe", files={"file": ("notaudio.wav", b"hello world")})
    assert response.status_code == 400
    assert "Could not decode" in response.json()["detail"]


def test_transcribe_413_on_oversized_upload(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.setenv("MAX_UPLOAD_BYTES", "1024")

    response = client.post("/transcribe", files={"file": ("big.wav", b"\0" * 4096)})

    assert response.status_code == 413
    assert "upload limit" in response.json()["detail"]
    assert list(tmp_path.iterdir()) == [], "rejected upload left a temp file behind"


def test_temp_file_removed_after_successful_request(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(
        api,
        "transcribe_and_extract",
        lambda *_a, **_k: MenuResult(options=[], raw_transcript="hi"),
    )

    assert client.post("/transcribe", files={"file": ("clip.wav", b"RIFF")}).status_code == 200
    assert list(tmp_path.iterdir()) == []


def test_spool_to_disk_cleans_up_a_partial_write(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A failure mid-copy must not orphan the temp file it had already created."""
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))

    with pytest.raises(OSError, match="disk full"):
        api._spool_to_disk(cast(UploadFile, _ExplodingUpload()), ".wav", 1_000_000)

    assert list(tmp_path.iterdir()) == []


def test_transcribe_503_when_whisper_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing backend is a deployment fault, not a bad upload."""

    def _boom(*_a: object, **_k: object) -> MenuResult:
        raise TranscriptionError("Whisper is not installed")

    monkeypatch.setattr(api, "transcribe_and_extract", _boom)
    response = client.post("/transcribe", files={"file": ("clip.wav", b"RIFF")})
    assert response.status_code == 503
    assert "not installed" in response.json()["detail"]
