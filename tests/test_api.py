"""Tests for the FastAPI service (Whisper + extractor mocked)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from phone_menu_transcriber import api
from phone_menu_transcriber.extraction import ExtractionError
from phone_menu_transcriber.models import MenuResult
from tests.conftest import EXPECTED_OPTIONS

client = TestClient(api.app)


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


def test_transcribe_422_on_silence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        api,
        "transcribe_and_extract",
        lambda *_a, **_k: MenuResult(options=[], raw_transcript="   "),
    )
    response = client.post("/transcribe", files={"file": ("clip.wav", b"RIFF....")})
    assert response.status_code == 422
    assert "No speech" in response.json()["detail"]


def test_transcribe_502_on_extraction_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*_a: object, **_k: object) -> MenuResult:
        raise ExtractionError("Could not reach Ollama")

    monkeypatch.setattr(api, "transcribe_and_extract", _boom)
    response = client.post("/transcribe", files={"file": ("clip.wav", b"RIFF....")})
    assert response.status_code == 502
    assert "Ollama" in response.json()["detail"]
