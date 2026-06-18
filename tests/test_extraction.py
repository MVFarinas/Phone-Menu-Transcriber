"""Tests for the pluggable LLM extractor (Ollama backend mocked)."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from phone_menu_transcriber import extraction
from phone_menu_transcriber.extraction import (
    ExtractionError,
    OllamaExtractor,
    build_extractor,
)


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


def _ollama_message(content: str) -> dict[str, Any]:
    return {"message": {"content": content}}


def test_schema_is_derived_from_pydantic_model() -> None:
    schema = extraction._menu_options_schema()
    item_props = schema["properties"]["options"]["items"]["properties"]
    assert set(item_props) == {"key", "action"}


def test_extract_parses_model_output(monkeypatch: pytest.MonkeyPatch) -> None:
    content = json.dumps({"options": [{"key": "1", "action": "Residential sales"}]})
    monkeypatch.setattr(httpx, "post", lambda *a, **k: _FakeResponse(_ollama_message(content)))

    result = OllamaExtractor().extract("press 1 for residential sales")

    assert [(o.key, o.action) for o in result.options] == [("1", "Residential sales")]
    assert result.raw_transcript == "press 1 for residential sales"


def test_extract_handles_empty_option_list(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        httpx, "post", lambda *a, **k: _FakeResponse(_ollama_message('{"options": []}'))
    )
    result = OllamaExtractor().extract("thanks for calling, goodbye")
    assert result.options == []


def test_connection_error_becomes_extraction_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom(*_args: Any, **_kwargs: Any) -> None:
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx, "post", _boom)

    with pytest.raises(ExtractionError, match="Could not reach Ollama"):
        OllamaExtractor().extract("press 1")


def test_invalid_json_becomes_extraction_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(httpx, "post", lambda *a, **k: _FakeResponse(_ollama_message("not json")))
    with pytest.raises(ExtractionError, match="valid JSON"):
        OllamaExtractor().extract("press 1")


def test_schema_violation_becomes_extraction_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    content = json.dumps({"options": [{"key": "1"}]})  # missing 'action'
    monkeypatch.setattr(httpx, "post", lambda *a, **k: _FakeResponse(_ollama_message(content)))
    with pytest.raises(ExtractionError, match="schema validation"):
        OllamaExtractor().extract("press 1")


def test_build_extractor_defaults_to_ollama(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EXTRACTOR_BACKEND", raising=False)
    monkeypatch.setenv("EXTRACTOR_MODEL", "qwen3:8b")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://example:1234/")

    extractor = build_extractor()

    assert isinstance(extractor, OllamaExtractor)
    assert extractor.model == "qwen3:8b"
    assert extractor.base_url == "http://example:1234"


def test_build_extractor_hosted_not_implemented(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EXTRACTOR_BACKEND", "hosted")
    with pytest.raises(ExtractionError, match="not implemented"):
        build_extractor()


def test_build_extractor_unknown_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXTRACTOR_BACKEND", "bogus")
    with pytest.raises(ExtractionError, match="Unknown EXTRACTOR_BACKEND"):
        build_extractor()
