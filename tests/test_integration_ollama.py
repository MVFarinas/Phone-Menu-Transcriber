"""Optional integration test that hits a real local Ollama server.

Skipped by default so CI stays offline. Run it locally with a running Ollama
and the model pulled:

    RUN_OLLAMA_INTEGRATION=1 pytest tests/test_integration_ollama.py
"""

from __future__ import annotations

import os

import pytest

from phone_menu_transcriber.extraction import OllamaExtractor

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_OLLAMA_INTEGRATION") != "1",
    reason="set RUN_OLLAMA_INTEGRATION=1 to run against a real Ollama server",
)


def test_real_ollama_extracts_options() -> None:
    model = os.environ.get("EXTRACTOR_MODEL", "qwen3:8b")
    extractor = OllamaExtractor(model=model)
    transcript = "For sales press 1. For support press 2. To speak to an operator press 0."

    result = extractor.extract(transcript)

    keys = {o.key for o in result.options}
    assert {"1", "2", "0"} <= keys
