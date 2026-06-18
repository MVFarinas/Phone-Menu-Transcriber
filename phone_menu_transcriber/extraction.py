"""Pluggable LLM extraction of menu options from a transcript.

The default backend is a locally hosted Qwen 3 8B served by Ollama (free,
offline, private). A hosted-API backend can be swapped in behind the same
``Extractor`` interface via the ``EXTRACTOR_BACKEND`` environment variable
without touching the rest of the codebase.
"""

from __future__ import annotations

import json
import os
from typing import Protocol

import httpx
from pydantic import ValidationError

from phone_menu_transcriber.models import MenuResult

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen3:8b"

_SYSTEM_PROMPT = (
    "You parse transcribed phone-menu (IVR) recordings into structured data. "
    "Given a transcript, extract every selectable option the caller can choose. "
    "For each option, return the key the caller presses and a short, cleaned "
    "description of what it does. Handle spelled-out digits ('press one' -> '1'), "
    "alternate verbs ('dial', 'select', 'enter'), multi-digit keys, and missing "
    "punctuation. Drop filler like 'please press' or 'for' from the description. "
    "If the transcript contains no selectable options, return an empty list."
)


class ExtractionError(RuntimeError):
    """Raised when the LLM backend cannot produce a valid ``MenuResult``."""


class Extractor(Protocol):
    """Anything that turns a transcript into a structured ``MenuResult``."""

    def extract(self, transcript: str) -> MenuResult:
        """Return the menu options found in ``transcript``."""
        ...


def _menu_options_schema() -> dict[str, object]:
    """JSON schema describing just the ``options`` the model must produce.

    Derived from the Pydantic model so the schema and the validation target
    never drift apart.
    """
    option_schema = MenuResult.model_json_schema()["$defs"]["MenuOption"]
    return {
        "type": "object",
        "properties": {
            "options": {"type": "array", "items": option_schema},
        },
        "required": ["options"],
    }


class OllamaExtractor:
    """Extract menu options using a local Ollama model (default: Qwen 3 8B)."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_OLLAMA_BASE_URL,
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def extract(self, transcript: str) -> MenuResult:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": transcript},
            ],
            "format": _menu_options_schema(),
            "think": False,
            "stream": False,
            "options": {"temperature": 0},
        }
        try:
            response = httpx.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ExtractionError(
                f"Could not reach Ollama at {self.base_url} "
                f"(is it running and is '{self.model}' pulled?): {exc}"
            ) from exc

        content = response.json().get("message", {}).get("content", "")
        return self._parse(content, transcript)

    @staticmethod
    def _parse(content: str, transcript: str) -> MenuResult:
        try:
            options = json.loads(content).get("options", [])
        except json.JSONDecodeError as exc:
            raise ExtractionError(f"Model did not return valid JSON: {exc}") from exc
        try:
            return MenuResult(options=options, raw_transcript=transcript)
        except ValidationError as exc:
            raise ExtractionError(f"Model output failed schema validation: {exc}") from exc


def build_extractor() -> Extractor:
    """Build the configured extractor backend from environment variables.

    Environment variables:
        EXTRACTOR_BACKEND: 'ollama' (default) or 'hosted'.
        EXTRACTOR_MODEL:   model name (default 'qwen3:8b').
        OLLAMA_BASE_URL:   Ollama server URL (default 'http://localhost:11434').
    """
    backend = os.environ.get("EXTRACTOR_BACKEND", "ollama").lower()
    model = os.environ.get("EXTRACTOR_MODEL", DEFAULT_MODEL)

    if backend == "ollama":
        base_url = os.environ.get("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)
        return OllamaExtractor(model=model, base_url=base_url)
    if backend == "hosted":
        raise ExtractionError(
            "The 'hosted' extractor backend is not implemented yet. "
            "Use EXTRACTOR_BACKEND=ollama (default), or add a HostedExtractor."
        )
    raise ExtractionError(f"Unknown EXTRACTOR_BACKEND: {backend!r}")
