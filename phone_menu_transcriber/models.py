"""Pydantic models shared by every extractor backend and the API layer."""

from __future__ import annotations

from pydantic import BaseModel, Field


class MenuOption(BaseModel):
    """A single press-N option extracted from a phone menu."""

    key: str = Field(
        ...,
        description="The digit or key the caller presses, e.g. '1' or '0'.",
    )
    action: str = Field(
        ...,
        description="What selecting this option does, e.g. 'Residential sales'.",
    )

    def __str__(self) -> str:
        return f"[{self.key}] {self.action}"


class MenuResult(BaseModel):
    """The structured result of transcribing and parsing a phone menu."""

    options: list[MenuOption] = Field(
        default_factory=list,
        description="Every press-N option found in the menu, in spoken order.",
    )
    raw_transcript: str = Field(
        default="",
        description="The full Whisper transcription the options were extracted from.",
    )
