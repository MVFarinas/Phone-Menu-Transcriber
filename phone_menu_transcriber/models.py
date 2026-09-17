"""Pydantic models shared by every extractor backend and the API layer."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

#: Digits, star and pound, one to six characters — covers single presses,
#: multi-digit extensions, and the ``*``/``#`` keys real menus use.
KEY_PATTERN = r"^[0-9*#]{1,6}$"

_SPELLED_KEYS = {
    "zero": "0",
    "oh": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "star": "*",
    "asterisk": "*",
    "pound": "#",
    "hash": "#",
    "hashtag": "#",
}

_LEADING_VERBS = frozenset({"press", "dial", "select", "enter", "push", "key", "hit"})

_PUNCTUATION = ".,;:!?'\"“”‘’"


class MenuOption(BaseModel):
    """A single press-N option extracted from a phone menu."""

    key: str = Field(
        ...,
        pattern=KEY_PATTERN,
        description="The digit or key the caller presses, e.g. '1' or '0'.",
    )
    action: str = Field(
        ...,
        min_length=1,
        description="What selecting this option does, e.g. 'Residential sales'.",
    )

    @field_validator("key", mode="before")
    @classmethod
    def _normalize_key(cls, value: object) -> object:
        """Clean up what the model returned before validating it.

        The prompt asks for a bare key and the JSON schema constrains it, but
        LLM output still drifts — ``"press one"``, ``"Pound"``, ``"2."``. Fixing
        those is much better than rejecting an otherwise good menu, so normalize
        first and let ``KEY_PATTERN`` reject only what is genuinely unusable.
        """
        # A JSON number is a common slip even under a string schema, and
        # rejecting {"key": 1} would throw away the whole menu over formatting.
        if isinstance(value, int) and not isinstance(value, bool):
            return str(value)
        if not isinstance(value, str):
            return value

        words = value.strip().lower().split()
        while words and words[0].strip(_PUNCTUATION) in _LEADING_VERBS:
            words.pop(0)

        cleaned = (word.strip(_PUNCTUATION) for word in words)
        return "".join(_SPELLED_KEYS.get(word, word) for word in cleaned)

    @field_validator("action", mode="before")
    @classmethod
    def _strip_action(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value

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
