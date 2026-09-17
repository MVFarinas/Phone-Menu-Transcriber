"""The shared transcribe-then-extract pipeline, independent of any entry point.

Both the CLI and the HTTP API drive this module. Keeping the orchestration here
means neither entry point has to import the other, which is what the README's
architecture diagram has always claimed.
"""

from __future__ import annotations

from enum import Enum

from phone_menu_transcriber.extraction import Extractor, build_extractor
from phone_menu_transcriber.models import MenuResult
from phone_menu_transcriber.transcription import transcribe_audio


class WhisperModel(str, Enum):
    """The Whisper checkpoints this project accepts.

    Mirrors ``whisper.available_models()`` — a test asserts the two match
    whenever Whisper is importable, so this cannot quietly fall behind the
    library. The ``.en`` entries are English-only builds and beat their
    multilingual counterparts on English phone menus; ``turbo`` is a distilled
    ``large-v3`` that runs far faster at close to the same accuracy.
    """

    TINY_EN = "tiny.en"
    TINY = "tiny"
    BASE_EN = "base.en"
    BASE = "base"
    SMALL_EN = "small.en"
    SMALL = "small"
    MEDIUM_EN = "medium.en"
    MEDIUM = "medium"
    LARGE_V1 = "large-v1"
    LARGE_V2 = "large-v2"
    LARGE_V3 = "large-v3"
    LARGE = "large"
    LARGE_V3_TURBO = "large-v3-turbo"
    TURBO = "turbo"


#: Accepted model names, derived from the enum so the two can never disagree.
WHISPER_MODELS = [model.value for model in WhisperModel]

DEFAULT_WHISPER_MODEL = WhisperModel.BASE


def transcribe_and_extract(
    audio_file: str,
    whisper_model: str = DEFAULT_WHISPER_MODEL.value,
    extractor: Extractor | None = None,
) -> MenuResult:
    """Transcribe ``audio_file`` and extract its menu options.

    The transcript is stamped onto the result here rather than taken from the
    extractor. The ``Extractor`` protocol only promises options, so a backend
    that left ``raw_transcript`` unset would otherwise be indistinguishable from
    silence to every caller downstream.
    """
    extractor = extractor or build_extractor()
    transcript = transcribe_audio(audio_file, whisper_model)
    if not transcript.strip():
        return MenuResult(options=[], raw_transcript=transcript)
    result = extractor.extract(transcript)
    return result.model_copy(update={"raw_transcript": transcript})
