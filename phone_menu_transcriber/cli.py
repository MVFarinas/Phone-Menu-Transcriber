"""Command-line interface for the phone menu transcriber."""

from __future__ import annotations

import argparse
import os
import sys

from phone_menu_transcriber.extraction import ExtractionError, Extractor, build_extractor
from phone_menu_transcriber.models import MenuResult
from phone_menu_transcriber.transcription import transcribe_audio

WHISPER_MODELS = ["tiny", "base", "small", "medium", "large"]


def transcribe_and_extract(
    audio_file: str,
    whisper_model: str = "base",
    extractor: Extractor | None = None,
) -> MenuResult:
    """Transcribe ``audio_file`` and extract its menu options.

    Shared by the CLI and the API so there is a single source of truth.
    """
    extractor = extractor or build_extractor()
    transcript = transcribe_audio(audio_file, whisper_model)
    if not transcript.strip():
        return MenuResult(options=[], raw_transcript=transcript)
    return extractor.extract(transcript)


def main() -> None:
    """Entry point: print menu options extracted from an audio file."""
    parser = argparse.ArgumentParser(
        description="Transcribe a phone menu audio file and extract press-N options.",
    )
    parser.add_argument("audio_file", help="Path to the audio file (.wav, .mp3, etc.)")
    parser.add_argument(
        "--model",
        choices=WHISPER_MODELS,
        default="base",
        help="Whisper model size (default: base)",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.audio_file):
        sys.exit(f"File not found: {args.audio_file}")

    try:
        result = transcribe_and_extract(args.audio_file, args.model)
    except ExtractionError as exc:
        sys.exit(str(exc))

    if not result.raw_transcript.strip():
        print("No speech detected in audio file.")
        return

    if not result.options:
        print("No menu options detected. Raw transcription:")
        print(result.raw_transcript)
        return

    for option in result.options:
        print(option)


if __name__ == "__main__":
    main()
