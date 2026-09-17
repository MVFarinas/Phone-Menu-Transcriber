"""Command-line interface for the phone menu transcriber."""

from __future__ import annotations

import argparse
import os
import sys

from phone_menu_transcriber.extraction import ExtractionError
from phone_menu_transcriber.pipeline import (
    DEFAULT_WHISPER_MODEL,
    WHISPER_MODELS,
    transcribe_and_extract,
)
from phone_menu_transcriber.transcription import TranscriptionError


def main() -> None:
    """Entry point: print menu options extracted from an audio file."""
    parser = argparse.ArgumentParser(
        description="Transcribe a phone menu audio file and extract press-N options.",
    )
    parser.add_argument("audio_file", help="Path to the audio file (.wav, .mp3, etc.)")
    parser.add_argument(
        "--model",
        choices=WHISPER_MODELS,
        default=DEFAULT_WHISPER_MODEL.value,
        metavar="MODEL",
        help=f"Whisper model size (default: {DEFAULT_WHISPER_MODEL.value}). "
        f"Choices: {', '.join(WHISPER_MODELS)}",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.audio_file):
        sys.exit(f"File not found: {args.audio_file}")

    try:
        result = transcribe_and_extract(args.audio_file, args.model)
    except (ExtractionError, TranscriptionError) as exc:
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
