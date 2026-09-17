"""Audio-to-text transcription using OpenAI Whisper."""

from __future__ import annotations

_MISSING_WHISPER = (
    "Whisper is not installed or could not be imported. It ships as an optional "
    "extra because it pulls in torch (~2.5 GB):\n"
    "    pip install 'phone-menu-transcriber[whisper]'\n"
    "ffmpeg must also be installed and on your PATH.\n"
    "Underlying error: {error}"
)


class TranscriptionError(RuntimeError):
    """Raised when audio cannot be transcribed at all.

    Distinct from the ``RuntimeError`` Whisper raises for an undecodable file:
    this means the transcription backend itself is unavailable, which is a
    deployment problem rather than a bad input.
    """


def transcribe_audio(audio_file: str, model_name: str = "base") -> str:
    """Use Whisper to transcribe the given audio file to text.

    Whisper (and its heavy ``torch`` dependency) is imported lazily so the rest
    of the package — models, the extractor, and the API wiring — can be imported
    and tested without pulling in the ML stack.

    Args:
        audio_file: Path to the audio file (.wav, .mp3, etc.).
        model_name: Whisper model size (see ``WHISPER_MODELS``).

    Returns:
        The transcribed text.

    Raises:
        TranscriptionError: If Whisper is not installed or cannot be imported.
    """
    try:
        import whisper
    except ImportError as exc:
        # Now that Whisper is an optional extra this is a realistic failure, and
        # a bare ImportError traceback is a poor way to report it.
        raise TranscriptionError(_MISSING_WHISPER.format(error=exc)) from exc

    model = whisper.load_model(model_name)
    result = model.transcribe(audio_file)
    return str(result["text"])
