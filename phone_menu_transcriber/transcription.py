"""Audio-to-text transcription using OpenAI Whisper."""

from __future__ import annotations


def transcribe_audio(audio_file: str, model_name: str = "base") -> str:
    """Use Whisper to transcribe the given audio file to text.

    Whisper (and its heavy ``torch`` dependency) is imported lazily so the rest
    of the package — models, the extractor, and the API wiring — can be imported
    and tested without pulling in the ML stack.

    Args:
        audio_file: Path to the audio file (.wav, .mp3, etc.).
        model_name: Whisper model size (tiny, base, small, medium, large).

    Returns:
        The transcribed text.
    """
    import whisper

    model = whisper.load_model(model_name)
    result = model.transcribe(audio_file)
    return str(result["text"])
