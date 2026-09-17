"""FastAPI service exposing transcription + menu extraction over HTTP."""

from __future__ import annotations

import os
import tempfile
from enum import Enum

from fastapi import FastAPI, File, HTTPException, Query, UploadFile

from phone_menu_transcriber.cli import transcribe_and_extract
from phone_menu_transcriber.extraction import ExtractionError
from phone_menu_transcriber.models import MenuResult

DEFAULT_MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # ~25 minutes of phone-quality audio
_CHUNK_BYTES = 1024 * 1024


class WhisperModel(str, Enum):
    """Whisper model sizes the service accepts.

    Declaring these as an enum (rather than validating a bare ``str`` by hand)
    puts the valid values in the OpenAPI schema and lets FastAPI reject the rest.
    A test keeps this in step with ``WHISPER_MODELS`` so the two cannot drift.
    """

    TINY = "tiny"
    BASE = "base"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


app = FastAPI(
    title="Phone Menu Transcriber",
    description="Transcribe a phone-menu recording and extract its press-N options.",
    version="0.1.0",
)


def _max_upload_bytes() -> int:
    """Upload cap, read at call time so deployments and tests can override it."""
    return int(os.environ.get("MAX_UPLOAD_BYTES", DEFAULT_MAX_UPLOAD_BYTES))


def _spool_to_disk(upload: UploadFile, suffix: str, limit: int) -> str:
    """Stream ``upload`` to a temp file, refusing anything larger than ``limit``.

    Returns the temp file's path; the caller owns deleting it. Any failure here
    removes the partial file first, so a disk-full or disconnected client cannot
    leave orphans behind.
    """
    fd, path = tempfile.mkstemp(suffix=suffix)
    written = 0
    try:
        with os.fdopen(fd, "wb") as dest:
            while chunk := upload.file.read(_CHUNK_BYTES):
                written += len(chunk)
                if written > limit:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Audio file exceeds the {limit:,}-byte upload limit.",
                    )
                dest.write(chunk)
    except Exception:
        os.unlink(path)
        raise
    return path


@app.get("/health")
def health() -> dict[str, str]:
    """Readiness probe for deployment."""
    return {"status": "ok"}


@app.post("/transcribe", response_model=MenuResult)
def transcribe(
    file: UploadFile = File(..., description="Audio file (.wav, .mp3, etc.)"),
    model: WhisperModel = Query(WhisperModel.BASE, description="Whisper model size"),
) -> MenuResult:
    """Transcribe an uploaded audio file and return its structured menu options.

    A transcript with no speech is a valid result, not a bad request, so it comes
    back as ``200`` with an empty option list — matching what the CLI reports.

    Deliberately ``def`` and not ``async def``: the body is entirely blocking
    (disk I/O, Whisper inference, a synchronous call to Ollama), so FastAPI has
    to run it in a worker thread. As a coroutine it would hold the event loop for
    the whole transcription and stall every other request, ``/health`` included.
    """
    suffix = os.path.splitext(file.filename or "")[1] or ".wav"
    tmp_path = _spool_to_disk(file, suffix, _max_upload_bytes())

    try:
        return transcribe_and_extract(tmp_path, model.value)
    except ExtractionError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except (OSError, RuntimeError) as exc:
        # Whisper shells out to ffmpeg and raises RuntimeError when it cannot
        # decode the input — an unusable upload is the client's problem, not a
        # server fault, so it must not surface as an opaque 500.
        raise HTTPException(
            status_code=400,
            detail=f"Could not decode audio file: {exc}",
        ) from exc
    finally:
        os.unlink(tmp_path)
