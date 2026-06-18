"""FastAPI service exposing transcription + menu extraction over HTTP."""

from __future__ import annotations

import os
import shutil
import tempfile

from fastapi import FastAPI, File, HTTPException, Query, UploadFile

from phone_menu_transcriber.cli import WHISPER_MODELS, transcribe_and_extract
from phone_menu_transcriber.extraction import ExtractionError
from phone_menu_transcriber.models import MenuResult

app = FastAPI(
    title="Phone Menu Transcriber",
    description="Transcribe a phone-menu recording and extract its press-N options.",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    """Readiness probe for deployment."""
    return {"status": "ok"}


@app.post("/transcribe", response_model=MenuResult)
async def transcribe(
    file: UploadFile = File(..., description="Audio file (.wav, .mp3, etc.)"),
    model: str = Query("base", description="Whisper model size"),
) -> MenuResult:
    """Transcribe an uploaded audio file and return its structured menu options."""
    if model not in WHISPER_MODELS:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown Whisper model {model!r}; choose one of {WHISPER_MODELS}.",
        )

    suffix = os.path.splitext(file.filename or "")[1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        result = transcribe_and_extract(tmp_path, model)
    except ExtractionError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    finally:
        os.unlink(tmp_path)

    if not result.raw_transcript.strip():
        raise HTTPException(status_code=422, detail="No speech detected in audio file.")

    return result
