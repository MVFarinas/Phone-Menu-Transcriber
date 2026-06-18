# Phone-Menu-Transcriber

Transcribes a recorded phone menu (IVR) and extracts the press-N options into
structured data — using **Whisper** for speech-to-text and a **locally hosted
LLM (Qwen 3 8B via Ollama)** for robust option extraction.

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)
![LLM](https://img.shields.io/badge/LLM-Ollama%20%C2%B7%20Qwen3--8B-black)
![Tests](https://img.shields.io/badge/tests-pytest-green)

## Why an LLM instead of string parsing?

Phone menus are messy: "press one", "for billing, dial 2", multi-digit
extensions, and missing punctuation all break naive parsing. Instead of slicing
the transcript around periods and digits, the transcript is handed to an LLM
that returns schema-validated JSON, so the output is reliable across phrasings.

## Architecture

```
CLI / Flutter app ──▶ FastAPI backend ──▶ Whisper (transcribe)
                       /transcribe         └─▶ Extractor interface (extract → JSON)
                       Pydantic-validated         ├─ OllamaExtractor (Qwen 3 8B) ← default
                                                  └─ HostedExtractor (optional, via env)
```

The extractor is pluggable: the default `OllamaExtractor` calls a local Ollama
server, and a hosted-API backend can be swapped in via `EXTRACTOR_BACKEND`
without code changes.

## Requirements

- Python 3.9+
- [FFmpeg](https://ffmpeg.org/download.html) installed and on your PATH
- [Ollama](https://ollama.com) running locally, with the model pulled:
  ```bash
  ollama pull qwen3:8b
  ```

## Install

```bash
pip install -e ".[dev]"   # dev extras add the test + lint toolchain
```

## Configuration

Copy `.env.example` to `.env` (or export the variables) to choose the backend:

| Variable | Description | Default |
|---|---|---|
| `EXTRACTOR_BACKEND` | `ollama` or `hosted` | `ollama` |
| `EXTRACTOR_MODEL` | model name / Ollama tag | `qwen3:8b` |
| `OLLAMA_BASE_URL` | Ollama server URL | `http://localhost:11434` |

## Usage (CLI)

```bash
phone-menu-transcriber <audio_file> [--model MODEL]
# or
python -m phone_menu_transcriber <audio_file> [--model MODEL]
```

| Argument | Description | Default |
|---|---|---|
| `audio_file` | Path to the audio file (.wav, .mp3, etc.) | required |
| `--model` | Whisper model size: `tiny`, `base`, `small`, `medium`, `large` | `base` |

**Example:**

```bash
phone-menu-transcriber examples/audio_prompt.wav
```

Running against `examples/audio_prompt.wav` produces:

```
[1] For residential sales
[2] For installer and integrator sales
[3] For product questions or technical support
[4] If you have a question about an existing order, or for any other customer service inquiries
[5] If you are a current supplier
[6] If you are a freight carrier and need to schedule a delivery appointment
[7] All other calls
```

## Usage (HTTP API)

```bash
uvicorn phone_menu_transcriber.api:app --reload
```

- `POST /transcribe` — upload an audio file, get structured menu options back.
- `GET /health` — readiness probe.
- Interactive docs at `http://localhost:8000/docs`.

```bash
curl -F "file=@examples/audio_prompt.wav" "http://localhost:8000/transcribe?model=base"
```

Returns:

```json
{
  "options": [
    { "key": "1", "action": "Residential sales" },
    { "key": "2", "action": "Installer and integrator sales" }
  ],
  "raw_transcript": "For residential sales press 1. ..."
}
```

## Development & verification

```bash
ruff check .          # lint
black --check .       # format check
mypy phone_menu_transcriber   # type check
pytest                # tests + coverage (LLM and Whisper are mocked — offline & free)
pre-commit run --all-files
```

The unit tests mock the LLM and Whisper, so they run offline and deterministically.
An optional live test runs against a real Ollama server:

```bash
RUN_OLLAMA_INTEGRATION=1 pytest tests/test_integration_ollama.py
```

## Examples

The `examples/` folder contains:
- `audio_prompt.wav` — a sample phone menu recording
- `transcribed_prompt.txt` — the expected Whisper transcription of the above

## Roadmap

- **Phase 1 (done):** tested, typed, packaged backend with a pluggable LLM extractor and FastAPI service.
- **Phase 2:** Dockerize and deploy the API to the cloud.
- **Phase 3:** Flutter (Dart) iOS/Android app over the API.
