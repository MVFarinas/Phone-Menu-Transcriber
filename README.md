# Phone-Menu-Transcriber

Transcribes a recorded phone menu (IVR) and extracts the press-N options into
structured data — using **Whisper** for speech-to-text and a **locally hosted
LLM (Qwen 3 8B via Ollama)** for robust option extraction.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)
![LLM](https://img.shields.io/badge/LLM-Ollama%20%C2%B7%20Qwen3--8B-black)
![Tests](https://img.shields.io/badge/tests-pytest-green)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

## Why an LLM instead of string parsing?

Phone menus are messy: "press one", "for billing, dial 2", multi-digit
extensions, and missing punctuation all break naive parsing. Instead of slicing
the transcript around periods and digits, the transcript is handed to an LLM
that returns schema-validated JSON, so the output is reliable across phrasings.

## Architecture

```
CLI ─┐
     ├─▶ pipeline.transcribe_and_extract ─▶ Whisper (transcribe)
API ─┘                                   └─▶ Extractor (extract → JSON)
                                              ├─ OllamaExtractor (Qwen 3 8B) ← default
                                              └─ HostedExtractor (optional, via env)
```

Both entry points drive `pipeline.py`; neither imports the other. The extractor
is pluggable: the default `OllamaExtractor` calls a local Ollama server, and a
hosted-API backend can be swapped in via `EXTRACTOR_BACKEND` without code
changes.

## Requirements

- Python 3.10+
- [FFmpeg](https://ffmpeg.org/download.html) installed and on your PATH
- [Ollama](https://ollama.com) running locally, with the model pulled:
  ```bash
  ollama pull qwen3:8b
  ```

## Install

```bash
pip install -e ".[whisper]"       # speech-to-text (pulls in torch, ~2.5 GB)
pip install -e ".[whisper,dev]"   # plus the test and lint toolchain
```

Whisper is an optional extra because of its size. The base install gives you the
models, the extractor and the API wiring — everything except transcription — and
is what the test suite runs against. Ask for transcription without it and you
get a clear error telling you which command to run.

## Configuration

Copy `.env.example` to `.env` (or export the variables) to choose the backend.
The file is read from the directory you run in; real environment variables take
precedence over it.

| Variable | Description | Default |
|---|---|---|
| `EXTRACTOR_BACKEND` | `ollama` or `hosted` | `ollama` |
| `EXTRACTOR_MODEL` | model name / Ollama tag | `qwen3:8b` |
| `OLLAMA_BASE_URL` | Ollama server URL | `http://localhost:11434` |
| `MAX_UPLOAD_BYTES` | API upload cap, in bytes | `26214400` (25 MB) |

## Usage (CLI)

```bash
phone-menu-transcriber <audio_file> [--model MODEL]
# or
python -m phone_menu_transcriber <audio_file> [--model MODEL]
```

| Argument | Description | Default |
|---|---|---|
| `audio_file` | Path to the audio file (.wav, .mp3, etc.) | required |
| `--model` | Whisper checkpoint (see below) | `base` |

**Models.** Anything Whisper itself offers:

| Family | Names | Notes |
|---|---|---|
| Multilingual | `tiny`, `base`, `small`, `medium`, `large` | `large` aliases the newest large |
| English-only | `tiny.en`, `base.en`, `small.en`, `medium.en` | better than their multilingual twins on English menus |
| Large revisions | `large-v1`, `large-v2`, `large-v3` | pin a specific release |
| Distilled | `turbo`, `large-v3-turbo` | near-`large` accuracy, far faster — the practical choice on low-power hardware |

**Example:**

```bash
phone-menu-transcriber examples/audio_prompt.wav --model base.en
```

Against `examples/audio_prompt.wav` the output takes the form:

```
[1] Residential sales
[2] Installer and integrator sales
[3] Product questions or technical support
[4] Existing order or other customer service inquiries
[5] Current supplier
[6] Freight carrier scheduling a delivery appointment
[7] All other calls
```

Exact wording varies with the Whisper checkpoint and the extraction model; the
keys and their ordering are what the schema guarantees.

## Usage (HTTP API)

```bash
uvicorn phone_menu_transcriber.api:app --reload
```

- `POST /transcribe` — upload an audio file, get structured menu options back.
- `GET /health` — readiness probe.
- Interactive docs at `http://localhost:8000/docs`.

```bash
curl -F "file=@examples/audio_prompt.wav" "http://localhost:8000/transcribe?model=base.en"
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

**Responses**

| Status | Meaning |
|---|---|
| `200` | Success. A recording with no speech is a valid result: empty `options`. |
| `400` | The upload could not be decoded as audio. |
| `413` | The upload exceeded `MAX_UPLOAD_BYTES`. |
| `422` | Unknown `model` value. |
| `502` | The extraction backend could not be reached. |
| `503` | Whisper is not installed or not usable on the server. |

## Development & verification

```bash
ruff check .          # lint
ruff format --check . # format check
mypy .                # type check
pytest                # tests + coverage (LLM and Whisper are mocked — offline & free)
pre-commit run --all-files
```

The unit tests mock the LLM and Whisper, so they run offline, deterministically,
and without the ML stack installed. An optional live test runs against a real
Ollama server:

```bash
RUN_OLLAMA_INTEGRATION=1 pytest tests/test_integration_ollama.py
```

CI runs the suite on Linux and Windows across Python 3.10–3.12, with a separate
job that installs the Whisper extra and exercises that path.

## Examples

- `examples/audio_prompt.wav` — a sample phone menu recording

## Roadmap

- **Phase 1 (done):** tested, typed, packaged backend with a pluggable LLM extractor and FastAPI service.
- **Phase 2:** Dockerize and deploy the API to the cloud.
- **Phase 3:** Flutter (Dart) iOS/Android app over the API.

## Hardware integration (Cyberdeck, Phase A)

This project is also the software payload for a planned "offline AI deck," a self-contained,
no-internet field device that runs the full transcribe -> extract pipeline on portable hardware.
Because the whole stack (Whisper plus a local Ollama model) already runs offline, no rebuild is
needed here; the deck work is hardware and integration only.

Independent work this repo needs to support that build:

- Generalize the framing beyond phone menus to a broader "voice -> structured notes" tool. The
  pipeline is unchanged, but the CLI help text, API docs, and examples currently assume IVR input.
- Add a headless / kiosk run mode suitable for a small built-in touchscreen (auto-start the API,
  render the JSON result on-device with no browser).
- Document and test a low-resource model path (`qwen3:4b` or `qwen3:1.7b`) so the deck can run on a
  Pi 5 instead of a mini-PC, trading accuracy for portability. Capture the accuracy difference.
  On the Whisper side, `turbo` is the corresponding trade.
- Confirm a fully offline boot: record via a built-in mic and produce structured output with
  networking physically disabled.

Cross-repo context and the two-phase build plan live in the knowledge base under
`projects/cyberdeck/`.

## License

MIT — see [LICENSE](LICENSE).
