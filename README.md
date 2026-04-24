# Phone-Menu-Transcriber

Transcribes a recorded phone menu and extracts the press-N options.

## Requirements

- Python 3.9+
- [FFmpeg](https://ffmpeg.org/download.html) installed and on your PATH
- `openai-whisper` Python package

## Install

```bash
pip install openai-whisper
```

## Usage

```bash
python src/main.py <audio_file> [--model MODEL]
```

| Argument | Description | Default |
|---|---|---|
| `audio_file` | Path to the audio file (.wav, .mp3, etc.) | required |
| `--model` | Whisper model size: `tiny`, `base`, `small`, `medium`, `large` | `base` |

**Examples:**

```bash
python src/main.py examples/audio_prompt.wav
python src/main.py examples/audio_prompt.wav --model small
```

## Sample Output

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

## Examples

The `examples/` folder contains:
- `audio_prompt.wav` — a sample phone menu recording
- `transcribed_prompt.txt` — the expected Whisper transcription of the above
