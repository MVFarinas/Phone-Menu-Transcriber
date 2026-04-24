import argparse
import os
import sys
import whisper


def transcribe_audio(audio_file: str, model_name: str = "base") -> str:
    """
    Use Whisper AI to transcribe the audio to text.
    """
    model = whisper.load_model(model_name)
    result = model.transcribe(audio_file)
    return result["text"]


def find_indices(prompt: str) -> tuple:
    """
    Finds the indices of periods (end of sentences) and digits in the prompt.
    Returns a tuple of two lists: end_indices and num_indices.
    """
    end_indices = []
    num_indices = []

    for i, char in enumerate(prompt):
        if char == ".":
            end_indices.append(i)
        elif char.isdigit():
            num_indices.append(i)

    return end_indices, num_indices


def make_options(prompt: str, end_indices: list, num_indices: list) -> list:
    """
    Creates menu options based on the indices of numbers and sentence endings.
    Returns a list of formatted option strings.
    """
    options = []

    for num_index in num_indices:
        # Default of -1 means slice from index 0 when no period precedes the number
        last_index = max((i for i in end_indices if i < num_index), default=-1)
        option = prompt[last_index + 1:num_index].strip()
        option = f"[{prompt[num_index]}] {option}".replace(", please press", "").replace(", press", "").strip()
        options.append(option)

    return options


def main(audio_file: str, model_name: str = "base") -> None:
    """
    Prints menu options extracted from the given audio file.
    """
    prompt = transcribe_audio(audio_file, model_name)

    if not prompt.strip():
        print("No speech detected in audio file.")
        return

    end_indices, num_indices = find_indices(prompt)
    options = make_options(prompt, end_indices, num_indices)

    if not options:
        print("No menu options detected. Raw transcription:")
        print(prompt)
        return

    for option in options:
        print(option)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Transcribe a phone menu audio file and extract press-N options."
    )
    parser.add_argument("audio_file", help="Path to the audio file (.wav, .mp3, etc.)")
    parser.add_argument(
        "--model",
        choices=["tiny", "base", "small", "medium", "large"],
        default="base",
        help="Whisper model size (default: base)",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.audio_file):
        sys.exit(f"File not found: {args.audio_file}")

    main(args.audio_file, args.model)
