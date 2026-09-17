"""Tests for the CLI orchestration (Whisper + extractor mocked)."""

from __future__ import annotations

import pytest

from phone_menu_transcriber import cli
from phone_menu_transcriber.models import MenuOption, MenuResult
from tests.conftest import EXPECTED_OPTIONS, FakeExtractor


def test_transcribe_and_extract_happy_path(
    monkeypatch: pytest.MonkeyPatch, example_transcript: str
) -> None:
    monkeypatch.setattr(cli, "transcribe_audio", lambda *_a, **_k: example_transcript)

    result = cli.transcribe_and_extract("ignored.wav", extractor=FakeExtractor())

    assert result.options == EXPECTED_OPTIONS
    assert result.raw_transcript == example_transcript


def test_transcribe_and_extract_stamps_the_transcript(
    monkeypatch: pytest.MonkeyPatch, example_transcript: str
) -> None:
    """The transcript must survive an extractor that does not echo it back.

    ``Extractor`` only promises options, so a backend returning a bare
    ``MenuResult`` used to make a real menu look like silence to the API.
    """

    class SilentExtractor:
        def extract(self, transcript: str) -> MenuResult:
            return MenuResult(options=[MenuOption(key="1", action="Sales")])

    monkeypatch.setattr(cli, "transcribe_audio", lambda *_a, **_k: example_transcript)

    result = cli.transcribe_and_extract("ignored.wav", extractor=SilentExtractor())

    assert result.raw_transcript == example_transcript
    assert len(result.options) == 1


def test_transcribe_and_extract_skips_llm_on_silence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "transcribe_audio", lambda *_a, **_k: "   ")

    def _should_not_run(_transcript: str) -> None:
        raise AssertionError("extractor must not be called on empty transcript")

    extractor = FakeExtractor()
    monkeypatch.setattr(extractor, "extract", _should_not_run)

    result = cli.transcribe_and_extract("ignored.wav", extractor=extractor)
    assert result.options == []


def test_main_prints_options(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    example_transcript: str,
    tmp_path: object,
) -> None:
    audio = f"{tmp_path}/clip.wav"
    open(audio, "w").close()
    monkeypatch.setattr(cli, "transcribe_audio", lambda *_a, **_k: example_transcript)
    monkeypatch.setattr(cli, "build_extractor", lambda: FakeExtractor())
    monkeypatch.setattr("sys.argv", ["phone-menu-transcriber", audio])

    cli.main()

    out = capsys.readouterr().out
    assert "[1] Residential sales" in out
    assert "[4] All other calls" in out


def test_main_errors_on_missing_file(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["phone-menu-transcriber", "does-not-exist.wav"])
    with pytest.raises(SystemExit):
        cli.main()
