"""Tests for the CLI entry point (the pipeline itself is mocked)."""

from __future__ import annotations

from pathlib import Path

import pytest

from phone_menu_transcriber import cli
from phone_menu_transcriber.extraction import ExtractionError
from phone_menu_transcriber.models import MenuResult
from tests.conftest import EXPECTED_OPTIONS


def _stub_result(monkeypatch: pytest.MonkeyPatch, result: MenuResult) -> None:
    monkeypatch.setattr(cli, "transcribe_and_extract", lambda *_a, **_k: result)


def _argv(monkeypatch: pytest.MonkeyPatch, audio: Path, *extra: str) -> None:
    monkeypatch.setattr("sys.argv", ["phone-menu-transcriber", str(audio), *extra])


@pytest.fixture
def audio(tmp_path: Path) -> Path:
    path = tmp_path / "clip.wav"
    path.touch()
    return path


def test_main_prints_options(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    example_transcript: str,
    audio: Path,
) -> None:
    _stub_result(
        monkeypatch,
        MenuResult(options=EXPECTED_OPTIONS, raw_transcript=example_transcript),
    )
    _argv(monkeypatch, audio)

    cli.main()

    out = capsys.readouterr().out
    assert "[1] Residential sales" in out
    assert "[4] All other calls" in out


def test_main_reports_silence(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], audio: Path
) -> None:
    _stub_result(monkeypatch, MenuResult(options=[], raw_transcript="   "))
    _argv(monkeypatch, audio)

    cli.main()

    assert "No speech detected" in capsys.readouterr().out


def test_main_falls_back_to_the_raw_transcript(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], audio: Path
) -> None:
    """A recording with speech but no options should still show the transcript."""
    _stub_result(monkeypatch, MenuResult(options=[], raw_transcript="Thanks for calling."))
    _argv(monkeypatch, audio)

    cli.main()

    out = capsys.readouterr().out
    assert "No menu options detected" in out
    assert "Thanks for calling." in out


def test_main_exits_on_extraction_error(monkeypatch: pytest.MonkeyPatch, audio: Path) -> None:
    def _boom(*_a: object, **_k: object) -> MenuResult:
        raise ExtractionError("Could not reach Ollama")

    monkeypatch.setattr(cli, "transcribe_and_extract", _boom)
    _argv(monkeypatch, audio)

    with pytest.raises(SystemExit, match="Could not reach Ollama"):
        cli.main()


def test_main_errors_on_missing_file(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["phone-menu-transcriber", "does-not-exist.wav"])
    with pytest.raises(SystemExit, match="File not found"):
        cli.main()


def test_main_rejects_an_unknown_model(monkeypatch: pytest.MonkeyPatch, audio: Path) -> None:
    _argv(monkeypatch, audio, "--model", "enormous")
    with pytest.raises(SystemExit):
        cli.main()
