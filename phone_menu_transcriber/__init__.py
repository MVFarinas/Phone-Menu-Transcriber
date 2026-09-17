"""Transcribe a recorded phone menu and extract its press-N options."""

from importlib.metadata import PackageNotFoundError, version

from phone_menu_transcriber.models import MenuOption, MenuResult

__all__ = ["MenuOption", "MenuResult", "__version__"]

try:
    __version__ = version("phone-menu-transcriber")
except PackageNotFoundError:  # running from a source tree with no install
    __version__ = "0.0.0+unknown"
