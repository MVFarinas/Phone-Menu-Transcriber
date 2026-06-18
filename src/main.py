"""Backwards-compatible shim.

The implementation moved into the ``phone_menu_transcriber`` package. This file
remains so the historical ``python src/main.py <audio_file>`` invocation keeps
working; new code should use the package CLI (``phone-menu-transcriber`` or
``python -m phone_menu_transcriber``).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phone_menu_transcriber.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
