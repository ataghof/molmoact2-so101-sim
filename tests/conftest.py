"""Import-path shim for the CPU test suite.

The package uses a ``src/`` layout; prepending ``<repo>/src`` here lets the suite run
without an editable install (e.g. bare ``pytest`` in CI before ``pip install -e .``).
"""
from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
