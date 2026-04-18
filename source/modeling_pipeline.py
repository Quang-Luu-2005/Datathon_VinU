#!/usr/bin/env python
"""Backward-compatible wrapper for the modular pipeline in `source/py`."""

from __future__ import annotations

import sys
from pathlib import Path


def _bootstrap_path() -> None:
    py_root = Path(__file__).resolve().parent / "py"
    if str(py_root) not in sys.path:
        sys.path.insert(0, str(py_root))


def main() -> None:
    _bootstrap_path()
    from run_modeling import main as run_main

    run_main()


if __name__ == "__main__":
    main()

