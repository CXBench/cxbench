#!/usr/bin/env python3
"""Alias for main_experiment.rejudge with the same arguments and defaults."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main_experiment import rejudge  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    return rejudge.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
