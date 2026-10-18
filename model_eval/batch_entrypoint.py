#!/usr/bin/env python3
from __future__ import annotations

from scripts.run_eval_batch import main as batch_main


def main() -> int:
    return batch_main()


if __name__ == "__main__":
    raise SystemExit(main())
