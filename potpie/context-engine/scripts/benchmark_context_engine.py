#!/usr/bin/env python3
"""Compatibility wrapper for the benchmark package.

Use this path from the repository root:

    uv run python app/src/context-engine/scripts/benchmark_context_engine.py mock
    uv run python app/src/context-engine/scripts/benchmark_context_engine.py http-e2e
    uv run python app/src/context-engine/scripts/benchmark_context_engine.py api
"""

from __future__ import annotations

import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from benchmarks.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
