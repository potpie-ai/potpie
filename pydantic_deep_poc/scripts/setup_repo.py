#!/usr/bin/env python3
"""CLI wrapper; implementation lives in poc.repo_setup."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from poc.repo_setup import base_worktree, create_worktree, setup  # noqa: E402

if __name__ == "__main__":
    setup()
    print("OK:", base_worktree())
