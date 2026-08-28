"""Default paths for engine-owned local persistence."""

from __future__ import annotations

import os
from pathlib import Path


def default_home() -> Path:
    """Return the configured Context Engine data home."""
    raw = os.getenv("CONTEXT_ENGINE_HOME")
    return Path(raw).expanduser() if raw else Path.home() / ".potpie"


__all__ = ["default_home"]
