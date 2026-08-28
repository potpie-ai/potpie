"""Root Potpie filesystem path resolution."""

from __future__ import annotations

import os
from pathlib import Path


def default_home() -> Path:
    """Return the configured Potpie home, or ``~/.potpie`` by default."""
    raw = os.getenv("CONTEXT_ENGINE_HOME")
    return Path(raw).expanduser() if raw else Path.home() / ".potpie"


__all__ = ["default_home"]
