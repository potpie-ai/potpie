from __future__ import annotations

import os
from pathlib import Path

_CONFIG_DIR_NAME = "potpie"


def config_dir() -> Path:
    base = os.getenv("XDG_CONFIG_HOME")
    if base:
        return Path(base) / _CONFIG_DIR_NAME
    return Path.home() / ".config" / _CONFIG_DIR_NAME


__all__ = ["config_dir"]
