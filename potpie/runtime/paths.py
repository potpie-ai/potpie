from __future__ import annotations

import os
from pathlib import Path

_CONFIG_DIR_NAME = "potpie"


def config_dir() -> Path:
    base = os.getenv("XDG_CONFIG_HOME")
    if base:
        return Path(base) / _CONFIG_DIR_NAME
    return Path.home() / ".config" / _CONFIG_DIR_NAME


def product_data_dir(environ: dict[str, str] | None = None) -> Path:
    """Return the root-owned product data directory.

    ``CONTEXT_ENGINE_HOME`` remains an on-disk location input during the data
    compatibility migration; engine code never resolves it directly.
    """

    values = os.environ if environ is None else environ
    raw = values.get("POTPIE_HOME") or values.get("CONTEXT_ENGINE_HOME")
    if raw:
        return Path(raw).expanduser()
    home = values.get("HOME")
    return (Path(home).expanduser() if home else Path.home()) / ".potpie"


__all__ = ["config_dir", "product_data_dir"]
