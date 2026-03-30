"""Persist Potpie API token for context-engine CLI (user config dir)."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import Any, Optional

_CREDENTIALS_FILENAME = "credentials.json"


def config_dir() -> Path:
    base = os.getenv("XDG_CONFIG_HOME")
    if base:
        return Path(base) / "context-engine"
    return Path.home() / ".config" / "context-engine"


def credentials_path() -> Path:
    return config_dir() / _CREDENTIALS_FILENAME


def read_credentials() -> dict[str, Any]:
    """Return parsed JSON or empty dict if missing/invalid."""
    path = credentials_path()
    if not path.is_file():
        return {}
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def get_stored_api_key() -> str:
    v = read_credentials().get("api_key")
    return str(v).strip() if v else ""


def get_stored_api_base_url() -> str:
    v = read_credentials().get("api_base_url")
    if not v:
        return ""
    return str(v).strip().rstrip("/")


def write_credentials(*, api_key: str, api_base_url: Optional[str] = None) -> None:
    """Write credentials file with mode 0o600.

    Merges with existing file: if ``api_base_url`` is ``None``, any stored ``api_base_url`` is kept.
    Pass ``api_base_url=""`` to clear the stored base URL.
    """
    d = config_dir()
    d.mkdir(parents=True, exist_ok=True)
    path = credentials_path()
    payload: dict[str, Any] = dict(read_credentials())
    payload["api_key"] = api_key.strip()
    if api_base_url is not None:
        u = api_base_url.strip().rstrip("/")
        if u:
            payload["api_base_url"] = u
        else:
            payload.pop("api_base_url", None)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    path.chmod(stat.S_IRUSR | stat.S_IWUSR)


def clear_credentials() -> None:
    """Remove credentials file if it exists."""
    path = credentials_path()
    try:
        path.unlink(missing_ok=True)
    except TypeError:
        if path.is_file():
            path.unlink()
