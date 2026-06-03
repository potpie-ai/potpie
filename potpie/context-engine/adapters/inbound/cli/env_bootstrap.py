"""Load repo ``.env`` for CLI so POTPIE_* and other vars match ``uv run`` / app startup."""

from __future__ import annotations

import os
from pathlib import Path

_loaded: bool = False


def _parse_env_line(line: str) -> tuple[str, str] | None:
    s = line.strip()
    if not s or s.startswith("#"):
        return None
    if s.lower().startswith("export "):
        s = s[7:].strip()
    if "=" not in s:
        return None
    key, val = s.split("=", 1)
    key = key.strip()
    if not key:
        return None
    val = val.strip()
    if len(val) >= 2 and val[0] == val[-1] and val[0] in ('"', "'"):
        val = val[1:-1]
    return key, val


def _load_env_file(path: Path) -> None:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return
    for line in text.splitlines():
        parsed = _parse_env_line(line)
        if not parsed:
            continue
        k, v = parsed
        if k not in os.environ:
            os.environ[k] = v


def load_cli_env() -> None:
    """Walk upward from ``cwd`` and merge the first ``.env`` found (do not override existing env)."""
    global _loaded
    if _loaded:
        return
    _loaded = True
    cur = Path.cwd().resolve()
    for _ in range(24):
        candidate = cur / ".env"
        if candidate.is_file():
            _load_env_file(candidate)
            return
        if cur.parent == cur:
            break
        cur = cur.parent
