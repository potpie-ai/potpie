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


_PROJECT_ROOT_MARKERS = ("pyproject.toml", ".git")


def load_cli_env() -> None:
    """Merge the project root's ``.env`` (never an arbitrary ancestor's).

    The previous behavior walked up to 24 parents and loaded the *first*
    ``.env`` it saw — running the CLI inside an untrusted repo loaded that
    repo's ``.env`` (arbitrary POTPIE_*/token injection). Anchor instead to
    the nearest ancestor that is a real project root (has ``pyproject.toml``
    or ``.git``) and load only that directory's ``.env`` (security review
    L-3). Existing env is never overridden.
    """
    global _loaded
    if _loaded:
        return
    _loaded = True
    cur = Path.cwd().resolve()
    for _ in range(24):
        if any((cur / m).exists() for m in _PROJECT_ROOT_MARKERS):
            candidate = cur / ".env"
            if candidate.is_file():
                _load_env_file(candidate)
            _load_monorepo_potpie_env(cur)
            return
        if cur.parent == cur:
            break
        cur = cur.parent


def _load_monorepo_potpie_env(start: Path) -> None:
    """Merge ``potpie/.env`` when the CLI runs inside the Potpie monorepo."""
    for ancestor in [start, *start.parents]:
        potpie_root = ancestor / "potpie"
        if not potpie_root.is_dir():
            continue
        if (
            not (potpie_root / "pyproject.toml").is_file()
            and not (potpie_root / "app" / "main.py").is_file()
        ):
            continue
        env_file = potpie_root / ".env"
        if env_file.is_file():
            _load_env_file(env_file)
            return
