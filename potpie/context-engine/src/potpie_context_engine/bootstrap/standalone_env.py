"""Load a trusted project ``.env`` for standalone Context Engine processes."""

from __future__ import annotations

import os
from collections.abc import Collection
from pathlib import Path

_loaded = False
_PROTECTED_DOTENV_KEYS = frozenset({"POTPIE_ENVIRONMENT"})
_PROJECT_ROOT_MARKERS = ("pyproject.toml", ".git")


def load_standalone_env(*, skip_keys: Collection[str] = _PROTECTED_DOTENV_KEYS) -> None:
    """Merge only the nearest trusted project root's ``.env`` once."""
    global _loaded
    if _loaded:
        return
    _loaded = True
    current = Path.cwd().resolve()
    for _ in range(24):
        if any((current / marker).exists() for marker in _PROJECT_ROOT_MARKERS):
            candidate = current / ".env"
            if candidate.is_file():
                _load_env_file(candidate, skip_keys=skip_keys)
            _load_monorepo_context_engine_env(current, skip_keys=skip_keys)
            return
        if current.parent == current:
            return
        current = current.parent


def _load_env_file(
    path: Path, *, skip_keys: Collection[str] = _PROTECTED_DOTENV_KEYS
) -> None:
    blocked_keys = _PROTECTED_DOTENV_KEYS.union(skip_keys)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    for line in lines:
        parsed = _parse_env_line(line)
        if parsed is None:
            continue
        key, value = parsed
        if key not in blocked_keys and key not in os.environ:
            os.environ[key] = value


def _load_monorepo_context_engine_env(
    start: Path, *, skip_keys: Collection[str]
) -> None:
    """Preserve the source checkout's historical ``potpie/.env`` fallback."""
    for ancestor in (start, *start.parents):
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
            _load_env_file(env_file, skip_keys=skip_keys)
            return


def _parse_env_line(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    if stripped.lower().startswith("export "):
        stripped = stripped[7:].strip()
    if "=" not in stripped:
        return None
    key, value = stripped.split("=", 1)
    key = key.strip()
    if not key:
        return None
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        value = value[1:-1]
    return key, value


__all__ = ["load_standalone_env"]
