"""Repo-source location normalization shared by CLI entrypoints."""

from __future__ import annotations

import subprocess
from pathlib import Path


def resolve_repo_location(location: str) -> str:
    """Resolve repo-source shorthand to a durable, matchable location.

    ``.`` / ``current`` and relative paths registered verbatim are hard to match
    back to a working tree. Prefer the current repo's normalized remote when
    available, otherwise store an absolute path.
    """

    raw = (location or "").strip()
    if raw.lower() in (".", "current"):
        cwd = Path.cwd().resolve()
        remote = current_git_remote(cwd)
        return remote or str(cwd)
    if raw.startswith((".", "~")):
        return str(Path(raw).expanduser().resolve(strict=False))
    return raw


def repo_identity_key(value: str) -> str | None:
    """Stable local key for matching repo sources/defaults.

    Git remotes are normalized to ``host/owner/repo`` and lower-cased. Paths are
    resolved but keep their original casing because filesystem semantics vary.
    """

    raw = (value or "").strip()
    if not raw:
        return None
    if raw.startswith((".", "~")) or Path(raw).is_absolute():
        return str(Path(raw).expanduser().resolve(strict=False))
    return normalize_repo_ref(raw)


def current_repo_identity(cwd: Path) -> str | None:
    remote = current_git_remote(cwd)
    if remote:
        return remote
    try:
        return str(cwd.resolve())
    except OSError:
        return str(cwd)


def current_git_remote(cwd: Path) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "-C", str(cwd), "remote", "get-url", "origin"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    return normalize_repo_ref(proc.stdout.strip())


def normalize_repo_ref(value: str) -> str | None:
    raw = (value or "").strip()
    if not raw:
        return None
    if raw.endswith(".git"):
        raw = raw[:-4]
    if raw.startswith("git@") and ":" in raw:
        host, path = raw[4:].split(":", 1)
        return f"{host}/{path}".strip("/").lower()
    if "://" in raw:
        from urllib.parse import urlparse

        parsed = urlparse(raw)
        if parsed.netloc and parsed.path:
            return f"{parsed.netloc}/{parsed.path.strip('/')}".lower()
    return raw.strip("/").lower()


__all__ = [
    "current_git_remote",
    "current_repo_identity",
    "normalize_repo_ref",
    "repo_identity_key",
    "resolve_repo_location",
]
