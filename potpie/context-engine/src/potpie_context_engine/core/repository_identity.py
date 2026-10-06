"""Canonical repository identity shared by graph write and read surfaces.

Only supported spellings are normalized: local paths remain path identities and
git remote spellings collapse to ``host/owner/repo``.  This module does not
look up aliases or inspect stored graph data, so adopting it cannot guess at or
silently migrate an existing repository entity.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse


def normalize_repo_ref(value: str) -> str | None:
    """Collapse a git remote to a comparable ``host/owner/repo`` spelling."""
    raw = (value or "").strip()
    if not raw:
        return None
    if raw.startswith("repo:"):
        raw = raw[len("repo:") :]
    raw = raw.rstrip("/")
    if raw.endswith(".git"):
        raw = raw[:-4]
    if raw.startswith("git@") and ":" in raw:
        host, path = raw[4:].split(":", 1)
        return f"{host}/{path}".strip("/").lower()
    if "://" in raw:
        parsed = urlparse(raw)
        host = parsed.hostname or ""
        try:
            port = parsed.port
        except ValueError:
            port = None
        if port:
            host = f"{host}:{port}"
        if host and parsed.path:
            return f"{host}/{parsed.path.strip('/')}".lower()
    return raw.strip("/").lower()


def repo_identity_key(value: str) -> str | None:
    """Return a stable comparison key for a local path or remote reference."""
    raw = (value or "").strip()
    if not raw:
        return None
    if raw.startswith((".", "~")) or Path(raw).is_absolute():
        return str(Path(raw).expanduser().resolve(strict=False))
    return normalize_repo_ref(raw)


__all__ = ["normalize_repo_ref", "repo_identity_key"]
