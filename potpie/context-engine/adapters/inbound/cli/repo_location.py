"""Repo-source location normalization shared by CLI entrypoints."""

from __future__ import annotations

from pathlib import Path
import subprocess


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
        return f"{host}/{path}".strip("/")
    if "://" in raw:
        from urllib.parse import urlparse

        parsed = urlparse(raw)
        if parsed.netloc and parsed.path:
            return f"{parsed.netloc}/{parsed.path.strip('/')}"
    return raw


__all__ = ["current_git_remote", "normalize_repo_ref", "resolve_repo_location"]
