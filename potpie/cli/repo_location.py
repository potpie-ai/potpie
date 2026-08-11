"""Repo-source location normalization shared by CLI entrypoints.

The two identity functions are re-exported from
:mod:`potpie_context_engine.domain.repo_identity` rather than defined here. This
module used to carry its own copy, and the copies drifted where it mattered
least visibly: the engine's keyed a credential-bearing remote by its full
``user:password@host`` netloc while this one keyed it by host alone, so
``potpie setup`` and ``potpie source add repo .`` disagreed about whether two
spellings of one repository were the same repository — and only one of them
dedup'd. Re-exporting keeps this import path (the CLI's own seam) while leaving
exactly one definition of what a repo *is*.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from potpie_context_engine.domain.repo_identity import (
    normalize_repo_ref,
    repo_identity_key,
)


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


__all__ = [
    "current_git_remote",
    "current_repo_identity",
    "normalize_repo_ref",
    "repo_identity_key",
    "resolve_repo_location",
]
