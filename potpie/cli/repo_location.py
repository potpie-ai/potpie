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

import os
from pathlib import Path
from typing import Final, Literal

from potpie_context_engine.domain.git_probe import current_git_remote as _current_git_remote
from potpie_context_engine.domain.repo_identity import (
    normalize_repo_ref,
    repo_identity_key,
)


#: The registered ref names the working tree the caller is standing in — either
#: exactly, or as the project directory the cwd sits inside.
REPO_MATCH_SELF: Final = "self"

#: The registered repository lives *inside* the caller's cwd: the cwd is a
#: workspace/parent directory that happens to contain a registered project. Not
#: the same fact as :data:`REPO_MATCH_SELF` and must not be reported as one —
#: see :func:`classify_repo_source_match`.
REPO_MATCH_CONTAINED: Final = "contained"

RepoMatch = Literal["self", "contained"]


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


def repo_identity_key_for_location(location: str) -> str | None:
    """The routing key for a *registered* repo location.

    ``repo_identity_key`` keys a path by the path and a remote by the remote,
    which is right for two strings compared against each other and wrong for
    the one comparison that matters: repo→pot routing keys the *current* working
    tree by its git remote (see :func:`current_repo_identity`). So
    ``source add repo /path/to/shop --default`` wrote ``repo_defaults`` under
    the absolute path while every later command looked the repo up under
    ``github.com/acme/shop`` — the flag reported success and changed nothing
    routing ever read.

    A location that resolves to a git working tree is therefore keyed by that
    tree's remote, exactly as ``.`` already is. Anything else — an
    ``owner/repo`` ref, a URL, a plain directory with no remote — keeps
    ``repo_identity_key``'s answer.
    """
    raw = (location or "").strip()
    if not raw:
        return None
    path = _local_directory(raw)
    if path is not None:
        remote = current_git_remote(path)
        if remote:
            return repo_identity_key(remote)
    return repo_identity_key(raw)


def _local_directory(location: str) -> Path | None:
    """``location`` as an existing local directory, or ``None``.

    Only path-shaped spellings are probed. ``owner/repo`` is a remote ref that
    would become a filesystem lookup — and, in a directory that happens to hold
    a matching subtree, a wrong one.
    """
    if not location.startswith(("/", "~", ".")):
        return None
    try:
        path = Path(location).expanduser()
        return path if path.is_dir() else None
    except OSError:
        return None


def classify_repo_source_match(
    ref: str, *, cwd: Path, remote: str | None
) -> RepoMatch | None:
    """How a registered repo ref relates to ``cwd``, or ``None`` for no relation.

    Two very different relations were collapsed into one boolean, and the
    collapse is what makes a workspace root behave like a project. ``ref``
    matching because the cwd *is* (or is inside) the registered project is
    :data:`REPO_MATCH_SELF` — the pot that owns this tree. ``ref`` matching
    because the registered project sits *underneath* the cwd is
    :data:`REPO_MATCH_CONTAINED`: standing in ``~/work`` with ``~/work/alpha``
    registered says nothing about which project the caller means, and with two
    registered children it says nothing twice.

    Callers that need the old boolean ask ``is not None``.
    """
    ref = (ref or "").strip()
    if not ref:
        return None
    source_path = Path(ref).expanduser()
    if source_path.is_absolute() or ref.startswith((".", "~")):
        try:
            resolved = source_path.resolve(strict=False)
        except OSError:
            resolved = source_path.absolute()
        if cwd == resolved or cwd.is_relative_to(resolved):
            return REPO_MATCH_SELF
        if resolved.is_relative_to(cwd):
            return REPO_MATCH_CONTAINED

    normalized_source = normalize_repo_ref(ref)
    if remote and normalized_source and normalized_source == remote:
        return REPO_MATCH_SELF
    return None


def current_repo_identity(cwd: Path) -> str | None:
    remote = current_git_remote(cwd)
    if remote:
        return remote
    try:
        return str(cwd.resolve())
    except OSError:
        return str(cwd)


def current_git_remote(cwd: Path) -> str | None:
    """The normalized ``origin`` remote of ``cwd``. See ``domain.git_probe`` for
    why the probe is shaped the way it is; this used to be the one site that
    had the Windows-safe shape, and the others have since joined it."""
    return _current_git_remote(cwd)


__all__ = [
    "REPO_MATCH_CONTAINED",
    "REPO_MATCH_SELF",
    "RepoMatch",
    "classify_repo_source_match",
    "current_git_remote",
    "current_repo_identity",
    "normalize_repo_ref",
    "repo_identity_key",
    "repo_identity_key_for_location",
    "resolve_repo_location",
]
