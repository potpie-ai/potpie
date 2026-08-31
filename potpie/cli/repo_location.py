"""Repo-source location normalization shared by CLI entrypoints."""

from __future__ import annotations

import subprocess
from pathlib import Path


class RepoLocationError(ValueError):
    """Raised when a repo's durable identity cannot be determined."""


def resolve_repo_location(location: str) -> str:
    """Resolve repo-source shorthand to a durable, matchable location.

    ``.`` / ``current`` and relative paths registered verbatim are hard to match
    back to a working tree. Prefer the current repo's normalized remote when
    available, otherwise store an absolute path.

    Determinism matters here: the stored location *is* the repo's identity, and
    silently substituting an absolute path when a ``git`` probe happened to be
    slow made two identical invocations register two different identities. A
    repo with no ``origin`` deterministically gets its absolute path; a probe
    that *fails* raises instead of guessing.
    """

    raw = (location or "").strip()
    if raw.lower() in (".", "current"):
        cwd = Path.cwd().resolve()
        remote, failure = git_remote_or_reason(cwd)
        if failure is not None:
            raise RepoLocationError(
                f"could not determine this repo's git remote ({failure}); "
                "refusing to register it under a different identity — "
                "pass the location explicitly, e.g. "
                "'potpie source add repo <owner>/<repo>'"
            )
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


# `git remote get-url` exits 2 only when the remote is not configured — a
# definitive "no origin". Exit 128 is overloaded: it covers "not a git
# repository" (also definitive) *and* fatal repository/config errors, which
# must not be papered over as "no origin" or registration silently stores a
# path identity for a repo that does have a remote. So 128 is classified from
# git's own message.
_GIT_NO_REMOTE_CODE = 2
_GIT_FATAL_CODE = 128
_GIT_NOT_A_REPOSITORY = (
    "not a git repository",
    "not a work tree",
    "this operation must be run in a work tree",
)


def git_remote_or_reason(cwd: Path) -> tuple[str | None, str | None]:
    """Return ``(remote, failure_reason)`` for the current work tree.

    ``(remote, None)`` when origin resolved, ``(None, None)`` when the repo
    definitively has no origin, and ``(None, reason)`` when the probe itself
    could not answer — the case callers must not paper over.
    """
    try:
        proc = subprocess.run(
            ["git", "-C", str(cwd), "remote", "get-url", "origin"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except FileNotFoundError:
        return None, "git is not installed"
    except subprocess.TimeoutExpired:
        return None, "git did not respond within 10s"
    except OSError as exc:
        return None, f"git could not be run: {exc}"
    if proc.returncode == _GIT_NO_REMOTE_CODE:
        return None, None
    if proc.returncode != 0:
        first_line = next(iter((proc.stderr or "").strip().splitlines()), "")
        if proc.returncode == _GIT_FATAL_CODE and _is_not_a_repository(first_line):
            return None, None
        return None, first_line or f"git exited {proc.returncode}"
    return normalize_repo_ref(proc.stdout.strip()), None


def _is_not_a_repository(message: str) -> bool:
    """True for git's "there is no work tree here", not for a fatal error."""

    lowered = message.lower()
    return any(marker in lowered for marker in _GIT_NOT_A_REPOSITORY)


def current_git_remote(cwd: Path) -> str | None:
    """Lenient probe for identity *lookups*, where a miss is recoverable.

    Registration uses :func:`git_remote_or_reason` instead, because there a
    silent miss writes the wrong identity.
    """
    remote, _failure = git_remote_or_reason(cwd)
    return remote


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


__all__ = [
    "RepoLocationError",
    "current_git_remote",
    "current_repo_identity",
    "git_remote_or_reason",
    "normalize_repo_ref",
    "repo_identity_key",
    "resolve_repo_location",
]
