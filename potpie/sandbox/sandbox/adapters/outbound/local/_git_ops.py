"""Shared git helpers for the local adapters.

Extracted from ``LocalGitWorkspaceProvider`` so the workspace adapter
and the repo-cache adapter both reuse the same validation / auth-URL /
error-sanitization logic without one importing the other.

These are intentionally module-level functions, not methods on a class:
the helpers are pure (no state) and used as utilities by multiple
adapters. Putting them in a private module (`_git_ops`) keeps them out
of the package's public surface while still letting both providers
import the same implementations.
"""

from __future__ import annotations

import re
import subprocess
from urllib.parse import quote, urlparse

from sandbox.domain.errors import RepoAuthFailed, RepoCacheUnavailable


_TOKEN_URL_PATTERN = re.compile(r"x-access-token:[^@\s]+@")

# Substrings git uses for auth/permission failures across providers
# (GitHub, GitLab, generic ssh helpers). Lowercased compare.
_AUTH_KEYWORDS = (
    "authentication",
    "permission denied",
    "could not read from remote repository",
    "repository not found",
)


def validate_repo_name(repo_name: str) -> None:
    if not repo_name or "/" not in repo_name:
        raise ValueError("repo_name must be in owner/repo format")
    if repo_name.startswith("/") or "\\" in repo_name or ".." in repo_name:
        raise ValueError("repo_name contains unsafe path components")


def validate_ref(ref: str) -> None:
    if not ref:
        raise ValueError("git ref cannot be empty")
    if ".." in ref or "\n" in ref or "\r" in ref:
        raise ValueError("git ref contains unsafe characters")


def default_github_url(repo_name: str) -> str:
    return f"https://github.com/{repo_name}.git"


def authenticated_url(repo_url: str, auth_token: str | None) -> str:
    if not auth_token:
        return repo_url
    parsed = urlparse(repo_url)
    if not parsed.scheme or not parsed.netloc:
        return repo_url
    host = parsed.hostname or parsed.netloc
    if parsed.port:
        host = f"{host}:{parsed.port}"
    netloc = f"x-access-token:{quote(auth_token, safe='')}@{host}"
    return parsed._replace(netloc=netloc).geturl()


def sanitize_git_error(message: str) -> str:
    """Replace any inline ``x-access-token:<token>@host`` with ``***``.

    A naive ``replace("x-access-token:", ...)`` only inserts ``***`` next
    to the secret; the token itself remains in the message. This matches
    the full credential segment up to ``@`` and replaces it wholesale.
    """
    return _TOKEN_URL_PATTERN.sub("x-access-token:***@", message)


def run(cmd: list[str], *, timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout, check=False
    )


def raise_git_error(operation: str, stderr: str) -> None:
    """Translate a non-zero git invocation into the right typed error.

    Auth/permission/not-found failures get ``RepoAuthFailed`` so upstream
    callers can show a clear "you need to re-auth" message; everything
    else falls through as ``RepoCacheUnavailable``.

    Used by both clone and fetch so they classify identically — the old
    code raised ``RepoCacheUnavailable`` for fetch failures regardless of
    whether the underlying issue was an expired token.
    """
    message = sanitize_git_error(stderr or "")
    lower = message.lower()
    if any(k in lower for k in _AUTH_KEYWORDS):
        raise RepoAuthFailed(f"{operation}: {message}")
    raise RepoCacheUnavailable(f"{operation}: {message}")
