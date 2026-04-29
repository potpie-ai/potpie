"""Auth-token resolution for the local-fs workspace provider.

The sandbox library treats auth as a strict input — callers pass an
``auth_token`` to :meth:`SandboxClient.get_workspace` if they have one and the
local adapter embeds it in the clone URL. That contract works for explicit
tokens but doesn't capture the *chain* potpie needs in production:

    GitHub App installation token  →  User OAuth token  →  ``GH_TOKEN`` env

This module provides a small pluggable resolver. The default
:func:`env_token_resolver` walks the env-var portion of the chain (the cheap
case that doesn't need a DB session). Production code that wants the full
chain can register a richer resolver via :func:`set_token_resolver` — the
plan calls for moving the existing ``app/modules/repo_manager/sync_helper.py``
implementation here once we're ready to delete the legacy module.

Until then this gives the local adapter a self-sufficient default so unit and
e2e tests on public repos work without dragging in the legacy auth chain.
"""

from __future__ import annotations

import os
from typing import Callable, Protocol


class TokenResolver(Protocol):
    """Resolve a clone token for ``repo_name`` and an optional ``user_id``.

    Returns ``None`` if no usable token is available — callers should treat
    ``None`` as "clone anonymously" (works for public repos).
    """

    def __call__(
        self, *, repo_name: str | None, user_id: str | None
    ) -> str | None:  # pragma: no cover - protocol
        ...


def env_token_resolver(
    *, repo_name: str | None = None, user_id: str | None = None
) -> str | None:
    """Resolve a token from environment variables only.

    Honours, in priority order:
      1. ``GH_TOKEN`` (single token)
      2. ``GITHUB_TOKEN`` (CI-style alias)
      3. ``GH_TOKEN_LIST`` (comma- or newline-separated; first non-empty wins)
      4. ``CODE_PROVIDER_TOKEN`` (legacy potpie variable)
    """
    del repo_name, user_id  # default resolver doesn't use them
    for var in ("GH_TOKEN", "GITHUB_TOKEN"):
        val = os.getenv(var)
        if val:
            return val
    raw = os.getenv("GH_TOKEN_LIST", "").strip()
    if raw:
        for part in raw.replace("\n", ",").split(","):
            part = part.strip()
            if part:
                return part
    val = os.getenv("CODE_PROVIDER_TOKEN")
    if val:
        return val
    return None


_resolver: TokenResolver = env_token_resolver


def set_token_resolver(resolver: TokenResolver | Callable[..., str | None]) -> None:
    """Override the process-wide token resolver.

    Production code can install a richer resolver here — for example a
    GitHub App installation token resolver that falls back to user OAuth.
    The default :func:`env_token_resolver` stays in place when this is
    never called.
    """
    global _resolver
    _resolver = resolver  # type: ignore[assignment]


def resolve_token(
    *, repo_name: str | None = None, user_id: str | None = None
) -> str | None:
    """Public entry point used by the local workspace provider."""
    return _resolver(repo_name=repo_name, user_id=user_id)


__all__ = [
    "TokenResolver",
    "env_token_resolver",
    "resolve_token",
    "set_token_resolver",
]
