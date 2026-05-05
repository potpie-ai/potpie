"""Remote auth adapter for the sandbox.

Implements :class:`sandbox.RemoteAuthProvider` so ``SandboxClient.push``
(and any future fetch path) injects a fresh credential per call. Pulls
the token from the same chain the clone path uses
(``app.modules.intelligence.tools.sandbox.client._resolve_auth``) so
both ends of the conversation — clone and push — agree on identity.

The ``RemoteAuth.kind`` field reflects the actual chain branch that
produced the token (``"context"``, ``"app"``, ``"user_oauth"``,
``"env"``), so logs let ops see whether a given push attributes to the
bot or to the user without dumping the token. This is a pure
observability hook — the sandbox sends the same bearer header for any
non-empty token.
"""

from __future__ import annotations

import logging

from sandbox import RemoteAuth, RepoIdentity


logger = logging.getLogger(__name__)


class PotpieRemoteAuthProvider:
    """Resolve a fresh GitHub credential for sandbox network ops."""

    async def auth_for_remote(
        self, *, repo: RepoIdentity, user_id: str | None = None
    ) -> RemoteAuth | None:
        # Lazy-import to avoid pulling DB / GitHub-App machinery into
        # tests that don't need it; the sandbox library itself is meant
        # to be import-light.
        try:
            from app.modules.intelligence.tools.sandbox.client import (
                _resolve_auth,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "PotpieRemoteAuthProvider: _resolve_auth unavailable: %r",
                exc,
            )
            return None

        resolved = _resolve_auth(user_id, repo.repo_name)
        if not resolved.token:
            logger.info(
                "sandbox.remote_auth: no credential available for %s "
                "(user_id=%s) — push/fetch will run anonymously",
                repo.repo_name,
                user_id,
            )
            return None
        # NEVER log the token itself — only the kind.
        logger.info(
            "sandbox.remote_auth: %s for %s (user_id=%s)",
            resolved.kind,
            repo.repo_name,
            user_id,
        )
        return RemoteAuth(token=resolved.token, kind=resolved.kind)
