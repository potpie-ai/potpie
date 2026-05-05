"""Bot identity + remote auth ports.

The sandbox library deliberately doesn't know how Potpie resolves the
GitHub App installation token or where the bot's email is configured.
It exposes two ports the application can implement:

* :class:`BotIdentityProvider` — given a repo, return the
  ``(name, email)`` to stamp on commits made by the sandbox. Without
  this, ``SandboxClient.commit`` falls back to whatever ``git config
  user.email`` happens to be — which is rarely what the caller wants
  for production agent commits.

* :class:`RemoteAuthProvider` — given a repo, return a fresh
  :class:`RemoteAuth` for ``git push``/``fetch``. The clone-time token
  is scrubbed from the bare's ``origin`` URL on purpose (caches are
  shared across users); without this port, push to a private remote
  has no credential and fails.

Both are optional. The library functions without them; production wires
them up in :func:`sandbox.bootstrap.container.build_sandbox_container`.
"""

from __future__ import annotations

from typing import Protocol

from sandbox.domain.models import Author, RemoteAuth, RepoIdentity


class BotIdentityProvider(Protocol):
    """Resolve the author/committer identity for sandbox commits."""

    async def identity_for_repo(
        self, *, repo: RepoIdentity, user_id: str | None = None
    ) -> Author | None:
        """Return the bot ``Author`` for a repo, or ``None`` to defer.

        Returning ``None`` keeps the caller's existing behavior (use
        ``git config`` defaults, or whatever the user passed). Returning
        an :class:`Author` makes that identity the default for every
        commit the sandbox issues.

        ``user_id`` is provided so adapters can opt to attribute commits
        to the human user instead of the bot for repos where the App
        isn't installed; the default Potpie adapter ignores it and
        always returns the bot.
        """
        ...


class RemoteAuthProvider(Protocol):
    """Resolve a token for git network operations against a remote."""

    async def auth_for_remote(
        self, *, repo: RepoIdentity, user_id: str | None = None
    ) -> RemoteAuth | None:
        """Return a fresh :class:`RemoteAuth`, or ``None`` if anonymous.

        Called per-operation (push/fetch) rather than at clone time —
        GitHub App installation tokens expire in 1 hour, so caching at
        acquire-time would break long-running conversations.
        """
        ...
