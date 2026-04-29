"""Repo cache provider port.

A `RepoCache` is the durable bare git mirror that backs all workspaces
for a given `(provider_host, repo_name)` pair. Splitting this concern
out of `WorkspaceProvider` lets the application layer:

* materialise the cache once at parse time (P5 in the roadmap) so the
  bare repo is already on disk before the first agent call,
* share the cache across users (the bare repo on disk doesn't care
  about per-user auth), and
* swap cache backends (local filesystem, Daytona-internal, future S3-
  backed) independently of how worktrees get checked out.

Adapters typically implement both `RepoCacheProvider` and
`WorkspaceProvider` — but the ports are kept separate so each can be
faked independently in tests.
"""

from __future__ import annotations

from typing import Protocol

from sandbox.domain.models import RepoCache, RepoCacheRequest


class RepoCacheProvider(Protocol):
    kind: str

    async def ensure_cache(self, request: RepoCacheRequest) -> RepoCache:
        """Ensure the bare repo exists at the requested ref.

        Creates the cache on first call; fetches the ref into the
        existing cache on subsequent calls. Returns a `RepoCache` that
        is `READY` and whose `location` points at the bare repo.

        Idempotent on `RepoCacheRequest.key()` — concurrent callers
        with the same key see the same cache (the application service
        is responsible for taking the lock).
        """
        ...

    async def get_cache(self, cache_id: str) -> RepoCache | None:
        ...

    async def delete_cache(self, cache: RepoCache) -> None:
        """Remove the bare repo from the backend.

        Does NOT cascade to workspaces — the application layer must
        delete dependent workspaces first (otherwise their worktrees
        become orphaned).
        """
        ...
