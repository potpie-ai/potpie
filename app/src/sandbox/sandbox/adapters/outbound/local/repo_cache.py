"""Local filesystem repo cache provider.

Owns the bare-repo concern: clones once into ``.repos/<owner>/<repo>/.bare``,
fetches new refs into the existing bare, and never holds a worktree.
``LocalGitWorkspaceProvider`` delegates here for the cache; it only forks
worktrees off the resulting bare repo.

The split mirrors the doc's three-entity model (RepoCache vs Workspace
vs Runtime) and removes the clone-on-create policy from the workspace
adapter (the doc explicitly forbids the adapter from owning that policy
— see "Adapter Responsibilities").
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from sandbox.adapters.outbound.local._git_ops import (
    authenticated_url,
    default_github_url,
    raise_git_error,
    run,
    validate_ref,
    validate_repo_name,
)
from sandbox.adapters.outbound.local.auth import resolve_token
from sandbox.domain.models import (
    RepoCache,
    RepoCacheRequest,
    WorkspaceLocation,
    WorkspaceState,
    WorkspaceStorageKind,
    new_id,
    utc_now,
)


class LocalRepoCacheProvider:
    """Bare-repo cache rooted under ``repos_base_path``."""

    kind = "local"

    def __init__(self, repos_base_path: str | Path | None = None) -> None:
        raw_base = repos_base_path or os.getenv("SANDBOX_REPOS_BASE_PATH") or ".repos"
        self.repos_base_path = Path(raw_base).expanduser().resolve()
        self.repos_base_path.mkdir(parents=True, exist_ok=True)
        self._by_id: dict[str, RepoCache] = {}
        # Idempotency: the same `RepoCacheRequest.key()` must return the
        # same cache object across calls. Without this, a second call
        # (e.g. via `LocalGitWorkspaceProvider` after the service has
        # already provisioned the cache) would mint a fresh id and the
        # workspace's `repo_cache_id` would diverge from the store row.
        self._by_key: dict[str, str] = {}

    async def ensure_cache(self, request: RepoCacheRequest) -> RepoCache:
        return await asyncio.to_thread(self._ensure_cache_sync, request)

    async def get_cache(self, cache_id: str) -> RepoCache | None:
        return self._by_id.get(cache_id)

    async def delete_cache(self, cache: RepoCache) -> None:
        self._by_id.pop(cache.id, None)
        self._by_key.pop(cache.key, None)
        path = cache.location.local_path
        if not path:
            return
        # Best-effort. Workspaces forked off this cache must already be
        # gone; otherwise their worktrees become orphaned. The service
        # layer is responsible for that ordering.
        await asyncio.to_thread(self._remove_bare_sync, Path(path))

    def bare_path(self, repo_name: str) -> Path:
        """Public accessor — workspace adapter needs this to add worktrees."""
        return self.repos_base_path / repo_name / ".bare"

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _ensure_cache_sync(self, request: RepoCacheRequest) -> RepoCache:
        validate_repo_name(request.repo.repo_name)
        validate_ref(request.base_ref)

        bare_path = self.bare_path(request.repo.repo_name)
        repo_url = request.repo.repo_url or default_github_url(
            request.repo.repo_name
        )
        token = request.auth_token or resolve_token(
            repo_name=request.repo.repo_name, user_id=request.user_id
        )
        # Cache rows are keyed by (host, repo_name) — multiple users
        # share the same on-disk bare. We clone with a tokenized URL
        # but immediately rewrite ``origin`` to the plain URL so the
        # next user's request can't piggy-back on the persisted token.
        # Fetches are always given an explicit URL for the same reason.
        fetch_url = authenticated_url(repo_url, token)

        if bare_path.exists() and (bare_path / "HEAD").exists():
            self._fetch_ref(bare_path, fetch_url, request.base_ref)
        else:
            self._clone_bare(bare_path, fetch_url, repo_url)
            self._fetch_ref(bare_path, fetch_url, request.base_ref)

        # If we already minted a cache for this key in this process,
        # refresh its timestamps and reuse the id. The on-disk fetch
        # above still ran (it's the only way to materialize the
        # requested ref into the bare); the in-memory id stays stable.
        key = request.key()
        existing_id = self._by_key.get(key)
        if existing_id is not None:
            existing = self._by_id.get(existing_id)
            if existing is not None:
                existing.last_fetched_at = utc_now()
                existing.last_used_at = utc_now()
                existing.updated_at = utc_now()
                return existing

        cache = RepoCache(
            id=new_id("rc"),
            key=key,
            repo=request.repo,
            location=WorkspaceLocation(
                kind=WorkspaceStorageKind.LOCAL_PATH,
                local_path=str(bare_path.resolve()),
            ),
            backend_kind=self.kind,
            state=WorkspaceState.READY,
            last_fetched_at=utc_now(),
        )
        self._by_id[cache.id] = cache
        self._by_key[key] = cache.id
        return cache

    @staticmethod
    def _clone_bare(bare_path: Path, fetch_url: str, plain_url: str) -> None:
        bare_path.parent.mkdir(parents=True, exist_ok=True)
        result = run(
            [
                "git",
                "clone",
                "--bare",
                "--filter=blob:none",
                "--",
                fetch_url,
                str(bare_path),
            ],
            timeout=600,
        )
        if result.returncode != 0:
            raise_git_error("git clone failed", result.stderr)
        # Strip the token from the persisted ``origin`` URL. Subsequent
        # fetches don't rely on it (we always pass a fresh tokenized
        # URL), and leaving it on disk under a cache that's shared
        # across users is a credential-leakage risk.
        scrub = run(
            [
                "git",
                "-C",
                str(bare_path),
                "remote",
                "set-url",
                "origin",
                plain_url,
            ],
            timeout=60,
        )
        if scrub.returncode != 0:
            raise_git_error("git remote set-url origin failed", scrub.stderr)

    @staticmethod
    def _fetch_ref(bare_path: Path, fetch_url: str, ref: str) -> None:
        # Pass the tokenized URL explicitly instead of relying on
        # ``origin`` so the credential never lands in ``.git/config``
        # of a cache shared between users.
        result = run(
            ["git", "-C", str(bare_path), "fetch", "--", fetch_url, ref],
            timeout=300,
        )
        if result.returncode != 0:
            raise_git_error("git fetch failed", result.stderr)

    @staticmethod
    def _remove_bare_sync(path: Path) -> None:
        if not path.exists():
            return
        import shutil

        shutil.rmtree(path, ignore_errors=True)
