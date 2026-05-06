"""Repository resource for PotpieRuntime library.

Sandbox-backed implementation of the public repository API.

The previous implementation was a thin wrapper around :class:`RepoManager`.
With the sandbox cutover, ``RepoManager`` is no longer the source of truth —
``SandboxClient`` is — so this module routes every public method through the
sandbox client.

Some legacy administrative methods (eviction, volume reports) don't have a
clean sandbox equivalent yet because the sandbox's
:class:`EvictionPolicy` port is currently a no-op (``NoOpEvictionPolicy``).
Those methods raise :class:`NotImplementedError` with a clear migration note
rather than silently no-op'ing — out-of-tree callers should pin to the
runtime version that still backed them with RepoManager, or wait for the
follow-up that promotes :class:`VolumeBasedEvictionPolicy` to default.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

from potpie.exceptions import RepositoryError
from potpie.resources.base import BaseResource
from potpie.types.repository import (
    RepositoryInfo,
    VolumeInfo,
)

if TYPE_CHECKING:
    from potpie.config import RuntimeConfig
    from potpie.core.database import DatabaseManager
    from potpie.core.neo4j import Neo4jManager

logger = logging.getLogger(__name__)


_ADMIN_DEPRECATION_NOTE = (
    "RepoManager-backed administrative APIs (eviction, volume info, "
    "register, list_repos, get_info, get_path, evict_stale_*) are no "
    "longer supported on the sandbox-backed runtime. Wait for the "
    "follow-up that promotes VolumeBasedEvictionPolicy + a public "
    "store enumeration API on SandboxClient."
)


class RepositoryResource(BaseResource):
    """Public repository resource, sandbox-backed."""

    def __init__(
        self,
        config: "RuntimeConfig",
        db_manager: "DatabaseManager",
        neo4j_manager: "Neo4jManager",
    ):
        super().__init__(config, db_manager, neo4j_manager)

    # ------------------------------------------------------------------
    # Lookups — answered against the sandbox metadata store.
    # ------------------------------------------------------------------
    async def is_available(
        self,
        repo_name: str,
        user_id: str,
        *,
        branch: Optional[str] = None,
        commit_id: Optional[str] = None,
    ) -> bool:
        """Return ``True`` if a sandbox ``RepoCache`` row exists for
        ``(repo_name, ref)``.

        Mirrors the legacy ``RepoManager.is_repo_available`` shape, but
        the truth source is now the sandbox metadata store.
        """
        ref = commit_id or branch
        if not ref:
            raise RepositoryError("Either branch or commit_id is required")
        try:
            from app.modules.intelligence.tools.sandbox.client import (
                get_sandbox_client,
            )
            from sandbox.domain.models import RepoCacheRequest, RepoIdentity

            client = get_sandbox_client()
            req = RepoCacheRequest(
                repo=RepoIdentity(repo_name=repo_name),
                base_ref=ref,
                user_id=user_id,
            )
            existing = await client.container.store.find_repo_cache_by_key(
                req.key()
            )
            return existing is not None
        except Exception as exc:
            raise RepositoryError(
                f"Failed to check repository availability: {exc}"
            ) from exc

    async def get_path(
        self,
        repo_name: str,
        user_id: str,
        *,
        branch: Optional[str] = None,
        commit_id: Optional[str] = None,
    ) -> Optional[str]:
        """Return the on-disk path of the sandbox bare cache for
        ``(repo_name, ref)``, or ``None`` if no cache exists yet.
        """
        ref = commit_id or branch
        if not ref:
            raise RepositoryError("Either branch or commit_id is required")
        try:
            from app.modules.intelligence.tools.sandbox.client import (
                get_sandbox_client,
            )
            from sandbox.domain.models import RepoCacheRequest, RepoIdentity

            client = get_sandbox_client()
            req = RepoCacheRequest(
                repo=RepoIdentity(repo_name=repo_name),
                base_ref=ref,
                user_id=user_id,
            )
            existing = await client.container.store.find_repo_cache_by_key(
                req.key()
            )
            if existing is None:
                return None
            return existing.location.local_path
        except Exception as exc:
            raise RepositoryError(f"Failed to get repository path: {exc}") from exc

    # ------------------------------------------------------------------
    # Bare cache + worktree creation.
    # ------------------------------------------------------------------
    async def prepare_for_parsing(
        self,
        repo_name: str,
        ref: str,
        user_id: str,
        *,
        repo_url: Optional[str] = None,
        auth_token: Optional[str] = None,
        is_commit: bool = False,
    ) -> str:
        """Provision the sandbox bare cache + an ANALYSIS workspace on
        ``ref`` and return the workspace's local path.

        Replaces the legacy RepoManager entry point parsing and the
        worktree-allocation step in one call. ``is_commit`` is a no-op
        in this implementation — the sandbox treats branch and commit
        refs uniformly.
        """
        del is_commit
        try:
            from app.modules.intelligence.tools.sandbox.client import (
                get_sandbox_client,
                provision_repo_cache,
            )
            from sandbox import WorkspaceMode

            await provision_repo_cache(
                user_id=user_id,
                repo_name=repo_name,
                base_ref=ref,
                auth_token=auth_token,
                repo_url=repo_url,
            )
            client = get_sandbox_client()
            handle = await client.acquire_session(
                user_id=user_id,
                project_id=f"parsing:{repo_name}",
                repo=repo_name,
                branch=ref,
                base_ref=ref,
                create_branch=False,
                auth_token=auth_token,
                mode=WorkspaceMode.ANALYSIS,
                repo_url=repo_url,
            )
            if handle.local_path is None:
                raise RepositoryError(
                    f"Sandbox returned a workspace with no local path for "
                    f"{repo_name}@{ref}; parsing requires a host-fs backend."
                )
            return handle.local_path
        except Exception as exc:
            raise RepositoryError(
                f"Failed to prepare repository for parsing: {exc}"
            ) from exc

    async def create_worktree(
        self,
        repo_name: str,
        ref: str,
        *,
        user_id: str | None = None,
        unique_id: str | None = None,
        auth_token: str | None = None,
        is_commit: bool = False,
        exists_ok: bool = False,
    ) -> Path:
        """Allocate (or look up) a per-``unique_id`` EDIT workspace.

        Idempotent on ``(user_id, unique_id, repo_name, ref)``. Returns
        the worktree's local path.
        """
        del is_commit, exists_ok
        if not user_id:
            raise RepositoryError("user_id is required")
        if not unique_id:
            raise RepositoryError("unique_id is required")
        try:
            from app.modules.intelligence.tools.sandbox.client import (
                get_sandbox_client,
            )
            from sandbox import WorkspaceMode

            client = get_sandbox_client()
            handle = await client.acquire_session(
                user_id=user_id,
                project_id=f"runtime:{unique_id}",
                repo=repo_name,
                branch=ref,
                base_ref=ref,
                create_branch=False,
                auth_token=auth_token,
                mode=WorkspaceMode.EDIT,
                conversation_id=unique_id,
            )
            if handle.local_path is None:
                raise RepositoryError(
                    f"Sandbox returned a workspace with no local path for "
                    f"{repo_name}@{ref}"
                )
            return Path(handle.local_path)
        except Exception as exc:
            raise RepositoryError(f"Failed to create worktree: {exc}") from exc

    async def delete_worktree(
        self,
        repo_name: str,
        ref: str,
        user_id: str,
        unique_id: str,
    ) -> bool:
        """Tear down the workspace allocated for ``unique_id``.

        Returns ``True`` if a workspace was destroyed, ``False`` if none
        existed. Looks the workspace up by re-acquiring the same key, then
        calling :meth:`SandboxClient.destroy_workspace`.
        """
        try:
            from app.modules.intelligence.tools.sandbox.client import (
                get_sandbox_client,
            )
            from sandbox import WorkspaceMode

            client = get_sandbox_client()
            try:
                handle = await client.get_workspace(
                    user_id=user_id,
                    project_id=f"runtime:{unique_id}",
                    repo=repo_name,
                    branch=ref,
                    base_ref=ref,
                    create_branch=False,
                    mode=WorkspaceMode.EDIT,
                    conversation_id=unique_id,
                )
            except Exception:
                return False
            await client.destroy_workspace(handle)
            return True
        except Exception as exc:
            raise RepositoryError(f"Failed to delete worktree: {exc}") from exc

    # ------------------------------------------------------------------
    # Admin operations — not yet supported on the sandbox-backed runtime.
    # ------------------------------------------------------------------
    async def register(
        self,
        repo_name: str,
        local_path: str,
        user_id: str,
        *,
        branch: Optional[str] = None,
        commit_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> str:
        del repo_name, local_path, user_id, branch, commit_id, metadata
        raise NotImplementedError(_ADMIN_DEPRECATION_NOTE)

    async def get_info(
        self,
        repo_name: str,
        user_id: str,
        *,
        branch: Optional[str] = None,
        commit_id: Optional[str] = None,
    ) -> Optional[RepositoryInfo]:
        del repo_name, user_id, branch, commit_id
        raise NotImplementedError(_ADMIN_DEPRECATION_NOTE)

    async def list_repos(
        self, user_id: str, *, limit: Optional[int] = None
    ) -> List[RepositoryInfo]:
        del user_id, limit
        raise NotImplementedError(_ADMIN_DEPRECATION_NOTE)

    async def evict(
        self,
        repo_name: str,
        user_id: str,
        *,
        branch: Optional[str] = None,
        commit_id: Optional[str] = None,
    ) -> bool:
        del repo_name, user_id, branch, commit_id
        raise NotImplementedError(_ADMIN_DEPRECATION_NOTE)

    async def evict_stale(self, max_age_days: int, user_id: str) -> List[str]:
        del max_age_days, user_id
        raise NotImplementedError(_ADMIN_DEPRECATION_NOTE)

    async def evict_stale_worktrees(
        self, max_age_days: int = 30, user_id: Optional[str] = None
    ) -> List[str]:
        del max_age_days, user_id
        raise NotImplementedError(_ADMIN_DEPRECATION_NOTE)

    async def get_volume_info(self, user_id: str) -> VolumeInfo:
        del user_id
        raise NotImplementedError(_ADMIN_DEPRECATION_NOTE)
