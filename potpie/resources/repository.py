"""Repository resource for PotpieRuntime library."""

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


class RepositoryResource(BaseResource):
    """Access and manage Git repositories via RepoManager.

    Wraps RepoManager with a clean library interface.
    Translates standard Python exceptions to library-specific exceptions.
    User context is passed per-operation, not stored in the resource.
    """

    def __init__(
        self,
        config: RuntimeConfig,
        db_manager: DatabaseManager,
        neo4j_manager: Neo4jManager,
    ):
        super().__init__(config, db_manager, neo4j_manager)

    def _get_repo_manager(self):
        """Get a RepoManager instance configured from RuntimeConfig."""
        from app.modules.repo_manager.repo_manager import RepoManager

        return RepoManager(repos_base_path=self._config.repos_base_path)

    async def is_available(
        self,
        repo_name: str,
        user_id: str,
        *,
        branch: Optional[str] = None,
        commit_id: Optional[str] = None,
    ) -> bool:
        """Check if a repository is available locally.

        Args:
            repo_name: Full repository name (e.g., "owner/repo")
            user_id: User ID who owns the repository
            branch: Branch name (optional)
            commit_id: Commit SHA (optional)

        Returns:
            True if repository is available locally, False otherwise

        Raises:
            RepositoryError: If check fails
        """
        repo_manager = self._get_repo_manager()
        try:
            return repo_manager.is_repo_available(
                repo_name=repo_name,
                branch=branch,
                commit_id=commit_id,
                user_id=user_id,
            )
        except Exception as e:
            raise RepositoryError(
                f"Failed to check repository availability: {e}"
            ) from e

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
        """Register a repository that has been downloaded/parsed.

        Args:
            repo_name: Full repository name (e.g., "owner/repo")
            local_path: Local filesystem path where repo is stored
            user_id: User ID who owns this repository
            branch: Branch name (optional)
            commit_id: Commit SHA (optional)
            metadata: Additional metadata about repository (optional)

        Returns:
            Repository identifier/tracking ID

        Raises:
            RepositoryError: If registration fails

        Example:
            repo_id = await runtime.repositories.register(
                repo_name="langchain-ai/langchain",
                local_path="/path/to/repo",
                user_id="user-123",
                branch="main"
            )
        """
        repo_manager = self._get_repo_manager()
        try:
            return repo_manager.register_repo(
                repo_name=repo_name,
                local_path=local_path,
                branch=branch,
                commit_id=commit_id,
                user_id=user_id,
                metadata=metadata or {},
            )
        except Exception as e:
            raise RepositoryError(f"Failed to register repository: {e}") from e

    async def get_path(
        self,
        repo_name: str,
        user_id: str,
        *,
        branch: Optional[str] = None,
        commit_id: Optional[str] = None,
    ) -> Optional[str]:
        """Get the local filesystem path for a repository.

        Args:
            repo_name: Full repository name (e.g., "owner/repo")
            user_id: User ID who owns the repository
            branch: Branch name (optional)
            commit_id: Commit SHA (optional)

        Returns:
            Local filesystem path if available, None otherwise

        Raises:
            RepositoryError: If lookup fails
        """
        repo_manager = self._get_repo_manager()
        try:
            return repo_manager.get_repo_path(
                repo_name=repo_name,
                branch=branch,
                commit_id=commit_id,
                user_id=user_id,
            )
        except Exception as e:
            raise RepositoryError(f"Failed to get repository path: {e}") from e

    async def get_info(
        self,
        repo_name: str,
        user_id: str,
        *,
        branch: Optional[str] = None,
        commit_id: Optional[str] = None,
    ) -> Optional[RepositoryInfo]:
        """Get information about a registered repository.

        Args:
            repo_name: Full repository name (e.g., "owner/repo")
            user_id: User ID who owns the repository
            branch: Branch name (optional)
            commit_id: Commit SHA (optional)

        Returns:
            RepositoryInfo with detailed information, None if not found

        Raises:
            RepositoryError: If lookup fails
        """
        repo_manager = self._get_repo_manager()
        try:
            info_dict = repo_manager.get_repo_info(
                repo_name=repo_name,
                branch=branch,
                commit_id=commit_id,
                user_id=user_id,
            )

            if info_dict is None:
                return None

            return RepositoryInfo.from_dict(info_dict)
        except Exception as e:
            raise RepositoryError(f"Failed to get repository info: {e}") from e

    async def list_repos(
        self, user_id: str, *, limit: Optional[int] = None
    ) -> List[RepositoryInfo]:
        """List all available repositories for a user.

        Args:
            user_id: User ID whose repositories to list
            limit: Maximum number of repos to return (optional)

        Returns:
            List of RepositoryInfo objects

        Raises:
            RepositoryError: If listing fails
        """
        repo_manager = self._get_repo_manager()
        try:
            repos_dicts = repo_manager.list_repos(user_id=user_id, limit=limit)
            return [RepositoryInfo.from_dict(r) for r in repos_dicts]
        except Exception as e:
            raise RepositoryError(f"Failed to list repositories: {e}") from e

    async def evict(
        self,
        repo_name: str,
        user_id: str,
        *,
        branch: Optional[str] = None,
        commit_id: Optional[str] = None,
    ) -> bool:
        """Evict a repository from local storage.

        Args:
            repo_name: Full repository name (e.g., "owner/repo")
            user_id: User ID who owns the repository
            branch: Branch name (optional)
            commit_id: Commit SHA (optional)

        Returns:
            True if repository was evicted, False if not found

        Raises:
            RepositoryError: If eviction fails
        """
        repo_manager = self._get_repo_manager()
        try:
            return repo_manager.evict_repo(
                repo_name=repo_name,
                branch=branch,
                commit_id=commit_id,
                user_id=user_id,
            )
        except Exception as e:
            raise RepositoryError(f"Failed to evict repository: {e}") from e

    async def evict_stale(self, max_age_days: int, user_id: str) -> List[str]:
        """Evict repositories that haven't been accessed in a while.

        Args:
            max_age_days: Maximum age in days since last access
            user_id: User ID whose repositories to evict

        Returns:
            List of repository identifiers that were evicted

        Raises:
            RepositoryError: If eviction fails
        """
        repo_manager = self._get_repo_manager()
        try:
            return repo_manager.evict_stale_repos(
                max_age_days=max_age_days, user_id=user_id
            )
        except Exception as e:
            raise RepositoryError(f"Failed to evict stale repositories: {e}") from e

    async def get_volume_info(self, user_id: str) -> VolumeInfo:
        """Get information about disk volume usage.

        Args:
            user_id: User ID for filtering (optional, uses all repos if None)

        Returns:
            VolumeInfo with total usage, limits, and counts

        Raises:
            RepositoryError: If volume check fails
        """
        repo_manager = self._get_repo_manager()
        try:
            total_bytes = repo_manager.get_total_volume_bytes(user_id=user_id)
            percentage = repo_manager.get_volume_percentage(user_id=user_id)
            repo_count = len(repo_manager.list_available_repos(user_id=user_id))

            volume_limit_bytes = self._config.repos_volume_limit_bytes or 0

            return VolumeInfo(
                total_volume_bytes=total_bytes,
                volume_limit_bytes=volume_limit_bytes,
                volume_percentage=percentage,
                repo_count=repo_count,
            )
        except Exception as e:
            raise RepositoryError(f"Failed to get volume info: {e}") from e

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
        """Prepare a repository for parsing.

        Orchestrates: eviction → ensure bare repo → create worktree.

        Args:
            repo_name: Full repository name (e.g., "owner/repo")
            ref: Branch name or commit SHA
            user_id: User ID performing parsing
            repo_url: Optional Git repository URL (derived if not provided)
            auth_token: Optional authentication token
            is_commit: Whether ref is a commit SHA

        Returns:
            Path to worktree directory ready for parsing

        Raises:
            RepositoryError: If preparation fails

        Example:
            worktree_path = await runtime.repositories.prepare_for_parsing(
                repo_name="langchain-ai/langchain",
                ref="main",
                user_id="user-123"
            )
        """
        repo_manager = self._get_repo_manager()
        try:
            return repo_manager.prepare_for_parsing(
                repo_name=repo_name,
                ref=ref,
                repo_url=repo_url,
                auth_token=auth_token,
                is_commit=is_commit,
                user_id=user_id,
            )
        except Exception as e:
            raise RepositoryError(
                f"Failed to prepare repository for parsing: {e}"
            ) from e

    async def evict_stale_worktrees(
        self, max_age_days: int = 30, user_id: Optional[str] = None
    ) -> List[str]:
        """Evict old worktrees to free disk space.

        Worktrees are evicted before bare repos when volume thresholds are reached.

        Args:
            max_age_days: Maximum age in days before eviction
            user_id: Optional user ID for multi-tenant tracking

        Returns:
            List of worktrees that were evicted

        Raises:
            RepositoryError: If eviction fails
        """
        repo_manager = self._get_repo_manager()
        try:
            return repo_manager.evict_stale_worktrees(
                max_age_days=max_age_days, user_id=user_id
            )
        except Exception as e:
            raise RepositoryError(f"Failed to evict stale worktrees: {e}") from e

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
        """Create a worktree for a repository.

        Args:
            repo_name: Full repository name (e.g., "owner/repo")
            ref: Branch name or commit SHA
            user_id: User ID creating the worktree
            unique_id: Optional unique identifier for worktree
            auth_token: Optional authentication token
            is_commit: Whether ref is a commit SHA

        Returns:
            Path to worktree directory

        Raises:
            RepositoryError: If creation fails

        Example:
            worktree_path = await runtime.repositories.create_worktree(
                repo_name="langchain-ai/langchain",
                ref="main",
                user_id="user-123",
                unique_id="parse-001"
            )
        """
        repo_manager = self._get_repo_manager()
        try:
            repo_manager.ensure_bare_repo(
                repo_name=repo_name, auth_token=auth_token, ref=ref, user_id=user_id
            )
            return repo_manager.create_worktree(
                repo_name=repo_name,
                ref=ref,
                user_id=user_id,
                unique_id=unique_id,
                auth_token=auth_token,
                is_commit=is_commit,
                exists_ok=exists_ok,
            )
        except Exception as e:
            raise RepositoryError(f"Failed to create worktree: {e}") from e

    async def delete_worktree(
        self,
        repo_name: str,
        ref: str,
        user_id: str,
        unique_id: str,
    ) -> bool:
        """Delete a specific worktree for a repository.

        Args:
            repo_name: Full repository name (e.g., "owner/repo")
            ref: Branch name or commit SHA
            user_id: User ID who owns the worktree
            unique_id: Unique identifier for worktree

        Returns:
            True if worktree was deleted, False if not found

        Raises:
            RepositoryError: If deletion fails

        Example:
            deleted = await runtime.repositories.delete_worktree(
                repo_name="langchain-ai/langchain",
                ref="main",
                user_id="user-123",
                unique_id="parse-001"
            )
        """
        repo_manager = self._get_repo_manager()
        try:
            worktree_path = repo_manager._get_unique_worktree_path(
                repo_name=repo_name, ref=ref, user_id=user_id, unique_id=unique_id
            )

            if worktree_path and worktree_path.exists():
                import shutil

                shutil.rmtree(worktree_path)
                logger.info(f"Deleted worktree: {worktree_path}")
                return True

            logger.warning(f"Worktree not found: {worktree_path}")
            return False

        except Exception as e:
            raise RepositoryError(f"Failed to delete worktree: {e}") from e
