"""Repositories resource for PotpieRuntime library.

Provides worktree creation for codegen and other workflows.
Uses RepoManager when enabled (Option A: RepoManager as worktree creation source).
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from git import GitCommandError, Repo

from potpie.exceptions import PotpieError
from potpie.resources.base import BaseResource

if TYPE_CHECKING:
    from potpie.config import RuntimeConfig
    from potpie.core.database import DatabaseManager
    from potpie.core.neo4j import Neo4jManager

logger = logging.getLogger(__name__)

CODEGEN_WORKTREE_PREFIX = "codegen_"


class RepositoriesResource(BaseResource):
    """Access repository worktrees for codegen and workflows.

    When REPO_MANAGER_ENABLED=true, uses RepoManager as the worktree creation
    source: if the repo is already registered (e.g. from parsing), creates
    a new isolated worktree from the base path. Avoids re-cloning.
    """

    def __init__(
        self,
        config: "RuntimeConfig",
        db_manager: "DatabaseManager",
        neo4j_manager: "Neo4jManager",
    ):
        super().__init__(config, db_manager, neo4j_manager)
        self._repo_manager = None

    def _get_repo_manager(self):
        """Get RepoManager instance if enabled."""
        if self._repo_manager is not None:
            return self._repo_manager
        enabled = (
            os.getenv("REPO_MANAGER_ENABLED", "false").lower()
            in ("true", "1", "yes", "y")
        )
        if not enabled:
            return None
        try:
            from app.modules.repo_manager import RepoManager

            self._repo_manager = RepoManager()
            logger.info("RepositoriesResource: RepoManager initialized")
            return self._repo_manager
        except Exception as e:
            logger.warning(
                "RepositoriesResource: Failed to initialize RepoManager: %s", e
            )
            return None

    def _get_base_path_from_repo_manager(
        self, repo_name: str, ref: str
    ) -> Optional[str]:
        """
        Get base git repository path from RepoManager if available.

        RepoManager stores:
        - Base repo at .repos/<owner>/<repo>
        - Worktrees at base_path/worktrees/<ref>/

        Returns base path (main repo root, parent of worktrees) or None if not found.
        """
        rm = self._get_repo_manager()
        if not rm:
            return None

        def _resolve_base(path: str) -> Optional[str]:
            """Resolve path to base repo root (main repo, not worktree)."""
            try:
                r = Repo(path)
                # For worktrees: common_dir points to main repo's .git
                if hasattr(r, "common_dir") and r.common_dir:
                    return str(Path(r.common_dir).parent)
                return path
            except Exception:
                return path if os.path.exists(path) else None

        # Try base repo path first (no branch/commit)
        base_path = rm.get_repo_path(repo_name)
        if base_path and os.path.exists(base_path):
            resolved = _resolve_base(base_path)
            if resolved:
                return resolved

        # Try path for this ref (might be worktree path)
        is_commit = (
            len(ref) >= 7
            and all(c in "0123456789abcdefABCDEF" for c in ref[:7])
        )
        branch = None if is_commit else ref
        commit_id = ref if is_commit else None
        path = rm.get_repo_path(repo_name, branch=branch, commit_id=commit_id)
        if path and os.path.exists(path):
            resolved = _resolve_base(path)
            if resolved:
                return resolved

        # Fallback: expected base path from RepoManager layout
        expected = rm._get_repo_local_path(repo_name)
        if expected.exists() and (expected / ".git").exists():
            return str(expected)

        return None

    def _create_worktree_from_base(
        self,
        base_path: str,
        ref: str,
        unique_id: str,
        exists_ok: bool,
    ) -> str:
        """
        Create a new worktree from base repo for codegen isolation.

        Worktree path: base_path/worktrees/codegen_<unique_id>/
        """
        base = Path(base_path)
        worktree_name = f"{CODEGEN_WORKTREE_PREFIX}{unique_id}".replace(
            ":", "-"
        ).replace("/", "_")
        worktrees_dir = base / "worktrees"
        worktree_path = worktrees_dir / worktree_name

        if worktree_path.exists():
            if exists_ok:
                try:
                    r = Repo(str(worktree_path))
                    if r.working_tree_dir:
                        # Clean the worktree to avoid stale state from previous runs
                        # This ensures patches are applied against a clean base
                        logger.info(
                            "[RepositoriesResource] Cleaning and reusing existing worktree at %s",
                            worktree_path,
                        )
                        try:
                            # Reset any uncommitted changes
                            r.git.reset("--hard")
                            # Remove untracked files (including .potpie_codegen/ patches)
                            r.git.clean("-fdx")
                            logger.info(
                                "[RepositoriesResource] Worktree cleaned successfully"
                            )
                        except GitCommandError as e:
                            logger.warning(
                                "[RepositoriesResource] Failed to clean worktree: %s",
                                e
                            )
                        return str(worktree_path)
                except Exception:
                    pass
            try:
                base_repo = Repo(base_path)
                base_repo.git.worktree("remove", str(worktree_path), force=True)
            except GitCommandError:
                shutil.rmtree(worktree_path, ignore_errors=True)

        worktrees_dir.mkdir(parents=True, exist_ok=True)

        is_commit = (
            len(ref) >= 7
            and all(c in "0123456789abcdefABCDEF" for c in ref[:7])
        )

        try:
            base_repo = Repo(base_path)
            if is_commit:
                base_repo.git.worktree(
                    "add", str(worktree_path), ref, "--detach"
                )
            else:
                try:
                    base_repo.git.worktree("add", str(worktree_path), ref)
                except GitCommandError:
                    # Branch might not exist, fetch or create
                    remotes = [r.name for r in base_repo.remotes]
                    if "origin" in remotes:
                        try:
                            base_repo.git.fetch("origin", f"{ref}:{ref}")
                            base_repo.git.worktree("add", str(worktree_path), ref)
                        except GitCommandError:
                            current = base_repo.head.commit.hexsha
                            base_repo.git.worktree(
                                "add", "-b", ref, str(worktree_path), current
                            )
                    else:
                        current = base_repo.head.commit.hexsha
                        base_repo.git.worktree(
                            "add", "-b", ref, str(worktree_path), current
                        )

            logger.info(
                "[RepositoriesResource] Created worktree at %s for %s",
                worktree_path,
                ref,
            )
            return str(worktree_path)

        except GitCommandError as e:
            raise PotpieError(
                f"Failed to create worktree for {ref}: {e}"
            ) from e

    async def create_worktree(
        self,
        repo_name: str,
        ref: str,
        user_id: str,
        unique_id: str,
        *,
        exists_ok: bool = False,
    ) -> str:
        """
        Create or get an isolated worktree for the given repo and ref.

        When RepoManager is enabled and has the repo (e.g. from parsing),
        creates a new worktree from the base path. Avoids re-cloning.

        Note: Git operations are run in an executor to avoid blocking the event loop.

        Args:
            repo_name: Full repository name (e.g., "owner/repo")
            ref: Branch name or commit SHA
            user_id: User ID (for RepoManager tracking)
            unique_id: Unique identifier for this worktree (e.g. task_splitting_id)
            exists_ok: If True, return existing worktree path when present

        Returns:
            Absolute path to the worktree

        Raises:
            PotpieError: If worktree creation fails or repo not available
        """
        if not repo_name or not ref:
            raise PotpieError("repo_name and ref are required")

        base_path = self._get_base_path_from_repo_manager(repo_name, ref)

        if base_path:
            try:
                # Run blocking git operations in executor to avoid blocking event loop
                loop = asyncio.get_event_loop()
                path = await loop.run_in_executor(
                    None,
                    self._create_worktree_from_base,
                    base_path,
                    ref,
                    unique_id,
                    exists_ok,
                )
                # Update last accessed for eviction tracking
                rm = self._get_repo_manager()
                if rm:
                    is_commit = (
                        len(ref) >= 7
                        and all(c in "0123456789abcdefABCDEF" for c in ref[:7])
                    )
                    rm.update_last_accessed(
                        repo_name,
                        branch=None if is_commit else ref,
                        commit_id=ref if is_commit else None,
                        user_id=user_id,
                    )
                return path
            except PotpieError:
                raise
            except Exception as e:
                raise PotpieError(
                    f"Failed to create worktree from RepoManager base: {e}"
                ) from e

        # Fallback: RepoManager doesn't have the repo
        raise PotpieError(
            f"Repository {repo_name}@{ref} not found in RepoManager. "
            "Ensure REPO_MANAGER_ENABLED=true and the project has been parsed "
            "so the repository is available locally."
        )
