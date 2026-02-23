"""
Shared helper for ensuring repositories are registered in the repo manager.
Used by parsing and conversation flows.
"""

import os
from typing import TYPE_CHECKING, Any, Dict, Optional

from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)

if TYPE_CHECKING:
    from app.modules.repo_manager.repo_manager import RepoManager
    from sqlalchemy.orm import Session


def get_or_create_worktree_path(
    repo_manager: "RepoManager",
    repo_name: str,
    branch: Optional[str],
    commit_id: Optional[str],
    user_id: Optional[str],
    sql_db: Optional["Session"] = None,
) -> Optional[str]:
    """
    Get the worktree path for a repo, cloning via prepare_for_parsing if missing.

    Tries multiple lookup strategies first (commit_id, branch). If no worktree is
    found, calls prepare_for_parsing to clone/recreate it. This handles the case
    where a worktree was evicted or the repo was never cloned locally.

    Args:
        repo_manager: RepoManager instance
        repo_name: Full repository name (e.g., 'owner/repo')
        branch: Branch name
        commit_id: Commit SHA
        user_id: User ID for security checks
        sql_db: Optional DB session used to fetch the user's GitHub auth token

    Returns:
        Path to the worktree, or None if it can't be obtained
    """
    if not repo_manager:
        return None

    def _lookup() -> Optional[str]:
        path = repo_manager.get_repo_path(
            repo_name, branch=branch, commit_id=commit_id, user_id=user_id
        )
        if path and os.path.exists(path):
            return path
        if commit_id:
            path = repo_manager.get_repo_path(
                repo_name, commit_id=commit_id, user_id=user_id
            )
            if path and os.path.exists(path):
                return path
        if branch:
            path = repo_manager.get_repo_path(
                repo_name, branch=branch, user_id=user_id
            )
            if path and os.path.exists(path):
                return path
        return None

    worktree_path = _lookup()
    if worktree_path:
        return worktree_path

    # Worktree not found — try to create it via prepare_for_parsing
    ref = commit_id if commit_id else branch
    if not ref:
        logger.warning(
            f"Cannot create worktree for {repo_name}: no ref (branch or commit_id)"
        )
        return None

    auth_token = None
    if sql_db and user_id:
        try:
            from app.modules.code_provider.github.github_service import GithubService

            auth_token = GithubService(sql_db).get_github_oauth_token(user_id)
        except Exception as e:
            logger.debug(f"Could not get GitHub OAuth token for {user_id}: {e}")

    try:
        logger.info(
            f"Worktree for {repo_name}@{ref} not found, calling prepare_for_parsing"
        )
        worktree_path_str = repo_manager.prepare_for_parsing(
            repo_name,
            ref,
            auth_token=auth_token,
            is_commit=bool(commit_id),
            user_id=user_id,
        )
        if worktree_path_str and os.path.exists(worktree_path_str):
            logger.info(
                f"Created worktree for {repo_name}@{ref} at {worktree_path_str}"
            )
            return worktree_path_str
    except Exception as e:
        logger.warning(
            f"prepare_for_parsing failed for {repo_name}@{ref}: {e}", exc_info=True
        )

    return None


def ensure_repo_registered(
    project_data: Dict[str, Any],
    user_id: str,
    repo_manager: "RepoManager",
    *,
    registered_from: str = "sync_helper",
) -> None:
    """
    Ensure that the repository for a project is registered in the repo manager.
    If the repo doesn't exist, attempts to register it from disk if found in
    repo manager's expected locations.

    Args:
        project_data: Dict with keys project_name, branch_name, commit_id (optional),
                      repo_path (optional). Same shape as project from DB.
        user_id: The user ID
        repo_manager: RepoManager instance
        registered_from: Metadata label for register_repo calls (for debugging)
    """
    if not repo_manager:
        return

    try:
        repo_name = project_data.get("project_name")
        branch = project_data.get("branch_name")
        commit_id = project_data.get("commit_id")
        repo_path = project_data.get("repo_path")

        if not repo_name:
            logger.warning(
                "Cannot ensure repo in repo manager: project_data has no project_name"
            )
            return

        # Check if repo is already available
        if repo_manager.is_repo_available(
            repo_name, branch=branch, commit_id=commit_id, user_id=user_id
        ):
            logger.debug(
                f"Repo {repo_name}@{commit_id or branch} already available in repo manager"
            )
            repo_manager.update_last_accessed(
                repo_name, branch=branch, commit_id=commit_id, user_id=user_id
            )
            return

        # Check if repo exists in repo manager's expected location but not registered
        try:
            expected_base_path = repo_manager._get_repo_local_path(repo_name)
        except Exception as e:
            logger.warning(f"Failed to get repo local path for {repo_name}: {e}")
            return

        # Check for worktree path (where repos are actually stored)
        ref = commit_id if commit_id else branch
        if ref:
            worktree_name = ref.replace("/", "_").replace("\\", "_")
            expected_worktree_path = expected_base_path / "worktrees" / worktree_name

            if expected_worktree_path.exists() and expected_worktree_path.is_dir():
                git_dir = expected_worktree_path / ".git"
                if git_dir.exists():
                    try:
                        repo_manager.register_repo(
                            repo_name=repo_name,
                            local_path=str(expected_worktree_path),
                            branch=branch,
                            commit_id=commit_id,
                            user_id=user_id,
                            metadata={"registered_from": registered_from},
                        )
                        logger.info(
                            f"Registered existing worktree {repo_name}@{ref} in repo manager from {registered_from}"
                        )
                        return
                    except Exception as e:
                        logger.warning(
                            f"Failed to register existing worktree {repo_name} in repo manager: {e}"
                        )

        # Check base repo path (for repos without worktrees)
        if expected_base_path.exists() and expected_base_path.is_dir():
            git_dir = expected_base_path / ".git"
            if git_dir.exists():
                try:
                    repo_manager.register_repo(
                        repo_name=repo_name,
                        local_path=str(expected_base_path),
                        branch=branch,
                        commit_id=commit_id,
                        user_id=user_id,
                        metadata={"registered_from": registered_from},
                    )
                    logger.info(
                        f"Registered existing base repo {repo_name} in repo manager from {registered_from}"
                    )
                    return
                except Exception as e:
                    logger.warning(
                        f"Failed to register existing base repo {repo_name} in repo manager: {e}"
                    )

        # For local repos (repo_path), check if it's a different location
        if repo_path and os.path.exists(repo_path):
            if not str(repo_path).startswith(str(repo_manager.repos_base_path)):
                logger.debug(
                    f"Repo {repo_name} has external path {repo_path}, not registering in repo manager. "
                    f"Repo manager base path: {repo_manager.repos_base_path}"
                )
            else:
                try:
                    repo_manager.register_repo(
                        repo_name=repo_name,
                        local_path=repo_path,
                        branch=branch,
                        commit_id=commit_id,
                        user_id=user_id,
                        metadata={"registered_from": registered_from},
                    )
                    logger.info(
                        f"Registered local repo {repo_name}@{commit_id or branch} in repo manager from {registered_from}"
                    )
                    return
                except Exception as e:
                    logger.warning(
                        f"Failed to register local repo {repo_name} in repo manager: {e}"
                    )

        # If we get here, repo doesn't exist in repo manager's directory structure
        expected_worktree_info = "N/A"
        if ref:
            worktree_name = ref.replace("/", "_").replace("\\", "_")
            expected_worktree_path = expected_base_path / "worktrees" / worktree_name
            expected_worktree_info = (
                f"{expected_worktree_path} (exists: {expected_worktree_path.exists()})"
            )

        logger.info(
            f"Repo {repo_name}@{commit_id or branch} not found in repo manager. "
            f"Project status: {project_data.get('status', 'N/A')}. "
            f"Repo manager base path: {repo_manager.repos_base_path}. "
            f"Expected base path: {expected_base_path} (exists: {expected_base_path.exists()}). "
            f"Expected worktree path: {expected_worktree_info}. "
            f"Project may need to be parsed first or repo manager may not be enabled during parsing."
        )

    except Exception as e:
        logger.warning(
            f"Error in ensure_repo_registered: {e}",
            exc_info=True,
        )
