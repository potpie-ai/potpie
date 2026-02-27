"""
Shared helper for ensuring repositories are registered in the repo manager.
Used by parsing and conversation flows.
"""

import asyncio
import os
import traceback
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

from app.modules.utils.email_helper import EmailHelper
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
) -> Tuple[Optional[str], Optional[str]]:
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
        Tuple of (worktree_path, failure_reason). On success: (path, None).
        On failure: (None, reason) where reason describes the failure (e.g. "no_ref",
        "auth_failed", "clone_failed", "worktree_add_failed").
    """
    if not repo_manager:
        return None, "repo_manager_not_initialized"

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
        return worktree_path, None

    # Worktree not found — try to create it via prepare_for_parsing
    ref = commit_id if commit_id else branch
    if not ref:
        logger.warning(
            f"[Repomanager] Cannot create worktree for {repo_name}: no ref (branch or commit_id)",
            repo_name=repo_name,
            user_id=user_id,
        )
        return None, "no_ref"

    # ============================================================================
    # AUTHENTICATION CHAIN: GitHub App -> User OAuth -> Environment Tokens
    # ============================================================================

    # PRIORITY 1: Try GitHub App token if available
    github_app_token = None
    try:
        from app.modules.code_provider.provider_factory import CodeProviderFactory

        provider = CodeProviderFactory.create_github_app_provider(repo_name)
        if provider and hasattr(provider, "client") and hasattr(provider.client, "_Github__requester"):
            requester = provider.client._Github__requester
            if hasattr(requester, "auth") and requester.auth:
                github_app_token = requester.auth.token
                logger.info(
                    f"[Repomanager] SyncHelper: Got GitHub App token for {repo_name}",
                    repo_name=repo_name,
                    user_id=user_id,
                    token_type="github_app",
                )
    except Exception as e:
        logger.info(
            f"[Repomanager] SyncHelper: GitHub App token not available for {repo_name}",
            repo_name=repo_name,
            user_id=user_id,
            reason=str(e),
        )

    # PRIORITY 2: Get User OAuth token from DB
    user_oauth_token = None
    if sql_db and user_id:
        try:
            from app.modules.code_provider.github.github_service import GithubService

            user_oauth_token = GithubService(sql_db).get_github_oauth_token(user_id)
            if user_oauth_token:
                logger.info(
                    f"[Repomanager] SyncHelper: Got user OAuth token for {repo_name}",
                    repo_name=repo_name,
                    user_id=user_id,
                    token_type="user_oauth",
                )
        except Exception as e:
            logger.info(
                f"[Repomanager] SyncHelper: Could not get user OAuth token for {user_id}",
                repo_name=repo_name,
                user_id=user_id,
                reason=str(e),
            )

    # Try authentication methods in priority order
    last_error = None

    # Try GitHub App token first
    if github_app_token:
        try:
            logger.info(
                f"[Repomanager] SyncHelper: Attempting Priority 1 - GitHub App token",
                repo_name=repo_name,
                user_id=user_id,
                ref=ref,
            )
            worktree_path_str = repo_manager.prepare_for_parsing(
                repo_name,
                ref,
                auth_token=github_app_token,
                is_commit=bool(commit_id),
                user_id=user_id,
            )
            if worktree_path_str and os.path.exists(worktree_path_str):
                logger.info(
                    f"[Repomanager] SyncHelper: SUCCESS with GitHub App token",
                    repo_name=repo_name,
                    user_id=user_id,
                    ref=ref,
                    worktree_path=worktree_path_str,
                    method="github_app_token",
                )
                return worktree_path_str, None
        except Exception as e:
            last_error = e
            logger.warning(
                f"[Repomanager] SyncHelper: GitHub App token failed, trying next",
                repo_name=repo_name,
                user_id=user_id,
                ref=ref,
                error=str(e),
            )

    # Try User OAuth token second
    if user_oauth_token:
        try:
            logger.info(
                f"[Repomanager] SyncHelper: Attempting Priority 2 - User OAuth token",
                repo_name=repo_name,
                user_id=user_id,
                ref=ref,
            )
            worktree_path_str = repo_manager.prepare_for_parsing(
                repo_name,
                ref,
                auth_token=user_oauth_token,
                is_commit=bool(commit_id),
                user_id=user_id,
            )
            if worktree_path_str and os.path.exists(worktree_path_str):
                logger.info(
                    f"[Repomanager] SyncHelper: SUCCESS with User OAuth token",
                    repo_name=repo_name,
                    user_id=user_id,
                    ref=ref,
                    worktree_path=worktree_path_str,
                    method="user_oauth_token",
                )
                return worktree_path_str, None
        except Exception as e:
            last_error = e
            logger.warning(
                f"[Repomanager] SyncHelper: User OAuth token failed, trying next",
                repo_name=repo_name,
                user_id=user_id,
                ref=ref,
                error=str(e),
            )

    # PRIORITY 3: Try environment token (will be fetched internally by prepare_for_parsing)
    try:
        logger.info(
            f"[Repomanager] SyncHelper: Attempting Priority 3 - Environment token",
            repo_name=repo_name,
            user_id=user_id,
            ref=ref,
        )
        worktree_path_str = repo_manager.prepare_for_parsing(
            repo_name,
            ref,
            auth_token=None,  # Will use environment token
            is_commit=bool(commit_id),
            user_id=user_id,
        )
        if worktree_path_str and os.path.exists(worktree_path_str):
            logger.info(
                f"[Repomanager] SyncHelper: SUCCESS with Environment token",
                repo_name=repo_name,
                user_id=user_id,
                ref=ref,
                worktree_path=worktree_path_str,
                method="environment_token",
            )
            return worktree_path_str, None
    except Exception as e:
        last_error = e
        logger.error(
            f"[Repomanager] SyncHelper: All authentication methods failed for {repo_name}@{ref}",
            repo_name=repo_name,
            user_id=user_id,
            ref=ref,
            error=str(e),
            suggestion="Check GitHub App installation, user OAuth scopes, and environment tokens",
        )

        # Send email alert for final auth failure (run to completion like _clone_to_repo_manager)
        try:
            asyncio.run(
                EmailHelper().send_parsing_failure_alert(
                    repo_name=repo_name,
                    branch_name=ref,
                    error_message=f"All authentication methods failed: {str(e)}",
                    auth_method="environment",
                    failure_type="cloning_auth",
                    user_id=user_id,
                    project_id=None,
                    stack_trace=traceback.format_exc(),
                )
            )
        except Exception as email_err:
            logger.exception(f"[Repomanager] Failed to send failure email: {email_err}")

    # Determine failure reason for richer error propagation
    failure_reason = "auth_failed"
    if last_error:
        err_str = str(last_error).lower()
        if "clone" in err_str or "git clone" in err_str:
            failure_reason = "clone_failed"
        elif "worktree" in err_str or "work tree" in err_str:
            failure_reason = "worktree_add_failed"
    return None, failure_reason


def get_or_create_edits_worktree_path(
    repo_manager: "RepoManager",
    project_id: str,
    conversation_id: str,
    user_id: Optional[str],
    sql_db: Optional["Session"] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Get or create a dedicated edits worktree for the given conversation.

    Creates a new branch ``agent/edits-{conversation_id}`` from the project's
    base branch and checks out a worktree on it.  Subsequent calls for the
    same conversation reuse the existing worktree (idempotent).

    Args:
        repo_manager: RepoManager instance
        project_id: Project ID used to look up repo details
        conversation_id: Conversation ID used to derive the unique branch name
        user_id: User ID for auth and multi-tenant isolation
        sql_db: DB session for project lookup and OAuth token retrieval

    Returns:
        Tuple of (worktree_path, failure_reason).  On success: (path, None).
        On failure: (None, reason).
    """
    if not repo_manager:
        return None, "repo_manager_not_initialized"

    if not project_id:
        return None, "project_id_required"

    if not conversation_id:
        return None, "conversation_id_required"

    # Sanitize conversation_id for use as a git branch name component
    sanitized_conv_id = (
        conversation_id.replace("/", "-").replace("\\", "-").replace(" ", "-")
    )
    new_branch_name = f"agent/edits-{sanitized_conv_id}"

    # Resolve project details (repo_name, base_branch)
    try:
        from app.modules.projects.projects_service import ProjectService

        project_service = ProjectService(sql_db)
        details = project_service.get_project_from_db_by_id_sync(project_id)  # type: ignore[arg-type]
        if not details or "project_name" not in details:
            return None, "project_not_found"

        repo_name = details["project_name"]
        base_branch = details.get("branch_name") or "main"
        project_user_id = details.get("user_id")
        effective_user_id = user_id or project_user_id
    except Exception as e:
        logger.warning(
            f"[Repomanager] get_or_create_edits_worktree_path: Failed to get project {project_id}: {e}",
            project_id=project_id,
            user_id=user_id,
        )
        return None, "project_lookup_failed"

    if not repo_name:
        return None, "no_repo_name"

    if not effective_user_id:
        return None, "no_user_id"

    # Fast path: return existing edits worktree
    existing = repo_manager.get_repo_path(
        repo_name, branch=new_branch_name, user_id=effective_user_id
    )
    if existing and os.path.exists(existing):
        logger.info(
            f"[Repomanager] Reusing existing edits worktree for {repo_name}:{new_branch_name}",
            repo_name=repo_name,
            user_id=effective_user_id,
        )
        return existing, None

    # ============================================================================
    # AUTHENTICATION CHAIN: GitHub App -> User OAuth -> Environment Tokens
    # ============================================================================

    github_app_token = None
    try:
        from app.modules.code_provider.provider_factory import CodeProviderFactory

        provider = CodeProviderFactory.create_github_app_provider(repo_name)
        if provider and hasattr(provider, "client") and hasattr(provider.client, "_Github__requester"):
            requester = provider.client._Github__requester
            if hasattr(requester, "auth") and requester.auth:
                github_app_token = requester.auth.token
    except Exception as e:
        logger.info(
            f"[Repomanager] GitHub App token not available for edits worktree: {e}",
            repo_name=repo_name,
            user_id=effective_user_id,
        )

    user_oauth_token = None
    if sql_db and effective_user_id:
        try:
            from app.modules.code_provider.github.github_service import GithubService

            user_oauth_token = GithubService(sql_db).get_github_oauth_token(effective_user_id)
        except Exception as e:
            logger.info(
                f"[Repomanager] Could not get OAuth token for edits worktree: {e}",
                repo_name=repo_name,
                user_id=effective_user_id,
            )

    last_error = None

    for auth_token, method_name in [
        (github_app_token, "github_app_token"),
        (user_oauth_token, "user_oauth_token"),
        (None, "environment_token"),
    ]:
        if auth_token is None and method_name != "environment_token":
            continue

        try:
            logger.info(
                f"[Repomanager] Creating edits worktree with {method_name} for {repo_name}:{new_branch_name}",
                repo_name=repo_name,
                user_id=effective_user_id,
                branch=new_branch_name,
            )
            worktree_path = repo_manager.create_worktree_with_new_branch(
                repo_name=repo_name,
                base_ref=base_branch,
                new_branch_name=new_branch_name,
                auth_token=auth_token,
                user_id=effective_user_id,
                unique_id=conversation_id,
                exists_ok=True,
            )
            worktree_path_str = str(worktree_path)
            if worktree_path_str and os.path.exists(worktree_path_str):
                logger.info(
                    f"[Repomanager] SUCCESS: edits worktree at {worktree_path_str}",
                    repo_name=repo_name,
                    user_id=effective_user_id,
                    branch=new_branch_name,
                    method=method_name,
                )
                return worktree_path_str, None
        except Exception as e:
            last_error = e
            logger.warning(
                f"[Repomanager] {method_name} failed for edits worktree: {e}",
                repo_name=repo_name,
                user_id=effective_user_id,
            )

    failure_reason = "auth_failed"
    if last_error:
        err_str = str(last_error).lower()
        if "clone" in err_str or "git clone" in err_str:
            failure_reason = "clone_failed"
        elif "worktree" in err_str or "work tree" in err_str:
            failure_reason = "worktree_add_failed"

    return None, failure_reason


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
                "[Repomanager] Cannot ensure repo: project_data has no project_name",
                user_id=user_id,
            )
            return

        # Check if repo is already available
        if repo_manager.is_repo_available(
            repo_name, branch=branch, commit_id=commit_id, user_id=user_id
        ):
            logger.info(
                f"[Repomanager] Repo {repo_name}@{commit_id or branch} already available",
                repo_name=repo_name,
                user_id=user_id,
            )
            repo_manager.update_last_accessed(
                repo_name, branch=branch, commit_id=commit_id, user_id=user_id
            )
            return

        # Check if repo exists in repo manager's expected location but not registered
        try:
            expected_base_path = repo_manager._get_repo_local_path(repo_name)
        except Exception as e:
            logger.warning(
                f"[Repomanager] Failed to get repo local path for {repo_name}: {e}",
                repo_name=repo_name,
                user_id=user_id,
            )
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
                            f"[Repomanager] Registered existing worktree {repo_name}@{ref} from {registered_from}",
                            repo_name=repo_name,
                            user_id=user_id,
                        )
                        return
                    except Exception as e:
                        logger.warning(
                            f"[Repomanager] Failed to register existing worktree {repo_name}: {e}",
                            repo_name=repo_name,
                            user_id=user_id,
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
                        f"[Repomanager] Registered existing base repo {repo_name} from {registered_from}",
                        repo_name=repo_name,
                        user_id=user_id,
                    )
                    return
                except Exception as e:
                    logger.warning(
                        f"[Repomanager] Failed to register existing base repo {repo_name}: {e}",
                        repo_name=repo_name,
                        user_id=user_id,
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
                        f"[Repomanager] Registered local repo {repo_name}@{commit_id or branch} from {registered_from}",
                        repo_name=repo_name,
                        user_id=user_id,
                    )
                    return
                except Exception as e:
                    logger.warning(
                        f"[Repomanager] Failed to register local repo {repo_name}: {e}",
                        repo_name=repo_name,
                        user_id=user_id,
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
            f"[Repomanager] Repo {repo_name}@{commit_id or branch} not found. "
            f"Project status: {project_data.get('status', 'N/A')}. "
            f"Expected base: {expected_base_path} (exists: {expected_base_path.exists()}). "
            f"Expected worktree: {expected_worktree_info}.",
            repo_name=repo_name,
            user_id=user_id,
        )

    except Exception as e:
        logger.warning(
            f"[Repomanager] Error in ensure_repo_registered: {e}",
            repo_name=repo_name,
            user_id=user_id,
            exc_info=True,
        )
