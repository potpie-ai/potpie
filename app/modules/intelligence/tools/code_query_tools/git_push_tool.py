"""
Git Push Tool

Pushes the current branch to the remote repository.
"""

import os
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from app.modules.intelligence.tools.tool_schema import OnyxTool

from app.modules.projects.projects_service import ProjectService
from app.modules.repo_manager import RepoManager
from app.modules.repo_manager.sync_helper import (
    get_or_create_worktree_path,
    get_or_create_edits_worktree_path,
)
from app.modules.code_provider.git_safe import safe_git_repo_operation, GitOperationError
from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)


class GitPushInput(BaseModel):
    project_id: str = Field(
        ..., description="Project ID that references the repository"
    )
    remote: str = Field(
        default="origin",
        description="Remote name to push to (default: origin)"
    )
    branch: Optional[str] = Field(
        None,
        description="Optional branch name to push. If not provided, pushes the current branch.",
    )
    force: bool = Field(
        default=False,
        description="Whether to force push (use with caution)"
    )
    conversation_id: Optional[str] = Field(
        None,
        description="Optional conversation ID. When provided, pushes from the edits worktree branch (agent/edits-{conversation_id}) instead of the base project branch.",
    )


class GitPushTool:
    name: str = "git_push"
    description: str = """Push the current branch to the remote repository.

        This tool pushes commits from the local worktree to the remote repository.
        Use it after creating a commit with `git_commit` to make the changes
        available for PR creation.

        The tool will:
        1. Identify the current branch (or use specified branch)
        2. Push to the specified remote (default: origin)
        3. Return the push result and remote URL

        Args:
            project_id: The repository project ID (UUID)
            remote: The remote name to push to (default: origin)
            branch: Optional branch name. If omitted, uses current branch.
            force: Whether to force push (default: False)

        Returns:
            Dictionary with:
            - success: bool indicating success
            - branch: The branch that was pushed
            - remote: The remote that was pushed to
            - remote_url: The URL of the remote
            - message: Success/error message

        Example:
            {
                "project_id": "550e8400-e29b-41d4-a716-446655440000",
                "remote": "origin",
                "branch": "feature/new-auth",
                "force": false
            }
        """
    args_schema: type[BaseModel] = GitPushInput

    def __init__(self, sql_db: Session, user_id: str):
        self.sql_db = sql_db
        self.user_id = user_id
        self.project_service = ProjectService(sql_db)

        # Initialize repo manager if enabled
        self.repo_manager = None
        try:
            repo_manager_enabled = (
                os.getenv("REPO_MANAGER_ENABLED", "false").lower() == "true"
            )
            if repo_manager_enabled:
                self.repo_manager = RepoManager()
                logger.info("GitPushTool: RepoManager initialized")
        except Exception as e:
            logger.warning(f"GitPushTool: Failed to initialize RepoManager: {e}")

    def _get_project_details(self, project_id: str) -> Dict[str, str]:
        """Get project details and validate user access."""
        details = self.project_service.get_project_from_db_by_id_sync(project_id)  # type: ignore[arg-type]
        if not details or "project_name" not in details:
            raise ValueError(f"Cannot find repo details for project_id: {project_id}")
        if details["user_id"] != self.user_id:
            raise ValueError(
                f"Cannot find repo details for project_id: {project_id} for current user"
            )
        return details

    def _get_worktree_path(
        self,
        project_id: str,
        repo_name: str,
        branch: Optional[str],
        commit_id: Optional[str],
        user_id: Optional[str],
        conversation_id: Optional[str] = None,
    ) -> tuple[Optional[str], Optional[str]]:
        """Get the worktree path.

        When ``conversation_id`` is provided, uses the edits worktree for that
        conversation.  Otherwise falls back to the base repo worktree.
        """
        if conversation_id:
            return get_or_create_edits_worktree_path(
                self.repo_manager,
                project_id=project_id,
                conversation_id=conversation_id,
                user_id=user_id,
                sql_db=self.sql_db,
            )
        return get_or_create_worktree_path(
            self.repo_manager, repo_name, branch, commit_id, user_id, self.sql_db
        )

    def _get_auth_token(self, repo_name: str) -> Optional[str]:
        """Get authentication token using priority chain (same as sync_helper).

        Priority 1: GitHub App token
        Priority 2: User OAuth token from DB
        Priority 3: Environment token (via repo_manager)
        """
        # Priority 1: Try GitHub App token if available
        try:
            from app.modules.code_provider.provider_factory import (
                CodeProviderFactory,
            )

            provider = CodeProviderFactory.create_github_app_provider(repo_name)
            if (
                provider
                and hasattr(provider, "client")
                and hasattr(provider.client, "_Github__requester")
            ):
                requester = provider.client._Github__requester
                if hasattr(requester, "auth") and requester.auth:
                    token = requester.auth.token
                    logger.info(
                        f"GitPushTool: Using GitHub App token for {repo_name}"
                    )
                    return token
        except Exception:
            pass

        # Priority 2: Get User OAuth token from DB
        if self.sql_db and self.user_id:
            try:
                from app.modules.code_provider.github.github_service import (
                    GithubService,
                )

                token = GithubService(self.sql_db).get_github_oauth_token(
                    self.user_id
                )
                if token:
                    logger.info(
                        f"GitPushTool: Using user OAuth token for {repo_name}"
                    )
                    return token
            except Exception:
                pass

        # Priority 3: Environment token (via repo_manager)
        if self.repo_manager:
            token = self.repo_manager._get_github_token()
            if token:
                logger.info(
                    f"GitPushTool: Using environment token for {repo_name}"
                )
                return token

        logger.warning("GitPushTool: No authentication token found")
        return None

    def _run(
        self,
        project_id: str,
        remote: str = "origin",
        branch: Optional[str] = None,
        force: bool = False,
        conversation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Push the current branch to remote."""
        try:
            # Check if repo manager is available
            if not self.repo_manager:
                return {
                    "success": False,
                    "error": "Repo manager is not enabled. Git push requires a local worktree.",
                }

            # Get project details
            details = self._get_project_details(project_id)
            repo_name = details["project_name"]
            project_branch = details.get("branch_name")
            commit_id = details.get("commit_id")
            user_id = details.get("user_id")

            # Get worktree path (edits worktree when conversation_id provided)
            worktree_path, failure_reason = self._get_worktree_path(
                project_id, repo_name, project_branch, commit_id, user_id, conversation_id
            )
            if not worktree_path:
                error_msg = f"Worktree not found for project {project_id}. The repository must be parsed and available in the repo manager."
                if failure_reason:
                    reason_hint = {
                        "no_ref": "missing branch or commit_id",
                        "auth_failed": "all auth methods failed (GitHub App, OAuth, env token)",
                        "clone_failed": "git clone failed",
                        "worktree_add_failed": "git worktree add failed",
                    }.get(failure_reason, failure_reason)
                    error_msg += f" Reason: {reason_hint}."
                return {
                    "success": False,
                    "error": error_msg,
                }

            def _push_operation(repo):
                # Get current branch if not specified
                current_branch = branch
                if not current_branch:
                    try:
                        current_branch = repo.active_branch.name
                    except Exception as e:
                        return {
                            "success": False,
                            "error": f"Could not determine current branch: {str(e)}",
                        }

                # Get remote URL
                try:
                    remote_obj = repo.remote(remote)
                    remote_url = next(remote_obj.urls, "unknown")
                except Exception:
                    remote_url = "unknown"

                # Inject auth URL for push (App -> OAuth -> env); restore in finally
                auth_token = self._get_auth_token(repo_name)
                auth_url = None
                plain_url = None
                if auth_token and self.repo_manager:
                    try:
                        plain_url = (
                            repo.git.remote("get-url", remote).strip().rstrip("/")
                        )
                        auth_url = self.repo_manager._build_authenticated_url(
                            plain_url, auth_token, repo_name=repo_name
                        )
                        if auth_url:
                            auth_url = auth_url.rstrip("/")
                        if auth_url:
                            repo.git.remote("set-url", remote, auth_url)
                    except Exception as e:
                        logger.warning(f"GitPushTool: Could not set auth URL: {e}")

                try:
                    logger.info(
                        f"GitPushTool: Pushing {current_branch} to {remote} "
                        f"(force: {force})"
                    )

                    # Push to remote
                    push_args = [remote, current_branch]
                    if force:
                        push_args.append("--force")
                    else:
                        push_args.append("--set-upstream")

                    push_info = repo.git.push(*push_args)

                    return {
                        "success": True,
                        "branch": current_branch,
                        "remote": remote,
                        "remote_url": remote_url,
                        "message": f"Successfully pushed {current_branch} to {remote}",
                        "push_output": push_info,
                    }
                finally:
                    if auth_url and plain_url and self.repo_manager:
                        try:
                            repo.git.remote("set-url", remote, plain_url)
                        except Exception:
                            pass

            # Execute push with safe git operation wrapper
            result = safe_git_repo_operation(
                worktree_path,
                _push_operation,
                operation_name="git_push",
                timeout=60.0,  # Push can take longer for large repos
            )

            return result

        except GitOperationError as e:
            logger.error(f"GitPushTool: Git operation error: {e}")
            return {
                "success": False,
                "error": f"Git operation failed: {str(e)}",
            }
        except ValueError as e:
            logger.error(f"GitPushTool: Value error: {e}")
            return {
                "success": False,
                "error": str(e),
            }
        except Exception as e:
            logger.exception("GitPushTool: Unexpected error")
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}",
            }

    async def _arun(
        self,
        project_id: str,
        remote: str = "origin",
        branch: Optional[str] = None,
        force: bool = False,
        conversation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Async wrapper for _run."""
        import asyncio

        return await asyncio.to_thread(
            self._run, project_id, remote, branch, force, conversation_id
        )


def git_push_tool(sql_db: Session, user_id: str) -> Optional[OnyxTool]:
    """
    Create git push tool if repo manager is enabled.

    Returns None if repo manager is not enabled.
    """
    repo_manager_enabled = os.getenv("REPO_MANAGER_ENABLED", "false").lower() == "true"
    if not repo_manager_enabled:
        logger.debug("GitPushTool not created: REPO_MANAGER_ENABLED is false")
        return None

    tool_instance = GitPushTool(sql_db, user_id)
    if not tool_instance.repo_manager:
        logger.debug("GitPushTool not created: RepoManager initialization failed")
        return None

    return OnyxTool.from_function(
        coroutine=tool_instance._arun,
        func=tool_instance._run,
        name="git_push",
        description=tool_instance.description,
        args_schema=GitPushInput,
    )
