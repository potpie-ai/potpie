"""
Checkout Worktree Branch Tool

Ensures the edits worktree branch is checked out and ready for direct file modifications.
This is the entry point for the new agent flow that writes changes directly to the worktree.
"""

import os
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from app.modules.intelligence.tools.tool_schema import OnyxTool

from app.modules.projects.projects_service import ProjectService
from app.modules.repo_manager import RepoManager
from app.modules.repo_manager.sync_helper import get_or_create_edits_worktree_path
from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)


class CheckoutWorktreeBranchInput(BaseModel):
    """Input for checking out the edits worktree branch."""

    project_id: str = Field(
        ..., description="Project ID that references the repository"
    )
    conversation_id: str = Field(
        ..., description="Conversation ID used to derive the branch name"
    )
    base_branch: str = Field(
        default="main",
        description="Base branch to create the edits branch from (default: main)",
    )


class CheckoutWorktreeBranchTool:
    """
    Tool to ensure the edits worktree branch is checked out and ready.

    This tool:
    1. Creates or reuses the edits worktree for the conversation
    2. Creates the branch 'agent/edits-{conversation_id}' from the base branch
    3. Checks out the branch in the worktree
    4. Returns the worktree path for direct file operations

    Use this at the start of the code generation flow to prepare the worktree
    for direct file writes and commits.
    """

    name: str = "checkout_worktree_branch"
    description: str = """Checkout the edits worktree branch for direct file modifications.

This tool prepares the worktree for the new agent flow by:
1. Creating or reusing the edits worktree for this conversation
2. Creating the branch 'agent/edits-{conversation_id}' from the base branch
3. Checking out the branch in the worktree

Use this at the START of code generation to enable direct worktree writes.
After calling this, you can:
- Use code_changes_manager to write files (they'll go directly to worktree)
- Use git_commit to commit changes
- Use bash_command to run tests in the worktree

Args:
    project_id: The repository project ID (UUID)
    conversation_id: The conversation ID (used to name the branch)
    base_branch: Base branch to create from (default: 'main')

Returns:
    Dictionary with:
    - success: bool indicating success
    - worktree_path: Path to the worktree for direct file operations
    - branch_name: The name of the created/checked out branch
    - error: Error message if failed

Example:
    {
        "project_id": "550e8400-e29b-41d4-a716-446655440000",
        "conversation_id": "conv-123-456",
        "base_branch": "main"
    }
"""
    args_schema: type[BaseModel] = CheckoutWorktreeBranchInput

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
                logger.info("CheckoutWorktreeBranchTool: RepoManager initialized")
        except Exception as e:
            logger.warning(
                f"CheckoutWorktreeBranchTool: Failed to initialize RepoManager: {e}"
            )

    def _get_project_details(self, project_id: str) -> Dict[str, str]:
        """Get project details and validate user access."""
        details = self.project_service.get_project_from_db_by_id_sync(
            project_id  # type: ignore[arg-type]
        )
        if not details or "project_name" not in details:
            raise ValueError(f"Cannot find repo details for project_id: {project_id}")
        if details["user_id"] != self.user_id:
            raise ValueError(
                f"Cannot find repo details for project_id: {project_id} for current user"
            )
        return details

    def _run(
        self,
        project_id: str,
        conversation_id: str,
        base_branch: str = "main",
    ) -> Dict[str, Any]:
        """Ensure the edits worktree branch is checked out."""
        try:
            # Validate repo manager
            if not self.repo_manager:
                return {
                    "success": False,
                    "error": "Repo manager not enabled. Set REPO_MANAGER_ENABLED=true",
                }

            # Get project details
            details = self._get_project_details(project_id)
            user_id = details.get("user_id")

            # Get or create edits worktree
            worktree_path, failure_reason = get_or_create_edits_worktree_path(
                self.repo_manager,
                project_id=project_id,
                conversation_id=conversation_id,
                user_id=user_id,
                sql_db=self.sql_db,
            )

            if not worktree_path:
                error_msg = "Failed to create edits worktree."
                if failure_reason:
                    reason_hint = {
                        "no_ref": "missing branch or commit_id",
                        "auth_failed": "all auth methods failed (GitHub App, OAuth, env token)",
                        "clone_failed": "git clone failed",
                        "worktree_add_failed": "git worktree add failed",
                        "project_not_found": "project not found in database",
                        "no_repo_name": "project has no repository name",
                        "no_user_id": "no user ID available",
                    }.get(failure_reason, failure_reason)
                    error_msg += f" Reason: {reason_hint}."
                return {
                    "success": False,
                    "error": error_msg,
                }

            # Determine branch name
            sanitized_conv_id = (
                conversation_id.replace("/", "-").replace("\\", "-").replace(" ", "-")
            )
            branch_name = f"agent/edits-{sanitized_conv_id}"

            # Check if branch is already checked out
            from app.modules.code_provider.git_safe import (
                safe_git_repo_operation,
                GitOperationError,
            )

            def _check_branch(repo):
                try:
                    current_branch = repo.active_branch.name
                    return current_branch
                except TypeError:
                    # Detached HEAD
                    return None

            current_branch = safe_git_repo_operation(
                worktree_path,
                _check_branch,
                operation_name="check_branch",
                timeout=5.0,
            )

            if current_branch == branch_name:
                logger.info(
                    f"CheckoutWorktreeBranchTool: Already on branch {branch_name}"
                )
                return {
                    "success": True,
                    "worktree_path": worktree_path,
                    "branch_name": branch_name,
                    "message": f"Already on branch {branch_name}",
                }

            # Checkout the branch (create if needed)
            def _checkout_branch(repo):
                existing_branch = next(
                    (b for b in repo.branches if b.name == branch_name), None
                )
                if existing_branch is not None:
                    try:
                        if repo.active_branch.name != branch_name:
                            existing_branch.checkout()
                            logger.info(
                                f"CheckoutWorktreeBranchTool: Checked out existing branch {branch_name}"
                            )
                        else:
                            logger.info(
                                f"CheckoutWorktreeBranchTool: Already on branch {branch_name}"
                            )
                    except TypeError:
                        # Detached HEAD
                        existing_branch.checkout()
                        logger.info(
                            f"CheckoutWorktreeBranchTool: Checked out branch {branch_name} from detached HEAD"
                        )
                else:
                    logger.info(
                        f"CheckoutWorktreeBranchTool: Creating and checking out new branch {branch_name}"
                    )
                    new_branch = repo.create_head(branch_name)
                    new_branch.checkout()

                return {
                    "success": True,
                    "branch": branch_name,
                }

            checkout_result = safe_git_repo_operation(
                worktree_path,
                _checkout_branch,
                operation_name="checkout_branch",
                timeout=10.0,
            )

            if not checkout_result.get("success"):
                return {
                    "success": False,
                    "error": checkout_result.get("error", "Failed to checkout branch"),
                }

            logger.info(
                f"CheckoutWorktreeBranchTool: Successfully prepared worktree at {worktree_path} "
                f"on branch {branch_name}"
            )

            return {
                "success": True,
                "worktree_path": worktree_path,
                "branch_name": branch_name,
                "message": f"Checked out branch {branch_name} in worktree",
            }

        except GitOperationError as e:
            logger.error(f"CheckoutWorktreeBranchTool: Git operation error: {e}")
            return {
                "success": False,
                "error": f"Git operation failed: {str(e)}",
            }
        except ValueError as e:
            logger.error(f"CheckoutWorktreeBranchTool: Value error: {e}")
            return {"success": False, "error": str(e)}
        except Exception as e:
            logger.exception("CheckoutWorktreeBranchTool: Unexpected error")
            return {"success": False, "error": f"Unexpected error: {str(e)}"}

    async def _arun(
        self,
        project_id: str,
        conversation_id: str,
        base_branch: str = "main",
    ) -> Dict[str, Any]:
        """Async wrapper for _run."""
        import asyncio

        return await asyncio.to_thread(
            self._run, project_id, conversation_id, base_branch
        )


def checkout_worktree_branch_tool(
    sql_db: Session, user_id: str
) -> Optional[OnyxTool]:
    """
    Create the checkout worktree branch tool if repo manager is enabled.

    Returns None if repo manager is not enabled.
    """
    repo_manager_enabled = (
        os.getenv("REPO_MANAGER_ENABLED", "false").lower() == "true"
    )
    if not repo_manager_enabled:
        logger.debug(
            "CheckoutWorktreeBranchTool not created: REPO_MANAGER_ENABLED is false"
        )
        return None

    tool_instance = CheckoutWorktreeBranchTool(sql_db, user_id)
    if not tool_instance.repo_manager:
        logger.debug(
            "CheckoutWorktreeBranchTool not created: RepoManager initialization failed"
        )
        return None

    return OnyxTool.from_function(
        coroutine=tool_instance._arun,
        func=tool_instance._run,
        name="checkout_worktree_branch",
        description=tool_instance.description,
        args_schema=CheckoutWorktreeBranchInput,
    )
