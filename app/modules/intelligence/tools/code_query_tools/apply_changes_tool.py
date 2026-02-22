"""
Apply Changes Tool

Applies code changes from CodeChangesManager (Redis) to the actual worktree filesystem.
This bridges the gap between stored changes and the git repository.
"""

import os
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from langchain_core.tools import StructuredTool

from app.modules.projects.projects_service import ProjectService
from app.modules.repo_manager import RepoManager
from app.modules.intelligence.tools.code_changes_manager import CodeChangesManager
from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)


class ApplyChangesInput(BaseModel):
    project_id: str = Field(
        ..., description="Project ID that references the repository"
    )
    conversation_id: str = Field(
        ..., description="Conversation ID where changes are stored in Redis"
    )
    file_path: Optional[str] = Field(
        None,
        description="Optional specific file to apply. If not provided, applies all changes.",
    )


class ApplyChangesTool:
    name: str = "apply_changes"
    description: str = """Apply code changes from CodeChangesManager to the worktree filesystem.

        This tool writes changes stored in Redis (from CodeChangesManager) to the actual
        repository worktree files. This is the bridge between the agent's change tracking
        system and the git repository.

        Use this tool when you need to:
        - Export agent-generated code changes to the actual filesystem
        - Prepare changes for git commit and PR creation
        - Apply modifications before running tests or verification

        The tool will:
        1. Fetch changes from Redis for the given conversation_id
        2. Write each changed file to the worktree filesystem
        3. Create parent directories if needed
        4. Return a summary of files applied

        Args:
            project_id: The repository project ID (UUID)
            conversation_id: The conversation ID where changes are stored
            file_path: Optional specific file to apply. If omitted, applies all changes.

        Returns:
            Dictionary with:
            - success: bool indicating success
            - files_applied: List of files that were written
            - files_skipped: List of files that were skipped (e.g., deleted files)
            - error: Error message if failed

        Example:
            {
                "project_id": "550e8400-e29b-41d4-a716-446655440000",
                "conversation_id": "conv-123-456",
                "file_path": null  # Apply all changes
            }
        """
    args_schema: type[BaseModel] = ApplyChangesInput

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
                logger.info("ApplyChangesTool: RepoManager initialized")
        except Exception as e:
            logger.warning(f"ApplyChangesTool: Failed to initialize RepoManager: {e}")

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
        repo_name: str,
        branch: Optional[str],
        commit_id: Optional[str],
        user_id: Optional[str],
    ) -> Optional[str]:
        """Get the worktree path for the project."""
        if not self.repo_manager:
            return None

        # Try to get worktree path with user_id for security
        worktree_path = self.repo_manager.get_repo_path(
            repo_name, branch=branch, commit_id=commit_id, user_id=user_id
        )
        if worktree_path and os.path.exists(worktree_path):
            return worktree_path

        # Try with just commit_id (with user_id)
        if commit_id:
            worktree_path = self.repo_manager.get_repo_path(
                repo_name, commit_id=commit_id, user_id=user_id
            )
            if worktree_path and os.path.exists(worktree_path):
                return worktree_path

        # Try with just branch (with user_id)
        if branch:
            worktree_path = self.repo_manager.get_repo_path(
                repo_name, branch=branch, user_id=user_id
            )
            if worktree_path and os.path.exists(worktree_path):
                return worktree_path

        return None

    def _run(
        self,
        project_id: str,
        conversation_id: str,
        file_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Apply changes from CodeChangesManager to the worktree."""
        try:
            # Check if repo manager is available
            if not self.repo_manager:
                return {
                    "success": False,
                    "error": "Repo manager is not enabled. Apply changes requires a local worktree.",
                    "files_applied": [],
                    "files_skipped": [],
                }

            # Get project details
            details = self._get_project_details(project_id)
            repo_name = details["project_name"]
            branch = details.get("branch_name")
            commit_id = details.get("commit_id")
            user_id = details.get("user_id")

            # Get worktree path
            worktree_path = self._get_worktree_path(
                repo_name, branch, commit_id, user_id
            )
            if not worktree_path:
                return {
                    "success": False,
                    "error": f"Worktree not found for project {project_id}. The repository must be parsed and available in the repo manager.",
                    "files_applied": [],
                    "files_skipped": [],
                }

            # Initialize CodeChangesManager with conversation_id
            changes_manager = CodeChangesManager(conversation_id=conversation_id)

            # Get all changes
            all_changes = changes_manager.changes

            if not all_changes:
                return {
                    "success": True,
                    "message": "No changes found to apply",
                    "files_applied": [],
                    "files_skipped": [],
                }

            # Filter to specific file if requested
            if file_path:
                if file_path not in all_changes:
                    return {
                        "success": False,
                        "error": f"File '{file_path}' not found in changes for conversation {conversation_id}",
                        "files_applied": [],
                        "files_skipped": [],
                    }
                files_to_apply = {file_path: all_changes[file_path]}
            else:
                files_to_apply = all_changes

            files_applied: List[str] = []
            files_skipped: List[str] = []

            for fpath, change in files_to_apply.items():
                # Skip deleted files (they don't have content to write)
                if change.change_type.value == "delete":
                    files_skipped.append(f"{fpath} (deleted)")
                    continue

                if change.content is None:
                    files_skipped.append(f"{fpath} (no content)")
                    continue

                # Construct full path in worktree
                full_path = os.path.join(worktree_path, fpath)

                # Ensure parent directory exists
                parent_dir = os.path.dirname(full_path)
                if parent_dir and not os.path.exists(parent_dir):
                    os.makedirs(parent_dir, exist_ok=True)
                    logger.info(f"ApplyChangesTool: Created directory {parent_dir}")

                # Write file content
                try:
                    with open(full_path, "w", encoding="utf-8") as f:
                        f.write(change.content)
                    files_applied.append(fpath)
                    logger.info(f"ApplyChangesTool: Applied changes to {fpath}")
                except Exception as e:
                    logger.error(f"ApplyChangesTool: Failed to write {fpath}: {e}")
                    files_skipped.append(f"{fpath} (write error: {str(e)})")

            return {
                "success": len(files_applied) > 0 or len(files_skipped) == 0,
                "files_applied": files_applied,
                "files_skipped": files_skipped,
                "worktree_path": worktree_path,
            }

        except ValueError as e:
            logger.error(f"ApplyChangesTool: Value error: {e}")
            return {
                "success": False,
                "error": str(e),
                "files_applied": [],
                "files_skipped": [],
            }
        except Exception as e:
            logger.exception("ApplyChangesTool: Unexpected error")
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}",
                "files_applied": [],
                "files_skipped": [],
            }

    async def _arun(
        self,
        project_id: str,
        conversation_id: str,
        file_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Async wrapper for _run."""
        import asyncio

        return await asyncio.to_thread(
            self._run, project_id, conversation_id, file_path
        )


def apply_changes_tool(sql_db: Session, user_id: str) -> Optional[StructuredTool]:
    """
    Create apply changes tool if repo manager is enabled.

    Returns None if repo manager is not enabled.
    """
    repo_manager_enabled = os.getenv("REPO_MANAGER_ENABLED", "false").lower() == "true"
    if not repo_manager_enabled:
        logger.debug("ApplyChangesTool not created: REPO_MANAGER_ENABLED is false")
        return None

    tool_instance = ApplyChangesTool(sql_db, user_id)
    if not tool_instance.repo_manager:
        logger.debug("ApplyChangesTool not created: RepoManager initialization failed")
        return None

    return StructuredTool.from_function(
        coroutine=tool_instance._arun,
        func=tool_instance._run,
        name="apply_changes",
        description=tool_instance.description,
        args_schema=ApplyChangesInput,
    )
