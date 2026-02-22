"""
Git Commit Tool

Stages and commits changes in the repository worktree.
"""

import os
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from langchain_core.tools import StructuredTool

from app.modules.projects.projects_service import ProjectService
from app.modules.repo_manager import RepoManager
from app.modules.code_provider.git_safe import safe_git_repo_operation, GitOperationError
from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)


class GitCommitInput(BaseModel):
    project_id: str = Field(
        ..., description="Project ID that references the repository"
    )
    commit_message: str = Field(
        ..., description="Commit message describing the changes"
    )
    files: Optional[List[str]] = Field(
        None,
        description="Optional list of specific files to commit. If not provided, commits all staged changes.",
    )


class GitCommitTool:
    name: str = "git_commit"
    description: str = """Stage and commit changes in the repository worktree.

        This tool stages modified files and creates a git commit in the worktree.
        Use it after applying changes with `apply_changes` to create a commit
        that can then be pushed and used for a PR.

        The tool will:
        1. Stage the specified files (or all changes if no files specified)
        2. Create a git commit with the provided message
        3. Return the commit hash

        Args:
            project_id: The repository project ID (UUID)
            commit_message: The commit message (should be descriptive)
            files: Optional list of specific file paths to commit. If omitted, commits all changes.

        Returns:
            Dictionary with:
            - success: bool indicating success
            - commit_hash: The created commit hash (if successful)
            - files_committed: List of files that were committed
            - error: Error message if failed

        Example:
            {
                "project_id": "550e8400-e29b-41d4-a716-446655440000",
                "commit_message": "Add new feature for user authentication",
                "files": null  # Commit all changes
            }
        """
    args_schema: type[BaseModel] = GitCommitInput

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
                logger.info("GitCommitTool: RepoManager initialized")
        except Exception as e:
            logger.warning(f"GitCommitTool: Failed to initialize RepoManager: {e}")

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
        commit_message: str,
        files: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Stage and commit changes in the worktree."""
        try:
            # Check if repo manager is available
            if not self.repo_manager:
                return {
                    "success": False,
                    "error": "Repo manager is not enabled. Git commit requires a local worktree.",
                }

            # Validate commit message
            if not commit_message or not commit_message.strip():
                return {
                    "success": False,
                    "error": "Commit message cannot be empty",
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
                }

            logger.info(
                f"GitCommitTool: Committing changes in {worktree_path} "
                f"(files: {files if files else 'all'})"
            )

            def _commit_operation(repo):
                # Stage files
                if files:
                    # Stage specific files
                    for file_path in files:
                        full_path = os.path.join(worktree_path, file_path)
                        if os.path.exists(full_path):
                            repo.git.add(file_path)
                        else:
                            logger.warning(
                                f"GitCommitTool: File not found for staging: {file_path}"
                            )
                else:
                    # Stage all changes
                    repo.git.add(".")

                # Check if there are changes to commit
                diff = repo.git.diff("--cached", "--name-only")
                if not diff.strip():
                    return {
                        "success": False,
                        "error": "No changes to commit. Stage files first using apply_changes.",
                    }

                staged_files = [f.strip() for f in diff.strip().split("\n") if f.strip()]

                # Create commit
                commit = repo.index.commit(commit_message)
                commit_hash = commit.hexsha

                return {
                    "success": True,
                    "commit_hash": commit_hash,
                    "files_committed": staged_files,
                    "message": f"Created commit {commit_hash[:8]}: {commit_message}",
                }

            # Execute commit with safe git operation wrapper
            result = safe_git_repo_operation(
                worktree_path,
                _commit_operation,
                operation_name="git_commit",
                timeout=30.0,
            )

            return result

        except GitOperationError as e:
            logger.error(f"GitCommitTool: Git operation error: {e}")
            return {
                "success": False,
                "error": f"Git operation failed: {str(e)}",
            }
        except ValueError as e:
            logger.error(f"GitCommitTool: Value error: {e}")
            return {
                "success": False,
                "error": str(e),
            }
        except Exception as e:
            logger.exception("GitCommitTool: Unexpected error")
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}",
            }

    async def _arun(
        self,
        project_id: str,
        commit_message: str,
        files: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Async wrapper for _run."""
        import asyncio

        return await asyncio.to_thread(
            self._run, project_id, commit_message, files
        )


def git_commit_tool(sql_db: Session, user_id: str) -> Optional[StructuredTool]:
    """
    Create git commit tool if repo manager is enabled.

    Returns None if repo manager is not enabled.
    """
    repo_manager_enabled = os.getenv("REPO_MANAGER_ENABLED", "false").lower() == "true"
    if not repo_manager_enabled:
        logger.debug("GitCommitTool not created: REPO_MANAGER_ENABLED is false")
        return None

    tool_instance = GitCommitTool(sql_db, user_id)
    if not tool_instance.repo_manager:
        logger.debug("GitCommitTool not created: RepoManager initialization failed")
        return None

    return StructuredTool.from_function(
        coroutine=tool_instance._arun,
        func=tool_instance._run,
        name="git_commit",
        description=tool_instance.description,
        args_schema=GitCommitInput,
    )
