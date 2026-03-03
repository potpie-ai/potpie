"""
Git Commit Tool

Stages and commits changes in the repository worktree.
"""

import json
import os
from typing import Dict, Any, Optional, List, Union
from pydantic import BaseModel, Field, field_validator
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

from app.modules.intelligence.tools.code_query_tools.apply_changes_tool import (
    _get_apply_result,
)

logger = setup_logger(__name__)


class GitCommitInput(BaseModel):
    project_id: str = Field(
        ..., description="Project ID that references the repository"
    )
    commit_message: str = Field(
        ..., description="Commit message describing the changes"
    )
    conversation_id: Optional[str] = Field(
        None,
        description="Conversation ID from apply_changes. When provided with files=None, stages only files_applied + files_deleted from the last apply_changes run.",
    )
    files: Optional[List[str]] = Field(
        None,
        description="Optional list of specific files to commit. If not provided, uses files from apply_changes when conversation_id is set; otherwise pass files explicitly (files_applied + files_deleted from apply_changes).",
    )

    @field_validator("files", mode="before")
    @classmethod
    def coerce_files_from_json_string(cls, v):
        """Accept a JSON-encoded string like '["a.py","b.py"]' in addition to a real list."""
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return parsed
            except (json.JSONDecodeError, ValueError):
                pass
        return v


class GitCommitTool:
    name: str = "git_commit"
    description: str = """Stage and commit changes in the repository worktree.

        This tool stages modified files and creates a git commit in the worktree.
        Use it after applying changes with `apply_changes` to create a commit
        that can then be pushed and used for a PR.

        The tool will:
        1. Stage the specified files (or files from apply_changes when conversation_id is provided)
        2. Create a git commit with the provided message
        3. Return the commit hash

        Pass conversation_id when committing after apply_changes to stage only those files.

        Args:
            project_id: The repository project ID (UUID)
            commit_message: The commit message (should be descriptive)
            conversation_id: Optional. When set with files=None, stages files_applied + files_deleted from apply_changes.
            files: Optional list of specific file paths to commit. If omitted and conversation_id is set, uses apply_changes result.

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
        project_id: str,
        repo_name: str,
        branch: Optional[str],
        commit_id: Optional[str],
        user_id: Optional[str],
        conversation_id: Optional[str] = None,
    ) -> tuple[Optional[str], Optional[str]]:
        """Get the worktree path.

        When ``conversation_id`` is provided, targets the edits worktree for
        that conversation (``agent/edits-{conversation_id}`` branch).
        Otherwise falls back to the base repo worktree via prepare_for_parsing.
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

    def _run(
        self,
        project_id: str,
        commit_message: str,
        conversation_id: Optional[str] = None,
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

            # Get worktree path (edits worktree when conversation_id provided)
            worktree_path, failure_reason = self._get_worktree_path(
                project_id, repo_name, branch, commit_id, user_id, conversation_id
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

            # Resolve files to stage: explicit list, or from apply_changes when conversation_id provided
            if files is not None:
                files_to_stage = files
            elif conversation_id:
                apply_result = _get_apply_result(conversation_id, project_id)
                if apply_result:
                    files_to_stage = apply_result.get("files_applied", []) + apply_result.get("files_deleted", [])
                else:
                    return {
                        "success": False,
                        "error": "No apply_changes result found. Run apply_changes first, or pass 'files' explicitly.",
                    }
            else:
                return {
                    "success": False,
                    "error": "Pass 'files' (files_applied + files_deleted from apply_changes) or 'conversation_id' to stage only intended changes.",
                }

            logger.info(
                f"GitCommitTool: Committing changes in {worktree_path} "
                f"(files: {len(files_to_stage)} paths)"
            )

            def _commit_operation(repo):
                # Stage additions, modifications, and deletions for the given paths
                if files_to_stage:
                    repo.git.add("-A", "--", *files_to_stage)

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
        conversation_id: Optional[str] = None,
        files: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Async wrapper for _run."""
        import asyncio

        return await asyncio.to_thread(
            self._run, project_id, commit_message, conversation_id, files
        )


def git_commit_tool(sql_db: Session, user_id: str) -> Optional[OnyxTool]:
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

    return OnyxTool.from_function(
        coroutine=tool_instance._arun,
        func=tool_instance._run,
        name="git_commit",
        description=tool_instance.description,
        args_schema=GitCommitInput,
    )
