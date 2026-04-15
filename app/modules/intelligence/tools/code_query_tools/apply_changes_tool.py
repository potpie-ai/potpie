"""
Apply Changes Tool

Applies code changes from CodeChangesManager (Redis) to the actual worktree filesystem.
This bridges the gap between stored changes and the git repository.
"""

import json
import os
from typing import Dict, Any, Optional, List

import redis

from app.core.config_provider import ConfigProvider
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from langchain_core.tools import StructuredTool

from app.modules.projects.projects_service import ProjectService
from app.modules.repo_manager import RepoManager
from app.modules.repo_manager.sync_helper import (
    get_or_create_worktree_path,
    get_or_create_edits_worktree_path,
)
from app.modules.intelligence.tools.code_changes_manager import CodeChangesManager
from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)

# Redis key prefix and TTL for apply changes result (used by git_commit)
APPLY_CHANGES_RESULT_KEY_PREFIX = "apply_changes_result"
APPLY_CHANGES_RESULT_TTL_SECONDS = 60 * 60  # 1 hour


def _store_apply_result(
    conversation_id: str,
    project_id: str,
    files_applied: List[str],
    files_deleted: List[str],
) -> None:
    """Store apply changes result in Redis for git_commit to consume."""
    try:
        config = ConfigProvider()
        client = redis.from_url(config.get_redis_url())
        key = f"{APPLY_CHANGES_RESULT_KEY_PREFIX}:{conversation_id}:{project_id}"
        data = json.dumps(
            {"files_applied": files_applied, "files_deleted": files_deleted}
        )
        client.setex(key, APPLY_CHANGES_RESULT_TTL_SECONDS, data)
    except Exception as e:
        logger.warning(f"ApplyChangesTool: Failed to store apply result: {e}")


def _get_apply_result(
    conversation_id: str, project_id: str
) -> Optional[Dict[str, List[str]]]:
    """Fetch apply changes result from Redis (files_applied + files_deleted)."""
    try:
        config = ConfigProvider()
        client = redis.from_url(config.get_redis_url())
        key = f"{APPLY_CHANGES_RESULT_KEY_PREFIX}:{conversation_id}:{project_id}"
        data = client.get(key)
        if data:
            raw = data.decode("utf-8") if isinstance(data, bytes) else data
            return json.loads(raw)
        return None
    except Exception as e:
        logger.warning(f"ApplyChangesTool: Failed to get apply result: {e}")
        return None


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
            - files_deleted: List of files that were removed from the worktree
            - files_skipped: List of files that were skipped (e.g., no content, path traversal)
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
        project_id: str,
        conversation_id: str,
        user_id: Optional[str],
    ) -> tuple[Optional[str], Optional[str]]:
        """Get the edits worktree path for this conversation. Returns (path, failure_reason)."""
        return get_or_create_edits_worktree_path(
            self.repo_manager,
            project_id=project_id,
            conversation_id=conversation_id,
            user_id=user_id,
            sql_db=self.sql_db,
        )

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
                    "files_deleted": [],
                    "files_skipped": [],
                }

            # Get project details
            details = self._get_project_details(project_id)
            user_id = details.get("user_id")

            # Get edits worktree path for this conversation
            worktree_path, failure_reason = self._get_worktree_path(
                project_id, conversation_id, user_id
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
                    "files_applied": [],
                    "files_deleted": [],
                    "files_skipped": [],
                }

            # Initialize CodeChangesManager with conversation_id
            changes_manager = CodeChangesManager(conversation_id=conversation_id)

            # Get all changes
            all_changes = changes_manager.changes

            if not all_changes:
                _store_apply_result(conversation_id, project_id, [], [])
                return {
                    "success": True,
                    "message": "No changes found to apply",
                    "files_applied": [],
                    "files_deleted": [],
                    "files_skipped": [],
                }

            # Filter to specific file if requested
            if file_path:
                if file_path not in all_changes:
                    return {
                        "success": False,
                        "error": f"File '{file_path}' not found in changes for conversation {conversation_id}",
                        "files_applied": [],
                        "files_deleted": [],
                        "files_skipped": [],
                    }
                files_to_apply = {file_path: all_changes[file_path]}
            else:
                files_to_apply = all_changes

            worktree_path_abs = os.path.abspath(worktree_path)
            files_applied: List[str] = []
            files_deleted: List[str] = []
            files_skipped: List[str] = []

            def _is_path_safe(fpath: str) -> tuple[bool, str]:
                """Validate fpath is inside worktree (prevents path traversal)."""
                full = os.path.abspath(os.path.join(worktree_path, fpath))
                try:
                    common = os.path.commonpath([worktree_path_abs, full])
                    if common != worktree_path_abs:
                        return False, f"{fpath} (path traversal)"
                    return True, full
                except ValueError:
                    return False, f"{fpath} (path traversal)"

            for fpath, change in files_to_apply.items():
                if change.change_type.value == "delete":
                    safe, path_or_reason = _is_path_safe(fpath)
                    if not safe:
                        logger.error(
                            f"ApplyChangesTool: Path traversal blocked: {fpath}"
                        )
                        files_skipped.append(path_or_reason)
                        continue
                    full_path = path_or_reason
                    try:
                        if os.path.exists(full_path):
                            os.remove(full_path)
                        files_deleted.append(fpath)
                        logger.info(f"ApplyChangesTool: Deleted {fpath}")
                    except Exception as e:
                        logger.error(f"ApplyChangesTool: Failed to delete {fpath}: {e}")
                        files_skipped.append(f"{fpath} (delete error: {str(e)})")
                    continue

                if change.content is None:
                    files_skipped.append(f"{fpath} (no content)")
                    continue

                safe, path_or_reason = _is_path_safe(fpath)
                if not safe:
                    logger.error(f"ApplyChangesTool: Path traversal blocked: {fpath}")
                    files_skipped.append(path_or_reason)
                    continue
                full_path = path_or_reason

                parent_dir = os.path.dirname(full_path)
                if parent_dir and not os.path.exists(parent_dir):
                    os.makedirs(parent_dir, exist_ok=True)
                    logger.info(f"ApplyChangesTool: Created directory {parent_dir}")

                try:
                    with open(full_path, "w", encoding="utf-8") as f:
                        f.write(change.content)
                    files_applied.append(fpath)
                    logger.info(f"ApplyChangesTool: Applied changes to {fpath}")
                except Exception as e:
                    logger.error(f"ApplyChangesTool: Failed to write {fpath}: {e}")
                    files_skipped.append(f"{fpath} (write error: {str(e)})")

            # Success when no write/delete/path-traversal failures occurred
            failure_skips = [
                s
                for s in files_skipped
                if "(write error" in s or "(delete error" in s or "(path traversal" in s
            ]
            success = len(failure_skips) == 0

            _store_apply_result(
                conversation_id, project_id, files_applied, files_deleted
            )

            return {
                "success": success,
                "files_applied": files_applied,
                "files_deleted": files_deleted,
                "files_skipped": files_skipped,
                "worktree_path": worktree_path,
            }

        except ValueError as e:
            logger.error(f"ApplyChangesTool: Value error: {e}")
            return {
                "success": False,
                "error": str(e),
                "files_applied": [],
                "files_deleted": [],
                "files_skipped": [],
            }
        except Exception as e:
            logger.exception("ApplyChangesTool: Unexpected error")
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}",
                "files_applied": [],
                "files_deleted": [],
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
