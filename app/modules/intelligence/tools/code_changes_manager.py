"""
Code Changes Manager Tool for Agent State Management

This tool allows agents to manage code changes in Redis, reducing token usage
by storing code modifications separately from response text. Changes are tracked
per-file and can be searched, retrieved, and serialized for persistence.

Changes persist across messages within a conversation (keyed by conversation_id).
"""

import uuid
import re
import json
import os
import inspect
import functools
import difflib
import time
import threading
import redis
from contextvars import ContextVar
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Callable, TypeVar
from enum import Enum
from dataclasses import dataclass, asdict
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from app.modules.utils.logger import setup_logger
from app.core.config_provider import ConfigProvider

logger = setup_logger(__name__)

# Redis key prefix and expiry for code changes
CODE_CHANGES_KEY_PREFIX = "code_changes"
CODE_CHANGES_TTL_SECONDS = 24 * 60 * 60  # 24 hours expiry

T = TypeVar("T")

# Maximum file size to read into memory (8MB - reduced from 10MB for safety margin)
# This prevents OOM kills when processing very large files
# Reduced to 8MB to leave headroom for Python object overhead and memory spikes
MAX_FILE_SIZE_BYTES = 8 * 1024 * 1024  # 8MB

# Database query timeout in seconds - prevents deadlocks in forked workers
DB_QUERY_TIMEOUT = 15.0  # 15 seconds max for any database query
DB_SESSION_CREATE_TIMEOUT = 10.0  # 10 seconds max for creating a new session

# Memory pressure threshold - skip non-critical operations if memory usage exceeds this
MEMORY_PRESSURE_THRESHOLD = 0.80  # 80% of worker memory limit


def _execute_with_timeout(
    operation: Callable[[], T],
    timeout: float,
    operation_name: str = "operation",
) -> T:
    """
    Execute an operation with a timeout using threading.

    This prevents deadlocks when database queries or other operations hang
    in forked Celery workers.

    Args:
        operation: The operation to execute
        timeout: Maximum time in seconds to wait
        operation_name: Name of the operation for logging

    Returns:
        The result of the operation

    Raises:
        TimeoutError: If the operation exceeds the timeout
        Exception: Any exception raised by the operation
    """
    result_container = {"value": None, "exception": None, "completed": False}

    def _run_operation():
        try:
            result_container["value"] = operation()
            result_container["completed"] = True
        except Exception as e:
            result_container["exception"] = e
            result_container["completed"] = True

    thread = threading.Thread(target=_run_operation, daemon=True)
    thread.start()
    thread.join(timeout=timeout)

    if not result_container["completed"]:
        logger.error(
            f"Operation '{operation_name}' timed out after {timeout} seconds. "
            f"This may indicate a deadlock or hung operation."
        )
        raise TimeoutError(
            f"Operation '{operation_name}' timed out after {timeout} seconds"
        )

    if result_container["exception"]:
        raise result_container["exception"]

    return result_container["value"]


def _check_memory_pressure() -> tuple[bool, Optional[float]]:
    """
    Check if worker is under memory pressure.

    Returns:
        Tuple of (is_under_pressure: bool, memory_usage_percent: float or None)
        Returns (False, None) if memory check fails or psutil not available
    """
    try:
        import psutil
        import os as os_module

        process = psutil.Process()
        memory_info = process.memory_info()
        rss_mb = memory_info.rss / 1024 / 1024

        # Get worker memory limit from environment
        max_memory_kb = int(os_module.getenv("CELERY_WORKER_MAX_MEMORY_KB", "2000000"))
        max_memory_mb = max_memory_kb / 1024

        if max_memory_mb <= 0:
            return False, None

        memory_usage_percent = rss_mb / max_memory_mb
        is_under_pressure = memory_usage_percent >= MEMORY_PRESSURE_THRESHOLD

        if is_under_pressure:
            logger.warning(
                f"Memory pressure detected: {memory_usage_percent:.1%} ({rss_mb:.1f}MB / {max_memory_mb:.1f}MB). "
                f"Skipping non-critical operations."
            )

        return is_under_pressure, memory_usage_percent
    except ImportError:
        # psutil not available - can't check memory
        return False, None
    except Exception as e:
        logger.debug(f"Failed to check memory pressure: {e}")
        return False, None


def _get_git_file_size(
    repo_path: str, file_path: str, ref: Optional[str] = None
) -> Optional[int]:
    """
    Get file size from git repository without loading content.
    Uses 'git cat-file -s' which is very efficient.

    Args:
        repo_path: Path to git repository
        file_path: Relative path to file
        ref: Branch/commit reference (defaults to HEAD)

    Returns:
        File size in bytes, or None if unable to determine
    """
    try:
        from app.modules.code_provider.git_safe import safe_git_repo_operation

        def _get_size(repo):
            from git.exc import GitCommandError

            actual_ref = ref or repo.active_branch.name
            try:
                # Use git cat-file -s to get object size without loading content
                # Format: ref:path
                blob_sha = repo.git.rev_parse(f"{actual_ref}:{file_path}")
                size_str = repo.git.cat_file("-s", blob_sha)
                return int(size_str.strip())
            except GitCommandError as e:
                if "does not exist" in str(e) or "path not in" in str(e):
                    return None
                raise

        return safe_git_repo_operation(
            repo_path,
            _get_size,
            max_retries=1,
            timeout=5.0,  # Quick operation, short timeout
            operation_name=f"get_file_size({file_path})",
        )
    except Exception as e:
        logger.debug(f"Could not get file size for {file_path}: {e}")
        return None


class ChangeType(str, Enum):
    """Type of code change"""

    ADD = "add"  # New file
    UPDATE = "update"  # Modified file
    DELETE = "delete"  # Deleted file


@dataclass
class FileChange:
    """Represents a change to a single file"""

    file_path: str
    change_type: ChangeType
    content: Optional[str] = None  # None for DELETE
    previous_content: Optional[str] = None  # For UPDATE/DELETE
    created_at: str = ""
    updated_at: str = ""
    description: Optional[str] = None  # Optional change description

    def __post_init__(self):
        now = datetime.now().isoformat()
        if not self.created_at:
            self.created_at = now
        if not self.updated_at:
            self.updated_at = now


class CodeChangesManager:
    """Manages code changes in Redis for a conversation, persisting across messages"""

    def __init__(self, conversation_id: Optional[str] = None):
        """
        Initialize the CodeChangesManager with Redis storage.

        Args:
            conversation_id: The conversation ID to use as part of the Redis key.
                           If None, a random session_id is used (backward compatible).
        """
        self._conversation_id = conversation_id
        self.session_id = (
            conversation_id[:8] if conversation_id else str(uuid.uuid4())[:8]
        )

        # Initialize Redis client
        config = ConfigProvider()
        self._redis_client = redis.from_url(config.get_redis_url())

        # In-memory cache for the current instance
        self._changes_cache: Optional[Dict[str, FileChange]] = None

        logger.info(
            f"CodeChangesManager: Initialized with conversation_id={conversation_id}, "
            f"session_id={self.session_id}, redis_key={self._redis_key}"
        )

    @property
    def _redis_key(self) -> str:
        """Get the Redis key for storing changes"""
        if self._conversation_id:
            return f"{CODE_CHANGES_KEY_PREFIX}:{self._conversation_id}"
        return f"{CODE_CHANGES_KEY_PREFIX}:session:{self.session_id}"

    @property
    def changes(self) -> Dict[str, FileChange]:
        """Get the changes dict, loading from Redis if needed"""
        if self._changes_cache is None:
            self._load_from_redis()
        # After _load_from_redis, _changes_cache is always set (even if empty dict)
        assert self._changes_cache is not None
        return self._changes_cache

    @changes.setter
    def changes(self, value: Dict[str, FileChange]) -> None:
        """Set the changes dict and persist to Redis"""
        self._changes_cache = value
        self._save_to_redis()

    def _load_from_redis(self) -> None:
        """Load changes from Redis into the cache"""
        try:
            data = self._redis_client.get(self._redis_key)
            if data:
                json_str = data.decode("utf-8") if isinstance(data, bytes) else data
                parsed = json.loads(json_str)
                self._changes_cache = {}
                for change_data in parsed.get("changes", []):
                    change = FileChange(
                        file_path=change_data["file_path"],
                        change_type=ChangeType(change_data["change_type"]),
                        content=change_data.get("content"),
                        previous_content=change_data.get("previous_content"),
                        created_at=change_data.get(
                            "created_at", datetime.now().isoformat()
                        ),
                        updated_at=change_data.get(
                            "updated_at", datetime.now().isoformat()
                        ),
                        description=change_data.get("description"),
                    )
                    self._changes_cache[change.file_path] = change
                logger.debug(
                    f"CodeChangesManager._load_from_redis: Loaded {len(self._changes_cache)} changes from Redis"
                )
            else:
                self._changes_cache = {}
                logger.debug(
                    "CodeChangesManager._load_from_redis: No existing data in Redis, starting fresh"
                )
        except Exception as e:
            logger.warning(
                f"CodeChangesManager._load_from_redis: Error loading from Redis: {e}, starting fresh"
            )
            self._changes_cache = {}

    def _save_to_redis(self) -> None:
        """Save changes to Redis with expiry"""
        try:
            if self._changes_cache is None:
                return

            data = {
                "session_id": self.session_id,
                "conversation_id": self._conversation_id,
                "changes": [
                    {
                        "file_path": change.file_path,
                        "change_type": change.change_type.value,
                        "content": change.content,
                        "previous_content": change.previous_content,
                        "created_at": change.created_at,
                        "updated_at": change.updated_at,
                        "description": change.description,
                    }
                    for change in self._changes_cache.values()
                ],
            }
            json_str = json.dumps(data)
            self._redis_client.setex(
                self._redis_key, CODE_CHANGES_TTL_SECONDS, json_str
            )
            logger.debug(
                f"CodeChangesManager._save_to_redis: Saved {len(self._changes_cache)} changes to Redis "
                f"(key={self._redis_key}, ttl={CODE_CHANGES_TTL_SECONDS}s)"
            )
        except Exception as e:
            logger.error(
                f"CodeChangesManager._save_to_redis: Error saving to Redis: {e}"
            )

    def _persist_change(self) -> None:
        """Persist current changes to Redis (call after any modification)"""
        self._save_to_redis()

    def add_file(
        self,
        file_path: str,
        content: str,
        description: Optional[str] = None,
    ) -> bool:
        """Add a new file"""
        logger.info(
            f"CodeChangesManager.add_file: Adding file '{file_path}' (content length: {len(content)} chars)"
        )
        # Access self.changes to ensure we load from Redis
        changes = self.changes
        if file_path in changes and changes[file_path].change_type != ChangeType.DELETE:
            logger.warning(
                f"CodeChangesManager.add_file: File '{file_path}' already exists (not deleted)"
            )
            return False  # File already exists (not deleted)

        change = FileChange(
            file_path=file_path,
            change_type=ChangeType.ADD,
            content=content,
            description=description,
        )
        self._changes_cache[file_path] = change
        self._persist_change()
        logger.info(
            f"CodeChangesManager.add_file: Successfully added file '{file_path}' (session: {self.session_id})"
        )
        return True

    def _get_current_content(
        self,
        file_path: str,
        project_id: Optional[str] = None,
        db: Optional[Session] = None,
    ) -> str:
        """Get current content of a file (from changes, repository via code provider, filesystem, or empty if new)"""
        start_time = time.time()
        logger.info(
            f"CodeChangesManager._get_current_content: [START] Getting content for '{file_path}' "
            f"(project_id={project_id}, db={'provided' if db else 'None'})"
        )

        if file_path in self.changes:
            existing = self.changes[file_path]
            if existing.change_type == ChangeType.DELETE:
                logger.info(
                    f"CodeChangesManager._get_current_content: File '{file_path}' is marked as deleted"
                )
                return ""  # Deleted file has no content
            content = existing.content or ""
            elapsed = time.time() - start_time
            logger.info(
                f"CodeChangesManager._get_current_content: [SUCCESS] Retrieved '{file_path}' from changes "
                f"({len(content)} chars, {len(content.split(chr(10)))} lines) in {elapsed:.3f}s"
            )
            return content

        # If file not in changes, try to fetch from repository using code provider
        if project_id and db:
            logger.info(
                f"CodeChangesManager._get_current_content: [STEP 1] Attempting to fetch '{file_path}' "
                f"from repository using project_id={project_id}"
            )
            try:
                from app.modules.code_provider.code_provider_service import (
                    CodeProviderService,
                )
                from app.modules.projects.projects_model import Project

                # Project.id is Text (string) in the database, so query directly with the string project_id
                # Note: get_project_from_db_by_id_sync has incorrect type hint (int), but actually accepts string
                query_start = time.time()
                logger.info(
                    f"CodeChangesManager._get_current_content: [STEP 1.1] Fetching project details for project_id={project_id} "
                    f"(type: {type(project_id).__name__}, session_id={id(db)})"
                )
                project_details = None
                try:
                    # Query directly - Project.id is Text column, so string works fine
                    # The method type hint says int, but the actual column accepts strings
                    # Wrap in timeout to prevent deadlocks
                    logger.debug(
                        f"CodeChangesManager._get_current_content: [STEP 1.1.1] Executing database query with timeout={DB_QUERY_TIMEOUT}s"
                    )
                    project = _execute_with_timeout(
                        lambda: db.query(Project)
                        .filter(Project.id == project_id)
                        .first(),
                        timeout=DB_QUERY_TIMEOUT,
                        operation_name=f"db_query_project_{project_id}",
                    )
                    query_elapsed = time.time() - query_start
                    logger.info(
                        f"CodeChangesManager._get_current_content: [STEP 1.1.2] Database query completed in {query_elapsed:.3f}s"
                    )

                    if project:
                        project_details = {
                            "project_name": project.repo_name,
                            "id": project.id,
                            "commit_id": project.commit_id,
                            "status": project.status,
                            "branch_name": project.branch_name,
                            "repo_path": project.repo_path,
                            "user_id": project.user_id,
                        }
                        logger.info(
                            f"CodeChangesManager._get_current_content: [STEP 1.1.3] Project details retrieved: "
                            f"repo={project_details.get('project_name')}, branch={project_details.get('branch_name')}"
                        )
                    else:
                        logger.warning(
                            f"CodeChangesManager._get_current_content: [STEP 1.1.3] Project not found in database for project_id={project_id}"
                        )
                        project_details = None
                except TimeoutError as timeout_err:
                    query_elapsed = time.time() - query_start
                    logger.error(
                        f"CodeChangesManager._get_current_content: [STEP 1.1.ERROR] Database query TIMED OUT after {query_elapsed:.3f}s "
                        f"for project_id={project_id}. This indicates a potential deadlock or hung database connection. "
                        f"Error: {timeout_err}"
                    )
                    project_details = None
                except Exception as e:
                    query_elapsed = time.time() - query_start
                    # Database session might be invalid in forked workers - try creating a new one
                    error_str = str(e).lower()
                    logger.warning(
                        f"CodeChangesManager._get_current_content: [STEP 1.1.ERROR] Database query failed after {query_elapsed:.3f}s: {e}"
                    )
                    if any(
                        keyword in error_str
                        for keyword in [
                            "connection",
                            "session",
                            "closed",
                            "invalid",
                            "fork",
                        ]
                    ):
                        logger.warning(
                            f"CodeChangesManager._get_current_content: [STEP 1.2] Database session error (likely from forked worker): {e}. "
                            f"Creating new session for project_id={project_id}"
                        )
                        try:
                            from app.core.database import SessionLocal

                            session_create_start = time.time()
                            logger.info(
                                "CodeChangesManager._get_current_content: [STEP 1.2.1] Attempting to close old session and create new one"
                            )
                            old_db = db  # Keep reference to old session for cleanup
                            try:
                                old_db.close()  # Close invalid session if possible
                                logger.debug(
                                    "CodeChangesManager._get_current_content: [STEP 1.2.1.1] Old session closed"
                                )
                            except Exception as close_err:
                                logger.debug(
                                    f"CodeChangesManager._get_current_content: [STEP 1.2.1.1] Error closing old session (ignored): {close_err}"
                                )
                                pass  # Ignore errors when closing invalid session

                            # Create new session with timeout protection
                            logger.debug(
                                f"CodeChangesManager._get_current_content: [STEP 1.2.2] Creating new session with timeout={DB_SESSION_CREATE_TIMEOUT}s"
                            )
                            db = _execute_with_timeout(
                                lambda: SessionLocal(),
                                timeout=DB_SESSION_CREATE_TIMEOUT,
                                operation_name="create_new_db_session",
                            )
                            session_create_elapsed = time.time() - session_create_start
                            logger.info(
                                f"CodeChangesManager._get_current_content: [STEP 1.2.3] New session created in {session_create_elapsed:.3f}s (session_id={id(db)})"
                            )

                            # Retry the query with new session
                            retry_query_start = time.time()
                            logger.info(
                                "CodeChangesManager._get_current_content: [STEP 1.2.4] Retrying database query with new session"
                            )
                            project = _execute_with_timeout(
                                lambda: db.query(Project)
                                .filter(Project.id == project_id)
                                .first(),
                                timeout=DB_QUERY_TIMEOUT,
                                operation_name=f"db_query_project_retry_{project_id}",
                            )
                            retry_query_elapsed = time.time() - retry_query_start
                            logger.info(
                                f"CodeChangesManager._get_current_content: [STEP 1.2.5] Retry query completed in {retry_query_elapsed:.3f}s"
                            )

                            if project:
                                project_details = {
                                    "project_name": project.repo_name,
                                    "id": project.id,
                                    "commit_id": project.commit_id,
                                    "status": project.status,
                                    "branch_name": project.branch_name,
                                    "repo_path": project.repo_path,
                                    "user_id": project.user_id,
                                }
                                logger.info(
                                    f"CodeChangesManager._get_current_content: [STEP 1.2.6] Project details retrieved with new session: "
                                    f"repo={project_details.get('project_name')}, branch={project_details.get('branch_name')}"
                                )
                            else:
                                logger.warning(
                                    f"CodeChangesManager._get_current_content: [STEP 1.2.6] Project not found with new session for project_id={project_id}"
                                )
                                project_details = None
                        except TimeoutError as timeout_err:
                            logger.error(
                                f"CodeChangesManager._get_current_content: [STEP 1.2.ERROR] Session creation or retry query TIMED OUT: {timeout_err}. "
                                f"This indicates a potential deadlock. Falling back to filesystem."
                            )
                            project_details = None
                            # Ensure new session is closed if created
                            if (
                                "db" in locals()
                                and "old_db" in locals()
                                and db != old_db
                            ):
                                try:
                                    db.close()
                                except Exception:
                                    pass
                        except Exception as retry_error:
                            logger.error(
                                f"CodeChangesManager._get_current_content: [STEP 1.2.ERROR] Failed to create new session and retry: {retry_error}",
                                exc_info=True,
                            )
                            project_details = None
                            # Ensure new session is closed if created
                            if (
                                "db" in locals()
                                and "old_db" in locals()
                                and db != old_db
                            ):
                                try:
                                    db.close()
                                except Exception:
                                    pass
                    else:
                        logger.warning(
                            f"CodeChangesManager._get_current_content: [STEP 1.1.ERROR] Error querying project for project_id '{project_id}': {e}",
                            exc_info=True,
                        )
                        project_details = None

                if project_details and "project_name" in project_details:
                    # Check memory pressure before expensive operations
                    is_pressure, mem_percent = _check_memory_pressure()
                    if mem_percent is not None:
                        logger.debug(
                            f"CodeChangesManager._get_current_content: [STEP 2.MEMORY] Memory usage: {mem_percent:.1%} "
                            f"for file '{file_path}'"
                        )
                    if is_pressure:
                        logger.warning(
                            f"CodeChangesManager._get_current_content: [STEP 2.MEMORY] Memory pressure detected ({mem_percent:.1%}). "
                            f"Skipping repository fetch for '{file_path}' to prevent OOM kill. Falling back to filesystem."
                        )
                        repo_content = None
                    else:
                        repo_content = None  # Initialize before try block
                        try:
                            # Pre-check file size for local git repositories before loading
                            repo_path = project_details.get("repo_path")
                            if repo_path and os.path.exists(repo_path):
                                file_size = _get_git_file_size(
                                    repo_path,
                                    file_path,
                                    project_details.get("branch_name"),
                                )
                                if file_size is not None:
                                    if file_size > MAX_FILE_SIZE_BYTES:
                                        logger.warning(
                                            f"CodeChangesManager._get_current_content: [STEP 2.SIZE] File '{file_path}' "
                                            f"is too large ({file_size} bytes, max {MAX_FILE_SIZE_BYTES} bytes). "
                                            f"Skipping to prevent memory issues. Falling back to filesystem."
                                        )
                                        repo_content = None
                                    else:
                                        logger.debug(
                                            f"CodeChangesManager._get_current_content: [STEP 2.SIZE] File '{file_path}' "
                                            f"size check passed: {file_size} bytes"
                                        )

                            service_init_start = time.time()
                            logger.info(
                                f"CodeChangesManager._get_current_content: [STEP 2] Initializing CodeProviderService "
                                f"for repo '{project_details['project_name']}'"
                            )
                            # Initialize CodeProviderService with timeout protection
                            cp_service = _execute_with_timeout(
                                lambda: CodeProviderService(db),
                                timeout=DB_QUERY_TIMEOUT,  # Service init might query DB
                                operation_name="CodeProviderService_init",
                            )
                            service_init_elapsed = time.time() - service_init_start
                            logger.info(
                                f"CodeChangesManager._get_current_content: [STEP 2.1] CodeProviderService initialized in {service_init_elapsed:.3f}s"
                            )

                            # Skip file fetch if we already determined it's too large or memory pressure
                            if repo_content is not None:
                                # This means size check failed - skip to filesystem
                                pass
                            else:
                                file_fetch_start = time.time()
                                logger.info(
                                    f"CodeChangesManager._get_current_content: [STEP 3] Fetching file content from repository "
                                    f"for '{file_path}' in repo '{project_details['project_name']}'"
                                )
                                # Wrap git operations in safe handler to prevent SIGSEGV and timeouts
                                from app.modules.code_provider.git_safe import (
                                    safe_git_operation,
                                    GitOperationError,
                                )

                                def _fetch_file_content():
                                    fetch_start = time.time()
                                    logger.debug(
                                        "CodeChangesManager._get_current_content: [STEP 3.1] Inside _fetch_file_content, calling cp_service.get_file_content"
                                    )
                                    try:
                                        result = cp_service.get_file_content(
                                            repo_name=project_details["project_name"],
                                            file_path=file_path,
                                            branch_name=project_details.get(
                                                "branch_name"
                                            ),
                                            start_line=None,
                                            end_line=None,
                                            project_id=project_id,
                                            commit_id=project_details.get("commit_id"),
                                        )
                                        fetch_elapsed = time.time() - fetch_start
                                        logger.debug(
                                            f"CodeChangesManager._get_current_content: [STEP 3.1.1] cp_service.get_file_content completed in {fetch_elapsed:.3f}s"
                                        )
                                        return result
                                    except Exception as fetch_err:
                                        fetch_elapsed = time.time() - fetch_start
                                        logger.error(
                                            f"CodeChangesManager._get_current_content: [STEP 3.1.ERROR] cp_service.get_file_content failed after {fetch_elapsed:.3f}s: {fetch_err}",
                                            exc_info=True,
                                        )
                                        raise

                                try:
                                    # Use max_total_timeout to prevent operations from running indefinitely
                                    # Even with 2 retries at 20s each, cap total time at 40s to prevent worker hangs
                                    # Reduced timeouts to fail faster and prevent worker crashes
                                    logger.debug(
                                        "CodeChangesManager._get_current_content: [STEP 3.2] Calling safe_git_operation with timeout=20.0s, max_retries=1"
                                    )
                                    repo_content = safe_git_operation(
                                        _fetch_file_content,
                                        max_retries=1,  # Reduced to 1 retry to fail faster
                                        timeout=20.0,  # 20 second timeout per attempt (reduced from 30s)
                                        max_total_timeout=40.0,  # Maximum 40 seconds total (reduced from 60s)
                                        operation_name=f"get_file_content({file_path})",
                                    )
                                    file_fetch_elapsed = time.time() - file_fetch_start
                                    logger.info(
                                        f"CodeChangesManager._get_current_content: [STEP 3.3] File fetch completed in {file_fetch_elapsed:.3f}s"
                                    )
                                except GitOperationError as git_error:
                                    file_fetch_elapsed = time.time() - file_fetch_start
                                    # If safe wrapper fails after retries, log and continue to filesystem fallback
                                    logger.warning(
                                        f"CodeChangesManager._get_current_content: [STEP 3.ERROR] Safe git operation failed after {file_fetch_elapsed:.3f}s: {git_error}. "
                                        f"Falling back to filesystem for '{file_path}'"
                                    )
                                    repo_content = None
                                except MemoryError as mem_error:
                                    file_fetch_elapsed = time.time() - file_fetch_start
                                    # Handle memory errors gracefully - this can happen with very large files
                                    logger.error(
                                        f"CodeChangesManager._get_current_content: [STEP 3.ERROR] Memory error fetching '{file_path}' from repository after {file_fetch_elapsed:.3f}s: {mem_error}. "
                                        f"This may indicate the file is too large or system memory is low. "
                                        f"Falling back to filesystem."
                                    )
                                    repo_content = None
                                except (SystemExit, KeyboardInterrupt) as e:
                                    file_fetch_elapsed = time.time() - file_fetch_start
                                    # Don't re-raise - catch and fallback to prevent worker crash
                                    logger.error(
                                        f"CodeChangesManager._get_current_content: [STEP 3.ERROR] System exit/interrupt during git operation for '{file_path}' after {file_fetch_elapsed:.3f}s: {e}. "
                                        f"Falling back to filesystem to prevent worker crash."
                                    )
                                    repo_content = None
                                except BaseException as e:
                                    file_fetch_elapsed = time.time() - file_fetch_start
                                    # Catch any other exceptions (including segfault-related errors) to prevent worker crash
                                    logger.error(
                                        f"CodeChangesManager._get_current_content: [STEP 3.ERROR] Unexpected error during git operation for '{file_path}' after {file_fetch_elapsed:.3f}s: {type(e).__name__}: {e}. "
                                        f"This may indicate a crash or resource issue. Falling back to filesystem.",
                                        exc_info=True,
                                    )
                                    repo_content = None
                        except TimeoutError as timeout_err:
                            logger.error(
                                f"CodeChangesManager._get_current_content: [STEP 2.ERROR] CodeProviderService initialization TIMED OUT: {timeout_err}. "
                                f"This indicates a potential deadlock. Falling back to filesystem.",
                                exc_info=True,
                            )
                            repo_content = None
                        except Exception as service_error:
                            # Catch errors creating CodeProviderService or any other service-related errors
                            logger.error(
                                f"CodeChangesManager._get_current_content: [STEP 2.ERROR] Error creating CodeProviderService or fetching file '{file_path}': {service_error}. "
                                f"Falling back to filesystem.",
                                exc_info=True,
                            )
                            repo_content = None

                    if repo_content:
                        # Check content size to prevent memory issues
                        content_size_bytes = len(repo_content.encode("utf-8"))
                        if content_size_bytes > MAX_FILE_SIZE_BYTES:
                            logger.warning(
                                f"CodeChangesManager._get_current_content: [STEP 3.4] File '{file_path}' "
                                f"from repository is too large ({content_size_bytes} bytes, max {MAX_FILE_SIZE_BYTES} bytes). "
                                f"Skipping to prevent memory issues. Falling back to filesystem."
                            )
                            repo_content = None
                        else:
                            lines = repo_content.split("\n")
                            total_elapsed = time.time() - start_time
                            logger.info(
                                f"CodeChangesManager._get_current_content: [SUCCESS] Successfully retrieved '{file_path}' "
                                f"from repository via code provider ({len(repo_content)} chars, {len(lines)} lines) in {total_elapsed:.3f}s"
                            )
                            return repo_content

                    if not repo_content:
                        logger.warning(
                            f"CodeChangesManager._get_current_content: [STEP 3.4] Repository returned empty or oversized content for '{file_path}'"
                        )
                else:
                    logger.warning(
                        f"CodeChangesManager._get_current_content: [STEP 1.3] Cannot fetch from repository - "
                        f"project_details={'missing project_name' if project_details else 'None'}"
                    )
            except MemoryError as mem_error:
                elapsed = time.time() - start_time
                # Handle memory errors gracefully
                logger.error(
                    f"CodeChangesManager._get_current_content: [ERROR] Memory error fetching '{file_path}' from repository after {elapsed:.3f}s: {mem_error}. "
                    f"This may indicate the file is too large or system memory is low. "
                    f"Falling back to filesystem."
                )
                # Fall through to try filesystem
            except Exception as e:
                elapsed = time.time() - start_time
                logger.warning(
                    f"CodeChangesManager._get_current_content: [ERROR] Error fetching '{file_path}' from repository after {elapsed:.3f}s: {str(e)}",
                    exc_info=True,
                )
                # Fall through to try filesystem

        # If not available via code provider, try to read from filesystem
        # WARNING: This fallback can cause issues if file should be in changes but isn't found
        # (e.g., due to path mismatch). This returns ORIGINAL file content, which may overwrite changes.
        # Skip filesystem fallback if under memory pressure to prevent OOM
        is_pressure, mem_percent = _check_memory_pressure()
        if is_pressure:
            logger.warning(
                f"CodeChangesManager._get_current_content: [STEP 4.MEMORY] Memory pressure detected ({mem_percent:.1%}). "
                f"Skipping filesystem fallback for '{file_path}' to prevent OOM kill."
            )
            codebase_content = None
        else:
            filesystem_start = time.time()
            logger.info(
                f"CodeChangesManager._get_current_content: [STEP 4] Attempting to read '{file_path}' from filesystem"
            )
            codebase_content = self._read_file_from_codebase(file_path)
            filesystem_elapsed = time.time() - filesystem_start

        if codebase_content is not None:
            if "filesystem_elapsed" not in locals():
                filesystem_elapsed = 0
        if codebase_content is not None:
            lines = codebase_content.split("\n")
            total_elapsed = time.time() - start_time
            logger.warning(
                f"CodeChangesManager._get_current_content: [SUCCESS] ⚠️ WARNING - Retrieved '{file_path}' from filesystem "
                f"({len(codebase_content)} chars, {len(lines)} lines) in {filesystem_elapsed:.3f}s (total: {total_elapsed:.3f}s). "
                f"This may be the ORIGINAL file content, not the current changes. "
                f"If this file should have pending changes, they may be overwritten. "
                f"Consider providing project_id/db for proper repository access."
            )
            return codebase_content
        else:
            logger.warning(
                f"CodeChangesManager._get_current_content: [STEP 4.1] File '{file_path}' not found in filesystem (checked in {filesystem_elapsed:.3f}s)"
            )

        # File doesn't exist in changes, repository, or filesystem - treat as new file
        total_elapsed = time.time() - start_time
        logger.warning(
            f"CodeChangesManager._get_current_content: [END] File '{file_path}' not found anywhere - treating as new file (empty content) after {total_elapsed:.3f}s. "
            f"This may be intentional for creating new files, but verify the file path is correct."
        )
        return ""

    def _apply_update(
        self,
        file_path: str,
        new_content: str,
        description: Optional[str] = None,
        preserve_previous: bool = True,
        preserve_change_type: bool = True,
        original_content: Optional[str] = None,
    ) -> bool:
        """Internal method to apply a content update

        Args:
            file_path: Path to the file
            new_content: New content for the file
            description: Optional description of the change
            preserve_previous: Whether to preserve previous_content when updating existing changes
            preserve_change_type: Whether to preserve ADD change type for new files
            original_content: Original content before any modifications (used when creating new FileChange)
        """
        logger.debug(
            f"CodeChangesManager._apply_update: Applying update to '{file_path}' (preserve_previous={preserve_previous}, preserve_change_type={preserve_change_type}, original_content={'provided' if original_content is not None else 'None'})"
        )
        previous_content = None
        if file_path in self.changes:
            existing = self.changes[file_path]

            # Don't update if file is marked for deletion (unless explicitly allowed)
            if existing.change_type == ChangeType.DELETE:
                logger.warning(
                    f"CodeChangesManager._apply_update: File '{file_path}' is marked for deletion. "
                    f"Not applying update. Use delete_file to unmark deletion first."
                )
                return False

            if preserve_previous and existing.previous_content:
                previous_content = existing.previous_content
            elif preserve_previous:
                previous_content = existing.content

            # Preserve original change type if it was ADD (for new files)
            # Only change to UPDATE if it was already UPDATE or if preserve_change_type is False
            original_change_type = existing.change_type
            if preserve_change_type and original_change_type == ChangeType.ADD:
                # Keep as ADD for new files
                new_change_type = ChangeType.ADD
                logger.debug(
                    f"CodeChangesManager._apply_update: Preserving ADD change type for new file '{file_path}'"
                )
            else:
                # Change to UPDATE for existing files or when preserve_change_type is False
                new_change_type = ChangeType.UPDATE

            # Update the change
            existing.content = new_content
            existing.change_type = new_change_type
            existing.updated_at = datetime.now().isoformat()
            if description:
                existing.description = description
            if previous_content:
                existing.previous_content = previous_content
            self._persist_change()
        else:
            # New file in changes (not yet committed)
            # Use original_content as previous_content if provided (this is the original file content before any modifications)
            # Default to UPDATE, but caller should use add_file() for new files
            change = FileChange(
                file_path=file_path,
                change_type=ChangeType.UPDATE,
                content=new_content,
                previous_content=(
                    original_content
                    if original_content is not None
                    else previous_content
                ),
                description=description,
            )
            self._changes_cache[file_path] = change
            self._persist_change()
            logger.debug(
                f"CodeChangesManager._apply_update: Created new FileChange for '{file_path}' "
                f"with previous_content={'set' if original_content is not None else 'None'}"
            )
        return True

    def update_file(
        self,
        file_path: str,
        content: str,
        description: Optional[str] = None,
        preserve_previous: bool = True,
    ) -> bool:
        """Update an existing file with full content"""
        logger.info(
            f"CodeChangesManager.update_file: Updating file '{file_path}' with full content (content length: {len(content)} chars)"
        )
        result = self._apply_update(file_path, content, description, preserve_previous)
        if not result:
            logger.warning(
                f"CodeChangesManager.update_file: Failed to update file '{file_path}' - file may be marked for deletion"
            )
            return False
        logger.info(
            f"CodeChangesManager.update_file: Successfully updated file '{file_path}'"
        )
        return result

    def update_file_lines(
        self,
        file_path: str,
        start_line: int,
        end_line: Optional[int] = None,
        new_content: str = "",
        description: Optional[str] = None,
        project_id: Optional[str] = None,
        db: Optional[Session] = None,
    ) -> Dict[str, Any]:
        """
        Update specific lines in a file (1-indexed)

        Args:
            file_path: Path to the file
            start_line: Starting line number (1-indexed, inclusive)
            end_line: Ending line number (1-indexed, inclusive). If None, only start_line is replaced
            new_content: Content to replace the lines with
            description: Optional description of the change

        Returns:
            Dict with success status and information about the change
        """
        logger.info(
            f"CodeChangesManager.update_file_lines: Updating lines {start_line}-{end_line or start_line} in '{file_path}' "
            f"(project_id={project_id}, db={'provided' if db else 'None'})"
        )
        try:
            current_content = self._get_current_content(
                file_path, project_id=project_id, db=db
            )
            lines = current_content.split("\n")
            logger.info(
                f"CodeChangesManager.update_file_lines: Retrieved content for '{file_path}' - "
                f"{len(lines)} lines, content length: {len(current_content)} chars"
            )

            if len(lines) == 1 and lines[0] == "":
                logger.warning(
                    f"CodeChangesManager.update_file_lines: ⚠️ WARNING - File '{file_path}' appears to be empty or not found. "
                    f"This may indicate the file doesn't exist in the repository. "
                    f"Operation will proceed, creating/updating the file with new content."
                )

            # Validate line numbers (1-indexed)
            logger.debug(
                f"CodeChangesManager.update_file_lines: Validating start_line={start_line} against file with {len(lines)} lines"
            )
            if start_line < 1 or start_line > len(lines):
                error_msg = (
                    f"Invalid start_line {start_line}. File has {len(lines)} lines. "
                    f"⚠️ NOTE: If you've performed insert_lines or delete_lines operations on this file, "
                    f"the line numbers have shifted. Please fetch the current file state to get the correct line numbers."
                )
                return {
                    "success": False,
                    "error": error_msg,
                }

            if end_line is None:
                end_line = start_line

            if end_line < start_line:
                return {
                    "success": False,
                    "error": f"end_line ({end_line}) must be >= start_line ({start_line})",
                }

            if end_line > len(lines):
                error_msg = (
                    f"Invalid end_line {end_line}. File has {len(lines)} lines. "
                    f"⚠️ NOTE: If you've performed insert_lines or delete_lines operations on this file, "
                    f"the line numbers have shifted. Please fetch the current file state to get the correct line numbers."
                )
                return {
                    "success": False,
                    "error": error_msg,
                }

            # Convert to 0-indexed for list operations
            start_idx = start_line - 1
            end_idx = end_line  # exclusive for slicing

            # Get the lines being replaced (for context)
            replaced_lines = lines[start_idx:end_idx]
            replaced_content = "\n".join(replaced_lines)

            # Split new_content into lines
            new_lines = new_content.split("\n")

            # Replace the lines
            updated_lines = lines[:start_idx] + new_lines + lines[end_idx:]
            updated_content = "\n".join(updated_lines)

            # Apply the update
            # Pass current_content as original_content if file is not in changes yet
            # (this preserves the original file content for diff generation)
            change_desc = description or f"Updated lines {start_line}-{end_line}"
            original_content = (
                current_content if file_path not in self.changes else None
            )
            update_success = self._apply_update(
                file_path,
                updated_content,
                change_desc,
                original_content=original_content,
            )
            if not update_success:
                return {
                    "success": False,
                    "error": f"Cannot update file '{file_path}': file is marked for deletion. Use delete_file to unmark deletion first, or clear the file from changes.",
                }

            # Calculate context around updated area
            context_lines_before = 3
            context_lines_after = 3

            # Get context before
            context_start_idx = max(0, start_idx - context_lines_before)
            context_lines_before_list = lines[context_start_idx:start_idx]

            # Get context after (in the updated file)
            context_after_start_idx = start_idx + len(new_lines)
            context_after_end_idx = min(
                len(updated_lines), context_after_start_idx + context_lines_after
            )
            context_lines_after_list = updated_lines[
                context_after_start_idx:context_after_end_idx
            ]

            # Get the updated section with context
            context_with_updated = (
                context_lines_before_list + new_lines + context_lines_after_list
            )

            # Calculate line numbers for context display
            context_start_line = context_start_idx + 1
            context_end_line = context_after_end_idx

            logger.info(
                f"CodeChangesManager.update_file_lines: Successfully updated lines {start_line}-{end_line} in '{file_path}' (replaced {len(replaced_lines)} lines with {len(new_lines)} new lines)"
            )
            return {
                "success": True,
                "file_path": file_path,
                "start_line": start_line,
                "end_line": end_line,
                "lines_replaced": len(replaced_lines),
                "lines_added": len(new_lines),
                "replaced_content": replaced_content,
                "updated_context": "\n".join(context_with_updated),
                "context_start_line": context_start_line,
                "context_end_line": context_end_line,
            }
        except Exception as e:
            logger.exception(
                "CodeChangesManager.update_file_lines: Error updating lines",
                file_path=file_path,
            )
            return {"success": False, "error": str(e)}

    def replace_in_file(
        self,
        file_path: str,
        pattern: str,
        replacement: str,
        count: int = 0,
        description: Optional[str] = None,
        case_sensitive: bool = False,
        word_boundary: bool = False,
        project_id: Optional[str] = None,
        db: Optional[Session] = None,
    ) -> Dict[str, Any]:
        """
        Replace pattern matches in a file using regex

        Args:
            file_path: Path to the file
            pattern: Regex pattern to search for
            replacement: Replacement string (supports \\1, \\2, etc. for groups)
            count: Maximum number of replacements (0 = all)
            description: Optional description of the change
            case_sensitive: Whether pattern matching is case-sensitive
            word_boundary: If True, wrap pattern with word boundaries (\\b) to match whole words only
            project_id: Optional project ID to fetch file content from repository
            db: Optional database session for repository access

        Returns:
            Dict with success status and replacement information
        """
        logger.info(
            f"CodeChangesManager.replace_in_file: Replacing pattern '{pattern}' in '{file_path}' "
            f"(count={count}, case_sensitive={case_sensitive}, word_boundary={word_boundary}, project_id={project_id}, db={'provided' if db else 'None'})"
        )
        try:
            current_content = self._get_current_content(
                file_path, project_id=project_id, db=db
            )

            # Apply word boundary if requested
            actual_pattern = pattern
            if word_boundary:
                # Wrap pattern with word boundaries to match whole words only
                actual_pattern = r"\b" + pattern + r"\b"
                logger.debug(
                    f"CodeChangesManager.replace_in_file: Applied word boundary, pattern is now: '{actual_pattern}'"
                )

            # Compile regex pattern
            flags = 0 if case_sensitive else re.IGNORECASE
            try:
                regex = re.compile(actual_pattern, flags)
            except re.error as e:
                return {"success": False, "error": f"Invalid regex pattern: {str(e)}"}

            # Find all matches first (for reporting)
            matches = list(regex.finditer(current_content))
            if not matches:
                return {
                    "success": False,
                    "error": f"Pattern '{pattern}' not found in file",
                }

            # Perform replacement
            if count == 0:
                # Replace all
                new_content = regex.sub(replacement, current_content)
                replace_count = len(matches)
            else:
                # Replace first N matches
                new_content = regex.sub(replacement, current_content, count=count)
                replace_count = min(count, len(matches))

            # Apply the update
            # Pass current_content as original_content if file is not in changes yet
            # (this preserves the original file content for diff generation)
            change_desc = (
                description
                or f"Replaced '{pattern}' with '{replacement}' ({replace_count} occurrences)"
            )
            original_content = (
                current_content if file_path not in self.changes else None
            )
            update_success = self._apply_update(
                file_path, new_content, change_desc, original_content=original_content
            )
            if not update_success:
                return {
                    "success": False,
                    "error": f"Cannot replace in file '{file_path}': file is marked for deletion. Use delete_file to unmark deletion first, or clear the file from changes.",
                }

            # Get match locations for reporting
            match_locations = []
            for match in matches[:replace_count]:
                # Calculate line number
                line_num = current_content[: match.start()].count("\n") + 1
                match_locations.append(
                    {
                        "line": line_num,
                        "match": match.group(0)[:100],  # First 100 chars
                        "position": match.start(),
                    }
                )

            logger.info(
                f"CodeChangesManager.replace_in_file: Successfully replaced {replace_count} occurrence(s) of pattern '{pattern}' in '{file_path}'"
            )
            return {
                "success": True,
                "file_path": file_path,
                "pattern": pattern,
                "replacement": replacement,
                "replacements_made": replace_count,
                "total_matches": len(matches),
                "match_locations": match_locations,
            }
        except Exception as e:
            logger.exception(
                "CodeChangesManager.replace_in_file: Error replacing pattern",
                file_path=file_path,
            )
            return {"success": False, "error": str(e)}

    def insert_lines(
        self,
        file_path: str,
        line_number: int,
        content: str,
        description: Optional[str] = None,
        insert_after: bool = True,
        project_id: Optional[str] = None,
        db: Optional[Session] = None,
    ) -> Dict[str, Any]:
        """
        Insert content at a specific line in a file

        Args:
            file_path: Path to the file
            line_number: Line number to insert at (1-indexed)
            content: Content to insert
            description: Optional description of the change
            insert_after: If True, insert after line_number; if False, insert before

        Returns:
            Dict with success status and insertion information
        """
        position = "after" if insert_after else "before"
        logger.info(
            f"CodeChangesManager.insert_lines: Inserting {len(content.split(chr(10)))} line(s) {position} line {line_number} in '{file_path}' "
            f"(project_id={project_id}, db={'provided' if db else 'None'})"
        )
        try:
            current_content = self._get_current_content(
                file_path, project_id=project_id, db=db
            )
            lines = current_content.split("\n")
            logger.info(
                f"CodeChangesManager.insert_lines: Retrieved content for '{file_path}' - "
                f"{len(lines)} lines, content length: {len(current_content)} chars"
            )

            if len(lines) == 1 and lines[0] == "":
                logger.warning(
                    f"CodeChangesManager.insert_lines: WARNING - File '{file_path}' appears to be empty or only contains empty string!"
                )

            # Validate line number
            logger.debug(
                f"CodeChangesManager.insert_lines: Validating line_number={line_number} against file with {len(lines)} lines"
            )
            if line_number < 1:
                return {"success": False, "error": "line_number must be >= 1"}

            # Calculate max valid line number based on insert mode
            if insert_after:
                # Can insert after line 1 through line (len(lines) + 1)
                max_valid_line = len(lines) + 1
            else:
                # Can insert before line 1 through line len(lines)
                max_valid_line = len(lines)

            if line_number > max_valid_line:
                return {
                    "success": False,
                    "error": f"line_number {line_number} is beyond end of file ({len(lines)} lines). "
                    f"Valid range: 1 to {max_valid_line} when inserting {'after' if insert_after else 'before'}.",
                }

            # Split content into lines
            new_lines = content.split("\n")

            # Convert to 0-indexed insertion index
            # When insert_after=True: insert after line N means insert at index N (0-indexed)
            # When insert_after=False: insert before line N means insert at index N-1 (0-indexed)
            if insert_after:
                # Clamp to valid range: can insert at indices 0 through len(lines)
                insert_idx = min(line_number, len(lines))
            else:
                # Clamp to valid range: can insert at indices 0 through len(lines)-1
                insert_idx = max(0, min(line_number - 1, len(lines) - 1))

            # Insert the lines
            updated_lines = lines[:insert_idx] + new_lines + lines[insert_idx:]
            updated_content = "\n".join(updated_lines)

            # Apply the update
            # Pass current_content as original_content if file is not in changes yet
            # (this preserves the original file content for diff generation)
            change_desc = (
                description
                or f"Inserted {len(new_lines)} lines {position} line {line_number}"
            )
            original_content = (
                current_content if file_path not in self.changes else None
            )
            update_success = self._apply_update(
                file_path,
                updated_content,
                change_desc,
                original_content=original_content,
            )
            if not update_success:
                return {
                    "success": False,
                    "error": f"Cannot insert lines in file '{file_path}': file is marked for deletion. Use delete_file to unmark deletion first, or clear the file from changes.",
                }

            # Calculate context around inserted area
            context_lines_before = 3
            context_lines_after = 3

            # Get context before (from original file)
            context_start_idx = max(0, insert_idx - context_lines_before)
            context_lines_before_list = lines[context_start_idx:insert_idx]

            # Get context after (in the updated file)
            context_after_start_idx = insert_idx + len(new_lines)
            context_after_end_idx = min(
                len(updated_lines), context_after_start_idx + context_lines_after
            )
            context_lines_after_list = updated_lines[
                context_after_start_idx:context_after_end_idx
            ]

            # Get the inserted section with context
            context_with_inserted = (
                context_lines_before_list + new_lines + context_lines_after_list
            )

            # Calculate line numbers for context display
            context_start_line = context_start_idx + 1
            context_end_line = context_after_end_idx

            logger.info(
                f"CodeChangesManager.insert_lines: Successfully inserted {len(new_lines)} line(s) {position} line {line_number} in '{file_path}'"
            )
            return {
                "success": True,
                "file_path": file_path,
                "line_number": line_number,
                "position": position,
                "lines_inserted": len(new_lines),
                "inserted_context": "\n".join(context_with_inserted),
                "context_start_line": context_start_line,
                "context_end_line": context_end_line,
            }
        except Exception as e:
            logger.exception(
                "CodeChangesManager.insert_lines: Error inserting lines",
                file_path=file_path,
            )
            return {"success": False, "error": str(e)}

    def delete_lines(
        self,
        file_path: str,
        start_line: int,
        end_line: Optional[int] = None,
        description: Optional[str] = None,
        project_id: Optional[str] = None,
        db: Optional[Session] = None,
    ) -> Dict[str, Any]:
        """
        Delete specific lines from a file (1-indexed)

        Args:
            file_path: Path to the file
            start_line: Starting line number (1-indexed, inclusive)
            end_line: Ending line number (1-indexed, inclusive). If None, only start_line is deleted
            description: Optional description of the change
            project_id: Optional project ID to fetch file content from repository
            db: Optional database session for repository access

        Returns:
            Dict with success status and deletion information
        """
        logger.info(
            f"CodeChangesManager.delete_lines: Deleting lines {start_line}-{end_line or start_line} from '{file_path}' "
            f"(project_id={project_id}, db={'provided' if db else 'None'})"
        )
        try:
            current_content = self._get_current_content(
                file_path, project_id=project_id, db=db
            )
            lines = current_content.split("\n")

            # Warn if file appears to be empty or non-existent
            if len(lines) == 1 and lines[0] == "":
                logger.warning(
                    f"CodeChangesManager.delete_lines: ⚠️ WARNING - File '{file_path}' appears to be empty or not found. "
                    f"This may indicate the file doesn't exist in the repository."
                )

            # Validate line numbers
            if start_line < 1 or start_line > len(lines):
                return {
                    "success": False,
                    "error": f"Invalid start_line {start_line}. File has {len(lines)} lines.",
                }

            if end_line is None:
                end_line = start_line

            if end_line < start_line:
                return {
                    "success": False,
                    "error": f"end_line ({end_line}) must be >= start_line ({start_line})",
                }

            if end_line > len(lines):
                error_msg = (
                    f"Invalid end_line {end_line}. File has {len(lines)} lines. "
                    f"⚠️ NOTE: If you've performed insert_lines or delete_lines operations on this file, "
                    f"the line numbers have shifted. Please fetch the current file state to get the correct line numbers."
                )
                return {
                    "success": False,
                    "error": error_msg,
                }

            # Convert to 0-indexed
            start_idx = start_line - 1
            end_idx = end_line  # exclusive for slicing

            # Get the lines being deleted (for reporting)
            deleted_lines = lines[start_idx:end_idx]
            deleted_content = "\n".join(deleted_lines)

            # Delete the lines
            updated_lines = lines[:start_idx] + lines[end_idx:]
            updated_content = "\n".join(updated_lines)

            # Apply the update
            # Pass current_content as original_content if file is not in changes yet
            # (this preserves the original file content for diff generation)
            change_desc = description or f"Deleted lines {start_line}-{end_line}"
            original_content = (
                current_content if file_path not in self.changes else None
            )
            update_success = self._apply_update(
                file_path,
                updated_content,
                change_desc,
                original_content=original_content,
            )
            if not update_success:
                return {
                    "success": False,
                    "error": f"Cannot delete lines from file '{file_path}': file is marked for deletion. Use delete_file to unmark deletion first, or clear the file from changes.",
                }

            logger.info(
                f"CodeChangesManager.delete_lines: Successfully deleted {len(deleted_lines)} line(s) ({start_line}-{end_line}) from '{file_path}'"
            )
            return {
                "success": True,
                "file_path": file_path,
                "start_line": start_line,
                "end_line": end_line,
                "lines_deleted": len(deleted_lines),
                "deleted_content": deleted_content,
            }
        except Exception as e:
            logger.exception(
                "CodeChangesManager.delete_lines: Error deleting lines",
                file_path=file_path,
            )
            return {"success": False, "error": str(e)}

    def delete_file(
        self,
        file_path: str,
        description: Optional[str] = None,
        preserve_content: bool = True,
    ) -> bool:
        """Mark a file for deletion"""
        logger.info(
            f"CodeChangesManager.delete_file: Marking file '{file_path}' for deletion (preserve_content={preserve_content})"
        )
        previous_content = None
        # Access self.changes to ensure we load from Redis
        changes = self.changes
        if file_path in changes:
            existing = changes[file_path]
            if preserve_content and existing.content:
                previous_content = existing.content
            existing.change_type = ChangeType.DELETE
            existing.content = None  # Clear content for deleted files
            existing.updated_at = datetime.now().isoformat()
            if description:
                existing.description = description
            if previous_content:
                existing.previous_content = previous_content
            self._persist_change()
        else:
            # New deletion record
            change = FileChange(
                file_path=file_path,
                change_type=ChangeType.DELETE,
                content=None,
                previous_content=previous_content,
                description=description,
            )
            self._changes_cache[file_path] = change
            self._persist_change()
        logger.info(
            f"CodeChangesManager.delete_file: Successfully marked file '{file_path}' for deletion"
        )
        return True

    def get_file(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Get change information for a specific file"""
        logger.debug(f"CodeChangesManager.get_file: Retrieving file '{file_path}'")
        if file_path not in self.changes:
            logger.debug(
                f"CodeChangesManager.get_file: File '{file_path}' not found in changes"
            )
            return None

        change = self.changes[file_path]
        result = asdict(change)
        # Convert enum to string for serialization
        result["change_type"] = change.change_type.value
        return result

    def list_files(
        self,
        change_type_filter: Optional[ChangeType] = None,
        path_pattern: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List all files with changes, optionally filtered"""
        logger.debug(
            f"CodeChangesManager.list_files: Listing files (filter: {change_type_filter}, pattern: {path_pattern})"
        )
        files = list(self.changes.values())

        # Filter by change type
        if change_type_filter:
            files = [f for f in files if f.change_type == change_type_filter]

        # Filter by path pattern (supports regex)
        if path_pattern:
            try:
                pattern = re.compile(path_pattern, re.IGNORECASE)
                files = [f for f in files if pattern.search(f.file_path)]
            except re.error:
                # If regex is invalid, fall back to simple substring match
                files = [
                    f for f in files if path_pattern.lower() in f.file_path.lower()
                ]

        # Sort by file path
        files.sort(key=lambda x: x.file_path)

        # Convert to dict format, ensuring change_type is correctly set
        result = []
        for f in files:
            file_dict = asdict(f)
            # Ensure change_type is the string value, not the enum
            file_dict["change_type"] = f.change_type.value
            result.append(file_dict)
        return result

    def search_content(
        self,
        pattern: str,
        file_pattern: Optional[str] = None,
        case_sensitive: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Search for pattern in file contents (grep-like functionality)

        Args:
            pattern: Regex pattern to search for
            file_pattern: Optional regex to filter files by path
            case_sensitive: Whether search is case-sensitive

        Returns:
            List of matches with file_path, line_number, and matched line
        """
        logger.info(
            f"CodeChangesManager.search_content: Searching for pattern '{pattern}' (file_pattern: {file_pattern}, case_sensitive: {case_sensitive})"
        )
        matches = []
        flags = 0 if case_sensitive else re.IGNORECASE

        try:
            content_regex = re.compile(pattern, flags)
            file_regex = (
                re.compile(file_pattern, re.IGNORECASE) if file_pattern else None
            )
        except re.error as e:
            return [{"error": f"Invalid regex pattern: {str(e)}"}]

        for file_path, change in self.changes.items():
            # Skip deleted files
            if change.change_type == ChangeType.DELETE or not change.content:
                continue

            # Filter by file pattern
            if file_regex and not file_regex.search(file_path):
                continue

            # Search in content
            lines = change.content.split("\n")
            for line_num, line in enumerate(lines, start=1):
                if content_regex.search(line):
                    matches.append(
                        {
                            "file_path": file_path,
                            "line_number": line_num,
                            "line": line.strip(),
                            "change_type": change.change_type.value,
                        }
                    )

        logger.info(
            f"CodeChangesManager.search_content: Found {len(matches)} matches across files"
        )
        return matches

    def clear_file(self, file_path: str) -> bool:
        """Clear changes for a specific file"""
        logger.info(
            f"CodeChangesManager.clear_file: Clearing changes for file '{file_path}'"
        )
        # Access self.changes to ensure we load from Redis
        changes = self.changes
        if file_path in changes:
            del self._changes_cache[file_path]
            self._persist_change()
            logger.info(
                f"CodeChangesManager.clear_file: Successfully cleared changes for file '{file_path}'"
            )
            return True
        logger.warning(
            f"CodeChangesManager.clear_file: File '{file_path}' not found in changes"
        )
        return False

    def clear_all(self) -> int:
        """Clear all changes and return count of cleared files"""
        # Access self.changes to ensure we load from Redis
        changes = self.changes
        count = len(changes)
        logger.info(
            f"CodeChangesManager.clear_all: Clearing all changes ({count} files) from session {self.session_id}"
        )
        self._changes_cache.clear()
        self._persist_change()
        logger.info(
            f"CodeChangesManager.clear_all: Successfully cleared all {count} file(s)"
        )
        return count

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all changes"""
        logger.debug(
            f"CodeChangesManager.get_summary: Generating summary for session {self.session_id}"
        )
        change_counts = {ct.value: 0 for ct in ChangeType}
        for change in self.changes.values():
            change_counts[change.change_type.value] += 1

        return {
            "session_id": self.session_id,
            "total_files": len(self.changes),
            "change_counts": change_counts,
            "files": [
                {
                    "file_path": change.file_path,
                    "change_type": change.change_type.value,
                    "description": change.description,
                    "updated_at": change.updated_at,
                }
                for change in sorted(self.changes.values(), key=lambda x: x.file_path)
            ],
        }

    def serialize(self) -> str:
        """Serialize all changes to JSON string for persistence"""
        logger.info(
            f"CodeChangesManager.serialize: Serializing {len(self.changes)} file changes to JSON"
        )
        data = {
            "session_id": self.session_id,
            "changes": [
                {
                    "file_path": change.file_path,
                    "change_type": change.change_type.value,
                    "content": change.content,
                    "previous_content": change.previous_content,
                    "created_at": change.created_at,
                    "updated_at": change.updated_at,
                    "description": change.description,
                }
                for change in self.changes.values()
            ],
        }
        return json.dumps(data, indent=2)

    def deserialize(self, json_str: str) -> bool:
        """Deserialize changes from JSON string"""
        logger.info(
            f"CodeChangesManager.deserialize: Deserializing changes from JSON (length: {len(json_str)} chars)"
        )
        try:
            data = json.loads(json_str)
            self.session_id = data.get("session_id", str(uuid.uuid4())[:8])
            # Initialize empty cache
            self._changes_cache = {}

            for change_data in data.get("changes", []):
                change = FileChange(
                    file_path=change_data["file_path"],
                    change_type=ChangeType(change_data["change_type"]),
                    content=change_data.get("content"),
                    previous_content=change_data.get("previous_content"),
                    created_at=change_data.get(
                        "created_at", datetime.now().isoformat()
                    ),
                    updated_at=change_data.get(
                        "updated_at", datetime.now().isoformat()
                    ),
                    description=change_data.get("description"),
                )
                self._changes_cache[change.file_path] = change
            self._persist_change()
            logger.info(
                f"CodeChangesManager.deserialize: Successfully deserialized {len(self._changes_cache)} file changes"
            )
            return True
        except (json.JSONDecodeError, KeyError, ValueError):
            logger.exception("CodeChangesManager.deserialize: Error deserializing JSON")
            return False

    def _read_file_from_codebase(self, file_path: str) -> Optional[str]:
        """
        Read file content from the codebase filesystem

        Args:
            file_path: Relative path to the file

        Returns:
            File content as string, or None if file doesn't exist or is too large
        """
        try:
            # Try relative to current working directory
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                if file_size > MAX_FILE_SIZE_BYTES:
                    logger.warning(
                        f"CodeChangesManager._read_file_from_codebase: File '{file_path}' "
                        f"is too large ({file_size} bytes, max {MAX_FILE_SIZE_BYTES} bytes). "
                        f"Skipping to prevent memory issues."
                    )
                    return None
                with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                    return f.read()

            # Try relative to workspace root (common patterns)
            workspace_root = os.getcwd()
            possible_paths = [
                file_path,
                os.path.join(workspace_root, file_path),
                os.path.join(workspace_root, "app", file_path),
            ]

            for path in possible_paths:
                if os.path.exists(path):
                    file_size = os.path.getsize(path)
                    if file_size > MAX_FILE_SIZE_BYTES:
                        logger.warning(
                            f"CodeChangesManager._read_file_from_codebase: File '{file_path}' "
                            f"(found at {path}) is too large ({file_size} bytes, max {MAX_FILE_SIZE_BYTES} bytes). "
                            f"Skipping to prevent memory issues."
                        )
                        return None
                    with open(path, "r", encoding="utf-8", errors="replace") as f:
                        logger.debug(
                            f"CodeChangesManager._read_file_from_codebase: Found file at {path}"
                        )
                        return f.read()

            logger.debug(
                f"CodeChangesManager._read_file_from_codebase: File '{file_path}' not found in codebase"
            )
            return None
        except MemoryError as e:
            logger.error(
                f"CodeChangesManager._read_file_from_codebase: Memory error reading '{file_path}': {str(e)}. "
                f"This may indicate the file is too large or system memory is low."
            )
            return None
        except Exception as e:
            logger.warning(
                f"CodeChangesManager._read_file_from_codebase: Error reading '{file_path}': {str(e)}"
            )
            return None

    def generate_diff(
        self,
        file_path: Optional[str] = None,
        context_lines: int = 3,
        project_id: Optional[str] = None,
        db: Optional[Session] = None,
    ) -> Dict[str, str]:
        """
        Generate unified diff between managed changes and repository/base files

        Args:
            file_path: Optional specific file path. If None, generate diffs for all files
            context_lines: Number of context lines to include in diff
            project_id: Optional project ID to fetch original content from repository
            db: Optional database session for repository access

        Returns:
            Dict with file_paths as keys and diff strings as values
        """
        logger.info(
            f"CodeChangesManager.generate_diff: Generating diff(s) (file_path: {file_path}, context_lines: {context_lines}, project_id: {project_id})"
        )
        diffs = {}

        files_to_diff = [file_path] if file_path else list(self.changes.keys())

        for fp in files_to_diff:
            if fp not in self.changes:
                logger.warning(
                    f"CodeChangesManager.generate_diff: File '{fp}' not in changes"
                )
                continue

            change = self.changes[fp]

            # For deleted files, show diff of what was removed
            if change.change_type == ChangeType.DELETE:
                old_content = change.previous_content or ""
                new_content = ""
                diff = self._create_unified_diff(
                    old_content, new_content, fp, fp, context_lines
                )
                diffs[fp] = diff
                continue

            # For added files, show entire file as additions
            if change.change_type == ChangeType.ADD:
                old_content = ""
                new_content = change.content or ""
                diff = self._create_unified_diff(
                    old_content, new_content, "/dev/null", fp, context_lines
                )
                diffs[fp] = diff
                continue

            # For updated files, compare with previous_content if available, otherwise repository/base
            new_content = change.content or ""

            # Use previous_content if available, otherwise try repository, then filesystem
            if change.previous_content is not None:
                old_content = change.previous_content
                diff = self._create_unified_diff(
                    old_content, new_content, fp, fp, context_lines
                )
            else:
                # Try to get from repository first if project_id/db provided
                old_content = None
                if project_id and db:
                    # Fetch directly from repository, bypassing changes
                    try:
                        from app.modules.code_provider.code_provider_service import (
                            CodeProviderService,
                        )
                        from app.modules.code_provider.git_safe import (
                            safe_git_operation,
                            GitOperationError,
                        )
                        from app.modules.projects.projects_model import Project

                        project = (
                            db.query(Project).filter(Project.id == project_id).first()
                        )
                        if project:
                            cp_service = CodeProviderService(db)

                            def _fetch_repo_content():
                                return cp_service.get_file_content(
                                    repo_name=project.repo_name,
                                    file_path=fp,
                                    branch_name=project.branch_name,
                                    start_line=None,
                                    end_line=None,
                                    project_id=project_id,
                                    commit_id=project.commit_id,
                                )

                            try:
                                repo_content = safe_git_operation(
                                    _fetch_repo_content,
                                    max_retries=1,
                                    timeout=20.0,
                                    max_total_timeout=25.0,
                                    operation_name=f"generate_diff_get_content({fp})",
                                )
                            except GitOperationError as git_error:
                                logger.warning(
                                    f"CodeChangesManager.generate_diff: Git operation timed out for {fp}: {git_error}"
                                )
                                repo_content = None

                            if repo_content:
                                old_content = repo_content
                    except Exception as e:
                        logger.warning(
                            f"CodeChangesManager.generate_diff: Error fetching from repository: {e}"
                        )
                        old_content = None

                # Fallback to filesystem if repository fetch failed
                if old_content is None:
                    old_content = self._read_file_from_codebase(fp)

                # If file doesn't exist anywhere, treat as new file
                if old_content is None or old_content == "":
                    old_content = ""
                    diff = self._create_unified_diff(
                        old_content, new_content, "/dev/null", fp, context_lines
                    )
                else:
                    diff = self._create_unified_diff(
                        old_content, new_content, fp, fp, context_lines
                    )

            diffs[fp] = diff

        logger.info(f"CodeChangesManager.generate_diff: Generated {len(diffs)} diff(s)")
        return diffs

    def _create_unified_diff(
        self,
        old_content: str,
        new_content: str,
        old_path: str,
        new_path: str,
        context_lines: int,
    ) -> str:
        """Create a unified diff string from old and new content"""
        old_lines = old_content.splitlines(keepends=True)
        new_lines = new_content.splitlines(keepends=True)

        # Generate unified diff
        # Use lineterm="\n" to ensure proper newlines between diff header lines
        diff = difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=old_path,
            tofile=new_path,
            lineterm="\n",
            n=context_lines,
        )

        # Join with newlines to ensure proper formatting
        # The diff lines already have newlines from lineterm="\n"
        return "".join(diff)

    def _generate_git_diff_patch(
        self,
        file_path: str,
        old_content: str,
        new_content: str,
        context_lines: int = 3,
    ) -> str:
        """
        Generate a git-style diff patch for a single file

        Args:
            file_path: Path to the file
            old_content: Original file content
            new_content: New file content
            context_lines: Number of context lines to include in diff

        Returns:
            Git-style diff patch string
        """
        old_lines = old_content.splitlines(keepends=True) if old_content else []
        new_lines = new_content.splitlines(keepends=True) if new_content else []

        # Generate unified diff
        # Use lineterm="\n" to ensure proper newlines between diff header lines
        diff_lines = list(
            difflib.unified_diff(
                old_lines,
                new_lines,
                fromfile=f"a/{file_path}",
                tofile=f"b/{file_path}",
                lineterm="\n",
                n=context_lines,
            )
        )

        if not diff_lines:
            return ""

        # Convert to git diff format
        # Start with git diff header
        git_diff = f"diff --git a/{file_path} b/{file_path}\n"

        # The unified_diff output format is:
        # --- a/file
        # +++ b/file
        # @@ -old_start,old_count +new_start,new_count @@
        # ...
        # We keep the --- and +++ lines as-is, and add the rest
        # The diff lines already have newlines from lineterm="\n"
        git_diff += "".join(diff_lines)

        return git_diff

    def export_changes(
        self,
        format: str = "dict",
        project_id: Optional[str] = None,
        db: Optional[Session] = None,
    ) -> Union[Dict[str, str], List[Dict[str, Any]], str]:
        """
        Export changes in different formats

        Args:
            format: 'dict' (file_path -> content), 'list' (list of change dicts),
                   'json' (JSON string), 'diff' (git-style diff patch)
            project_id: Optional project ID for fetching original content from repository (for diff format)
            db: Optional database session for repository access (for diff format)
        """
        logger.info(
            f"CodeChangesManager.export_changes: Exporting {len(self.changes)} file changes in '{format}' format"
        )
        if format == "dict":
            return {
                file_path: change.content or ""
                for file_path, change in self.changes.items()
                if change.change_type != ChangeType.DELETE and change.content
            }
        elif format == "list":
            return [
                {
                    "file_path": change.file_path,
                    "change_type": change.change_type.value,
                    "content": change.content,
                    "description": change.description,
                }
                for change in self.changes.values()
            ]
        elif format == "json":
            return self.serialize()
        elif format == "diff":
            # Generate git-style diff patches for all files
            patches = []
            for file_path, change in sorted(self.changes.items()):
                if change.change_type == ChangeType.DELETE:
                    old_content = change.previous_content or ""
                    new_content = ""
                elif change.change_type == ChangeType.ADD:
                    old_content = ""
                    new_content = change.content or ""
                else:  # UPDATE
                    new_content = change.content or ""
                    # Try to get old content from previous_content, repository, or filesystem
                    if change.previous_content is not None:
                        old_content = change.previous_content
                    elif project_id and db:
                        # Try repository with timeout protection
                        try:
                            from app.modules.code_provider.code_provider_service import (
                                CodeProviderService,
                            )
                            from app.modules.code_provider.git_safe import (
                                safe_git_operation,
                                GitOperationError,
                            )
                            from app.modules.projects.projects_model import Project

                            project = (
                                db.query(Project)
                                .filter(Project.id == project_id)
                                .first()
                            )
                            if project:
                                cp_service = CodeProviderService(db)

                                def _fetch_content_for_format():
                                    return cp_service.get_file_content(
                                        repo_name=project.repo_name,
                                        file_path=file_path,
                                        branch_name=project.branch_name,
                                        start_line=None,
                                        end_line=None,
                                        project_id=project_id,
                                        commit_id=project.commit_id,
                                    )

                                try:
                                    repo_content = safe_git_operation(
                                        _fetch_content_for_format,
                                        max_retries=1,
                                        timeout=20.0,
                                        max_total_timeout=25.0,
                                        operation_name=f"format_diff_get_content({file_path})",
                                    )
                                except GitOperationError:
                                    repo_content = None

                                old_content = repo_content or ""
                            else:
                                old_content = ""
                        except Exception:
                            old_content = ""
                    else:
                        # Fallback to filesystem
                        old_content = self._read_file_from_codebase(file_path) or ""

                patch = self._generate_git_diff_patch(
                    file_path, old_content, new_content
                )
                if patch:
                    patches.append(patch)

            return "\n".join(patches)
        else:
            raise ValueError(
                f"Unknown format: {format}. Supported formats: 'dict', 'list', 'json', 'diff'"
            )


# Context variable for code changes manager - provides isolation per execution context
# This ensures parallel agent runs have separate, isolated state
_code_changes_manager_ctx: ContextVar[Optional[CodeChangesManager]] = ContextVar(
    "_code_changes_manager_ctx", default=None
)

# Context variable for conversation_id - used to persist changes across messages
_conversation_id_ctx: ContextVar[Optional[str]] = ContextVar(
    "_conversation_id_ctx", default=None
)


def _set_conversation_id(conversation_id: Optional[str]) -> None:
    """Set the conversation_id for the current execution context.

    This should be called at the start of each agent run to enable
    changes persistence across messages in the same conversation.
    """
    _conversation_id_ctx.set(conversation_id)
    logger.info(f"CodeChangesManager: Set conversation_id to {conversation_id}")


def _get_conversation_id() -> Optional[str]:
    """Get the conversation_id for the current execution context."""
    return _conversation_id_ctx.get()


def _get_code_changes_manager() -> CodeChangesManager:
    """Get the current code changes manager for this execution context, creating a new one if needed.

    Uses ContextVar to ensure each async execution context (agent run) has its own isolated instance.
    The manager uses Redis storage keyed by conversation_id for persistence across messages.
    """
    manager = _code_changes_manager_ctx.get()
    conversation_id = _get_conversation_id()

    # If we have a manager but conversation_id changed, create a new one
    if manager is not None and manager._conversation_id != conversation_id:
        logger.info(
            f"CodeChangesManager: conversation_id changed from {manager._conversation_id} to {conversation_id}, creating new manager"
        )
        manager = None

    if manager is None:
        logger.info(
            f"CodeChangesManager: Creating new manager instance for conversation_id={conversation_id}"
        )
        manager = CodeChangesManager(conversation_id=conversation_id)
        _code_changes_manager_ctx.set(manager)
        logger.info(
            f"CodeChangesManager: Created new manager with session_id={manager.session_id}, "
            f"redis_key={manager._redis_key}, existing_changes={len(manager.changes)}"
        )
    return manager


def _init_code_changes_manager(conversation_id: Optional[str] = None) -> None:
    """Initialize the code changes manager for a new agent run.

    This loads existing changes from Redis for the conversation (if any exist)
    instead of starting fresh. This ensures changes persist across messages.

    Args:
        conversation_id: The conversation ID to use for Redis key. If None,
                        uses a random session_id (backward compatible, no persistence).
    """
    _set_conversation_id(conversation_id)

    old_manager = _code_changes_manager_ctx.get()
    old_session = old_manager.session_id if old_manager else None
    old_count = len(old_manager.changes) if old_manager else 0
    logger.info(
        f"CodeChangesManager: Initializing manager for conversation_id={conversation_id} "
        f"(previous session: {old_session}, previous file count: {old_count})"
    )

    # Create manager - it will load existing changes from Redis automatically
    new_manager = CodeChangesManager(conversation_id=conversation_id)
    _code_changes_manager_ctx.set(new_manager)

    logger.info(
        f"CodeChangesManager: Initialized with session_id={new_manager.session_id}, "
        f"loaded {len(new_manager.changes)} existing changes from Redis"
    )


def _reset_code_changes_manager() -> None:
    """Reset the code changes manager for a new agent run.

    DEPRECATED: Use _init_code_changes_manager(conversation_id) instead.
    This function is kept for backward compatibility but now initializes
    with the current conversation_id context rather than creating a fresh instance.
    """
    conversation_id = _get_conversation_id()
    _init_code_changes_manager(conversation_id)


# Pydantic models for tool inputs
class AddFileInput(BaseModel):
    file_path: str = Field(description="Path to the file to add (e.g., 'src/main.py')")
    content: str = Field(description="Full content of the file to add")
    description: Optional[str] = Field(
        default=None, description="Optional description of what this file does"
    )


class UpdateFileInput(BaseModel):
    file_path: str = Field(description="Path to the file to update")
    content: str = Field(description="New content for the file")
    description: Optional[str] = Field(
        default=None, description="Optional description of the change"
    )
    preserve_previous: bool = Field(
        default=True,
        description="Whether to preserve previous content for reference",
    )


class DeleteFileInput(BaseModel):
    file_path: str = Field(description="Path to the file to delete")
    description: Optional[str] = Field(
        default=None, description="Optional reason for deletion"
    )
    preserve_content: bool = Field(
        default=True,
        description="Whether to preserve file content before deletion",
    )


class GetFileInput(BaseModel):
    file_path: str = Field(description="Path to the file to retrieve")


class ListFilesInput(BaseModel):
    change_type_filter: Optional[str] = Field(
        default=None,
        description="Filter by change type: 'add', 'update', or 'delete'",
    )
    path_pattern: Optional[str] = Field(
        default=None,
        description="Optional regex pattern to filter files by path (e.g., '.*\\.py$' for Python files)",
    )


class SearchContentInput(BaseModel):
    pattern: str = Field(
        description="Regex pattern to search for in file contents (grep-like search)"
    )
    file_pattern: Optional[str] = Field(
        default=None,
        description="Optional regex pattern to filter files by path before searching",
    )
    case_sensitive: bool = Field(
        default=False, description="Whether search should be case-sensitive"
    )


class ClearFileInput(BaseModel):
    file_path: str = Field(description="Path to the file to clear from changes")


class ExportChangesInput(BaseModel):
    format: str = Field(
        default="dict",
        description="Export format: 'dict' (file_path -> content), 'list' (list of changes), 'json' (JSON string), or 'diff' (git-style diff patch)",
    )
    project_id: Optional[str] = Field(
        default=None,
        description="Optional project ID to fetch original content from repository for accurate diff. Use project_id from conversation context.",
    )


class UpdateFileLinesInput(BaseModel):
    file_path: str = Field(description="Path to the file to update")
    start_line: int = Field(description="Starting line number (1-indexed, inclusive)")
    end_line: Optional[int] = Field(
        default=None,
        description="Ending line number (1-indexed, inclusive). If None, only start_line is replaced",
    )
    new_content: str = Field(description="Content to replace the lines with")
    description: Optional[str] = Field(
        default=None, description="Optional description of the change"
    )
    project_id: str = Field(
        ...,
        description="REQUIRED: Project ID (from context) to fetch file content from repository. "
        "Use the project_id from the conversation context. Without this, the tool cannot access existing file content.",
    )


class ReplaceInFileInput(BaseModel):
    file_path: str = Field(description="Path to the file to update")
    pattern: str = Field(
        description="Regex pattern to search for (supports capturing groups with \\1, \\2, etc. in replacement)"
    )
    replacement: str = Field(
        description="Replacement string (use \\1, \\2, etc. for captured groups)"
    )
    count: int = Field(
        default=0,
        description="Maximum number of replacements (0 = replace all occurrences)",
    )
    description: Optional[str] = Field(
        default=None, description="Optional description of the change"
    )
    case_sensitive: bool = Field(
        default=False, description="Whether pattern matching is case-sensitive"
    )
    word_boundary: bool = Field(
        default=False,
        description="If True, wrap pattern with word boundaries (\\b) to match whole words only. "
        "Useful to prevent matching partial strings within words (e.g., 'test function' won't match 'another_test function').",
    )
    project_id: Optional[str] = Field(
        default=None,
        description="Optional project ID (from context) to fetch file content from repository. "
        "Use the project_id from the conversation context for better content retrieval.",
    )


class InsertLinesInput(BaseModel):
    file_path: str = Field(description="Path to the file to update")
    line_number: int = Field(description="Line number to insert at (1-indexed)")
    content: str = Field(description="Content to insert")
    description: Optional[str] = Field(
        default=None, description="Optional description of the change"
    )
    insert_after: bool = Field(
        default=True,
        description="If True, insert after line_number; if False, insert before",
    )
    project_id: str = Field(
        ...,
        description="REQUIRED: Project ID (from context) to fetch file content from repository. "
        "Use the project_id from the conversation context. Without this, the tool cannot access existing file content.",
    )


class DeleteLinesInput(BaseModel):
    file_path: str = Field(description="Path to the file to update")
    start_line: int = Field(description="Starting line number (1-indexed, inclusive)")
    end_line: Optional[int] = Field(
        default=None,
        description="Ending line number (1-indexed, inclusive). If None, only start_line is deleted",
    )
    description: Optional[str] = Field(
        default=None, description="Optional description of the change"
    )
    project_id: Optional[str] = Field(
        default=None,
        description="Optional project ID (from context) to fetch file content from repository. "
        "Use the project_id from the conversation context for better content retrieval.",
    )


# Wrapper functions to convert kwargs to Pydantic models for PydanticAI Tool compatibility
def _wrap_tool(func, input_model):
    """
    Wrap a tool function to preserve signature for PydanticAI Tool introspection.
    PydanticAI Tool expects functions that can accept keyword arguments matching the model fields.
    We create a wrapper that preserves the function signature but accepts kwargs.
    """

    @functools.wraps(func)
    def wrapped_func(input_data: input_model) -> str:  # type: ignore
        return func(input_data)

    # Use functools.wraps but update the annotation to use the input_model
    # This preserves the signature for PydanticAI introspection
    sig = inspect.signature(func)
    param = list(sig.parameters.values())[0]
    new_param = param.replace(annotation=input_model)
    new_sig = sig.replace(parameters=[new_param])

    # Use functools.update_wrapper with signature preservation
    functools.update_wrapper(wrapped_func, func)
    wrapped_func.__annotations__ = {"input_data": input_model, "return": str}

    return wrapped_func


# Tool functions
def add_file_tool(input_data: AddFileInput) -> str:
    """Add a new file to the code changes manager"""
    logger.info(f"Tool add_file_tool: Adding file '{input_data.file_path}'")
    try:
        manager = _get_code_changes_manager()
        success = manager.add_file(
            file_path=input_data.file_path,
            content=input_data.content,
            description=input_data.description,
        )

        if success:
            summary = manager.get_summary()
            return f"✅ Added file '{input_data.file_path}'\n\nTotal files in changes: {summary['total_files']}"
        else:
            return f"❌ File '{input_data.file_path}' already exists in changes. Use update_file to modify it."
    except Exception:
        logger.exception(
            "Tool add_file_tool: Error adding file", file_path=input_data.file_path
        )
        return "❌ Error adding file"


def update_file_tool(input_data: UpdateFileInput) -> str:
    """Update a file in the code changes manager"""
    logger.info(f"Tool update_file_tool: Updating file '{input_data.file_path}'")
    try:
        manager = _get_code_changes_manager()
        success = manager.update_file(
            file_path=input_data.file_path,
            content=input_data.content,
            description=input_data.description,
            preserve_previous=input_data.preserve_previous,
        )

        if success:
            summary = manager.get_summary()
            return f"✅ Updated file '{input_data.file_path}'\n\nTotal files in changes: {summary['total_files']}"
        else:
            return f"❌ Error updating file '{input_data.file_path}'"
    except Exception:
        logger.exception(
            "Tool update_file_tool: Error updating file", file_path=input_data.file_path
        )
        return "❌ Error updating file"


def delete_file_tool(input_data: DeleteFileInput) -> str:
    """Mark a file for deletion in the code changes manager"""
    logger.info(
        f"Tool delete_file_tool: Marking file '{input_data.file_path}' for deletion"
    )
    try:
        manager = _get_code_changes_manager()
        success = manager.delete_file(
            file_path=input_data.file_path,
            description=input_data.description,
            preserve_content=input_data.preserve_content,
        )

        if success:
            summary = manager.get_summary()
            return f"✅ Marked file '{input_data.file_path}' for deletion\n\nTotal files in changes: {summary['total_files']}"
        else:
            return f"❌ Error deleting file '{input_data.file_path}'"
    except Exception:
        logger.exception(
            "Tool delete_file_tool: Error deleting file", file_path=input_data.file_path
        )
        return "❌ Error deleting file"


def get_file_tool(input_data: GetFileInput) -> str:
    """Get comprehensive change information and metadata for a specific file"""
    logger.info(f"Tool get_file_tool: Retrieving file '{input_data.file_path}'")
    try:
        manager = _get_code_changes_manager()
        file_data = manager.get_file(input_data.file_path)

        if file_data:
            change_emoji = {"add": "➕", "update": "✏️", "delete": "🗑️"}
            emoji = change_emoji.get(file_data["change_type"], "📄")

            result = f"{emoji} **{file_data['file_path']}**\n\n"
            result += f"**Change Type:** {file_data['change_type']}\n"
            result += f"**Created:** {file_data['created_at']}\n"
            result += f"**Last Updated:** {file_data['updated_at']}\n"

            if file_data.get("description"):
                result += f"**Description:** {file_data['description']}\n"

            # Line count information
            if file_data["change_type"] == "delete":
                result += "\n⚠️ **File marked for deletion**\n"
                if file_data.get("previous_content"):
                    prev_lines = file_data["previous_content"].split("\n")
                    result += f"**Original Lines:** {len(prev_lines)}\n"
                    result += f"**Original Size:** {len(file_data['previous_content'])} chars\n"
                    result += f"\n**Previous content preview (first 300 chars):**\n```\n{file_data['previous_content'][:300]}...\n```\n"
            else:
                lines = []
                if file_data.get("content"):
                    lines = file_data["content"].split("\n")
                    result += f"\n**Current Lines:** {len(lines)}\n"
                    result += f"**Current Size:** {len(file_data['content'])} chars\n"
                    content_preview = file_data["content"][:500]
                    result += f"\n**Content preview (first 500 chars):**\n```\n{content_preview}\n```\n"
                    if len(file_data["content"]) > 500:
                        result += f"\n... ({len(file_data['content']) - 500} more characters)\n"

                if (
                    file_data.get("previous_content")
                    and file_data["change_type"] == "update"
                    and lines
                ):
                    prev_lines = file_data["previous_content"].split("\n")
                    result += f"\n**Previous Lines:** {len(prev_lines)}\n"
                    result += f"**Previous Size:** {len(file_data['previous_content'])} chars\n"
                    result += (
                        f"**Line Change:** {len(lines) - len(prev_lines):+d} lines\n"
                    )

            result += "\n💡 **Tip:** Use `get_file_diff` to see the diff for this file against the repository branch."

            return result
        else:
            return f"❌ File '{input_data.file_path}' not found in changes"
    except Exception:
        logger.exception(
            "Tool get_file_tool: Error retrieving file", file_path=input_data.file_path
        )
        return "❌ Error retrieving file"


def list_files_tool(input_data: ListFilesInput) -> str:
    """List all files with changes, optionally filtered"""
    logger.info(
        f"Tool list_files_tool: Listing files (filter: {input_data.change_type_filter}, pattern: {input_data.path_pattern})"
    )
    try:
        manager = _get_code_changes_manager()

        change_type_filter = None
        if input_data.change_type_filter:
            try:
                change_type_filter = ChangeType(input_data.change_type_filter.lower())
            except ValueError:
                return f"❌ Invalid change type '{input_data.change_type_filter}'. Valid types: add, update, delete"

        files = manager.list_files(
            change_type_filter=change_type_filter,
            path_pattern=input_data.path_pattern,
        )

        if not files:
            filter_text = ""
            if input_data.change_type_filter:
                filter_text += f" with change type '{input_data.change_type_filter}'"
            if input_data.path_pattern:
                filter_text += f" matching pattern '{input_data.path_pattern}'"
            return f"📋 No files found{filter_text}"

        result = f"📋 **Files in Changes** ({len(files)} files)\n\n"

        change_emoji = {"add": "➕", "update": "✏️", "delete": "🗑️"}

        for file_data in files:
            emoji = change_emoji.get(file_data["change_type"], "📄")
            result += (
                f"{emoji} **{file_data['file_path']}** ({file_data['change_type']})\n"
            )
            if file_data.get("description"):
                result += f"   Description: {file_data['description']}\n"
            result += f"   Updated: {file_data['updated_at']}\n\n"

        return result
    except Exception:
        logger.exception("Tool list_files_tool: Error listing files")
        return "❌ Error listing files"


def search_content_tool(input_data: SearchContentInput) -> str:
    """Search for pattern in file contents (grep-like functionality)"""
    logger.info(
        f"Tool search_content_tool: Searching for pattern '{input_data.pattern}' (file_pattern: {input_data.file_pattern})"
    )
    try:
        manager = _get_code_changes_manager()
        matches = manager.search_content(
            pattern=input_data.pattern,
            file_pattern=input_data.file_pattern,
            case_sensitive=input_data.case_sensitive,
        )

        if not matches:
            filter_text = ""
            if input_data.file_pattern:
                filter_text = f" in files matching '{input_data.file_pattern}'"
            return (
                f"🔍 No matches found for pattern '{input_data.pattern}'{filter_text}"
            )

        # Group matches by file
        matches_by_file: Dict[str, List[Dict[str, Any]]] = {}
        for match in matches:
            if "error" in match:
                return f"❌ {match['error']}"
            file_path = match["file_path"]
            if file_path not in matches_by_file:
                matches_by_file[file_path] = []
            matches_by_file[file_path].append(match)

        result = f"🔍 **Search Results** ({len(matches)} matches in {len(matches_by_file)} files)\n\n"
        result += f"Pattern: `{input_data.pattern}`\n\n"

        for file_path, file_matches in matches_by_file.items():
            result += f"📄 **{file_path}** ({len(file_matches)} matches):\n"
            for match in file_matches[:10]:  # Show first 10 matches per file
                result += f"  Line {match['line_number']}: {match['line']}\n"
            if len(file_matches) > 10:
                result += f"  ... and {len(file_matches) - 10} more matches\n"
            result += "\n"

        return result
    except Exception:
        logger.exception(
            "Tool search_content_tool: Error searching content",
            pattern=input_data.pattern,
        )
        return "❌ Error searching content"


def clear_file_tool(input_data: ClearFileInput) -> str:
    """Clear changes for a specific file"""
    logger.info(f"Tool clear_file_tool: Clearing file '{input_data.file_path}'")
    try:
        manager = _get_code_changes_manager()
        success = manager.clear_file(input_data.file_path)

        if success:
            summary = manager.get_summary()
            return f"✅ Cleared changes for '{input_data.file_path}'\n\nTotal files in changes: {summary['total_files']}"
        else:
            return f"❌ File '{input_data.file_path}' not found in changes"
    except Exception:
        logger.exception(
            "Tool clear_file_tool: Error clearing file", file_path=input_data.file_path
        )
        return "❌ Error clearing file"


def clear_all_changes_tool() -> str:
    """Clear all changes from the code changes manager"""
    logger.info("Tool clear_all_changes_tool: Clearing all changes")
    try:
        manager = _get_code_changes_manager()
        count = manager.clear_all()
        return f"✅ Cleared all changes ({count} files removed)"
    except Exception:
        logger.exception("Tool clear_all_changes_tool: Error clearing all changes")
        return "❌ Error clearing all changes"


def get_changes_summary_tool() -> str:
    """Get a summary of all code changes"""
    logger.info("Tool get_changes_summary_tool: Getting summary")
    try:
        manager = _get_code_changes_manager()
        summary = manager.get_summary()

        result = f"📊 **Code Changes Summary** (Session: {summary['session_id']})\n\n"
        result += f"Total files: {summary['total_files']}\n\n"

        change_emoji = {"add": "➕", "update": "✏️", "delete": "🗑️"}

        result += "**Change Types:**\n"
        for change_type, count in summary["change_counts"].items():
            emoji = change_emoji.get(change_type, "📄")
            result += f"{emoji} {change_type.title()}: {count}\n"

        if summary["files"]:
            result += "\n**Files:**\n"
            for file_info in summary["files"][:10]:  # Show first 10
                emoji = change_emoji.get(file_info["change_type"], "📄")
                result += (
                    f"{emoji} {file_info['file_path']} ({file_info['change_type']})\n"
                )
            if len(summary["files"]) > 10:
                result += f"... and {len(summary['files']) - 10} more files\n"

        return result
    except Exception:
        logger.exception("Tool get_summary_tool: Error getting summary")
        return "❌ Error getting summary"


def update_file_lines_tool(input_data: UpdateFileLinesInput) -> str:
    """Update specific lines in a file using line numbers"""
    logger.info(
        f"Tool update_file_lines_tool: Updating lines {input_data.start_line}-{input_data.end_line or input_data.start_line} "
        f"in '{input_data.file_path}' (project_id={input_data.project_id})"
    )
    try:
        manager = _get_code_changes_manager()
        db = None
        if input_data.project_id:
            logger.info(
                f"Tool update_file_lines_tool: Project ID provided ({input_data.project_id}), fetching database session"
            )
            from app.core.database import get_db

            db = next(get_db())
            logger.debug("Tool update_file_lines_tool: Database session obtained")
        # project_id is now required, so this shouldn't happen, but keep for safety
        if not input_data.project_id:
            logger.error(
                "Tool update_file_lines_tool: ERROR - project_id is required but was not provided!"
            )
            return "❌ Error: project_id is required to update file lines. Please provide the project_id from the conversation context."
        result = manager.update_file_lines(
            file_path=input_data.file_path,
            start_line=input_data.start_line,
            end_line=input_data.end_line,
            new_content=input_data.new_content,
            description=input_data.description,
            project_id=input_data.project_id,
            db=db,
        )

        if result.get("success"):
            context_str = ""
            if result.get("updated_context"):
                context_start = result.get("context_start_line", result["start_line"])
                context_end = result.get("context_end_line", result["end_line"])
                context_str = f"\nUpdated lines with context (lines {context_start}-{context_end}):\n```{input_data.file_path}\n{result['updated_context']}\n```"
            return (
                f"✅ Updated lines {result['start_line']}-{result['end_line']} in '{input_data.file_path}'\n\n"
                + f"Replaced {result['lines_replaced']} lines with {result['lines_added']} new lines\n"
                + f"Replaced content:\n```\n{result['replaced_content'][:200]}{'...' if len(result['replaced_content']) > 200 else ''}\n```"
                + context_str
            )
        else:
            return f"❌ Error updating lines: {result.get('error', 'Unknown error')}"
    except Exception:
        logger.exception(
            "Tool update_file_lines_tool: Error updating file lines",
            file_path=input_data.file_path,
        )
        return "❌ Error updating file lines"


def replace_in_file_tool(input_data: ReplaceInFileInput) -> str:
    """Replace pattern matches in a file using regex"""
    logger.info(
        f"Tool replace_in_file_tool: Replacing pattern '{input_data.pattern}' in '{input_data.file_path}' "
        f"(project_id={input_data.project_id})"
    )
    try:
        manager = _get_code_changes_manager()
        db = None
        if input_data.project_id:
            logger.info(
                f"Tool replace_in_file_tool: Project ID provided ({input_data.project_id}), fetching database session"
            )
            from app.core.database import get_db

            db = next(get_db())
            logger.debug("Tool replace_in_file_tool: Database session obtained")
        result = manager.replace_in_file(
            file_path=input_data.file_path,
            pattern=input_data.pattern,
            replacement=input_data.replacement,
            count=input_data.count,
            description=input_data.description,
            case_sensitive=input_data.case_sensitive,
            word_boundary=input_data.word_boundary,
            project_id=input_data.project_id,
            db=db,
        )

        if result.get("success"):
            locations_str = "\n".join(
                [
                    f"  Line {loc['line']}: {loc['match']}"
                    for loc in result["match_locations"][:5]
                ]
            )
            if len(result["match_locations"]) > 5:
                locations_str += (
                    f"\n  ... and {len(result['match_locations']) - 5} more"
                )

            return (
                f"✅ Replaced pattern '{input_data.pattern}' in '{input_data.file_path}'\n\n"
                + f"Made {result['replacements_made']} replacement(s) out of {result['total_matches']} match(es)\n\n"
                + f"Match locations:\n{locations_str}"
            )
        else:
            return f"❌ Error replacing pattern: {result.get('error', 'Unknown error')}"
    except Exception:
        logger.exception(
            "Tool replace_in_file_tool: Error replacing in file",
            file_path=input_data.file_path,
            pattern=input_data.pattern,
        )
        return "❌ Error replacing in file"


def insert_lines_tool(input_data: InsertLinesInput) -> str:
    """Insert content at a specific line in a file"""
    position = "after" if input_data.insert_after else "before"
    logger.info(
        f"Tool insert_lines_tool: Inserting lines {position} line {input_data.line_number} "
        f"in '{input_data.file_path}' (project_id={input_data.project_id})"
    )
    try:
        manager = _get_code_changes_manager()
        db = None
        if input_data.project_id:
            logger.info(
                f"Tool insert_lines_tool: Project ID provided ({input_data.project_id}), fetching database session"
            )
            from app.core.database import get_db

            db = next(get_db())
            logger.debug("Tool insert_lines_tool: Database session obtained")
        # project_id is now required, so this shouldn't happen, but keep for safety
        if not input_data.project_id:
            logger.error(
                "Tool insert_lines_tool: ERROR - project_id is required but was not provided!"
            )
            return "❌ Error: project_id is required to insert lines. Please provide the project_id from the conversation context."
        result = manager.insert_lines(
            file_path=input_data.file_path,
            line_number=input_data.line_number,
            content=input_data.content,
            description=input_data.description,
            insert_after=input_data.insert_after,
            project_id=input_data.project_id,
            db=db,
        )

        if result.get("success"):
            position = "after" if result["position"] == "after" else "before"
            context_str = ""
            if result.get("inserted_context"):
                context_start = result.get("context_start_line", input_data.line_number)
                context_end = result.get(
                    "context_end_line",
                    input_data.line_number + result["lines_inserted"],
                )
                context_str = f"\n\nInserted lines with context (lines {context_start}-{context_end}):\n```{input_data.file_path}\n{result['inserted_context']}\n```"
            return f"✅ Inserted {result['lines_inserted']} line(s) {position} line {input_data.line_number} in '{input_data.file_path}'{context_str}"
        else:
            return f"❌ Error inserting lines: {result.get('error', 'Unknown error')}"
    except Exception:
        logger.exception(
            "Tool insert_lines_tool: Error inserting lines",
            file_path=input_data.file_path,
            line_number=input_data.line_number,
        )
        return "❌ Error inserting lines"


def delete_lines_tool(input_data: DeleteLinesInput) -> str:
    """Delete specific lines from a file"""
    logger.info(
        f"Tool delete_lines_tool: Deleting lines {input_data.start_line}-{input_data.end_line or input_data.start_line} "
        f"from '{input_data.file_path}' (project_id={input_data.project_id})"
    )
    try:
        manager = _get_code_changes_manager()
        db = None
        if input_data.project_id:
            logger.info(
                f"Tool delete_lines_tool: Project ID provided ({input_data.project_id}), fetching database session"
            )
            from app.core.database import get_db

            db = next(get_db())
            logger.debug("Tool delete_lines_tool: Database session obtained")
        result = manager.delete_lines(
            file_path=input_data.file_path,
            start_line=input_data.start_line,
            end_line=input_data.end_line,
            description=input_data.description,
            project_id=input_data.project_id,
            db=db,
        )

        if result.get("success"):
            deleted_preview = result["deleted_content"][:200]
            return (
                f"✅ Deleted lines {result['start_line']}-{result['end_line']} from '{input_data.file_path}'\n\n"
                + f"Deleted {result['lines_deleted']} line(s)\n"
                + f"Deleted content:\n```\n{deleted_preview}{'...' if len(result['deleted_content']) > 200 else ''}\n```"
            )
        else:
            return f"❌ Error deleting lines: {result.get('error', 'Unknown error')}"
    except Exception:
        logger.exception(
            "Tool delete_lines_tool: Error deleting lines",
            file_path=input_data.file_path,
        )
        return "❌ Error deleting lines"


class ShowUpdatedFileInput(BaseModel):
    file_paths: Optional[List[str]] = Field(
        default=None,
        description="Optional list of file paths to show. If not provided, shows all updated files.",
    )


class ShowDiffInput(BaseModel):
    file_path: Optional[str] = Field(
        default=None,
        description="Optional file path to show diff for a specific file. If not provided, shows diffs for all changed files.",
    )
    context_lines: int = Field(
        default=3,
        description="Number of context lines to include around changes in the diff (default: 3)",
    )
    project_id: Optional[str] = Field(
        default=None,
        description="Optional project ID to fetch original content from repository for accurate diffs against the branch. Use project_id from conversation context.",
    )


def show_updated_file_tool(input_data: ShowUpdatedFileInput) -> str:
    """
    Display the complete updated content of one or more files. This tool streams the full file content
    directly into the agent response without going through the LLM, allowing users to see
    the complete edited files. Use this when the user asks to see the updated file content.
    If no file_paths are provided, shows all changed files.
    """
    logger.info(
        f"Tool show_updated_file_tool: Showing updated content for '{input_data.file_paths or 'all files'}'"
    )
    try:
        manager = _get_code_changes_manager()
        summary = manager.get_summary()

        if summary["total_files"] == 0:
            return "📋 **No files to display**\n\nNo files have been modified yet."

        # Determine which files to show
        if input_data.file_paths:
            files_to_show = input_data.file_paths
        else:
            # Show all files
            files_to_show = [f["file_path"] for f in summary["files"]]

        if not files_to_show:
            return "📋 **No files to display**\n\nNo matching files found."

        change_emoji = {"add": "➕", "update": "✏️", "delete": "🗑️"}
        result = "\n\n---\n\n## 📝 **Updated Files**\n\n"

        if len(files_to_show) > 1:
            result += f"Showing {len(files_to_show)} files:\n\n"

        # Display each file
        for file_path in files_to_show:
            file_data = manager.get_file(file_path)

            if not file_data:
                result += f"❌ File '{file_path}' not found in changes\n\n"
                continue

            if file_data["change_type"] == "delete":
                result += f"⚠️ **{file_path}** - marked for deletion\n\n"
                continue

            content = file_data.get("content")
            if not content:
                result += f"❌ No content found for '{file_path}'\n\n"
                continue

            # Format the result with markdown code block
            change_type = file_data["change_type"]
            emoji = change_emoji.get(change_type, "📄")

            result += f"{emoji} **Updated File: {file_path}** ({change_type})\n\n"
            result += f"```\n{content}\n```\n\n"

            logger.info(
                f"Tool show_updated_file_tool: Successfully formatted file content for '{file_path}' "
                f"({len(content)} chars)"
            )

        result += "---\n\n"
        logger.info(
            f"Tool show_updated_file_tool: Successfully displayed {len(files_to_show)} file(s)"
        )

        return result
    except Exception:
        logger.exception(
            "Tool show_updated_file_tool: Error showing updated files",
        )
        return "❌ Error showing updated files"


def show_diff_tool(input_data: ShowDiffInput) -> str:
    """
    Display unified diffs showing changes between managed code and the actual codebase.
    This tool streams the formatted diffs directly into the agent response without going through
    the LLM, allowing users to see exactly what was changed. Use this at the end of your response
    to show all the code changes you've made. The content is automatically shown to the user
    without consuming LLM context.
    """
    logger.info(
        f"Tool show_diff_tool: Displaying diff(s) (file_path: {input_data.file_path}, context_lines: {input_data.context_lines}, project_id: {input_data.project_id})"
    )
    try:
        manager = _get_code_changes_manager()
        summary = manager.get_summary()

        if summary["total_files"] == 0:
            return (
                "📋 **No code changes to display**\n\nNo files have been modified yet."
            )

        # Get database session if project_id provided
        db = None
        if input_data.project_id:
            logger.info(
                f"Tool show_diff_tool: Project ID provided ({input_data.project_id}), fetching database session"
            )
            from app.core.database import get_db

            db = next(get_db())

        # Generate git-style diffs for each file
        files_to_diff = (
            [input_data.file_path]
            if input_data.file_path
            else list(manager.changes.keys())
        )
        git_diffs = []

        for file_path in files_to_diff:
            if file_path not in manager.changes:
                continue

            change = manager.changes[file_path]

            # Get old content
            if change.change_type == ChangeType.DELETE:
                old_content = change.previous_content or ""
                new_content = ""
            elif change.change_type == ChangeType.ADD:
                old_content = ""
                new_content = change.content or ""
            else:  # UPDATE
                new_content = change.content or ""
                if change.previous_content is not None:
                    old_content = change.previous_content
                else:
                    # Try to get from repository first if project_id/db provided
                    old_content = None
                    if input_data.project_id and db:
                        try:
                            from app.modules.code_provider.code_provider_service import (
                                CodeProviderService,
                            )
                            from app.modules.code_provider.git_safe import (
                                safe_git_operation,
                                GitOperationError,
                            )
                            from app.modules.projects.projects_model import Project

                            project = (
                                db.query(Project)
                                .filter(Project.id == input_data.project_id)
                                .first()
                            )
                            if project:
                                cp_service = CodeProviderService(db)

                                def _fetch_old_content():
                                    return cp_service.get_file_content(
                                        repo_name=project.repo_name,
                                        file_path=file_path,
                                        branch_name=project.branch_name,
                                        start_line=None,
                                        end_line=None,
                                        project_id=input_data.project_id,
                                        commit_id=project.commit_id,
                                    )

                                try:
                                    # Use timeout to prevent blocking worker
                                    repo_content = safe_git_operation(
                                        _fetch_old_content,
                                        max_retries=1,
                                        timeout=20.0,
                                        max_total_timeout=25.0,
                                        operation_name=f"show_diff_get_old_content({file_path})",
                                    )
                                except GitOperationError as git_error:
                                    logger.warning(
                                        f"Tool show_diff_tool: Git operation timed out: {git_error}"
                                    )
                                    repo_content = None

                                if repo_content:
                                    old_content = repo_content
                        except Exception as e:
                            logger.warning(
                                f"Tool show_diff_tool: Error fetching from repository: {e}"
                            )
                            old_content = None

                    # Fallback to filesystem
                    if old_content is None:
                        old_content = manager._read_file_from_codebase(file_path)

                    # If file doesn't exist, treat as new file
                    if old_content is None or old_content == "":
                        old_content = ""

            # Generate git-style diff
            git_diff = manager._generate_git_diff_patch(
                file_path=file_path,
                old_content=old_content or "",
                new_content=new_content or "",
                context_lines=input_data.context_lines,
            )

            if git_diff:
                git_diffs.append(git_diff)

        if not git_diffs:
            return "📋 **No diffs to display**\n\nNo changes found."

        # Combine all diffs into a single string
        combined_diff = "\n".join(git_diffs)

        # Write diff to .data folder as JSON
        try:
            data_dir = ".data"
            os.makedirs(data_dir, exist_ok=True)

            # Generate unique filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"diff_{timestamp}_{uuid.uuid4().hex[:8]}.json"
            filepath = os.path.join(data_dir, filename)

            # Get reasoning hash from reasoning manager
            reasoning_hash = None
            try:
                from app.modules.intelligence.tools.reasoning_manager import (
                    _get_reasoning_manager,
                )

                reasoning_manager = _get_reasoning_manager()
                reasoning_hash = reasoning_manager.get_reasoning_hash()
                # If not finalized yet, try to finalize it
                if not reasoning_hash:
                    reasoning_hash = reasoning_manager.finalize_and_save()
            except Exception as e:
                logger.warning(
                    f"Tool show_diff_tool: Failed to get reasoning hash: {e}"
                )

            # Create JSON with model_patch and reasoning_hash fields
            diff_data = {"model_patch": combined_diff}
            if reasoning_hash:
                diff_data["reasoning_hash"] = reasoning_hash

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(diff_data, f, indent=2, ensure_ascii=False)

            logger.info(
                f"Tool show_diff_tool: Diff written to {filepath} "
                f"(reasoning_hash: {reasoning_hash})"
            )
        except Exception as e:
            logger.warning(
                f"Tool show_diff_tool: Failed to write diff to .data folder: {e}"
            )

        # Output clean diff format
        result = "--generated diff--\n\n"
        result += "```\n"
        result += combined_diff
        result += "\n```\n\n--generated diff--\n"

        return result
    except Exception:
        project_id = getattr(input_data, "project_id", None)
        logger.exception(
            "Tool show_diff_tool: Error displaying diff", project_id=project_id
        )
        return "❌ Error displaying diff"


class GetFileDiffInput(BaseModel):
    file_path: str = Field(description="Path to the file to get diff for")
    context_lines: int = Field(
        default=3,
        description="Number of context lines to include around changes in the diff (default: 3)",
    )
    project_id: Optional[str] = Field(
        default=None,
        description="Optional project ID to fetch original content from repository for accurate diff against the branch. Use project_id from conversation context.",
    )


def get_file_diff_tool(input_data: GetFileDiffInput) -> str:
    """
    Get the diff for a specific file against the repository branch.
    This shows what has changed in this file compared to the original repository version.
    """
    logger.info(
        f"Tool get_file_diff_tool: Getting diff for '{input_data.file_path}' (context_lines: {input_data.context_lines}, project_id: {input_data.project_id})"
    )
    try:
        manager = _get_code_changes_manager()
        file_data = manager.get_file(input_data.file_path)

        if not file_data:
            return f"❌ File '{input_data.file_path}' not found in changes"

        # Get database session if project_id provided
        db = None
        if input_data.project_id:
            logger.info(
                f"Tool get_file_diff_tool: Project ID provided ({input_data.project_id}), fetching database session"
            )
            from app.core.database import get_db

            db = next(get_db())

        # Generate diff for this specific file
        diffs = manager.generate_diff(
            file_path=input_data.file_path,
            context_lines=input_data.context_lines,
            project_id=input_data.project_id,
            db=db,
        )

        if not diffs or input_data.file_path not in diffs:
            return f"❌ No diff generated for '{input_data.file_path}'"

        diff_content = diffs[input_data.file_path]
        change_emoji = {"add": "➕", "update": "✏️", "delete": "🗑️"}
        emoji = change_emoji.get(file_data["change_type"], "📄")

        result = f"📝 **Diff for {input_data.file_path}** ({emoji} {file_data['change_type']})\n\n"
        if file_data.get("description"):
            result += f"*{file_data['description']}*\n\n"
        result += f"**Last updated:** {file_data['updated_at']}\n\n"
        result += "```diff\n"
        result += diff_content
        result += "\n```\n"

        return result
    except Exception:
        logger.exception(
            "Tool get_file_diff_tool: Error getting file diff",
            project_id=input_data.project_id,
            file_path=input_data.file_path,
        )
        return "❌ Error getting file diff"


class GetComprehensiveMetadataInput(BaseModel):
    project_id: Optional[str] = Field(
        default=None,
        description="Optional project ID for logging purposes. Use project_id from conversation context.",
    )


def get_comprehensive_metadata_tool(input_data: GetComprehensiveMetadataInput) -> str:
    """
    Get comprehensive metadata about all code changes in the current session.
    This shows the complete state of all files being managed, including timestamps,
    descriptions, change types, and line counts. Use this to review your session progress
    and understand what files have been modified.
    """
    logger.info("Tool get_comprehensive_metadata_tool: Getting comprehensive metadata")
    try:
        manager = _get_code_changes_manager()
        summary = manager.get_summary()

        result = (
            f"📊 **Complete Session State** (Session ID: {summary['session_id']})\n\n"
        )
        result += f"**Total Files Changed:** {summary['total_files']}\n\n"

        change_emoji = {"add": "➕", "update": "✏️", "delete": "🗑️"}

        # Summary by change type
        result += "**Summary by Change Type:**\n"
        for change_type, count in summary["change_counts"].items():
            if count > 0:
                emoji = change_emoji.get(change_type, "📄")
                result += f"- {emoji} {change_type.title()}: {count}\n"
        result += "\n"

        # Detailed file information
        if summary["files"]:
            result += "**Detailed File Information:**\n\n"
            for file_info in summary["files"]:
                emoji = change_emoji.get(file_info["change_type"], "📄")
                result += f"{emoji} **{file_info['file_path']}**\n"
                result += f"  - Type: {file_info['change_type']}\n"
                result += f"  - Last Updated: {file_info['updated_at']}\n"
                if file_info.get("description"):
                    result += f"  - Description: {file_info['description']}\n"

                # Get file data for line counts
                file_data = manager.get_file(file_info["file_path"])
                if file_data:
                    if file_data["change_type"] == "delete":
                        if file_data.get("previous_content"):
                            lines = file_data["previous_content"].split("\n")
                            result += f"  - Original Lines: {len(lines)}\n"
                    else:
                        if file_data.get("content"):
                            lines = file_data["content"].split("\n")
                            result += f"  - Current Lines: {len(lines)}\n"
                        if file_data.get("previous_content"):
                            prev_lines = file_data["previous_content"].split("\n")
                            result += f"  - Original Lines: {len(prev_lines)}\n"
                result += "\n"
        else:
            result += "No files have been modified yet.\n"

        result += "\n💡 **Tip:** Use `get_file_from_changes` to see detailed information about a specific file, "
        result += "or `get_file_diff` to see the diff for a file against the repository branch."

        return result
    except Exception:
        logger.exception(
            "Tool get_comprehensive_metadata_tool: Error getting metadata",
            project_id=input_data.project_id,
        )
        return "❌ Error getting metadata"


def export_changes_tool(input_data: ExportChangesInput) -> str:
    """Export all changes in the specified format"""
    logger.info(
        f"Tool export_changes_tool: Exporting changes in '{input_data.format}' format"
    )
    try:
        manager = _get_code_changes_manager()

        # Get database session if project_id provided (needed for diff format)
        db = None
        if input_data.project_id:
            logger.info(
                f"Tool export_changes_tool: Project ID provided ({input_data.project_id}), fetching database session"
            )
            from app.core.database import get_db

            db = next(get_db())

        exported = manager.export_changes(
            format=input_data.format, project_id=input_data.project_id, db=db
        )

        if input_data.format == "json":
            # Return JSON directly (might be long, but that's expected)
            return f"📦 **Exported Changes (JSON)**\n\n```json\n{exported}\n```"
        elif input_data.format == "dict":
            if not isinstance(exported, dict):
                return f"❌ Expected dict format, got {type(exported)}"
            result = f"📦 **Exported Changes (Dictionary)** - {len(exported)} files\n\n"
            items_list = list(exported.items())[:5]  # Show first 5
            for file_path, content in items_list:
                result += f"**{file_path}** ({len(content)} chars):\n```\n{content[:200]}...\n```\n\n"
            if len(exported) > 5:
                result += f"... and {len(exported) - 5} more files\n"
            return result
        elif input_data.format == "diff":
            # Return diff patch format with "Generated Diff:" heading
            if not exported or not isinstance(exported, str):
                return "❌ No diff generated or invalid format"
            return f"Generated Diff:\n\n```\n{exported}\n```"
        else:  # list format
            if not isinstance(exported, list):
                return f"❌ Expected list format, got {type(exported)}"
            result = f"📦 **Exported Changes (List)** - {len(exported)} files\n\n"
            for change in exported[:5]:  # Show first 5
                if isinstance(change, dict):
                    result += f"**{change.get('file_path', 'unknown')}** ({change.get('change_type', 'unknown')})\n"
            if len(exported) > 5:
                result += f"... and {len(exported) - 5} more files\n"
            return result
    except Exception:
        logger.exception(
            "Tool export_changes_tool: Error exporting changes",
            format=input_data.format,
        )
        return "❌ Error exporting changes"


# Create the structured tools
class SimpleTool:
    """Simple tool wrapper that mimics StructuredTool interface"""

    def __init__(self, name: str, description: str, func, args_schema):
        self.name = name
        self.description = description
        self.func = func
        self.args_schema = args_schema


def create_code_changes_management_tools() -> List[SimpleTool]:
    """Create all code changes management tools"""

    tools = [
        SimpleTool(
            name="add_file_to_changes",
            description="Add a new file to the code changes manager. Use this to track new files you're creating instead of including full code in your response. This reduces token usage in conversation history.",
            func=add_file_tool,
            args_schema=AddFileInput,
        ),
        SimpleTool(
            name="update_file_in_changes",
            description="Update an existing file in the code changes manager with full content. Use this only when you need to replace the entire file. For targeted changes, prefer update_file_lines, replace_in_file, insert_lines, or delete_lines.",
            func=update_file_tool,
            args_schema=UpdateFileInput,
        ),
        SimpleTool(
            name="update_file_lines",
            description="Update specific lines in a file using line numbers. Use this for targeted line-by-line replacements. Lines are 1-indexed. Specify start_line and optionally end_line to replace a range. CRITICAL: You MUST preserve proper indentation - match the indentation of surrounding lines exactly. BEST PRACTICES: (1) Always fetch the file with line numbers first (get_file_from_changes with_line_numbers=true) to see exact indentation and current line numbers. (2) After updating, verify changes by fetching the updated lines to ensure indentation and content are correct. (3) For sequential operations: If you've performed insert_lines or delete_lines on this file, the line numbers have shifted - always fetch the current file state before using update_file_lines to get correct line numbers. (4) You MUST provide project_id from the conversation context to access existing file content from the repository.",
            func=update_file_lines_tool,
            args_schema=UpdateFileLinesInput,
        ),
        SimpleTool(
            name="replace_in_file",
            description="Replace pattern matches in a file using regex. Use this to replace text patterns throughout a file. Supports regex capturing groups (\\1, \\2, etc.) in replacement. Set count=0 to replace all occurrences. Use word_boundary=True to match whole words only (prevents matching partial strings within words). BEST PRACTICES: (1) After replacing, verify changes by fetching the file (get_file_from_changes) to confirm replacements were applied correctly. (2) For sequential operations: Always provide project_id from conversation context for proper content retrieval, especially when performing multiple operations on the same file. (3) Use word_boundary=True when you want to match whole words/phrases and avoid partial matches within variable or function names.",
            func=replace_in_file_tool,
            args_schema=ReplaceInFileInput,
        ),
        SimpleTool(
            name="insert_lines",
            description="Insert content at a specific line number in a file. Use this to add new code at a specific location. Lines are 1-indexed. Set insert_after=False to insert before the specified line. CRITICAL: You MUST preserve proper indentation - match the indentation level of the line you're inserting after/before, or maintain consistent indentation for the code block you're adding. BEST PRACTICES: (1) Always fetch the file with line numbers first (get_file_from_changes with_line_numbers=true) to see exact indentation and current line numbers. (2) After inserting, verify changes by fetching the inserted lines in context to ensure indentation and placement are correct. (3) For sequential operations: If you've performed insert_lines or delete_lines on this file, line numbers have shifted - always fetch the current file state before subsequent operations. (4) You MUST provide project_id from the conversation context to access existing file content from the repository.",
            func=insert_lines_tool,
            args_schema=InsertLinesInput,
        ),
        SimpleTool(
            name="delete_lines",
            description="Delete specific lines from a file using line numbers. Use this to remove unwanted code. Lines are 1-indexed. Specify start_line and optionally end_line to delete a range. BEST PRACTICES: (1) Always fetch the file with line numbers first (get_file_from_changes with_line_numbers=true) to get correct line numbers, especially after previous insert/delete operations. (2) After deleting, verify changes by fetching the file to confirm lines were removed correctly. (3) For sequential operations: Line numbers shift after deletions - always fetch current file state before subsequent line-based operations. (4) Provide project_id from conversation context for proper content retrieval, especially when performing sequential operations on the same file.",
            func=delete_lines_tool,
            args_schema=DeleteLinesInput,
        ),
        SimpleTool(
            name="delete_file_in_changes",
            description="Mark a file for deletion in the code changes manager. File content is preserved by default so you can reference it later.",
            func=delete_file_tool,
            args_schema=DeleteFileInput,
        ),
        SimpleTool(
            name="get_file_from_changes",
            description="Get change information and content for a specific file from the code changes manager. BEST PRACTICES: (1) Use with_line_numbers=true before line-based operations (update_file_lines, insert_lines, delete_lines) to see exact line numbers and indentation. (2) Use after editing operations to verify changes were applied correctly. (3) Essential for sequential operations - fetch current state between operations to get correct line numbers after insertions/deletions.",
            func=get_file_tool,
            args_schema=GetFileInput,
        ),
        SimpleTool(
            name="list_files_in_changes",
            description="List all files in the code changes manager, optionally filtered by change type (add/update/delete) or file path pattern (regex).",
            func=list_files_tool,
            args_schema=ListFilesInput,
        ),
        SimpleTool(
            name="search_content_in_changes",
            description="Search for a pattern in file contents using regex (grep-like functionality). Supports filtering by file path pattern. Returns matching lines with line numbers.",
            func=search_content_tool,
            args_schema=SearchContentInput,
        ),
        SimpleTool(
            name="clear_file_from_changes",
            description="Remove a specific file from the code changes manager (discard its changes).",
            func=clear_file_tool,
            args_schema=ClearFileInput,
        ),
        SimpleTool(
            name="clear_all_changes",
            description="Clear all files from the code changes manager (discard all changes).",
            func=clear_all_changes_tool,
            args_schema=None,
        ),
        SimpleTool(
            name="get_changes_summary",
            description="Get a summary overview of all code changes including file counts by change type.",
            func=get_changes_summary_tool,
            args_schema=None,
        ),
        SimpleTool(
            name="export_changes",
            description="Export all code changes in various formats (dict, list, json, or diff). Use 'json' format for persistence, 'diff' format for git-style patch.",
            func=export_changes_tool,
            args_schema=ExportChangesInput,
        ),
        SimpleTool(
            name="show_updated_file",
            description="Display the complete updated content of one or more files. This tool streams the full file content directly into the agent response without going through the LLM, allowing users to see the complete edited files. If no file_paths provided, shows ALL changed files. Use when the user asks to see updated files OR to showcase the final result of files you just edited. The content is automatically shown to the user without consuming LLM context.",
            func=show_updated_file_tool,
            args_schema=ShowUpdatedFileInput,
        ),
        SimpleTool(
            name="show_diff",
            description="Display unified diffs showing changes between managed code and the actual codebase. This tool streams the formatted diffs directly into the agent response without going through the LLM, allowing users to see exactly what was changed. Use this at the end of your response to show all the code changes you've made. The content is automatically shown to the user without consuming LLM context. BEST PRACTICES: (1) Use project_id to fetch original content from repository for accurate diffs against the branch. (2) When changes are spread across many lines, increase context_lines parameter (default 3) to show more surrounding context in the diff.",
            func=show_diff_tool,
            args_schema=ShowDiffInput,
        ),
        SimpleTool(
            name="get_file_diff",
            description="Get the diff for a specific file against the repository branch. Shows what has changed in this file compared to the original repository version. BEST PRACTICES: (1) Use project_id from conversation context to get accurate diffs against the branch. (2) When changes are spread across many lines, increase context_lines parameter (default 3) to show more surrounding context in the diff.",
            func=get_file_diff_tool,
            args_schema=GetFileDiffInput,
        ),
        SimpleTool(
            name="get_session_metadata",
            description="Get comprehensive metadata about all code changes in the current session. Shows complete state of all files being managed, including timestamps, descriptions, change types, and line counts. Use this to review your session progress and understand what files have been modified. This is your session state - all your work is tracked here.",
            func=get_comprehensive_metadata_tool,
            args_schema=GetComprehensiveMetadataInput,
        ),
    ]

    return tools
