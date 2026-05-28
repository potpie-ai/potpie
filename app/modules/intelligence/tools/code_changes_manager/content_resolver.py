"""File content retrieval from changes, repository, and filesystem."""

import os
import time
from typing import Dict, Optional, Any

from sqlalchemy.orm import Session

from app.modules.utils.logger import setup_logger

from .constants import (
    MAX_FILE_SIZE_BYTES,
    DB_QUERY_TIMEOUT,
    DB_SESSION_CREATE_TIMEOUT,
)
from .models import ChangeType, FileChange
from .utils import _execute_with_timeout, _check_memory_pressure, _get_git_file_size

logger = setup_logger(__name__)


def read_file_from_codebase(file_path: str) -> Optional[str]:
    """
    Read file content from the codebase filesystem.

    Args:
        file_path: Relative path to the file

    Returns:
        File content as string, or None if file doesn't exist or is too large
    """
    try:
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            if file_size > MAX_FILE_SIZE_BYTES:
                logger.warning(
                    f"content_resolver.read_file_from_codebase: File '{file_path}' "
                    f"is too large ({file_size} bytes, max {MAX_FILE_SIZE_BYTES} bytes)."
                )
                return None
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                return f.read()

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
                        f"content_resolver.read_file_from_codebase: File '{file_path}' "
                        f"(found at {path}) is too large ({file_size} bytes, max {MAX_FILE_SIZE_BYTES} bytes)."
                    )
                    return None
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    logger.debug(f"content_resolver.read_file_from_codebase: Found file at {path}")
                    return f.read()

        logger.debug(f"content_resolver.read_file_from_codebase: File '{file_path}' not found")
        return None
    except MemoryError as e:
        logger.error(
            f"content_resolver.read_file_from_codebase: Memory error reading '{file_path}': {e}"
        )
        return None
    except Exception as e:
        logger.warning(
            f"content_resolver.read_file_from_codebase: Error reading '{file_path}': {e}"
        )
        return None


def get_current_content(
    file_path: str,
    changes: Dict[str, FileChange],
    project_id: Optional[str] = None,
    db: Optional[Session] = None,
) -> str:
    """Get current content of a file (from changes, repository via code provider, filesystem, or empty if new)."""
    start_time = time.time()
    logger.info(
        f"content_resolver.get_current_content: [START] Getting content for '{file_path}' "
        f"(project_id={project_id}, db={'provided' if db else 'None'})"
    )

    if file_path in changes:
        existing = changes[file_path]
        if existing.change_type == ChangeType.DELETE:
            logger.info(f"content_resolver.get_current_content: File '{file_path}' is marked as deleted")
            return ""
        content = existing.content or ""
        elapsed = time.time() - start_time
        logger.info(
            f"content_resolver.get_current_content: [SUCCESS] Retrieved '{file_path}' from changes "
            f"({len(content)} chars) in {elapsed:.3f}s"
        )
        return content

    # If file not in changes, try to fetch from repository using code provider
    if project_id and db:
        logger.info(
            f"content_resolver.get_current_content: [STEP 1] Attempting to fetch '{file_path}' "
            f"from repository using project_id={project_id}"
        )
        try:
            from app.modules.code_provider.code_provider_service import CodeProviderService
            from app.modules.projects.projects_model import Project

            project_details: Optional[Dict[str, Any]] = None
            try:
                project = _execute_with_timeout(
                    lambda: db.query(Project).filter(Project.id == project_id).first(),
                    timeout=DB_QUERY_TIMEOUT,
                    operation_name=f"db_query_project_{project_id}",
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
                else:
                    project_details = None
            except TimeoutError as timeout_err:
                logger.error(
                    f"content_resolver.get_current_content: [STEP 1.1.ERROR] Database query TIMED OUT: {timeout_err}"
                )
                project_details = None
            except Exception as e:
                error_str = str(e).lower()
                if any(
                    kw in error_str
                    for kw in ["connection", "session", "closed", "invalid", "fork"]
                ):
                    try:
                        from app.core.database import SessionLocal

                        old_db = db
                        try:
                            old_db.close()
                        except Exception:
                            pass

                        db = _execute_with_timeout(
                            lambda: SessionLocal(),
                            timeout=DB_SESSION_CREATE_TIMEOUT,
                            operation_name="create_new_db_session",
                        )

                        project = _execute_with_timeout(
                            lambda: db.query(Project).filter(Project.id == project_id).first(),
                            timeout=DB_QUERY_TIMEOUT,
                            operation_name=f"db_query_project_retry_{project_id}",
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
                        else:
                            project_details = None
                    except (TimeoutError, Exception):
                        project_details = None
                else:
                    project_details = None

            if project_details and "project_name" in project_details:
                is_pressure, mem_percent = _check_memory_pressure()
                if is_pressure:
                    logger.warning(
                        f"content_resolver.get_current_content: [STEP 2.MEMORY] Memory pressure ({mem_percent:.1%}). "
                        f"Skipping repository fetch for '{file_path}'."
                    )
                    repo_content = None
                else:
                    repo_content = None
                    try:
                        repo_path = project_details.get("repo_path")
                        if repo_path and os.path.exists(repo_path):
                            file_size = _get_git_file_size(
                                repo_path,
                                file_path,
                                project_details.get("branch_name"),
                            )
                            if file_size is not None and file_size > MAX_FILE_SIZE_BYTES:
                                logger.warning(
                                    f"content_resolver.get_current_content: [STEP 2.SIZE] File '{file_path}' "
                                    f"too large ({file_size} bytes). Skipping."
                                )
                                repo_content = None

                        cp_service = _execute_with_timeout(
                            lambda: CodeProviderService(db),
                            timeout=DB_QUERY_TIMEOUT,
                            operation_name="CodeProviderService_init",
                        )

                        if repo_content is None:
                            from app.modules.code_provider.git_safe import (
                                safe_git_operation,
                                GitOperationError,
                            )

                            def _fetch_file_content():
                                return cp_service.get_file_content(
                                    repo_name=project_details["project_name"],
                                    file_path=file_path,
                                    branch_name=project_details.get("branch_name"),
                                    start_line=None,
                                    end_line=None,
                                    project_id=project_id,
                                    commit_id=project_details.get("commit_id"),
                                )

                            try:
                                repo_content = safe_git_operation(
                                    _fetch_file_content,
                                    max_retries=1,
                                    timeout=20.0,
                                    max_total_timeout=40.0,
                                    operation_name=f"get_file_content({file_path})",
                                )
                            except GitOperationError as git_error:
                                logger.warning(
                                    f"content_resolver.get_current_content: [STEP 3.ERROR] Git failed: {git_error}"
                                )
                                repo_content = None
                            except (MemoryError, SystemExit, KeyboardInterrupt, BaseException) as e:
                                logger.error(
                                    f"content_resolver.get_current_content: [STEP 3.ERROR] Error: {e}"
                                )
                                repo_content = None
                    except TimeoutError:
                        repo_content = None
                    except Exception as service_error:
                        logger.error(
                            f"content_resolver.get_current_content: [STEP 2.ERROR] Service error: {service_error}"
                        )
                        repo_content = None

                if repo_content:
                    content_size_bytes = len(repo_content.encode("utf-8"))
                    if content_size_bytes > MAX_FILE_SIZE_BYTES:
                        repo_content = None
                    else:
                        total_elapsed = time.time() - start_time
                        logger.info(
                            f"content_resolver.get_current_content: [SUCCESS] Retrieved '{file_path}' "
                            f"from repository ({len(repo_content)} chars) in {total_elapsed:.3f}s"
                        )
                        return repo_content
        except MemoryError as mem_error:
            logger.error(
                f"content_resolver.get_current_content: [ERROR] Memory error: {mem_error}"
            )
        except Exception as e:
            logger.warning(
                f"content_resolver.get_current_content: [ERROR] Error fetching from repository: {e}"
            )

    # Fallback to filesystem
    is_pressure, mem_percent = _check_memory_pressure()
    if is_pressure:
        logger.warning(
            f"content_resolver.get_current_content: [STEP 4.MEMORY] Memory pressure ({mem_percent:.1%}). "
            f"Skipping filesystem for '{file_path}'."
        )
        codebase_content = None
        filesystem_elapsed = 0.0
    else:
        filesystem_start = time.time()
        codebase_content = read_file_from_codebase(file_path)
        filesystem_elapsed = time.time() - filesystem_start

    if codebase_content is not None:
        total_elapsed = time.time() - start_time
        logger.warning(
            f"content_resolver.get_current_content: [SUCCESS] Retrieved '{file_path}' from filesystem "
            f"({len(codebase_content)} chars) in {filesystem_elapsed:.3f}s (total: {total_elapsed:.3f}s). "
            f"May be ORIGINAL content, not current changes."
        )
        return codebase_content

    logger.warning(
        f"content_resolver.get_current_content: [END] File '{file_path}' not found - treating as new file."
    )
    return ""
