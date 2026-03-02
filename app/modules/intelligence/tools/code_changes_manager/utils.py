"""Utility functions for code changes management."""

import threading
import os as os_module
from typing import Callable, TypeVar, Optional

from app.modules.utils.logger import setup_logger

from .constants import MEMORY_PRESSURE_THRESHOLD

logger = setup_logger(__name__)

T = TypeVar("T")


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
            timeout=5.0,
            operation_name=f"get_file_size({file_path})",
        )
    except Exception as e:
        logger.debug(f"Could not get file size for {file_path}: {e}")
        return None
