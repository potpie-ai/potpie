"""
Safe Git Operations Wrapper

This module provides utilities to safely execute git operations in multiprocessing
contexts (like Celery workers) where GitPython/libgit2 can cause SIGSEGV.

The main issue is that GitPython uses libgit2 which doesn't handle forking well.
This module provides retry logic and error handling to work around these issues.
"""

import logging
import time
import signal
import os
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Callable, TypeVar, Optional, Any
from functools import wraps

logger = logging.getLogger(__name__)

T = TypeVar("T")


class GitOperationError(Exception):
    """Base exception for git operation errors"""

    pass


class GitSegfaultError(GitOperationError):
    """Raised when a git operation might have caused a SIGSEGV"""

    pass


class GitOperationTimeoutError(GitOperationError):
    """Raised when a git operation times out"""

    pass


def safe_git_operation(
    operation: Callable[[], T],
    max_retries: int = 3,
    retry_delay: float = 0.5,
    operation_name: str = "git operation",
    timeout: Optional[float] = 30.0,
    max_total_timeout: Optional[float] = None,
) -> T:
    """
    Safely execute a git operation with retry logic and timeout protection.

    This function wraps git operations to handle SIGSEGV and other errors
    that can occur when using GitPython in multiprocessing contexts.

    Args:
        operation: The git operation to execute (callable that returns T)
        max_retries: Maximum number of retry attempts (default: 3)
        retry_delay: Initial delay between retries in seconds (default: 0.5)
        operation_name: Name of the operation for logging (default: "git operation")
        timeout: Maximum time in seconds to wait for operation (default: 30.0, None = no timeout)
        max_total_timeout: Maximum total time in seconds for all retries combined (default: None)

    Returns:
        The result of the operation

    Raises:
        GitOperationError: If the operation fails after all retries
        GitOperationTimeoutError: If the operation times out
    """
    last_exception = None
    start_time = time.time()

    def _execute_with_timeout():
        """Execute operation with timeout using ThreadPoolExecutor for better control"""
        if timeout is None:
            return operation()

        # Use ThreadPoolExecutor instead of raw threading for better timeout control
        # This allows the executor to properly manage the thread lifecycle
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"git-op-{operation_name[:20]}")
        try:
            future = executor.submit(operation)
            try:
                result = future.result(timeout=timeout)
                return result
            except FutureTimeoutError:
                # Cancel the future (though the thread may still run)
                future.cancel()
                logger.warning(
                    f"Git operation '{operation_name}' timed out after {timeout} seconds"
                )
                raise GitOperationTimeoutError(
                    f"Git operation '{operation_name}' timed out after {timeout} seconds"
                )
        finally:
            # Shutdown the executor - this will wait for the thread to finish
            # but in practice, if it's hanging, we'll continue anyway
            executor.shutdown(wait=False)  # Don't wait for hanging operations

    for attempt in range(max_retries):
        try:
            # Check if we've exceeded the maximum total timeout
            if max_total_timeout is not None:
                elapsed = time.time() - start_time
                if elapsed >= max_total_timeout:
                    logger.warning(
                        f"Git operation '{operation_name}' exceeded maximum total timeout "
                        f"of {max_total_timeout} seconds after {elapsed:.2f} seconds"
                    )
                    raise GitOperationTimeoutError(
                        f"Git operation '{operation_name}' exceeded maximum total timeout "
                        f"of {max_total_timeout} seconds"
                    )
            
            # Execute the operation with timeout
            result = _execute_with_timeout()
            if attempt > 0:
                logger.info(
                    f"Git operation '{operation_name}' succeeded on attempt {attempt + 1}"
                )
            return result

        except GitOperationTimeoutError as e:
            # Timeout - retry with exponential backoff
            logger.warning(
                f"Git operation '{operation_name}' timed out on attempt {attempt + 1}: {e}"
            )
            last_exception = e
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (2**attempt))  # Exponential backoff
                continue
            raise

        except SystemExit as e:
            # SystemExit might indicate a crash
            logger.warning(
                f"Git operation '{operation_name}' raised SystemExit on attempt {attempt + 1}: {e}"
            )
            last_exception = e
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (2**attempt))  # Exponential backoff
                continue
            raise GitSegfaultError(
                f"Git operation '{operation_name}' failed after {max_retries} attempts: {e}"
            ) from e

        except Exception as e:
            error_str = str(e).lower()

            # Check for common git errors that might indicate issues
            if any(
                keyword in error_str
                for keyword in [
                    "segmentation fault",
                    "sigsegv",
                    "signal 11",
                    "memory",
                    "corrupted",
                ]
            ):
                logger.warning(
                    f"Git operation '{operation_name}' may have crashed on attempt {attempt + 1}: {e}"
                )
                last_exception = e
                if attempt < max_retries - 1:
                    # Longer delay for potential crashes
                    time.sleep(retry_delay * (2**attempt) * 2)
                    continue
                raise GitSegfaultError(
                    f"Git operation '{operation_name}' failed after {max_retries} attempts: {e}"
                ) from e

            # For other exceptions, log and re-raise immediately
            logger.error(
                f"Git operation '{operation_name}' failed on attempt {attempt + 1}: {e}",
                exc_info=True,
            )
            raise

    # Should not reach here, but just in case
    raise GitOperationError(
        f"Git operation '{operation_name}' failed after {max_retries} attempts: {last_exception}"
    ) from last_exception


def safe_git_repo_operation(
    repo_path: str,
    operation: Callable[[Any], T],
    max_retries: int = 3,
    retry_delay: float = 0.5,
    operation_name: str = "git repo operation",
    timeout: Optional[float] = 30.0,
) -> T:
    """
    Safely execute a git operation on a repository with retry logic and timeout protection.

    This function creates a fresh Repo object for each attempt to avoid
    issues with stale repository handles in forked processes.

    Args:
        repo_path: Path to the git repository
        operation: Function that takes a Repo object and returns T
        max_retries: Maximum number of retry attempts (default: 3)
        retry_delay: Initial delay between retries in seconds (default: 0.5)
        operation_name: Name of the operation for logging (default: "git repo operation")
        timeout: Maximum time in seconds to wait for operation (default: 30.0, None = no timeout)

    Returns:
        The result of the operation

    Raises:
        GitOperationError: If the operation fails after all retries
        GitOperationTimeoutError: If the operation times out
    """
    from git import Repo, InvalidGitRepositoryError

    def _execute_with_fresh_repo():
        """Create a fresh Repo object and execute the operation"""
        try:
            repo = Repo(repo_path)
            return operation(repo)
        except InvalidGitRepositoryError:
            # Don't retry for invalid repos
            raise
        except Exception as e:
            # Close any open handles before retrying
            logger.debug(f"Error in git operation, will retry: {e}")
            raise

    return safe_git_operation(
        _execute_with_fresh_repo,
        max_retries=max_retries,
        retry_delay=retry_delay,
        operation_name=operation_name,
        timeout=timeout,
    )


def install_sigsegv_handler():
    """
    Install a signal handler for SIGSEGV to log crashes.

    Note: This is a best-effort attempt. SIGSEGV handlers may not always
    be reliable, but they can help with debugging.
    """

    def sigsegv_handler(signum, frame):
        logger.error(
            f"Received SIGSEGV (signal {signum}) in process {os.getpid()}. "
            "This may be related to GitPython/libgit2 operations in multiprocessing context."
        )
        # Re-raise to allow normal crash handling
        raise SystemExit(1)

    try:
        signal.signal(signal.SIGSEGV, sigsegv_handler)
        logger.debug("Installed SIGSEGV handler for git operations")
    except (ValueError, OSError) as e:
        # Some systems don't allow SIGSEGV handlers
        logger.debug(f"Could not install SIGSEGV handler: {e}")
