"""Unit tests for git_safe.py - safe git operation wrappers."""

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from unittest.mock import MagicMock, patch
import time

import pytest

from app.modules.code_provider.git_safe import (
    safe_git_operation,
    safe_git_repo_operation,
    install_sigsegv_handler,
    GitOperationError,
    GitSegfaultError,
    GitOperationTimeoutError,
)


pytestmark = pytest.mark.unit


class TestSafeGitOperation:
    """Tests for safe_git_operation function."""

    def test_immediate_success(self):
        """Operation succeeds on first attempt without retries."""
        operation = MagicMock(return_value="success_result")

        result = safe_git_operation(operation, max_retries=3)

        assert result == "success_result"
        operation.assert_called_once()

    def test_success_after_retry_timeout(self):
        """Operation succeeds after first attempt times out."""
        operation = MagicMock(return_value="success_result")

        with patch("app.modules.code_provider.git_safe.time.sleep"):
            with patch(
                "app.modules.code_provider.git_safe.ThreadPoolExecutor"
            ) as mock_executor_class:
                mock_executor = MagicMock()
                mock_future = MagicMock()
                mock_future.result.side_effect = [
                    FutureTimeoutError(),
                    "success_result",
                ]
                mock_executor.submit.return_value = mock_future
                mock_executor_class.return_value = mock_executor

                result = safe_git_operation(operation, max_retries=3, retry_delay=0.1)

        assert result == "success_result"
        assert mock_future.result.call_count == 2

    def test_max_retries_exceeded_timeout(self):
        """Operation fails after max retries with timeout errors."""
        operation = MagicMock(return_value="result")

        with patch("app.modules.code_provider.git_safe.time.sleep"):
            with patch(
                "app.modules.code_provider.git_safe.ThreadPoolExecutor"
            ) as mock_executor_class:
                mock_executor = MagicMock()
                mock_future = MagicMock()
                mock_future.result.side_effect = [
                    FutureTimeoutError(),
                    FutureTimeoutError(),
                    FutureTimeoutError(),
                ]
                mock_executor.submit.return_value = mock_future
                mock_executor_class.return_value = mock_executor

                with pytest.raises(GitOperationTimeoutError):
                    safe_git_operation(operation, max_retries=3, retry_delay=0.1)

    def test_individual_operation_timeout(self):
        """Operation times out on individual attempt."""
        operation = MagicMock()

        def slow_operation():
            time.sleep(10)
            return "result"

        with patch(
            "app.modules.code_provider.git_safe.ThreadPoolExecutor"
        ) as mock_executor_class:
            mock_executor = MagicMock()
            mock_future = MagicMock()
            mock_future.result.side_effect = FutureTimeoutError()
            mock_executor.submit.return_value = mock_future
            mock_executor_class.return_value = mock_executor

            with pytest.raises(GitOperationTimeoutError):
                safe_git_operation(slow_operation, timeout=0.1, max_retries=1)

    def test_max_total_timeout(self):
        """Operation fails when max_total_timeout is exceeded across retries."""
        operation = MagicMock(return_value="result")

        with patch("app.modules.code_provider.git_safe.time.time") as mock_time:
            mock_time.side_effect = [0, 5, 10, 15]
            with patch("app.modules.code_provider.git_safe.time.sleep"):
                with patch(
                    "app.modules.code_provider.git_safe.ThreadPoolExecutor"
                ) as mock_executor_class:
                    mock_executor = MagicMock()
                    mock_future = MagicMock()
                    mock_future.result.side_effect = [
                        FutureTimeoutError(),
                        FutureTimeoutError(),
                    ]
                    mock_executor.submit.return_value = mock_future
                    mock_executor_class.return_value = mock_executor

                    with pytest.raises(
                        GitOperationTimeoutError, match="exceeded maximum total timeout"
                    ):
                        safe_git_operation(
                            operation,
                            max_retries=3,
                            retry_delay=0.1,
                            max_total_timeout=5.0,
                        )

    def test_segfault_detection_and_retry(self):
        """Segmentation fault error triggers retry with longer delay."""
        operation = MagicMock(return_value="success_result")

        with patch("app.modules.code_provider.git_safe.time.sleep") as mock_sleep:
            with patch(
                "app.modules.code_provider.git_safe.ThreadPoolExecutor"
            ) as mock_executor_class:
                mock_executor = MagicMock()
                mock_future = MagicMock()
                mock_future.result.side_effect = [
                    Exception("git: Segmentation fault (core dumped)"),
                    "success_result",
                ]
                mock_executor.submit.return_value = mock_future
                mock_executor_class.return_value = mock_executor

                result = safe_git_operation(operation, max_retries=3, retry_delay=0.1)

        assert result == "success_result"
        assert mock_sleep.call_count == 1

    def test_sigsegv_detection(self):
        """SIGSEGV keyword in error triggers segfault retry path."""
        operation = MagicMock(return_value="success_result")

        with patch("app.modules.code_provider.git_safe.time.sleep"):
            with patch(
                "app.modules.code_provider.git_safe.ThreadPoolExecutor"
            ) as mock_executor_class:
                mock_executor = MagicMock()
                mock_future = MagicMock()
                mock_future.result.side_effect = [
                    Exception("git: SIGSEGV detected"),
                    "success_result",
                ]
                mock_executor.submit.return_value = mock_future
                mock_executor_class.return_value = mock_executor

                result = safe_git_operation(operation, max_retries=3, retry_delay=0.1)

        assert result == "success_result"

    def test_signal_11_detection(self):
        """'signal 11' keyword in error triggers segfault retry path."""
        operation = MagicMock(return_value="success_result")

        with patch("app.modules.code_provider.git_safe.time.sleep"):
            with patch(
                "app.modules.code_provider.git_safe.ThreadPoolExecutor"
            ) as mock_executor_class:
                mock_executor = MagicMock()
                mock_future = MagicMock()
                mock_future.result.side_effect = [
                    Exception("git process error: signal 11"),
                    "success_result",
                ]
                mock_executor.submit.return_value = mock_future
                mock_executor_class.return_value = mock_executor

                result = safe_git_operation(operation, max_retries=3, retry_delay=0.1)

        assert result == "success_result"

    def test_memory_error_detection(self):
        """'memory' keyword in error triggers segfault retry path."""
        operation = MagicMock(return_value="success_result")

        with patch("app.modules.code_provider.git_safe.time.sleep"):
            with patch(
                "app.modules.code_provider.git_safe.ThreadPoolExecutor"
            ) as mock_executor_class:
                mock_executor = MagicMock()
                mock_future = MagicMock()
                mock_future.result.side_effect = [
                    Exception("git: out of memory"),
                    "success_result",
                ]
                mock_executor.submit.return_value = mock_future
                mock_executor_class.return_value = mock_executor

                result = safe_git_operation(operation, max_retries=3, retry_delay=0.1)

        assert result == "success_result"

    def test_corrupted_error_detection(self):
        """'corrupted' keyword in error triggers segfault retry path."""
        operation = MagicMock(return_value="success_result")

        with patch("app.modules.code_provider.git_safe.time.sleep"):
            with patch(
                "app.modules.code_provider.git_safe.ThreadPoolExecutor"
            ) as mock_executor_class:
                mock_executor = MagicMock()
                mock_future = MagicMock()
                mock_future.result.side_effect = [
                    Exception("git: corrupted loose object"),
                    "success_result",
                ]
                mock_executor.submit.return_value = mock_future
                mock_executor_class.return_value = mock_executor

                result = safe_git_operation(operation, max_retries=3, retry_delay=0.1)

        assert result == "success_result"

    def test_segfault_error_raised_after_max_attempts(self):
        """GitSegfaultError raised when segfault persists after all retries."""
        operation = MagicMock(return_value="result")

        with patch("app.modules.code_provider.git_safe.time.sleep"):
            with patch(
                "app.modules.code_provider.git_safe.ThreadPoolExecutor"
            ) as mock_executor_class:
                mock_executor = MagicMock()
                mock_future = MagicMock()
                mock_future.result.side_effect = [
                    Exception("git: Segmentation fault"),
                    Exception("git: Segmentation fault"),
                    Exception("git: Segmentation fault"),
                ]
                mock_executor.submit.return_value = mock_future
                mock_executor_class.return_value = mock_executor

                with pytest.raises(GitSegfaultError, match="failed after 3 attempts"):
                    safe_git_operation(operation, max_retries=3, retry_delay=0.1)

    def test_systemexit_triggers_retry(self):
        """SystemExit during operation triggers retry logic."""
        operation = MagicMock(return_value="success_result")

        with patch("app.modules.code_provider.git_safe.time.sleep"):
            with patch(
                "app.modules.code_provider.git_safe.ThreadPoolExecutor"
            ) as mock_executor_class:
                mock_executor = MagicMock()
                mock_future = MagicMock()
                mock_future.result.side_effect = [SystemExit(1), "success_result"]
                mock_executor.submit.return_value = mock_future
                mock_executor_class.return_value = mock_executor

                result = safe_git_operation(operation, max_retries=3, retry_delay=0.1)

        assert result == "success_result"

    def test_systemexit_raises_gitsegfault_after_max_attempts(self):
        """SystemExit that persists raises GitSegfaultError after max retries."""
        operation = MagicMock(return_value="result")

        with patch("app.modules.code_provider.git_safe.time.sleep"):
            with patch(
                "app.modules.code_provider.git_safe.ThreadPoolExecutor"
            ) as mock_executor_class:
                mock_executor = MagicMock()
                mock_future = MagicMock()
                mock_future.result.side_effect = [
                    SystemExit(1),
                    SystemExit(1),
                    SystemExit(1),
                ]
                mock_executor.submit.return_value = mock_future
                mock_executor_class.return_value = mock_executor

                with pytest.raises(GitSegfaultError, match="failed after 3 attempts"):
                    safe_git_operation(operation, max_retries=3, retry_delay=0.1)

    def test_non_retryable_error_raises_immediately(self):
        """ValueError is not retryable and raises immediately."""
        operation = MagicMock(return_value="result")

        with patch(
            "app.modules.code_provider.git_safe.ThreadPoolExecutor"
        ) as mock_executor_class:
            mock_executor = MagicMock()
            mock_future = MagicMock()
            mock_future.result.side_effect = ValueError("invalid value")
            mock_executor.submit.return_value = mock_future
            mock_executor_class.return_value = mock_executor

            with pytest.raises(ValueError, match="invalid value"):
                safe_git_operation(operation, max_retries=3, timeout=30.0)

    def test_invalid_git_repository_error_raises_immediately(self):
        """InvalidGitRepositoryError is not retryable and raises immediately."""
        from git import InvalidGitRepositoryError

        operation = MagicMock(return_value="result")

        with patch(
            "app.modules.code_provider.git_safe.ThreadPoolExecutor"
        ) as mock_executor_class:
            mock_executor = MagicMock()
            mock_future = MagicMock()
            mock_future.result.side_effect = InvalidGitRepositoryError("not a git repo")
            mock_executor.submit.return_value = mock_future
            mock_executor_class.return_value = mock_executor

            with pytest.raises(InvalidGitRepositoryError):
                safe_git_operation(operation, max_retries=3, timeout=30.0)

    def test_operation_with_no_timeout(self):
        """Operation executes with no timeout when timeout=None."""
        operation = MagicMock(return_value="result")

        result = safe_git_operation(operation, timeout=None)

        assert result == "result"
        operation.assert_called_once()

    def test_exponential_backoff_on_timeout_retry(self):
        """Retry delay doubles with each timeout attempt (exponential backoff)."""
        operation = MagicMock(return_value="result")

        with patch("app.modules.code_provider.git_safe.time.sleep") as mock_sleep:
            with patch(
                "app.modules.code_provider.git_safe.ThreadPoolExecutor"
            ) as mock_executor_class:
                mock_executor = MagicMock()
                mock_future = MagicMock()
                mock_future.result.side_effect = [
                    FutureTimeoutError(),
                    FutureTimeoutError(),
                    FutureTimeoutError(),
                ]
                mock_executor.submit.return_value = mock_future
                mock_executor_class.return_value = mock_executor

                with pytest.raises(GitOperationTimeoutError):
                    safe_git_operation(operation, max_retries=3, retry_delay=0.5)

        assert mock_sleep.call_count == 2
        mock_sleep.assert_any_call(0.5 * (2**0))
        mock_sleep.assert_any_call(0.5 * (2**1))

    def test_thread_pool_executor_used_for_timeout(self):
        """ThreadPoolExecutor is used when timeout is set."""
        operation = MagicMock(return_value="result")

        with patch(
            "app.modules.code_provider.git_safe.ThreadPoolExecutor"
        ) as mock_executor_class:
            mock_executor = MagicMock()
            mock_future = MagicMock()
            mock_future.result.return_value = "result"
            mock_executor.submit.return_value = mock_future
            mock_executor_class.return_value = mock_executor

            result = safe_git_operation(operation, timeout=30.0)

        assert result == "result"
        mock_executor_class.assert_called_once()
        mock_executor.submit.assert_called_once_with(operation)
        mock_executor.shutdown.assert_called_once_with(wait=False)

    def test_executor_shutdown_on_exception(self):
        """Executor is shutdown even when operation raises exception."""
        operation = MagicMock(return_value="result")

        with patch(
            "app.modules.code_provider.git_safe.ThreadPoolExecutor"
        ) as mock_executor_class:
            mock_executor = MagicMock()
            mock_future = MagicMock()
            mock_future.result.side_effect = ValueError("fail")
            mock_executor.submit.return_value = mock_future
            mock_executor_class.return_value = mock_executor

            with pytest.raises(ValueError):
                safe_git_operation(operation, timeout=30.0)

            mock_executor.shutdown.assert_called_once_with(wait=False)

    def test_future_cancel_on_timeout(self):
        """Future is cancelled when operation times out."""
        operation = MagicMock(return_value="result")

        with patch(
            "app.modules.code_provider.git_safe.ThreadPoolExecutor"
        ) as mock_executor_class:
            mock_executor = MagicMock()
            mock_future = MagicMock()
            mock_future.result.side_effect = FutureTimeoutError()
            mock_executor.submit.return_value = mock_future
            mock_executor_class.return_value = mock_executor

            with pytest.raises(GitOperationTimeoutError):
                safe_git_operation(operation, timeout=0.1, max_retries=1)

            assert mock_future.cancel.call_count >= 1

    def test_success_on_later_attempt_logs_info(self):
        """Successful retry after failure logs info message."""
        operation = MagicMock(return_value="success")

        with patch("app.modules.code_provider.git_safe.time.sleep"):
            with patch(
                "app.modules.code_provider.git_safe.ThreadPoolExecutor"
            ) as mock_executor_class:
                mock_executor = MagicMock()
                mock_future = MagicMock()
                mock_future.result.side_effect = [
                    FutureTimeoutError(),
                    "success",
                ]
                mock_executor.submit.return_value = mock_future
                mock_executor_class.return_value = mock_executor

                with patch("app.modules.code_provider.git_safe.logger") as mock_logger:
                    result = safe_git_operation(
                        operation,
                        max_retries=3,
                        retry_delay=0.1,
                        operation_name="test_op",
                    )

        assert result == "success"
        info_calls = [str(c) for c in mock_logger.info.call_args_list]
        assert any("succeeded on attempt 2" in c for c in info_calls)


class TestSafeGitRepoOperation:
    """Tests for safe_git_repo_operation function."""

    def test_repo_instantiation_per_attempt(self):
        """A fresh Repo object is created for each attempt."""
        mock_operation = MagicMock(return_value="result")

        with patch(
            "app.modules.code_provider.git_safe.safe_git_operation"
        ) as mock_safe_op:
            mock_safe_op.return_value = "result"

            result = safe_git_repo_operation(
                "/some/path", mock_operation, max_retries=3, timeout=30.0
            )

        assert result == "result"
        mock_safe_op.assert_called_once()
        args, kwargs = mock_safe_op.call_args
        assert kwargs["max_retries"] == 3
        assert kwargs["timeout"] == 30.0

    def test_fresh_repo_lifecycle(self):
        """Operation receives fresh Repo object each time."""
        mock_repo = MagicMock()
        mock_operation = MagicMock(return_value="result")

        with patch("git.Repo", return_value=mock_repo) as mock_repo_cls:
            result = safe_git_repo_operation(
                "/some/path", mock_operation, max_retries=1, timeout=30.0
            )

        assert result == "result"
        mock_repo_cls.assert_called_once_with("/some/path")

    def test_invalid_repository_raises_immediately(self):
        """InvalidGitRepositoryError is raised without retry."""
        from git import InvalidGitRepositoryError

        mock_operation = MagicMock(return_value="result")

        with patch(
            "app.modules.code_provider.git_safe.safe_git_operation"
        ) as mock_safe_op:
            mock_safe_op.side_effect = InvalidGitRepositoryError("not a repo")

            with pytest.raises(InvalidGitRepositoryError):
                safe_git_repo_operation("/some/path", mock_operation, max_retries=3)

    def test_repo_error_triggers_retry(self):
        """Errors during repo operation trigger retry."""
        mock_operation = MagicMock(return_value="result")

        with patch(
            "app.modules.code_provider.git_safe.safe_git_operation"
        ) as mock_safe_op:
            mock_safe_op.return_value = "result"

            result = safe_git_repo_operation(
                "/some/path", mock_operation, max_retries=3, timeout=30.0
            )

        assert result == "result"


class TestInstallSigsegvHandler:
    """Tests for install_sigsegv_handler function."""

    def test_install_sigsegv_handler_does_not_crash(self):
        """install_sigsegv_handler should not raise any exception."""
        try:
            install_sigsegv_handler()
        except Exception as e:
            pytest.fail(f"install_sigsegv_handler raised exception: {e}")

    def test_install_sigsegv_handler_logs_debug(self):
        """install_sigsegv_handler logs debug message on success."""
        with patch("app.modules.code_provider.git_safe.signal.signal") as mock_signal:
            with patch("app.modules.code_provider.git_safe.logger") as mock_logger:
                mock_signal.return_value = None

                install_sigsegv_handler()

                mock_signal.assert_called_once()
                mock_logger.debug.assert_called()

    def test_install_sigsegv_handler_handles_value_error(self):
        """install_sigsegv_handler handles ValueError gracefully."""
        with patch(
            "app.modules.code_provider.git_safe.signal.signal",
            side_effect=ValueError("unsupported signal"),
        ):
            with patch("app.modules.code_provider.git_safe.logger") as mock_logger:
                install_sigsegv_handler()
                mock_logger.debug.assert_called()

    def test_install_sigsegv_handler_handles_os_error(self):
        """install_sigsegv_handler handles OSError gracefully."""
        with patch(
            "app.modules.code_provider.git_safe.signal.signal",
            side_effect=OSError("permission denied"),
        ):
            with patch("app.modules.code_provider.git_safe.logger") as mock_logger:
                install_sigsegv_handler()
                mock_logger.debug.assert_called()


class TestGitSafeExceptions:
    """Tests for exception classes."""

    def test_git_operation_error_is_exception(self):
        """GitOperationError is a proper Exception subclass."""
        err = GitOperationError("test error")
        assert isinstance(err, Exception)
        assert str(err) == "test error"

    def test_git_segfault_error_is_git_operation_error(self):
        """GitSegfaultError inherits from GitOperationError."""
        err = GitSegfaultError("segfault error")
        assert isinstance(err, GitOperationError)
        assert isinstance(err, Exception)
        assert str(err) == "segfault error"

    def test_git_operation_timeout_error_is_git_operation_error(self):
        """GitOperationTimeoutError inherits from GitOperationError."""
        err = GitOperationTimeoutError("timeout error")
        assert isinstance(err, GitOperationError)
        assert isinstance(err, Exception)
        assert str(err) == "timeout error"
