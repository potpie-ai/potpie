"""Unit tests for app.modules.utils.logger."""
from unittest.mock import MagicMock

import pytest

from app.modules.utils import logger as logger_module
from app.modules.utils.logger import (
    SHOW_STACK_TRACES,
    SENSITIVE_PATTERNS,
    filter_sensitive_data,
    filter_sensitive_value,
)


pytestmark = pytest.mark.unit


class TestFilterSensitiveData:
    def test_returns_non_string_unchanged(self):
        assert filter_sensitive_data(123) == 123
        assert filter_sensitive_data(None) is None

    def test_redacts_password_equals(self):
        out = filter_sensitive_data('login password=secret123 ok')
        assert "secret123" not in out
        assert "***REDACTED***" in out

    def test_redacts_token_equals(self):
        out = filter_sensitive_data('access_token=abc123xyz')
        assert "abc123xyz" not in out
        assert "***REDACTED***" in out

    def test_redacts_bearer_token(self):
        out = filter_sensitive_data('Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.xxx')
        assert "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9" not in out
        assert "Bearer ***REDACTED***" in out

    def test_redacts_api_key(self):
        out = filter_sensitive_data('api_key=sk-12345')
        assert "sk-12345" not in out
        assert "***REDACTED***" in out

    def test_passes_through_safe_text(self):
        msg = "User logged in successfully"
        assert filter_sensitive_data(msg) == msg

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (
                "postgres://db-user:db-password@database.internal/app",
                "postgres://db-user:***REDACTED***@database.internal/app",
            ),
            (
                "rediss://cache-user:cache-password@cache.internal/0",
                "rediss://cache-user:***REDACTED***@cache.internal/0",
            ),
            (
                "mongodb+srv://mongo-user:mongo-password@cluster.example/app",
                "mongodb+srv://mongo-user:***REDACTED***@cluster.example/app",
            ),
            (
                "postgresql+asyncpg://db-user:db-password@database.internal/app",
                "postgresql+asyncpg://db-user:***REDACTED***@database.internal/app",
            ),
            (
                "amqps://queue-user:queue-password@broker.internal/vhost",
                "amqps://queue-user:***REDACTED***@broker.internal/vhost",
            ),
        ],
    )
    def test_connection_url_password_is_redacted(self, value, expected):
        assert filter_sensitive_data(value) == expected


class TestFilterSensitiveValue:
    def test_nested_values_and_sensitive_keys_are_redacted(self):
        value = {
            "headers": {
                "Authorization": "Bearer live-token",
                "X-Request-ID": "request-123",
            },
            "credentials": {
                "client_secret": "oauth-secret",
                "password": "database-password",
            },
        }

        filtered = filter_sensitive_value(value)

        assert filtered["headers"]["Authorization"] == "***REDACTED***"
        assert filtered["headers"]["X-Request-ID"] == "request-123"
        assert filtered["credentials"] == "***REDACTED***"

    @pytest.mark.parametrize(
        "key",
        [
            "token",
            "auth_token",
            "private_key",
            "client_id",
            "credential",
            "session_id",
        ],
    )
    def test_sensitive_structured_field_is_redacted(self, key):
        assert filter_sensitive_value("raw-credential", key) == "***REDACTED***"

    def test_bytes_value_is_decoded_and_redacted(self):
        assert (
            filter_sensitive_value(b"password=database-password")
            == "password=***REDACTED***"
        )

    def test_tuple_value_preserves_type_and_redacts_nested_fields(self):
        filtered = filter_sensitive_value(("safe", {"token": "raw-token"}))

        assert filtered == ("safe", {"token": "***REDACTED***"})

    def test_set_value_preserves_type_and_redacts_members(self):
        filtered = filter_sensitive_value({"safe", "Bearer raw-token"})

        assert filtered == {"safe", "Bearer ***REDACTED***"}

    def test_self_referential_value_is_redacted_without_recursion_error(self):
        value = {"safe": "visible"}
        value["self"] = value

        filtered = filter_sensitive_value(value)

        assert filtered == {
            "safe": "visible",
            "self": "***REDACTED***",
        }

    def test_excessively_nested_value_is_redacted(self):
        value = "deep-value"
        for _ in range(100):
            value = {"nested": [({"value"}, value)]}

        filtered = filter_sensitive_value(value)

        assert "***REDACTED***" in repr(filtered)
        assert "deep-value" not in repr(filtered)


class TestLoggerConstants:
    def test_sensitive_patterns_non_empty(self):
        assert len(SENSITIVE_PATTERNS) > 0

    def test_show_stack_traces_bool(self):
        assert isinstance(SHOW_STACK_TRACES, bool)

    @pytest.mark.parametrize("diagnose_enabled", [False, True])
    def test_development_sink_uses_diagnose_opt_in(
        self, monkeypatch, diagnose_enabled
    ):
        base_logger = MagicMock()
        patched_logger = MagicMock()
        base_logger.patch.return_value = patched_logger

        monkeypatch.setenv("ENV", "development")
        monkeypatch.setattr(logger_module, "_LOGGING_CONFIGURED", False)
        monkeypatch.setattr(logger_module, "_logger", base_logger)
        monkeypatch.setattr(
            logger_module,
            "ENABLE_LOG_DIAGNOSE",
            diagnose_enabled,
            raising=False,
        )
        monkeypatch.setattr(logger_module.logging, "basicConfig", MagicMock())
        monkeypatch.setattr(
            logger_module.logging,
            "getLogger",
            MagicMock(return_value=MagicMock()),
        )

        logger_module.configure_logging()

        patched_logger.add.assert_called_once()
        assert patched_logger.add.call_args.kwargs["diagnose"] is diagnose_enabled
