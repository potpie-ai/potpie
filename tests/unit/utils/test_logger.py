"""Unit tests for app.modules.utils.logger."""
import pytest

from app.modules.utils.logger import filter_sensitive_data, SENSITIVE_PATTERNS, SHOW_STACK_TRACES


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


class TestLoggerConstants:
    def test_sensitive_patterns_non_empty(self):
        assert len(SENSITIVE_PATTERNS) > 0

    def test_show_stack_traces_bool(self):
        assert isinstance(SHOW_STACK_TRACES, bool)
