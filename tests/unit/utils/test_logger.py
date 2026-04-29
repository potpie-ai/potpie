"""Unit tests for app.modules.utils.logger."""
import json
import pytest

from app.modules.utils.logger import (
    SHOW_STACK_TRACES,
    SENSITIVE_PATTERNS,
    filter_sensitive_data,
    production_log_sink,
    truncate_traceback,
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


class TestLoggerConstants:
    def test_sensitive_patterns_non_empty(self):
        assert len(SENSITIVE_PATTERNS) > 0

    def test_show_stack_traces_bool(self):
        assert isinstance(SHOW_STACK_TRACES, bool)


class TestProductionLogSink:
    def test_truncate_traceback_keeps_last_lines(self):
        traceback_text = "\n".join(f"line {i}" for i in range(12))

        assert truncate_traceback(traceback_text, max_lines=10) == "\n".join(
            f"line {i}" for i in range(2, 12)
        )

    def test_production_log_sink_truncates_exception_traceback(self, capsys):
        traceback_text = "\n".join(f"trace {i}" for i in range(15))
        payload = {
            "record": {
                "time": {"repr": "2026-04-29 12:00:00"},
                "level": {"name": "ERROR"},
                "name": "test.logger",
                "function": "failing_fn",
                "line": 42,
                "message": "something failed",
                "extra": {},
                "exception": {
                    "type": {"name": "ValueError"},
                    "value": "boom",
                    "traceback": traceback_text,
                },
            }
        }

        production_log_sink(json.dumps(payload))

        out = capsys.readouterr().out.strip()
        log_data = json.loads(out)
        assert log_data["exception"]["traceback"] == "\n".join(
            f"trace {i}" for i in range(5, 15)
        )
