"""Compatibility wrapper for the legacy app.modules.utils.logger import path."""

from __future__ import annotations

import json
import os
import sys
from typing import Any

from observability import configure, get_logger, log_context as log_context
from observability.redaction import (
    SENSITIVE_PATTERNS as SENSITIVE_PATTERNS,
    sanitize_log_text,
)

SHOW_STACK_TRACES = os.getenv("LOG_STACK_TRACES", "true").lower() in (
    "true",
    "1",
    "yes",
)


def setup_logger(name: str):
    """Return the new observability structured logger under the legacy name."""
    return get_logger(name)


def configure_logging(level: str | None = None) -> None:
    """Legacy entrypoint retained for modules that have not migrated yet."""
    del level
    configure()


def filter_sensitive_data(text: Any) -> Any:
    """Backward-compatible alias for observability redaction."""
    if not isinstance(text, str):
        return text
    return sanitize_log_text(text)


def truncate_traceback(text: Any, max_lines: int = 10) -> Any:
    """Keep only the most relevant tail of a traceback for production logs."""
    if not isinstance(text, str):
        return text
    if max_lines < 1:
        max_lines = 1

    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    return "\n".join(lines[-max_lines:])


def production_log_sink(message: str) -> None:
    """Legacy JSONL sink retained for callers that import it directly."""
    try:
        full_record = json.loads(message)
        record = full_record.get("record", full_record)
    except (json.JSONDecodeError, AttributeError):
        sys.stdout.write(message)
        sys.stdout.flush()
        return

    exception = None
    exc = record.get("exception")
    if exc:
        exception = {
            "type": (
                exc.get("type", {}).get("name", "Exception")
                if isinstance(exc.get("type"), dict)
                else str(exc.get("type", "Exception"))
            ),
            "value": filter_sensitive_data(str(exc.get("value", ""))),
            "traceback": filter_sensitive_data(
                truncate_traceback(str(exc.get("traceback", "")))
            ),
        }

    log_data = {
        "timestamp": record.get("time", {}).get("repr", ""),
        "level": record.get("level", {}).get("name", "INFO"),
        "logger": record.get("extra", {}).get("name", record.get("name", "unknown")),
        "function": record.get("function", ""),
        "line": record.get("line", 0),
        "message": filter_sensitive_data(str(record.get("message", ""))),
    }

    extras = record.get("extra", {})
    for key, value in extras.items():
        if key == "name":
            continue
        log_data[key] = (
            filter_sensitive_data(value) if isinstance(value, str) else value
        )

    if exception:
        log_data["exception"] = exception

    sys.stdout.write(json.dumps(log_data, default=str) + "\n")
    sys.stdout.flush()
