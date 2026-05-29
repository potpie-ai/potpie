"""Compatibility wrapper for the legacy app.modules.utils.logger import path."""

from __future__ import annotations

import os

from observability import configure, get_logger, log_context
from observability.redaction import SENSITIVE_PATTERNS, sanitize_log_text

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


def filter_sensitive_data(text: str) -> str:
    """Backward-compatible alias for observability redaction."""
    return sanitize_log_text(text)
