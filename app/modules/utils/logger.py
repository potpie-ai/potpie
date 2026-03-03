import json
import logging
import os
import re
import sys
from contextlib import contextmanager
from typing import Optional

from loguru import logger as _loguru_logger

_LOGGING_CONFIGURED = False
_logger = _loguru_logger

# Control whether to include stack traces in error logs
# Set LOG_STACK_TRACES=false to disable stack traces
SHOW_STACK_TRACES = os.getenv("LOG_STACK_TRACES", "true").lower() in (
    "true",
    "1",
    "yes",
)

# Sensitive data patterns to redact in logs
SENSITIVE_PATTERNS = [
    # Credentials in key=value format
    (
        re.compile(r'(password|passwd|pwd)=["\']?([^"\'\s&]+)', re.IGNORECASE),
        r"\1=***REDACTED***",
    ),
    (
        re.compile(
            r'(token|access_token|refresh_token|id_token)=["\']?([^"\'\s&]+)',
            re.IGNORECASE,
        ),
        r"\1=***REDACTED***",
    ),
    (
        re.compile(
            r'(secret|client_secret|api_secret)=["\']?([^"\'\s&]+)', re.IGNORECASE
        ),
        r"\1=***REDACTED***",
    ),
    (
        re.compile(r'(api[_-]?key|apikey)=["\']?([^"\'\s&]+)', re.IGNORECASE),
        r"\1=***REDACTED***",
    ),
    (
        re.compile(r'(auth|authorization)=["\']?([^"\'\s&]+)', re.IGNORECASE),
        r"\1=***REDACTED***",
    ),
    # Bearer tokens
    (
        re.compile(r"Bearer\s+([A-Za-z0-9\-._~+/]+=*)", re.IGNORECASE),
        r"Bearer ***REDACTED***",
    ),
    # Basic auth
    (re.compile(r"Basic\s+([A-Za-z0-9+/]+=*)", re.IGNORECASE), r"Basic ***REDACTED***"),
    # Redis/Database URLs with passwords
    (
        re.compile(
            r"(redis|postgresql|mysql|mongodb)://([^:]+):([^@]+)@", re.IGNORECASE
        ),
        r"\1://\2:***REDACTED***@",
    ),
    # OAuth authorization codes (typically 20-100 chars alphanumeric)
    (
        re.compile(r"([?&]code=)([A-Za-z0-9\-._~]{20,100})([&\s]|$)", re.IGNORECASE),
        r"\1***REDACTED***\3",
    ),
    # Generic secrets in quotes
    (
        re.compile(
            r'("(?:password|token|secret|api_key)"\s*:\s*)"([^"]+)"', re.IGNORECASE
        ),
        r'\1"***REDACTED***"',
    ),
    (
        re.compile(
            r"('(?:password|token|secret|api_key)'\s*:\s*)'([^']+)'", re.IGNORECASE
        ),
        r"\1'***REDACTED***'",
    ),
]


def filter_sensitive_data(text: str) -> str:
    """
    Filter sensitive data from log messages.

    Args:
        text: Log message text to filter

    Returns:
        Filtered text with sensitive data redacted
    """
    if not isinstance(text, str):
        return text

    filtered = text
    for pattern, replacement in SENSITIVE_PATTERNS:
        filtered = pattern.sub(replacement, filtered)

    return filtered


def production_log_sink(message):
    """Custom sink for production that outputs flat JSON format for better machine readability.

    When serialize=True, loguru outputs JSON string. We parse it and reformat as flat JSON
    for easier parsing by log aggregation tools (ELK, Datadog, Splunk, CloudWatch, etc.).

    Also filters sensitive data patterns to prevent credential leakage.
    """
    try:
        # Parse the serialized JSON from loguru
        full_record = json.loads(message)
        record = full_record.get("record", full_record)
    except (json.JSONDecodeError, AttributeError):
        # Fallback: if message is not JSON, output as-is (shouldn't happen with serialize=True)
        sys.stdout.write(message)
        sys.stdout.flush()
        return

    # Extract exception info if present
    exception = None
    exc = record.get("exception")
    if exc:
        tb = str(exc.get("traceback", ""))
        tb_truncated = "\n".join(tb.splitlines()[-10:]) if tb else ""
        exception = {
            "type": (
                exc.get("type", {}).get("name", "Exception")
                if isinstance(exc.get("type"), dict)
                else str(exc.get("type", "Exception"))
            ),
            "value": filter_sensitive_data(str(exc.get("value", ""))),
            "traceback": filter_sensitive_data(tb_truncated),
        }

    # Build flat JSON structure - easier for log parser
