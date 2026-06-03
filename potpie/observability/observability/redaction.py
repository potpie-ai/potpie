"""Sensitive-data sanitization before logs reach any sink/infra API.

Runs as a logging.Filter so every handler/sink gets scrubbed messages and
structured fields. Keep this package app.*-free: the sanitizer only uses local
regexes and conservative replacements.
"""

from __future__ import annotations

import logging
import re

REDACTION = r"\1=***REDACTED***"
EMAIL_REDACTION = "***REDACTED_EMAIL***"
PHONE_REDACTION = "***REDACTED_PHONE***"
PERSON_NAME_REDACTION = r"\1=***REDACTED_NAME***"
API_KEY_REDACTION = "***REDACTED_API_KEY***"
CLOUD_KEY_REDACTION = "***REDACTED_CLOUD_KEY***"

# (compiled pattern, replacement).
SENSITIVE_PATTERNS: list[tuple[re.Pattern, str]] = [
    # Cloud provider key shapes that often leak without a key=value prefix.
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), CLOUD_KEY_REDACTION),
    (re.compile(r"\bASIA[0-9A-Z]{16}\b"), CLOUD_KEY_REDACTION),
    (re.compile(r"\bAIza[0-9A-Za-z_-]{30,45}\b"), CLOUD_KEY_REDACTION),
    (re.compile(r"\bya29\.[0-9A-Za-z_-]+\b"), CLOUD_KEY_REDACTION),
    (re.compile(r"\bgh[pousr]_[0-9A-Za-z_]{36,255}\b"), API_KEY_REDACTION),
    (re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"), API_KEY_REDACTION),
    (re.compile(r"\bxox[baprs]-[0-9A-Za-z-]{10,}\b"), API_KEY_REDACTION),
    (
        re.compile(
            r"-----BEGIN PRIVATE KEY-----.*?-----END PRIVATE KEY-----", re.DOTALL
        ),
        "-----BEGIN PRIVATE KEY-----***REDACTED***-----END PRIVATE KEY-----",
    ),
    # PII shapes.
    (
        re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
        EMAIL_REDACTION,
    ),
    (re.compile(r"\b(?:\+?\d[\d\s().-]{7,}\d)\b"), PHONE_REDACTION),
    (
        re.compile(
            r'\b(name|full_name|first_name|last_name)=["\']?([^"\'\s&]+)', re.IGNORECASE
        ),
        PERSON_NAME_REDACTION,
    ),
    # Common key=value / JSON / dict secret shapes.
    (re.compile(r'(password|passwd|pwd)=["\']?([^"\'\s&]+)', re.IGNORECASE), REDACTION),
    (
        re.compile(
            r'(token|access_token|refresh_token|id_token)=["\']?([^"\'\s&]+)',
            re.IGNORECASE,
        ),
        REDACTION,
    ),
    (
        re.compile(
            r'(secret|client_secret|api_secret)=["\']?([^"\'\s&]+)', re.IGNORECASE
        ),
        REDACTION,
    ),
    (re.compile(r'(api[_-]?key|apikey)=["\']?([^"\'\s&]+)', re.IGNORECASE), REDACTION),
    (
        re.compile(
            r"(aws_access_key_id|aws_secret_access_key|aws_session_token)"
            r'=["\']?([^"\'\s&]+)',
            re.IGNORECASE,
        ),
        REDACTION,
    ),
    (
        re.compile(
            r"(azure_account_key|azure_storage_key|accountkey)" r'=["\']?([^"\'\s&;]+)',
            re.IGNORECASE,
        ),
        REDACTION,
    ),
    (
        re.compile(r'(private_key|client_email)=["\']?([^"\'\s&]+)', re.IGNORECASE),
        REDACTION,
    ),
    (
        re.compile(
            r'(auth|authorization)=["\']?'
            r'((?:Bearer|Basic)\s+[^"\'\s&]+|[^"\'\s&]+)',
            re.IGNORECASE,
        ),
        REDACTION,
    ),
    (
        re.compile(r"Bearer\s+([A-Za-z0-9\-._~+/]+=*)", re.IGNORECASE),
        r"Bearer ***REDACTED***",
    ),
    (re.compile(r"Basic\s+([A-Za-z0-9+/]+=*)", re.IGNORECASE), r"Basic ***REDACTED***"),
    (
        re.compile(
            r"(redis|postgresql|mysql|mongodb)://([^:]+):([^@]+)@", re.IGNORECASE
        ),
        r"\1://\2:***REDACTED***@",
    ),
    (
        re.compile(r"([?&]code=)([A-Za-z0-9\-._~]{20,100})([&\s]|$)", re.IGNORECASE),
        r"\1***REDACTED***\3",
    ),
    (
        re.compile(
            r'("(?:password|token|secret|api_key|apiKey|private_key|'
            r"client_email|aws_access_key_id|aws_secret_access_key|"
            r'aws_session_token|azure_account_key|account_key)"\s*:\s*)"([^"]+)"',
            re.IGNORECASE,
        ),
        r'\1"***REDACTED***"',
    ),
    (
        re.compile(
            r"('(?:password|token|secret|api_key|apiKey|private_key|"
            r"client_email|aws_access_key_id|aws_secret_access_key|"
            r"aws_session_token|azure_account_key|account_key)'\s*:\s*)'([^']+)'",
            re.IGNORECASE,
        ),
        r"\1'***REDACTED***'",
    ),
]


def sanitize_log_text(text: str) -> str:
    """Scrub sensitive data from log text before it reaches infra sinks."""
    if not isinstance(text, str):
        return text
    out = text
    for pattern, replacement in SENSITIVE_PATTERNS:
        out = pattern.sub(replacement, out)
    return out


def redact(text: str) -> str:
    """Backward-compatible alias for sanitize_log_text()."""
    return sanitize_log_text(text)


def sanitize_log_value(value):
    """Scrub log values recursively before they reach infra sinks."""
    if isinstance(value, str):
        return sanitize_log_text(value)
    if isinstance(value, dict):
        return {k: sanitize_log_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(sanitize_log_value(v) for v in value)
    return value


class RedactionFilter(logging.Filter):
    """Scrubs the formatted message, string args, and structured fields.

    Exception traceback text is scrubbed by the sink formatters (they own
    exception rendering), mirroring the original sink-level behavior.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:
            msg = str(record.msg)
        record.msg = redact(msg)
        record.args = None
        for attr in ("obs_fields", "obs_context"):
            data = getattr(record, attr, None)
            if isinstance(data, dict):
                setattr(record, attr, sanitize_log_value(data))
        return True
