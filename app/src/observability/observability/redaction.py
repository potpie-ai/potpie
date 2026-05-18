"""Sensitive-data redaction.

Ported verbatim from app/modules/utils/logger.py SENSITIVE_PATTERNS /
filter_sensitive_data (already app.*-free). Runs as a logging.Filter so it
applies to every handler regardless of sink (stdlib-core requirement).

KNOWN GAP (unchanged from the audit, deliberately not silently "fixed" here):
patterns redact credentials/tokens but NOT emails or arbitrary user IDs. The
email/PII decision is tracked separately (audit-remediation issue), not in
this port — porting must not change behavior.
"""

from __future__ import annotations

import logging
import re

REDACTION = r"\1=***REDACTED***"

# (compiled pattern, replacement) — verbatim port.
SENSITIVE_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r'(password|passwd|pwd)=["\']?([^"\'\s&]+)', re.IGNORECASE),
     REDACTION),
    (re.compile(r'(token|access_token|refresh_token|id_token)=["\']?([^"\'\s&]+)',
                re.IGNORECASE),
     REDACTION),
    (re.compile(r'(secret|client_secret|api_secret)=["\']?([^"\'\s&]+)',
                re.IGNORECASE),
     REDACTION),
    (re.compile(r'(api[_-]?key|apikey)=["\']?([^"\'\s&]+)', re.IGNORECASE),
     REDACTION),
    (re.compile(r'(auth|authorization)=["\']?([^"\'\s&]+)', re.IGNORECASE),
     REDACTION),
    (re.compile(r"Bearer\s+([A-Za-z0-9\-._~+/]+=*)", re.IGNORECASE),
     r"Bearer ***REDACTED***"),
    (re.compile(r"Basic\s+([A-Za-z0-9+/]+=*)", re.IGNORECASE),
     r"Basic ***REDACTED***"),
    (re.compile(r"(redis|postgresql|mysql|mongodb)://([^:]+):([^@]+)@",
                re.IGNORECASE),
     r"\1://\2:***REDACTED***@"),
    (re.compile(r"([?&]code=)([A-Za-z0-9\-._~]{20,100})([&\s]|$)", re.IGNORECASE),
     r"\1***REDACTED***\3"),
    (re.compile(r'("(?:password|token|secret|api_key)"\s*:\s*)"([^"]+)"',
                re.IGNORECASE),
     r'\1"***REDACTED***"'),
    (re.compile(r"('(?:password|token|secret|api_key)'\s*:\s*)'([^']+)'",
                re.IGNORECASE),
     r"\1'***REDACTED***'"),
]


def redact(text: str) -> str:
    """Scrub sensitive data from a string. Non-strings pass through unchanged."""
    if not isinstance(text, str):
        return text
    out = text
    for pattern, replacement in SENSITIVE_PATTERNS:
        out = pattern.sub(replacement, out)
    return out


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
                setattr(record, attr, {
                    k: (redact(v) if isinstance(v, str) else v)
                    for k, v in data.items()
                })
        return True
