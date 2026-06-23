from __future__ import annotations

import re

SENSITIVE_PATTERNS: list[tuple[re.Pattern, str]] = [
    (
        re.compile(r"(password|passwd|pwd)=['\"]?([^'\"\s&]+)", re.IGNORECASE),
        r"\1=***REDACTED***",
    ),
    (
        re.compile(
            r"(token|access_token|refresh_token|id_token)=['\"]?([^'\"\s&]+)",
            re.IGNORECASE,
        ),
        r"\1=***REDACTED***",
    ),
    (
        re.compile(
            r"(secret|client_secret|api[_-]?key|apikey)=['\"]?([^'\"\s&]+)",
            re.IGNORECASE,
        ),
        r"\1=***REDACTED***",
    ),
    (
        re.compile(r"Bearer\s+([A-Za-z0-9\-._~+/]+=*)", re.IGNORECASE),
        "Bearer ***REDACTED***",
    ),
]


def sanitize_log_text(text: str) -> str:
    if not isinstance(text, str):
        return text
    out = text
    for pattern, replacement in SENSITIVE_PATTERNS:
        out = pattern.sub(replacement, out)
    return out


def redact(text: str) -> str:
    return sanitize_log_text(text)
