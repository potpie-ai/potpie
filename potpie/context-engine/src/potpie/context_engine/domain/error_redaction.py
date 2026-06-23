"""Strip credentials from error text before it crosses a trust boundary.

PyGithub / httpx / git exception strings routinely embed tokenized clone
URLs (``https://x-access-token:<token>@host/...``), ``Authorization``
headers and bearer tokens. Returning that text to the agent re-enters it
into the model context and can be persisted into graph properties; in HTTP
500 bodies it leaks to the caller. Run every outbound error string through
:func:`redact_secrets` first (security review M-1 / L-1).
"""

from __future__ import annotations

import re

_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    # user:password@host  (covers x-access-token:<tok>@, oauth2:<tok>@, …)
    (re.compile(r"://[^/\s:@]+:[^/\s@]+@"), "://***:***@"),
    # GitHub tokens (classic / fine-grained / oauth / server / refresh)
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}"), "***"),
    (re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}"), "***"),
    # Bearer / token / apikey style "key = value" pairs. The value run is
    # masked to the end of its segment so multi-token values like
    # ``Authorization: Bearer <tok>`` are fully removed.
    (
        re.compile(
            r"(?i)\b(authorization|bearer|token|api[_-]?key|secret|"
            r"password|access[_-]?token)\b\s*[:=]\s*"
            r"[^\r\n,;\"'\s][^\r\n,;\"']*"
        ),
        r"\1=***",
    ),
)


def redact_secrets(text: str | None) -> str:
    """Return ``text`` with embedded credentials masked."""
    if not text:
        return ""
    out = str(text)
    for pat, repl in _PATTERNS:
        out = pat.sub(repl, out)
    return out


def safe_error(exc: object, *, limit: int = 300) -> str:
    """Redacted, length-bounded message for an exception."""
    return redact_secrets(str(exc))[:limit]
