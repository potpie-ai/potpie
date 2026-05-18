"""Sensitive-data redaction.

CONTRACT: redaction runs as a logging.Filter attached to every handler, so it
applies regardless of which sink is active (stdlib-core requirement).

EDGE CASES / GAPS (from the audit):
 - Must scrub the *formatted* message (record.getMessage()), the raw args,
   AND exc_info traceback text — secrets leak in tracebacks too.
 - Current regex patterns redact credentials/tokens but NOT emails or user
   IDs; the audit found 18+ email-at-INFO sites. Decision needed in Phase 2:
   add an email/PII pattern (may be noisy) vs. rely on callers. Tracked as a
   gap, not silently inherited.
 - Patterns will be ported verbatim from the current
   app/modules/utils/logger.py SENSITIVE_PATTERNS (already app.*-free).
"""

from __future__ import annotations

import logging

SENSITIVE_PATTERNS: list = []  # ported in Phase 2


def redact(text: str) -> str:
    """STUB (Phase 1): contract only."""
    raise NotImplementedError("Phase 1 scaffold — ported in Phase 2")


class RedactionFilter(logging.Filter):
    """STUB (Phase 1): contract only. Scrubs message + args + exc_text."""

    def filter(self, record: logging.LogRecord) -> bool:
        raise NotImplementedError("Phase 1 scaffold — implemented in Phase 2")
