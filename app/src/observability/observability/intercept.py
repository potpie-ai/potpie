"""Bridge: route third-party stdlib `logging` records into our sinks.

CONTRACT: lets ambient `logging.getLogger(__name__)` users (incl. context-
engine, uvicorn, sqlalchemy, celery) flow through the configured sinks with
correct level and redaction — no caller changes needed.

EDGE CASES (from the audit's current InterceptHandler):
 - Frame-depth walk must skip the logging module so file/line point at the
   real caller.
 - Do NOT use logging.basicConfig(force=True) blindly (EC2): it nukes all
   handlers and breaks idempotency. Install our handler, tag it, and on
   re-configure remove only our tagged handler.
 - Per-library levels (uvicorn/sqlalchemy/httpx/...) are applied here, mirror
   the audit's existing dial-down map.
"""

from __future__ import annotations

import logging

HANDLER_TAG = "observability.managed"


class InterceptHandler(logging.Handler):
    """STUB (Phase 1): contract only. Ported from current logger.py."""

    def emit(self, record: logging.LogRecord) -> None:
        raise NotImplementedError("Phase 1 scaffold — ported in Phase 2")


def install_intercept(level: str, library_levels: dict[str, str]) -> None:
    """STUB (Phase 1): idempotent, tag-based handler management (EC2)."""
    raise NotImplementedError("Phase 1 scaffold — implemented in Phase 2")
