"""Central logging configuration — the one place logging is set up.

Before this module the codebase had ~200 ad-hoc ``getLogger(__name__)``
sites and *no* configuration: format and level were whatever uvicorn /
root defaults happened to be, and the one structured ``extra={"audit":...}``
payload was dropped on the floor (no formatter rendered it).

:func:`configure_logging` is idempotent and shared by the HTTP and MCP
entrypoints. It is deliberately **stdlib-only** — no structlog hard dependency.
The ~200 existing ``getLogger`` sites get structured JSON +
trace correlation for free because the *handler/formatter* is the swappable
seam, not the logger objects. The active correlation ids (see
:mod:`potpie_context_engine.bootstrap.observability_context`) are injected into every record via a
filter, so any log line can be tied back to its event / pot / batch.

Env (fail-safe defaults — current behavior preserved unless opted in):

* ``CONTEXT_ENGINE_LOG_LEVEL``   — default ``INFO``
* ``CONTEXT_ENGINE_LOG_FORMAT``  — ``plain`` (default) | ``json``
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from potpie_context_engine.bootstrap.observability_context import CORRELATION_KEYS, get_correlation

_CONFIGURED = False

# LogRecord attributes that are intrinsic — everything else a caller passed
# via ``extra=`` is treated as a structured field in JSON mode.
_RESERVED = set(vars(logging.makeLogRecord({})).keys()) | {
    "message",
    "asctime",
    "taskName",
}


class CorrelationFilter(logging.Filter):
    """Attach the active correlation ids to every record as attributes."""

    def filter(self, record: logging.LogRecord) -> bool:
        corr = get_correlation()
        for key in CORRELATION_KEYS:
            if not hasattr(record, key):
                setattr(record, key, corr.get(key))
        return True


class JsonFormatter(logging.Formatter):
    """One JSON object per line: timestamp, level, logger, msg, correlation,
    plus any structured ``extra=`` fields (including the audit channel)."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key in CORRELATION_KEYS:
            val = getattr(record, key, None)
            if val is not None:
                payload[key] = val
        for key, value in record.__dict__.items():
            if key in _RESERVED or key in CORRELATION_KEYS:
                continue
            if key.startswith("_"):
                continue
            payload[key] = _jsonify(value)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, separators=(",", ":"))


def _jsonify(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except Exception:  # noqa: BLE001
        return repr(value)


def configure_logging() -> None:
    """Install the root handler + correlation filter. Idempotent."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    level_name = os.getenv("CONTEXT_ENGINE_LOG_LEVEL", "INFO").strip().upper()
    level = getattr(logging, level_name, logging.INFO)
    fmt = os.getenv("CONTEXT_ENGINE_LOG_FORMAT", "plain").strip().lower()

    handler = logging.StreamHandler()
    handler.addFilter(CorrelationFilter())
    if fmt == "json":
        handler.setFormatter(JsonFormatter())
    else:
        # Plain mode preserves prior behavior but still surfaces the trace.
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)s %(name)s "
                "[trace=%(trace_id)s event=%(event_id)s pot=%(pot_id)s] "
                "%(message)s"
            )
        )

    root = logging.getLogger()
    # Replace any pre-existing handlers so we own the format deterministically.
    for h in list(root.handlers):
        root.removeHandler(h)
    root.addHandler(handler)
    root.setLevel(level)

    _CONFIGURED = True
