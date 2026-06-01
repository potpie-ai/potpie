"""Production sink: flat JSONL to stdout. Ports logger.py production_log_sink.

Flattens obs_context + obs_fields to top level (the audit's 'structure lost'
fix), serialises exception type/value/traceback, redacts exception text when
config.redact is on (message/fields are already scrubbed by RedactionFilter).
default=str so non-serialisable extras never crash logging.

GAP (unchanged): stdout JSONL is inert without an aggregator — where it ships
is a separate, untracked decision, not solved here.
"""

from __future__ import annotations

import json
import logging
import sys
import traceback
from datetime import datetime, timezone

from ..config import ObservabilityConfig
from ..redaction import redact

_STD = set(vars(logging.makeLogRecord({})))


def truncate_traceback_text(text: str, max_lines: int = 10) -> str:
    """Keep the traceback tail so production JSON logs stay bounded."""
    if max_lines < 1:
        max_lines = 1
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    return "\n".join(lines[-max_lines:])


class JsonFormatter(logging.Formatter):
    def __init__(self, redact_exc: bool) -> None:
        super().__init__()
        self._redact_exc = redact_exc

    def format(self, record: logging.LogRecord) -> str:
        data = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "function": record.funcName,
            "line": record.lineno,
            "message": record.getMessage(),
        }
        for attr in ("obs_context", "obs_fields"):
            extra = getattr(record, attr, None)
            if isinstance(extra, dict):
                for key, value in extra.items():
                    target_key = f"extra_{key}" if key in data else key
                    data[target_key] = value
        if record.exc_info:
            etype, eval_, _ = record.exc_info
            tb = truncate_traceback_text(
                "".join(traceback.format_exception(*record.exc_info))
            )
            data["exception"] = {
                "type": getattr(etype, "__name__", str(etype)),
                "value": redact(str(eval_)) if self._redact_exc else str(eval_),
                "traceback": redact(tb) if self._redact_exc else tb,
            }
        return json.dumps(data, default=str)


class JsonStdoutSink:
    name = "json_stdout"

    def setup(self, config: ObservabilityConfig) -> None:
        return None

    def build_handler(self, config: ObservabilityConfig) -> logging.Handler | None:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter(redact_exc=config.redact))
        return handler

    def instrument(self, config: ObservabilityConfig) -> None:
        return None

    def shutdown(self, config: ObservabilityConfig) -> None:
        return None
