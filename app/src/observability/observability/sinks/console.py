"""Development sink: human-readable line. Ports logger.py dev sink/_filter.

Format: `TS | LEVEL    | logger:func:line | message | k: v, ...`
Context + structured fields are appended to the tail (mirrors the original
dev `_filter` behavior). Minimal ANSI level color only when stdout is a TTY.
"""

from __future__ import annotations

import logging
import sys
import traceback
from datetime import datetime

from ..config import ObservabilityConfig
from ..redaction import redact

_COLORS = {
    "DEBUG": "\033[36m", "INFO": "\033[32m", "WARNING": "\033[33m",
    "ERROR": "\033[31m", "CRITICAL": "\033[41m",
}
_RESET = "\033[0m"


class TextFormatter(logging.Formatter):
    def __init__(self, color: bool, redact_exc: bool) -> None:
        super().__init__()
        self._color = color
        self._redact_exc = redact_exc

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S")
        level = record.levelname
        if self._color and level in _COLORS:
            level_s = f"{_COLORS[level]}{level:<8}{_RESET}"
        else:
            level_s = f"{level:<8}"
        line = (
            f"{ts} | {level_s} | "
            f"{record.name}:{record.funcName}:{record.lineno} | "
            f"{record.getMessage()}"
        )
        extras: dict = {}
        for attr in ("obs_context", "obs_fields"):
            data = getattr(record, attr, None)
            if isinstance(data, dict):
                extras.update(data)
        if extras:
            line += " | " + ", ".join(f"{k}: {v}" for k, v in extras.items())
        if record.exc_info:
            tb = "".join(traceback.format_exception(*record.exc_info))
            line += "\n" + (redact(tb) if self._redact_exc else tb)
        return line


class ConsoleSink:
    name = "console"

    def setup(self, config: ObservabilityConfig) -> None:
        return None

    def build_handler(self, config: ObservabilityConfig) -> logging.Handler | None:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            TextFormatter(
                color=sys.stdout.isatty(),
                redact_exc=config.redact,
            )
        )
        return handler

    def instrument(self, config: ObservabilityConfig) -> None:
        return None
