from __future__ import annotations

import contextvars
import logging
from contextlib import contextmanager

_RESERVED = {"exc_info", "stack_info", "stacklevel", "extra"}
_context_var: contextvars.ContextVar[dict] = contextvars.ContextVar(
    "integrations_log_context", default={}
)


class StructuredLogger:
    def __init__(self, logger: logging.Logger, bound_fields: dict | None = None) -> None:
        self._logger = logger
        self._bound_fields = bound_fields or {}

    def bind(self, **fields) -> "StructuredLogger":
        return StructuredLogger(self._logger, {**self._bound_fields, **fields})

    def debug(self, msg, *args, **kwargs) -> None:
        self._log(logging.DEBUG, msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs) -> None:
        self._log(logging.INFO, msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs) -> None:
        self._log(logging.WARNING, msg, *args, **kwargs)

    warn = warning

    def error(self, msg, *args, **kwargs) -> None:
        self._log(logging.ERROR, msg, *args, **kwargs)

    def exception(self, msg, *args, **kwargs) -> None:
        kwargs.setdefault("exc_info", True)
        self.error(msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs) -> None:
        self._log(logging.CRITICAL, msg, *args, **kwargs)

    fatal = critical

    def isEnabledFor(self, level: int) -> bool:  # noqa: N802
        return self._logger.isEnabledFor(level)

    def getChild(self, suffix: str) -> "StructuredLogger":  # noqa: N802
        return StructuredLogger(self._logger.getChild(suffix), self._bound_fields)

    def _log(self, level: int, msg, *args, **kwargs) -> None:
        passthrough = {key: kwargs.pop(key) for key in list(kwargs) if key in _RESERVED}
        extra = dict(passthrough.pop("extra", {}) or {})
        fields = {
            **_context_var.get(),
            **self._bound_fields,
            **extra.pop("obs_fields", {}),
            **kwargs,
        }
        passthrough["extra"] = {**extra, "obs_fields": fields}
        self._logger.log(level, msg, *args, **passthrough)

    def __getattr__(self, name: str):
        return getattr(self._logger, name)


def get_logger(name: str) -> StructuredLogger:
    return StructuredLogger(logging.getLogger(name))


@contextmanager
def log_context(**fields):
    token = _context_var.set({**_context_var.get(), **fields})
    try:
        yield
    finally:
        _context_var.reset(token)
