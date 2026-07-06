from __future__ import annotations

import contextvars
import logging
import os
from contextlib import contextmanager
from typing import Any

from .config import ObservabilityConfig
from .redaction import sanitize_log_text

__all__ = ["ObservabilityConfig", "configure", "get_logger", "log_context"]

_state: dict = {"configured_pid": None, "config": None}
_context_var: contextvars.ContextVar[dict] = contextvars.ContextVar(
    "observability_context", default={}
)
_RESERVED = {"exc_info", "stack_info", "stacklevel", "extra"}
_SENSITIVE_FIELD_NAMES = {
    "api_key",
    "apikey",
    "access_token",
    "authorization",
    "client_secret",
    "id_token",
    "password",
    "passwd",
    "pwd",
    "refresh_token",
    "secret",
    "token",
}


class _ObsFieldsFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "obs_fields"):
            record.obs_fields = {}
        return True


class StructuredLogger:
    def __init__(
        self, logger: logging.Logger, bound_fields: dict | None = None
    ) -> None:
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
        cfg = _state.get("config")
        redact_enabled = bool(getattr(cfg, "redact", False))
        fields = {
            **_context_var.get(),
            **self._bound_fields,
            **extra.pop("obs_fields", {}),
            **kwargs,
        }
        if redact_enabled:
            fields = _redact_value(fields)
            msg = _redact_value(msg)
            args = _redact_value(args)
        msg, args = _render_message(msg, args)
        passthrough["extra"] = {**extra, "obs_fields": fields}
        self._logger.log(level, msg, *args, **passthrough)

    def __getattr__(self, name: str):
        return getattr(self._logger, name)


def get_logger(name: str) -> StructuredLogger:
    return StructuredLogger(logging.getLogger(name))


def configure(config: ObservabilityConfig | None = None) -> None:
    cfg = config or ObservabilityConfig.from_env()
    logging.basicConfig(
        level=getattr(logging, cfg.level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s %(obs_fields)s",
    )
    for handler in logging.getLogger().handlers:
        if not any(
            isinstance(log_filter, _ObsFieldsFilter) for log_filter in handler.filters
        ):
            handler.addFilter(_ObsFieldsFilter())
    _state["configured_pid"] = os.getpid()
    _state["config"] = cfg


@contextmanager
def log_context(**fields):
    token = _context_var.set({**_context_var.get(), **fields})
    try:
        yield
    finally:
        _context_var.reset(token)


def _push_context(**fields) -> object:
    return _context_var.set({**_context_var.get(), **fields})


def _pop_context(token: object) -> None:
    try:
        _context_var.reset(token)  # type: ignore[arg-type]
    except (LookupError, ValueError):
        pass


def _render_message(msg: Any, args: tuple[Any, ...]) -> tuple[Any, tuple[Any, ...]]:
    if not args:
        return msg, args
    try:
        return msg % args, ()
    except Exception:
        try:
            return msg.format(*args), ()
        except Exception:
            return msg, ()


def _redact_value(value: Any) -> Any:
    if isinstance(value, str):
        return sanitize_log_text(value)
    if isinstance(value, dict):
        return {
            key: "***REDACTED***"
            if str(key).lower() in _SENSITIVE_FIELD_NAMES
            else _redact_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return type(value)(_redact_value(item) for item in value)
    return value
