"""Observability package — FROZEN PUBLIC CONTRACT.

Only the three functions below are the behavioral public API. The config and
sink protocol are also re-exported as data/typing contracts.

    get_logger(name)        -> StructuredLogger   (ambient; thin stdlib wrapper)
    configure(config)       -> None               (idempotent composition root)
    log_context(**fields)   -> context manager    (correlation-id propagation)

Design: stdlib `logging` core, consumers stay ambient, backends (loguru /
Sentry / logfire) plug in via the single `Sink` seam. No `app.*` imports
anywhere in this package — it must remain liftable into any Python service.

=========================== CONTRACT EDGE CASES ============================
EC1  stdlib loggers reject arbitrary kwargs (only exc_info/stack_info/
     stacklevel/extra). The codebase already uses loguru-style
     `logger.info("msg", key=val)`. `get_logger` returns a `StructuredLogger`
     that maps **fields -> record.obs_fields so structured sinks emit them as
     fields, not f-string blobs.

EC2  `configure()` is idempotent AND fork-aware. It removes its own previously
     installed handlers (tag-based) before re-adding — never
     `basicConfig(force=True)`. It records the configuring PID so a forked
     Celery worker (Phase 3, worker_process_init) can safely re-configure.

EC3  `log_context()` uses contextvars (async-safe). contextvars do NOT cross
     thread-pool / process / Celery task boundaries automatically; integrations
     must re-bind at each hop (Phase 3).

EC4  Logs emitted before `configure()` would otherwise hit the default root
     handler. A minimal safety handler (stderr, WARNING) is installed at
     import and removed by `configure()`, so pre-configure logs are not lost.
"""

from __future__ import annotations

import atexit
import contextvars
import logging
import os
import sys
from contextlib import contextmanager

from .config import ObservabilityConfig
from .sink import Sink

__all__ = ["get_logger", "configure", "log_context", "ObservabilityConfig", "Sink"]

_TAG = "_observability_managed"
HANDLER_TAG = "sink"
SAFETY_TAG = "safety"

_context_var: contextvars.ContextVar[dict] = contextvars.ContextVar(
    "observability_context", default={}
)
_state: dict = {"configured_pid": None, "sinks": [], "config": None}

_RESERVED = ("exc_info", "stack_info", "stacklevel")


class StructuredLogger:
    """Thin stdlib wrapper accepting loguru-style **fields (EC1).

    `logger.info("msg", conversation_id=cid)` -> record.obs_fields, so sinks
    emit it as a queryable field instead of interpolating into the message.
    """

    def __init__(
        self,
        logger: logging.Logger,
        extra: dict | None = None,
    ) -> None:
        self._logger = logger
        self._extra = extra or {}

    def _prepare(self, kwargs: dict, stacklevel_offset: int) -> dict:
        passthru = {k: kwargs.pop(k) for k in _RESERVED if k in kwargs}
        extra = kwargs.pop("extra", None) or {}
        fields = {**self._extra, **extra, **kwargs}
        passthru["extra"] = {"obs_fields": fields}
        passthru["stacklevel"] = int(passthru.get("stacklevel", 1)) + stacklevel_offset
        return passthru

    def debug(self, msg, *args, **kwargs) -> None:
        self._log(logging.DEBUG, msg, *args, stacklevel_offset=2, **kwargs)

    def info(self, msg, *args, **kwargs) -> None:
        self._log(logging.INFO, msg, *args, stacklevel_offset=2, **kwargs)

    def warning(self, msg, *args, **kwargs) -> None:
        self._log(logging.WARNING, msg, *args, stacklevel_offset=2, **kwargs)

    warn = warning

    def error(self, msg, *args, **kwargs) -> None:
        self._log(logging.ERROR, msg, *args, stacklevel_offset=2, **kwargs)

    def exception(self, msg, *args, **kwargs) -> None:
        kwargs.setdefault("exc_info", True)
        self.error(msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs) -> None:
        self._log(logging.CRITICAL, msg, *args, stacklevel_offset=2, **kwargs)

    fatal = critical

    def log(self, log_level, msg, *args, **kwargs) -> None:
        self._log(log_level, msg, *args, stacklevel_offset=2, **kwargs)

    def _log(self, log_level, msg, *args, stacklevel_offset: int, **kwargs) -> None:
        self._logger.log(
            log_level,
            msg,
            *args,
            **self._prepare(kwargs, stacklevel_offset),
        )

    def isEnabledFor(self, level: int) -> bool:  # noqa: N802 - stdlib API name
        return self._logger.isEnabledFor(level)

    def getChild(self, suffix: str) -> "StructuredLogger":  # noqa: N802
        return StructuredLogger(self._logger.getChild(suffix), self._extra)

    def bind(self, **fields) -> "StructuredLogger":
        """Return a logger with default structured fields, loguru-compatible."""
        return StructuredLogger(self._logger, {**self._extra, **fields})

    def __getattr__(self, name: str):
        return getattr(self._logger, name)


class ContextFilter(logging.Filter):
    """Inject log_context() correlation IDs (and ensure obs_fields exists) onto
    every record — works for ambient stdlib loggers too, not just ours."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "obs_fields"):
            record.obs_fields = {}
        record.obs_context = dict(_context_var.get())
        return True


def get_logger(name: str) -> StructuredLogger:
    """Return an ambient structured logger. Thin wrapper over getLogger(name)."""
    return StructuredLogger(logging.getLogger(name), {})


@contextmanager
def log_context(**fields):
    """Bind correlation IDs to all logs in scope (async-safe, stackable; EC3)."""
    current = _context_var.get()
    token = _context_var.set({**current, **fields})
    try:
        yield
    finally:
        _context_var.reset(token)


# --- internal helpers for cross-hop rebind (EC3) — used by integrations.* ---
# Not part of the frozen public API; the Celery integration uses these to
# bracket a task with prerun/postrun signals where a contextmanager doesn't fit.


def _push_context(**fields) -> object:
    cur = _context_var.get()
    return _context_var.set({**cur, **fields})


def _pop_context(token: object) -> None:
    try:
        _context_var.reset(token)  # type: ignore[arg-type]
    except (ValueError, LookupError):
        # Token from another context (e.g. task crashed before postrun) —
        # swallow rather than mask the real error.
        pass


def _iter_managed(root: logging.Logger):
    return [h for h in root.handlers if getattr(h, _TAG, None) is not None]


def _install_safety_handler() -> None:
    """EC4: minimal pre-configure handler so early logs are not dropped."""
    root = logging.getLogger()
    if any(getattr(h, _TAG, None) == SAFETY_TAG for h in root.handlers):
        return
    h = logging.StreamHandler(sys.stderr)
    h.setLevel(logging.WARNING)
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    setattr(h, _TAG, SAFETY_TAG)
    root.addHandler(h)


def configure(config: "ObservabilityConfig | None" = None) -> None:
    """Composition root: wire sinks + filters onto the root logger (EC2).

    Idempotent and fork-aware: removes our previously installed/safety handlers
    before re-adding, so repeat calls (tests, Celery worker re-init) never
    duplicate handlers. Sentry/logfire backend init lands in Phase 3.
    """
    from .intercept import apply_library_levels
    from .redaction import RedactionFilter
    from .sink import registry

    cfg = config or ObservabilityConfig()
    root = logging.getLogger()

    # Shutdown previously-active sinks (best-effort) before tearing down
    # their handlers — gives batch backends (Sentry, logfire) a chance to
    # flush. Never raise out of shutdown.
    prev_sinks = list(_state.get("sinks") or ())
    prev_cfg = _state.get("config") or cfg
    for s in prev_sinks:
        try:
            getattr(s, "shutdown", lambda *_: None)(prev_cfg)
        except Exception:
            pass

    for h in _iter_managed(root):
        try:
            h.close()
        except Exception:
            pass
        root.removeHandler(h)

    ctx_filter = ContextFilter()
    red_filter = RedactionFilter() if cfg.redact else None
    new_sinks: list = []

    for name in cfg.sinks:
        sink = registry.resolve(name)
        sink.setup(cfg)
        new_sinks.append(sink)
        handler = sink.build_handler(cfg)
        if handler is not None:
            handler.addFilter(ctx_filter)  # inject context first
            if red_filter is not None:
                handler.addFilter(red_filter)  # then scrub message + context
            # Respect sink-provided level (e.g. Sentry's event_level=ERROR);
            # only default to cfg.level when the sink did not set one.
            if handler.level == logging.NOTSET:
                handler.setLevel(cfg.level)
            setattr(handler, _TAG, HANDLER_TAG)
            root.addHandler(handler)
        sink.instrument(cfg)  # tracing-only sinks (logfire) still run this

    root.setLevel(cfg.level)
    apply_library_levels(cfg.library_levels)
    _state["configured_pid"] = os.getpid()
    _state["sinks"] = new_sinks
    _state["config"] = cfg


def _shutdown_all_sinks() -> None:
    """atexit hook: flush every active sink so batched events (Sentry,
    logfire) aren't lost on process exit. Never raise."""
    sinks = list(_state.get("sinks") or ())
    cfg = _state.get("config") or ObservabilityConfig()
    for s in sinks:
        try:
            getattr(s, "shutdown", lambda *_: None)(cfg)
        except Exception:
            pass


_install_safety_handler()
atexit.register(_shutdown_all_sinks)
