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
_state: dict = {"configured_pid": None}

_RESERVED = ("exc_info", "stack_info", "stacklevel")


class StructuredLogger(logging.LoggerAdapter):
    """stdlib LoggerAdapter accepting loguru-style **fields (EC1).

    `logger.info("msg", conversation_id=cid)` -> record.obs_fields, so sinks
    emit it as a queryable field instead of interpolating into the message.
    """

    def process(self, msg, kwargs):
        passthru = {k: kwargs.pop(k) for k in _RESERVED if k in kwargs}
        extra = kwargs.pop("extra", None) or {}
        fields = {**(self.extra or {}), **extra, **kwargs}
        kwargs.clear()
        kwargs.update(passthru)
        kwargs["extra"] = {"obs_fields": fields}
        return msg, kwargs


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

    for h in _iter_managed(root):
        try:
            h.close()
        except Exception:
            pass
        root.removeHandler(h)

    ctx_filter = ContextFilter()
    red_filter = RedactionFilter() if cfg.redact else None

    for name in cfg.sinks:
        sink = registry.resolve(name)
        sink.setup(cfg)
        handler = sink.build_handler(cfg)
        if handler is None:
            continue
        handler.addFilter(ctx_filter)  # inject context first
        if red_filter is not None:
            handler.addFilter(red_filter)  # then scrub message + context
        handler.setLevel(cfg.level)
        setattr(handler, _TAG, HANDLER_TAG)
        root.addHandler(handler)

    root.setLevel(cfg.level)
    apply_library_levels(cfg.library_levels)
    _state["configured_pid"] = os.getpid()


_install_safety_handler()
