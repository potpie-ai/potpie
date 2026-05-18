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
These are the load-bearing decisions found while scaffolding. They are part
of the contract; changing them is a breaking change.

EC1  stdlib loggers reject arbitrary kwargs (only exc_info/stack_info/
     stacklevel/extra). The codebase already uses loguru-style
     `logger.info("msg", key=val)` (73+ sites). `get_logger` therefore
     returns a `StructuredLogger` (LoggerAdapter) that maps **fields -> extra
     so those call sites keep working post-migration. This is why we do NOT
     hand back a raw `logging.Logger`.

EC2  `configure()` must be idempotent AND fork-safe. It runs at import in
     both the web app and Celery. Network-backed sinks (Sentry/logfire) that
     open sockets must (re)init in the worker process, not at import — see
     integrations/celery.py. Calling configure() twice must not duplicate
     handlers (we tag our handlers and replace, never blanket
     basicConfig(force=True)).

EC3  `log_context()` uses contextvars (async-safe). contextvars do NOT cross
     thread-pool / process / Celery task boundaries automatically. Correlation
     IDs WILL be lost across `run_in_executor` and Celery hops unless
     explicitly re-bound. This is the root cause of the audit's weak
     background-task correlation; integrations/* must re-bind at each hop.

EC4  Logs emitted before `configure()` go to the default root handler. Every
     entrypoint MUST call `configure()` first. Phase 2 must install a minimal
     safety handler so pre-configure logs are not silently dropped.
"""

from __future__ import annotations

import logging

from .config import ObservabilityConfig
from .sink import Sink

__all__ = ["get_logger", "configure", "log_context", "ObservabilityConfig", "Sink"]


class StructuredLogger(logging.LoggerAdapter):
    """stdlib LoggerAdapter that accepts loguru-style **fields.

    Contract: `logger.info("msg", conversation_id=cid)` maps `conversation_id`
    into `record.extra` so structured sinks can emit it as a field instead of
    interpolating it into the message (fixes the audit's 748 f-string blobs).

    STUB: kwarg->extra mapping is the contract. Implementation and sink-side
    field emission land in Phase 2.
    """

    def process(self, msg, kwargs):  # noqa: D401
        raise NotImplementedError("Phase 1 scaffold — implemented in Phase 2")


def get_logger(name: str) -> StructuredLogger:
    """Return an ambient structured logger. Thin wrapper over getLogger(name).

    STUB (Phase 1): contract signature only.
    """
    raise NotImplementedError("Phase 1 scaffold — implemented in Phase 2")


def configure(config: "ObservabilityConfig | None" = None) -> None:
    """Composition root: wire sinks, init Sentry + logfire. Idempotent (EC2).

    STUB (Phase 1): contract signature only.
    """
    raise NotImplementedError("Phase 1 scaffold — implemented in Phase 2/3")


def log_context(**fields):
    """Context manager binding correlation IDs to all logs in scope (EC3).

    STUB (Phase 1): contract signature only.
    """
    raise NotImplementedError("Phase 1 scaffold — implemented in Phase 2")
