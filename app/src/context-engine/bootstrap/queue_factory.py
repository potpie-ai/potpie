"""Select ``ContextGraphJobQueuePort`` (Hatchet, host Celery adapter, or no-op)."""

from __future__ import annotations

import importlib
import logging
import os

from domain.ports.context_graph_job_queue import (
    ContextGraphJobQueuePort,
    NoOpContextGraphJobQueue,
)

from bootstrap.potpie_path import ensure_potpie_repo_on_sys_path

logger = logging.getLogger(__name__)

# CONTEXT_GRAPH_JOB_QUEUE_BACKEND: celery | hatchet | noop
# When unset, defaults to celery (host Potpie adapter). If that module is not importable (e.g. standalone
# CLI without ``app`` on PYTHONPATH), falls back to noop. Set CONTEXT_GRAPH_JOB_QUEUE_BACKEND=celery
# explicitly to fail fast when the adapter is missing. Hatchet: CONTEXT_GRAPH_JOB_QUEUE_BACKEND=hatchet.
#
# Celery adapter lives in the host app (e.g. Potpie) so this package stays decoupled:
# set CONTEXT_GRAPH_CELERY_QUEUE_MODULE (default: app.modules.context_graph.celery_job_queue).


def _celery_queue_module_path() -> str:
    return os.environ.get(
        "CONTEXT_GRAPH_CELERY_QUEUE_MODULE",
        "app.modules.context_graph.celery_job_queue",
    )


def _import_celery_queue_adapter() -> ContextGraphJobQueuePort:
    mod = importlib.import_module(_celery_queue_module_path())
    return mod.CeleryContextGraphJobQueue()


def _resolve_celery(*, explicit_backend_was_set: bool) -> ContextGraphJobQueuePort:
    """Load host Celery adapter, or noop when default and host app is not on PYTHONPATH."""
    try:
        q = _import_celery_queue_adapter()
        logger.info("Context graph job queue: celery (host adapter)")
        return q
    except ImportError as exc:
        if explicit_backend_was_set:
            raise ImportError(
                f"Could not import CONTEXT_GRAPH_CELERY_QUEUE_MODULE={_celery_queue_module_path()!r}: {exc}"
            ) from exc
        logger.warning(
            "Context graph job queue: Celery adapter not importable (%s). "
            "Using noop queue — async enqueue will apply episode steps inline. "
            "For Potpie: run with repo root on PYTHONPATH. For standalone CLI: set "
            "CONTEXT_GRAPH_JOB_QUEUE_BACKEND=noop to silence, or point CONTEXT_GRAPH_CELERY_QUEUE_MODULE "
            "at an importable module.",
            exc,
        )
        return NoOpContextGraphJobQueue()


def get_context_graph_job_queue() -> ContextGraphJobQueuePort:
    """
    Resolve the queue adapter from ``CONTEXT_GRAPH_JOB_QUEUE_BACKEND``.

    - ``celery`` (default) — host Celery tasks (see ``CONTEXT_GRAPH_CELERY_QUEUE_MODULE``).
    - ``hatchet`` — Hatchet ``event.push``; requires ``HATCHET_CLIENT_TOKEN`` and a running Hatchet worker.
    - ``noop`` — no broker I/O (tests / inline-only).

    Implicit **celery** (env unset): if the host Celery module cannot be imported, falls back to **noop**.
    Explicit **celery**: raises if the module is missing. **Hatchet** failures fall through to Celery
    then noop as documented in ``_resolve_hatchet``. Noop: enqueue is a no-op; async ingest may
    still run agent planning and episode apply inline.
    """
    ensure_potpie_repo_on_sys_path()
    explicit = os.getenv("CONTEXT_GRAPH_JOB_QUEUE_BACKEND")
    raw = (explicit or "celery").strip().lower()
    explicit_set = bool(explicit and explicit.strip())
    if raw in ("noop", "none", "disabled"):
        logger.info("Context graph job queue: noop")
        return NoOpContextGraphJobQueue()
    if raw == "celery":
        return _resolve_celery(explicit_backend_was_set=explicit_set)
    if raw == "hatchet":
        return _resolve_hatchet(
            explicit_backend_was_set=bool(explicit and explicit.strip())
        )
    raise ValueError(
        f"Unknown CONTEXT_GRAPH_JOB_QUEUE_BACKEND={raw!r}; expected hatchet, celery, or noop"
    )


def _resolve_hatchet(*, explicit_backend_was_set: bool) -> ContextGraphJobQueuePort:
    from adapters.outbound.hatchet.hatchet_job_queue import HatchetContextGraphJobQueue

    try:
        q = HatchetContextGraphJobQueue.from_env()
        logger.info("Context graph job queue: hatchet (event.push)")
        return q
    except Exception as exc:
        if explicit_backend_was_set:
            logger.exception("Hatchet queue requested but initialization failed")
            raise
        logger.warning(
            "Context graph job queue: falling back to celery (Hatchet unavailable: %s). "
            "Set HATCHET_CLIENT_TOKEN and run the Hatchet worker, or set "
            "CONTEXT_GRAPH_JOB_QUEUE_BACKEND=celery to silence this.",
            exc,
        )
        try:
            q = _import_celery_queue_adapter()
            logger.info("Context graph job queue: celery (host adapter)")
            return q
        except ImportError as celery_exc:
            logger.warning(
                "Context graph job queue: Celery adapter not importable (%s). "
                "Using noop queue — async enqueue will apply episode steps inline. "
                "For real workers: install hatchet-sdk + token, run from Potpie with `app` on "
                "PYTHONPATH, or set CONTEXT_GRAPH_JOB_QUEUE_BACKEND=noop to silence.",
                celery_exc,
            )
            return NoOpContextGraphJobQueue()
