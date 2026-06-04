"""Celery integration (extra: observability[celery]) — NEW; fixes audit gaps.

The audit found: Celery workers never init Sentry; no Sentry CeleryIntegration;
correlation IDs lost across the broker hop (EC3). This wires:

 - worker_process_init -> configure(profiles.celery()). Backend init (Sentry/
   logfire sockets) happens INSIDE the forked worker (EC2) — not at import,
   which would leave pre-fork sockets dangling.
 - task_prerun/task_postrun -> bracket each task with log_context bound to
   task_id + task_name, plus best-effort scrape of well-known correlation
   keys from kwargs (conversation_id/run_id/project_id/user_id). This re-binds
   correlation across the broker hop where contextvars cannot survive (EC3).
   Once callers migrate (Phase 4), they can layer their own log_context()
   inside the task for richer fields.

EDGE CASES:
 - prefork is the standard pool; solo/threads/gevent also fork-or-not safely
   because configure() is idempotent + PID-tracked.
 - task_postrun fires even if the task raised — token cleanup is best-effort.
 - kwargs may be missing/None on retries/chains; degrade silently.
"""

from __future__ import annotations

from typing import Any

_HINT = "celery integration requires celery — install observability[celery]"
_KNOWN_CORRELATION_KEYS = (
    "conversation_id",
    "run_id",
    "project_id",
    "user_id",
    "request_id",
)


def install_celery_observability(celery_app: Any = None, config: Any = None) -> None:
    """Wire signals so workers initialise observability and propagate context.

    celery_app is accepted (parity with similar libs) but not strictly needed;
    Celery signals are global. config defaults to profiles.celery().
    """
    try:
        from celery.signals import (
            task_postrun,
            task_prerun,
            worker_process_init,
        )
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(_HINT) from exc

    from .. import _pop_context, _push_context, configure
    from ..profiles import celery as celery_profile

    cfg = config or celery_profile()
    _pending: dict[Any, Any] = {}

    @worker_process_init.connect
    def _init_worker(**_kwargs):  # pragma: no cover — fires inside worker
        configure(cfg)

    @task_prerun.connect
    def _on_prerun(task_id=None, task=None, args=None, kwargs=None, **_):  # noqa: D401
        fields = {"task_id": task_id, "task_name": getattr(task, "name", None)}
        if isinstance(kwargs, dict):
            for key in _KNOWN_CORRELATION_KEYS:
                if key in kwargs:
                    fields[key] = kwargs[key]
        _pending[task_id] = _push_context(**fields)

    @task_postrun.connect
    def _on_postrun(task_id=None, **_):
        token = _pending.pop(task_id, None)
        if token is not None:
            _pop_context(token)
