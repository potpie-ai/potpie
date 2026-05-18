"""Celery integration (extra: observability[celery]) — NEW; fixes audit gaps.

The audit found: Celery workers never init Sentry, no Sentry CeleryIntegration,
and correlation IDs are lost across the task boundary (EC3). This wires:

 - worker_process_init: call configure() in the worker AND do the fork-unsafe
   backend init here (sentry_sdk.init / logfire.configure) — NOT at import
   (EC2). prefork forks after import; sockets opened pre-fork are broken.
 - task_prerun/postrun: re-bind log_context from task headers
   (conversation_id/run_id/user_id) so queued runs are traceable end-to-end —
   contextvars do not survive the broker hop (EC3).
 - Sentry CeleryIntegration enabled here so task exceptions are captured.

EDGE CASES:
 - solo/threads/gevent pools fork differently than prefork — init guard must
   be per-process, idempotent.
 - logfire instrument_pydantic_ai MUST be False here (OTel contextvar bug in
   prefork async generators).
 - task headers may be missing on retries/chains — degrade, never raise.
"""

from __future__ import annotations


def install_celery_observability(celery_app) -> None:
    """STUB (Phase 1): contract only. Implemented in Phase 3."""
    raise NotImplementedError("Phase 1 scaffold — implemented in Phase 3")
