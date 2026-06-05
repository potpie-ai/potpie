"""Celery beat task: periodic sandbox storage sweep.

The reactive eviction path only fires when new work arrives
(``SandboxService.get_or_create_workspace``). This task relieves disk
pressure that builds up while the system is *idle* — a host can fill
with stale worktrees nobody is allocating against. It also refreshes the
``size_bytes`` the LRU ranking depends on and winds down idle runtimes.

Idempotent and best-effort: re-running early only costs the size-walk.
Mirrors ``archive_pot_cleanup``'s thread + ``asyncio.run`` pattern so the
async client runs cleanly regardless of the Celery worker pool (the
prefork/solo pools don't own an event loop the task can borrow).
"""

from __future__ import annotations

import asyncio
import os
import threading

from app.celery.celery_app import celery_app, logger

# Routed onto the context-graph-etl queue on purpose: that queue already
# has a beat-driven consumer deployed (the per-minute windowed-batch
# flush), so the sweep is guaranteed a worker wherever periodic tasks
# run, without standing up new infrastructure for a low-frequency
# maintenance job.
_SWEEP_QUEUE = "context-graph-etl"


def _run_sweep() -> dict:
    from app.modules.intelligence.tools.sandbox.client import (
        get_sandbox_client,
    )

    client = get_sandbox_client()
    idle_stop = float(os.getenv("SANDBOX_RUNTIME_IDLE_STOP_SECS", str(30 * 60)))
    idle_destroy = float(
        os.getenv("SANDBOX_RUNTIME_IDLE_DESTROY_SECS", str(6 * 60 * 60))
    )
    return asyncio.run(
        client.sweep_storage(
            idle_stop_seconds=idle_stop,
            idle_destroy_seconds=idle_destroy,
        )
    )


@celery_app.task(
    name="app.modules.intelligence.tools.sandbox.tasks.sandbox_storage_sweep",
    queue=_SWEEP_QUEUE,
)
def sandbox_storage_sweep() -> dict:
    """Beat-scheduled: refresh sizes, prune idle runtimes, evict to low-water.

    Runs every ``SANDBOX_STORAGE_SWEEP_INTERVAL_SECS`` (see
    ``celery_app.conf.beat_schedule``). The sweep itself is bounded by
    ``SANDBOX_STORAGE_SWEEP_TIMEOUT_SECS`` so a wedged backend can't pin
    a worker forever — a timeout is reported and the next tick retries.
    """
    result: dict = {}
    error: list[BaseException] = []

    def _runner() -> None:
        try:
            result.update(_run_sweep())
        except BaseException as exc:  # noqa: BLE001
            error.append(exc)
            logger.exception("sandbox_storage_sweep failed")

    worker = threading.Thread(target=_runner, name="sandbox-storage-sweep", daemon=True)
    worker.start()
    worker.join(timeout=float(os.getenv("SANDBOX_STORAGE_SWEEP_TIMEOUT_SECS", "600")))
    if worker.is_alive():
        logger.error("sandbox_storage_sweep timed out; leaving it to the next tick")
        return {"status": "timeout"}
    if error:
        return {"status": "error", "error": str(error[0])}
    logger.info("sandbox_storage_sweep ok: %s", result)
    return {"status": "ok", **result}
