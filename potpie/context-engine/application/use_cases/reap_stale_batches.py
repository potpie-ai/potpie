"""Reap reconciliation batches stuck in-flight past their lease.

A batch goes in-flight at ``claim_batch_by_id`` and is expected to reach
``done``/``failed`` within one agent run. If the worker dies mid-run (OOM
kill, ``worker_max_memory_per_child``, pod restart/deploy, or the Celery
hard ``task_time_limit`` after a hung model/sandbox call) the task message
is **not redelivered** (``task_reject_on_worker_lost=False``) and the row
is **never re-claimable** (``claim_batch_by_id`` only takes ``pending``).
There is no autoretry on these tasks. So absent this sweep, such a batch —
and its events, frozen at ``processing`` — would be stuck forever and
invisible (no terminal status, no error surfaced).

This task converts "stuck forever / invisible" into "failed / visible":
the batch is marked ``failed`` and its still-in-flight events become
``failed`` (the user-facing bulk-retry path can then re-drive them).
Events the batch had already completed stay ``reconciled`` — see
``ReconciliationLedgerPort.fail_inflight_events``.

Correctness rests entirely on the lease: the caller must pass a lease that
**exceeds** Celery's ``task_time_limit``. Celery hard-kills a live task at
the time limit, so any batch whose ``claimed_at`` is older than that lease
is definitively dead — never merely slow. With that invariant the reaper
can never race a live worker, so no double-execution and no schema-level
heartbeat is required. (A ``heartbeat_at`` column would let the lease be
much shorter / recovery faster; that's a deliberate follow-up, not needed
for correctness.)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from bootstrap import sentry_metrics_runtime
from domain.ports.batch_repository import BatchRepositoryPort
from domain.ports.reconciliation_ledger import ReconciliationLedgerPort

logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class ReapOutcome:
    batches_reaped: int
    events_failed: int
    errors: int
    reaped_batch_ids: list[str]


def reap_stale_batches(
    *,
    batches: BatchRepositoryPort,
    reco_ledger: ReconciliationLedgerPort,
    lease_seconds: float,
) -> ReapOutcome:
    """Fail every batch stuck ``claimed``/``running`` longer than the lease.

    Per stale batch the *events* are failed first, then the *batch* — so a
    crash between the two writes leaves the batch still in-flight (it gets
    reaped again next tick) rather than ``failed`` with events orphaned at
    ``processing``. ``mark_batch_failed`` moves the row out of
    ``claimed``/``running`` (and out of the partial-unique ``pending``
    space), so a row is never reaped twice.
    """
    stale = batches.list_stale_in_flight_batches(lease_seconds)
    if not stale:
        return ReapOutcome(0, 0, 0, [])

    reaped_ids: list[str] = []
    events_failed = 0
    errors = 0
    reason = (
        f"reaped: in-flight > {int(lease_seconds)}s lease "
        "(worker presumed dead — no retry/redelivery for this task)"
    )

    for batch in stale:
        try:
            event_ids = [
                ref.event_id for ref in batches.list_events_for_batch(batch.id)
            ]
            events_failed += reco_ledger.fail_inflight_events(event_ids, reason)
            batches.mark_batch_failed(batch.id, reason)
            reaped_ids.append(batch.id)
            # The only true dead-letter in the system — every firing is a
            # lost/stuck batch. High-signal: alert on rate(ce.batch.reaped_total).
            try:
                from bootstrap.observability_runtime import get_observability

                get_observability().counter(
                    "ce.batch.reaped_total",
                    1,
                    attributes={"pot_id": batch.pot_id},
                )
            except Exception:  # noqa: BLE001 — never break the reaper
                pass
            try:
                sentry_metrics_runtime.count(
                    "ce.batch.reaped_total",
                    1,
                    attributes={"result": "reaped"},
                )
            except Exception:  # noqa: BLE001 — never break the reaper
                pass
            logger.warning(
                "reaped stale batch %s (pot %s, status=%s, attempt=%d, claimed_at=%s)",
                batch.id,
                batch.pot_id,
                batch.status,
                batch.attempt_count,
                batch.claimed_at,
            )
        except Exception:
            logger.exception("reap_stale_batches: failed to reap batch %s", batch.id)
            errors += 1

    return ReapOutcome(
        batches_reaped=len(reaped_ids),
        events_failed=events_failed,
        errors=errors,
        reaped_batch_ids=reaped_ids,
    )
