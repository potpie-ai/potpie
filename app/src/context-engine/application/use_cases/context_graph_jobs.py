"""Job-runner adapter: thin shim between the Celery task and the verb.

One background job exists: a per-batch processor (``handle_process_batch``).
It is session-scoped and rebuilds the container per call so host-side
session-bound resolvers (e.g. ``SqlalchemyPotResolution``) stay valid.

Backfill is no longer a standalone enumerate-then-submit sweep. A source
attach (GitHub ``repository.added`` / Linear ``linear_team.added``) emits a
single ``agent_reconciliation`` event; the reconciliation agent — planner
on, via the backfill playbooks — enumerates and seeds the graph through this
same per-batch path.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from sqlalchemy.orm import Session

from application.use_cases.process_batch import process_batch
from bootstrap.container import ContextEngineContainer
from observability import log_context


def handle_process_batch(
    db: Session,
    batch_id: str,
    *,
    build_container: Callable[[Session], ContextEngineContainer],
) -> dict[str, Any]:
    """Claim one batch by id and run the reconciliation agent over it.

    Called by the host worker once per ``jobs.enqueue_batch`` event. If the
    batch is already claimed/running/done (a redundant enqueue races and
    loses), ``claim_batch_by_id`` returns ``None`` and this is a no-op.
    """
    container = build_container(db)
    if container.reconciliation_agent is None:
        return {"status": "skipped", "reason": "no_reconciliation_agent"}
    batches_repo = container.batch_repository(db)
    batch = batches_repo.claim_batch_by_id(batch_id)
    if batch is None:
        return {"status": "skipped", "reason": "not_pending", "batch_id": batch_id}
    # Bind batch_id / pot_id at the job boundary so every log emitted by
    # process_batch and the reconciliation agent it drives carries them as
    # structured fields. Single seam, no body changes downstream.
    with log_context(batch_id=batch.id, pot_id=batch.pot_id):
        outcome = process_batch(
            batch=batch,
            agent=container.reconciliation_agent,
            batches=batches_repo,
            reco_ledger=container.reconciliation_ledger(db),
            checkpoints=container.agent_checkpoint_store(db),
            pots=container.pots,
            policy=container.policy(),
            stream_publisher=container.event_stream_publisher,
            execution_log=container.agent_execution_log(db),
        )
    return {
        "status": "ok" if outcome.ok else "failed",
        "batch_id": outcome.batch_id,
        "completed_event_ids": outcome.completed_event_ids,
        "tool_call_count": outcome.tool_call_count,
        "error": outcome.error,
    }
