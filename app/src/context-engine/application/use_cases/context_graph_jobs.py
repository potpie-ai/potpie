"""Job-runner adapters: thin shims between Hatchet/Celery tasks and verbs.

Two background jobs exist: a backfill sweep (``handle_backfill_pot``) and
a per-batch processor (``handle_process_batch``). Both are session-scoped
and rebuild the container per call so host-side session-bound resolvers
(e.g. ``SqlalchemyPotResolution``) stay valid.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from sqlalchemy.orm import Session

from application.use_cases.backfill_pot import backfill_pot_context
from application.use_cases.process_batch import process_batch
from bootstrap.container import ContextEngineContainer


def handle_backfill_pot(
    db: Session,
    pot_id: str,
    *,
    target_repo_name: str | None = None,
    build_container: Callable[[Session], ContextEngineContainer],
) -> dict[str, Any]:
    container = build_container(db)
    if container.context_graph is None:
        return {"status": "error", "error": "context_graph_unavailable"}
    return backfill_pot_context(
        settings=container.settings,
        pots=container.pots,
        connectors=container.connectors,
        ledger=container.ledger(db),
        ingestion=container.ingestion_submission(db),
        pot_id=pot_id,
        target_repo_name=target_repo_name,
    )


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
