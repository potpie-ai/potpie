"""Run the reconciliation agent over one debounced batch.

Worker entrypoint: load the batch + its events + any prior agent checkpoint,
call ``agent.run_batch``, and persist the outcome (mark events processed, drop
checkpoint on success, or mark batch failed).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from domain.actor import SYSTEM_ACTOR
from domain.context_events import ContextEvent
from domain.ports.agent_checkpoint_store import AgentCheckpointStorePort
from domain.ports.batch_repository import BatchRepositoryPort
from domain.ports.policy import (
    ACTION_APPLY_WRITE,
    RESOURCE_APPLY,
    PolicyPort,
)
from domain.ports.pot_resolution import PotResolutionPort
from domain.ports.reconciliation_agent import ReconciliationAgentPort
from domain.ports.reconciliation_ledger import (
    ContextEventRow,
    ReconciliationLedgerPort,
)
from domain.reconciliation_batch import (
    BATCH_STATUS_DONE,
    BatchAgentContext,
    ReconciliationBatch,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class ProcessBatchOutcome:
    """Worker-level outcome for one batch processing attempt."""

    batch_id: str
    ok: bool
    completed_event_ids: list[str]
    tool_call_count: int
    error: str | None = None


def process_batch(
    *,
    batch: ReconciliationBatch,
    agent: ReconciliationAgentPort,
    batches: BatchRepositoryPort,
    reco_ledger: ReconciliationLedgerPort,
    checkpoints: AgentCheckpointStorePort,
    pots: PotResolutionPort,
    policy: PolicyPort | None = None,
) -> ProcessBatchOutcome:
    """Process a claimed batch end-to-end.

    The caller must have already transitioned the batch to ``claimed`` (via
    ``BatchRepositoryPort.claim_batch_by_id``); this function moves it to
    ``running`` before invoking the agent and to ``done`` / ``failed`` after.

    When a :class:`PolicyPort` is supplied, ``apply.write`` is authorized
    once before the agent runs — this is the single policy boundary for
    every mutation the agent issues over the batch.
    """
    if batch.status == BATCH_STATUS_DONE:
        return ProcessBatchOutcome(
            batch_id=batch.id,
            ok=True,
            completed_event_ids=[],
            tool_call_count=0,
        )

    if policy is not None:
        decision = policy.authorize(
            actor=SYSTEM_ACTOR,
            resource=RESOURCE_APPLY,
            action=ACTION_APPLY_WRITE,
            context={"pot_id": batch.pot_id, "batch_id": batch.id},
        )
        if not decision.allowed:
            logger.warning(
                "batch %s denied apply.write by policy: %s",
                batch.id,
                decision.reason,
            )
            batches.mark_batch_failed(batch.id, f"policy:{decision.reason}")
            refs = batches.list_events_for_batch(batch.id)
            for ref in refs:
                if ref.processed_at is None:
                    reco_ledger.record_event_failed(
                        ref.event_id, f"policy:{decision.reason}"
                    )
            return ProcessBatchOutcome(
                batch_id=batch.id,
                ok=False,
                completed_event_ids=[],
                tool_call_count=0,
                error=f"policy:{decision.reason}",
            )

    refs = batches.list_events_for_batch(batch.id)
    pending_event_ids = [r.event_id for r in refs if r.processed_at is None]
    events: list[ContextEvent] = []
    for ref in refs:
        if ref.processed_at is not None:
            continue
        row = reco_ledger.get_event_by_id(ref.event_id)
        if row is None:
            logger.warning(
                "batch %s references missing event %s; skipping",
                batch.id,
                ref.event_id,
            )
            continue
        events.append(_event_from_row(row))

    if not events:
        # Nothing to do — close the batch.
        batches.mark_batch_done(batch.id, completed_event_ids=[])
        checkpoints.delete(batch.id)
        return ProcessBatchOutcome(
            batch_id=batch.id,
            ok=True,
            completed_event_ids=[],
            tool_call_count=0,
        )

    repo_name = _primary_repo_name_for_pot(pots, batch.pot_id)

    prior = checkpoints.load(batch.id)
    ctx = BatchAgentContext(
        batch_id=batch.id,
        pot_id=batch.pot_id,
        repo_name=repo_name,
        events=events,
        prior_messages_json=prior.messages_json if prior else None,
        attempt_number=batch.attempt_count,
    )

    batches.mark_batch_running(batch.id)

    try:
        outcome = agent.run_batch(ctx, checkpoints=checkpoints)
    except Exception as exc:
        logger.exception("batch %s agent run failed", batch.id)
        batches.mark_batch_failed(batch.id, str(exc))
        for eid in pending_event_ids:
            reco_ledger.record_event_failed(eid, str(exc))
        return ProcessBatchOutcome(
            batch_id=batch.id,
            ok=False,
            completed_event_ids=[],
            tool_call_count=0,
            error=str(exc),
        )

    completed = list(outcome.completed_event_ids)

    if outcome.ok:
        batches.mark_batch_done(batch.id, completed_event_ids=completed)
        for eid in completed:
            reco_ledger.record_event_reconciled(eid)
        checkpoints.delete(batch.id)
    else:
        err = outcome.error or "agent_returned_not_ok"
        batches.mark_batch_failed(batch.id, err)
        for eid in pending_event_ids:
            if eid not in completed:
                reco_ledger.record_event_failed(eid, err)

    return ProcessBatchOutcome(
        batch_id=batch.id,
        ok=outcome.ok,
        completed_event_ids=completed,
        tool_call_count=outcome.tool_call_count,
        error=outcome.error,
    )


def _event_from_row(row: ContextEventRow) -> ContextEvent:
    return ContextEvent(
        event_id=row.id,
        source_system=row.source_system,
        event_type=row.event_type,
        action=row.action,
        pot_id=row.pot_id,
        provider=row.provider,
        provider_host=row.provider_host,
        repo_name=row.repo_name,
        source_id=row.source_id,
        source_event_id=row.source_event_id,
        payload=row.payload,
        occurred_at=row.occurred_at,
        received_at=row.received_at,
        ingestion_kind=row.ingestion_kind,
        idempotency_key=row.idempotency_key,
        source_channel=row.source_channel,
        actor=row.actor,
    )


def _primary_repo_name_for_pot(
    pots: PotResolutionPort, pot_id: str
) -> str | None:
    resolved = pots.resolve_pot(pot_id)
    if resolved is None:
        return None
    primary = resolved.primary_repo()
    return primary.repo_name if primary else None


