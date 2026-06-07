"""Run the reconciliation agent over one debounced batch.

Worker entrypoint: load the batch + its events + any prior agent checkpoint,
call ``agent.run_batch``, and persist the outcome (mark events processed, drop
checkpoint on success, or mark batch failed).

Large batches are split into chunks of
``CONTEXT_ENGINE_MAX_CHUNK_EVENTS`` (default 20) and the agent is invoked
once per chunk. Each chunk runs with a fresh pydantic-ai message history
to bound prompt size; the graph context is shared via the agent's
constructor, so a mutation made by chunk N is visible to chunks N+1+ via
graph reads. Chunks run sequentially within a batch — mutation ordering
across chunks is preserved.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

from application.services.agent_work_events import (
    WorkEventRecord,
    build_work_events,
)
from bootstrap.observability_context import correlation_scope
from bootstrap.observability_runtime import get_observability
from domain.actor import SYSTEM_ACTOR
from domain.context_events import ContextEvent
from domain.ports.observability import SPAN_KIND_INTERNAL
from domain.ports.agent_checkpoint_store import AgentCheckpointStorePort
from domain.ports.agent_execution_log import (
    AgentExecutionLogPort,
    NoOpAgentExecutionLog,
)
from domain.ports.batch_repository import BatchRepositoryPort
from domain.ports.event_stream import (
    EventStreamPublisherPort,
    NoOpEventStreamPublisher,
)
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
    BatchAgentOutcome,
    ReconciliationBatch,
)

logger = logging.getLogger(__name__)


def _chunk_size_default() -> int:
    """Read ``CONTEXT_ENGINE_MAX_CHUNK_EVENTS`` with a safe fallback.

    The default (20) is calibrated to fit comfortably inside the agent's
    instruction + tool surface without bloating prompt tokens. Operators
    tune lower for slower / cheaper models, higher for capable models that
    can hold more events in working memory.
    """
    raw = os.getenv("CONTEXT_ENGINE_MAX_CHUNK_EVENTS")
    if not raw:
        return 20
    try:
        v = int(raw)
        return v if v > 0 else 20
    except ValueError:
        logger.warning(
            "CONTEXT_ENGINE_MAX_CHUNK_EVENTS not an int: %r; using default 20",
            raw,
        )
        return 20


@dataclass(slots=True, frozen=True)
class ProcessBatchOutcome:
    """Worker-level outcome for one batch processing attempt."""

    batch_id: str
    ok: bool
    completed_event_ids: list[str]
    tool_call_count: int
    error: str | None = None
    chunks_run: int = 1
    """How many chunked agent invocations the batch actually triggered.
    ``1`` for batches that fit in one chunk (the common case)."""


def process_batch(
    *,
    batch: ReconciliationBatch,
    agent: ReconciliationAgentPort,
    batches: BatchRepositoryPort,
    reco_ledger: ReconciliationLedgerPort,
    checkpoints: AgentCheckpointStorePort,
    pots: PotResolutionPort,
    policy: PolicyPort | None = None,
    stream_publisher: EventStreamPublisherPort | None = None,
    execution_log: AgentExecutionLogPort | None = None,
) -> ProcessBatchOutcome:
    """Process a claimed batch end-to-end.

    The caller must have already transitioned the batch to ``claimed`` (via
    ``BatchRepositoryPort.claim_batch_by_id``); this function moves it to
    ``running`` before invoking the agent and to ``done`` / ``failed`` after.

    When a :class:`PolicyPort` is supplied, ``apply.write`` is authorized
    once before the agent runs — this is the single policy boundary for
    every mutation the agent issues over the batch.
    """
    # ``stream`` is the coarse per-pot status channel (list-row indicators).
    # ``exec_log`` is the durable agent execution log — the live token/tool
    # stream *and* the crash-resume substrate. ``checkpoints`` is retained
    # only for call-site compatibility; the execution log subsumes it.
    del checkpoints
    stream = stream_publisher or NoOpEventStreamPublisher()
    exec_log: AgentExecutionLogPort = execution_log or NoOpAgentExecutionLog()

    resume = exec_log.load_resume_state(batch.id)
    already_done: set[str] = set(resume.completed_event_ids) if resume else set()
    # Continue the append-only log from where a crashed run left off so
    # seqs stay monotonic and the UI never re-renders stale records.
    _seq = [resume.last_seq if resume else 0]

    def _emit(
        record_type: str,
        payload: dict[str, Any],
        *,
        event_id: str | None = None,
    ) -> None:
        _seq[0] += 1
        try:
            exec_log.append(
                batch_id=batch.id,
                seq=_seq[0],
                record_type=record_type,  # type: ignore[arg-type]
                payload=payload,
                event_id=event_id,
            )
        except Exception:  # noqa: BLE001 - liveness must not fail ingestion
            logger.warning(
                "execution-log %s emit failed for batch %s",
                record_type,
                batch.id,
                exc_info=True,
            )

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
            denied_event_ids = [
                ref.event_id for ref in refs if ref.processed_at is None
            ]
            reco_ledger.record_events_failed(
                denied_event_ids, f"policy:{decision.reason}"
            )
            _safe_publish_status(
                stream,
                pot_id=batch.pot_id,
                event_ids=denied_event_ids,
                status="failed",
                message=f"policy:{decision.reason}",
            )
            for eid in denied_event_ids:
                _safe_publish_end(
                    stream,
                    pot_id=batch.pot_id,
                    event_id=eid,
                    status="failed",
                    error=f"policy:{decision.reason}",
                )
            return ProcessBatchOutcome(
                batch_id=batch.id,
                ok=False,
                completed_event_ids=[],
                tool_call_count=0,
                error=f"policy:{decision.reason}",
            )

    refs = batches.list_events_for_batch(batch.id)
    # A resumed run skips events a crashed attempt already finished — their
    # graph mutations are durable (idempotent on entity_keys) and the
    # completed set is durable, so re-running them is wasted work and would
    # double-render in the UI. They still count as completed below.
    pending_event_ids = [
        r.event_id
        for r in refs
        if r.processed_at is None and r.event_id not in already_done
    ]
    events: list[ContextEvent] = []
    for ref in refs:
        if ref.processed_at is not None or ref.event_id in already_done:
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
        # Nothing left to run. Close the batch, crediting any events a
        # crashed attempt had already finished.
        done_ids = sorted(already_done)
        batches.mark_batch_done(batch.id, completed_event_ids=done_ids)
        reco_ledger.record_events_reconciled(done_ids)
        _emit(
            "run_finished",
            {
                "status": "done",
                "summary": "resumed: all events already reconciled"
                if done_ids
                else "no events to process",
                "completed_event_ids": done_ids,
            },
        )
        exec_log.clear(batch.id)
        return ProcessBatchOutcome(
            batch_id=batch.id,
            ok=True,
            completed_event_ids=done_ids,
            tool_call_count=0,
        )

    repo_name = _primary_repo_name_for_pot(pots, batch.pot_id)

    batches.mark_batch_running(batch.id)
    _claim_events_for_processing(reco_ledger, pending_event_ids)

    # Tell live subscribers each event is now in flight. Best-effort —
    # publish failures must not abort ingestion.
    processing_status = {
        "status": "processing",
        "stage": "executing",
        "message": f"batch {batch.id} running",
    }
    for eid in pending_event_ids:
        _emit("status", processing_status, event_id=eid)
    _safe_publish_status(
        stream,
        pot_id=batch.pot_id,
        event_ids=pending_event_ids,
        status=processing_status["status"],
        stage=processing_status["stage"],
        message=processing_status["message"],
    )

    capabilities = _capability_metadata(agent)
    run_ids = _start_runs_for_pending(
        reco_ledger,
        pending_event_ids=pending_event_ids,
        capabilities=capabilities,
    )

    # Chunked execution. The common case (one chunk) is the same as the
    # legacy single-agent-run path. Large batches split into sequential
    # chunks — each runs with a fresh message history so prompt size stays
    # bounded. The mutation tool's idempotency keeps cross-chunk safety:
    # if chunk N+1 sees the same entity_key chunk N upserted, it's a
    # no-op upsert.
    chunk_size = _chunk_size_default()
    event_chunks = _chunk_events(events, chunk_size)
    chunks_total = len(event_chunks)

    logger.info(
        "context-ingest start: pot=%s batch=%s repo=%s events=%d chunks=%d ids=[%s]",
        batch.pot_id,
        batch.id,
        repo_name or "-",
        len(pending_event_ids),
        chunks_total,
        _fmt_event_ids(pending_event_ids),
    )

    _emit(
        "run_started",
        {
            "title": "Agent run started",
            "attempt": batch.attempt_count,
            "chunks_total": chunks_total,
            "model": capabilities.get("agent_version"),
            "resumed": resume is not None,
            "resumed_completed": len(already_done),
        },
    )

    # Already-done events (durable, from a crashed attempt) stay credited.
    aggregated_completed: list[str] = sorted(already_done)
    aggregated_tool_calls = 0
    failure_outcome: BatchAgentOutcome | None = None

    for idx, chunk in enumerate(event_chunks, start=1):
        chunk_pending_ids = [e.event_id for e in chunk]
        if chunks_total > 1:
            # Emit the chunk marker *before* capturing ``start_seq`` so the
            # agent's seq allocation continues strictly after it.
            _emit(
                "chunk_marker",
                {
                    "title": f"Chunk {idx}/{chunks_total}",
                    "index": idx,
                    "total": chunks_total,
                },
            )
            # Coarse list-row hint: "chunk 2 / 8" (the per-pot status
            # channel, kept off the per-tool-call render path).
            for eid in chunk_pending_ids:
                _safe_publish_status_one(
                    stream,
                    pot_id=batch.pot_id,
                    event_id=eid,
                    status="processing",
                    stage=f"chunk_{idx}_of_{chunks_total}",
                    message=f"chunk {idx}/{chunks_total}",
                )

        # First chunk continues from the durable message-history checkpoint
        # whenever one exists (crash mid-run) so the agent resumes with full
        # context. Later chunks start fresh — they share state through the
        # graph, not the message history.
        prior_msgs = resume.messages_json if (resume is not None and idx == 1) else None
        chunk_ctx = BatchAgentContext(
            batch_id=batch.id,
            pot_id=batch.pot_id,
            repo_name=repo_name,
            events=chunk,
            prior_messages_json=prior_msgs,
            attempt_number=batch.attempt_count,
            chunk_index=idx - 1,
            chunks_total=chunks_total,
            start_seq=_seq[0],
        )

        _obs = get_observability()
        _chunk_label = f"{idx}/{chunks_total}"
        _event_ids_csv = ",".join(chunk_pending_ids)
        try:
            # Baggage so pydantic-ai's own (child) spans inherit the ids and
            # are traceable back to this agent run; the span itself carries
            # them as attributes for the deterministic case.
            with (
                correlation_scope(chunk=_chunk_label),
                _obs.baggage(
                    pot_id=batch.pot_id,
                    batch_id=batch.id,
                    chunk=_chunk_label,
                    event_ids=_event_ids_csv,
                ),
                _obs.span(
                    "agent.run_batch",
                    kind=SPAN_KIND_INTERNAL,
                    attributes={
                        "pot_id": batch.pot_id,
                        "batch_id": batch.id,
                        "agent.chunk": _chunk_label,
                        "agent.attempt": batch.attempt_count,
                        "agent.event_count": len(chunk_pending_ids),
                        "agent.event_ids": _event_ids_csv,
                    },
                ) as _span,
            ):
                chunk_outcome = agent.run_batch(chunk_ctx, execution_log=exec_log)
                _span.set_attribute("agent.ok", bool(chunk_outcome.ok))
                _span.set_attribute("agent.tool_calls", chunk_outcome.tool_call_count)
                if not chunk_outcome.ok:
                    _span.set_error(chunk_outcome.error or "agent run not ok")
            _obs.histogram(
                "ce.agent.tool_calls",
                float(chunk_outcome.tool_call_count),
                attributes={"pot_id": batch.pot_id},
            )
        except Exception as exc:
            logger.exception(
                "batch %s chunk %d/%d agent run failed", batch.id, idx, chunks_total
            )
            failure_outcome = BatchAgentOutcome(ok=False, error=str(exc))
            # The crashed agent may have streamed records up to its last
            # durable checkpoint — advance past them so the terminal
            # ``run_failed`` record doesn't collide on (batch, seq).
            crashed = exec_log.load_resume_state(batch.id)
            if crashed is not None:
                _seq[0] = max(_seq[0], crashed.last_seq)
            # Flush failure trace ONLY to this chunk's runs so other chunks'
            # event traces aren't polluted.
            chunk_run_ids = {
                eid: run_ids[eid] for eid in chunk_pending_ids if eid in run_ids
            }
            _flush_failure_trace(
                reco_ledger,
                run_ids=chunk_run_ids,
                outcome=failure_outcome,
            )
            break

        # The agent allocated execution-log seqs from ``start_seq``; continue
        # strictly after them for chunk markers / terminal records.
        _seq[0] = max(_seq[0], chunk_outcome.last_seq)

        # Apply this chunk's outcome to ONLY this chunk's event runs.
        chunk_completed = list(chunk_outcome.completed_event_ids)
        aggregated_completed.extend(chunk_completed)
        aggregated_tool_calls += chunk_outcome.tool_call_count
        chunk_run_ids = {
            eid: run_ids[eid] for eid in chunk_pending_ids if eid in run_ids
        }
        _flush_outcome_trace(
            reco_ledger,
            run_ids=chunk_run_ids,
            outcome=chunk_outcome,
            completed_event_ids=set(chunk_completed),
        )

        # Mid-batch chunk failure (agent returned ok=False without raising).
        if not chunk_outcome.ok:
            failure_outcome = chunk_outcome
            break

    if failure_outcome is None:
        completed = aggregated_completed
        logger.info(
            "context-ingest done: pot=%s batch=%s reconciled=%d tool_calls=%d chunks=%d",
            batch.pot_id,
            batch.id,
            len(completed),
            aggregated_tool_calls,
            chunks_total,
        )
        batches.mark_batch_done(batch.id, completed_event_ids=completed)
        reco_ledger.record_events_reconciled(completed)
        _emit(
            "run_finished",
            {
                "status": "done",
                "title": "Batch complete",
                "summary": f"{len(completed)} event(s) reconciled",
                "completed_event_ids": completed,
            },
        )
        # Drop the resume checkpoint; the append-only log rows stay for
        # post-run history / replay.
        exec_log.clear(batch.id)
        _safe_publish_status(
            stream,
            pot_id=batch.pot_id,
            event_ids=completed,
            status="done",
            stage="completed",
        )
        for eid in completed:
            _safe_publish_end(
                stream,
                pot_id=batch.pot_id,
                event_id=eid,
                status="done",
            )
        get_observability().counter(
            "ce.events.reconciled_total",
            len(completed),
            attributes={"pot_id": batch.pot_id},
        )
        return ProcessBatchOutcome(
            batch_id=batch.id,
            ok=True,
            completed_event_ids=completed,
            tool_call_count=aggregated_tool_calls,
            chunks_run=chunks_total,
        )

    # Failure path — mark the batch failed. Events that DID complete before
    # the failing chunk stay processed (each chunk marks done immediately
    # above, but mark_batch_done is only called on full success — so we
    # need to defer "mark events reconciled" too). To preserve the
    # "complete events stay complete" invariant, we record reconciled here
    # for the chunks that finished successfully.
    err = failure_outcome.error or "agent_returned_not_ok"
    batches.mark_batch_failed(batch.id, err)
    reco_ledger.record_events_reconciled(aggregated_completed)
    failed_event_ids = [
        eid for eid in pending_event_ids if eid not in aggregated_completed
    ]
    logger.warning(
        "context-ingest failed: pot=%s batch=%s error=%s reconciled=%d failed=%d ids=[%s]",
        batch.pot_id,
        batch.id,
        err,
        len(aggregated_completed),
        len(failed_event_ids),
        _fmt_event_ids(failed_event_ids),
    )
    reco_ledger.record_events_failed(failed_event_ids, err)
    _obs_fin = get_observability()
    if aggregated_completed:
        _obs_fin.counter(
            "ce.events.reconciled_total",
            len(aggregated_completed),
            attributes={"pot_id": batch.pot_id},
        )
    _obs_fin.counter(
        "ce.events.failed_total",
        len(failed_event_ids),
        attributes={"pot_id": batch.pot_id},
    )
    _safe_publish_status(
        stream,
        pot_id=batch.pot_id,
        event_ids=aggregated_completed,
        status="done",
        stage="completed",
    )
    _safe_publish_status(
        stream,
        pot_id=batch.pot_id,
        event_ids=failed_event_ids,
        status="failed",
        message=err,
    )
    for eid in aggregated_completed:
        _safe_publish_end(
            stream,
            pot_id=batch.pot_id,
            event_id=eid,
            status="done",
        )
    for eid in failed_event_ids:
        _safe_publish_end(
            stream,
            pot_id=batch.pot_id,
            event_id=eid,
            status="failed",
            error=err,
        )
    # Terminal record. The resume checkpoint is intentionally NOT cleared —
    # a retry re-dispatch resumes with full context + the durable completed
    # set so finished events aren't redone.
    _emit(
        "run_failed",
        {
            "status": "failed",
            "title": "Batch failed",
            "error": err,
            "completed_event_ids": aggregated_completed,
        },
    )
    return ProcessBatchOutcome(
        batch_id=batch.id,
        ok=False,
        completed_event_ids=aggregated_completed,
        tool_call_count=aggregated_tool_calls,
        error=err,
        chunks_run=chunks_total,
    )


def _capability_metadata(agent: ReconciliationAgentPort) -> dict[str, str | None]:
    meta: dict[str, object] = {}
    fn = getattr(agent, "capability_metadata", None)
    if callable(fn):
        try:
            raw = fn()
        except Exception:
            logger.debug("agent.capability_metadata() raised", exc_info=True)
            raw = None
        if isinstance(raw, dict):
            meta = raw
    return {
        "agent_name": _str_or_none(meta.get("agent")),
        "agent_version": _str_or_none(meta.get("version")),
        "toolset_version": _str_or_none(meta.get("toolset_version")),
    }


def _start_runs_for_pending(
    reco_ledger: ReconciliationLedgerPort,
    *,
    pending_event_ids: list[str],
    capabilities: dict[str, str | None],
) -> dict[str, str]:
    """Open one reconciliation_run per pending event so the UI can show progress.

    The batched agent runs once over all events; we materialize a run row per
    event up front and later fan the same work-event log into each one.
    Skipped on individual error so a single bad event doesn't poison the whole
    batch.
    """
    run_ids: dict[str, str] = {}
    for eid in pending_event_ids:
        try:
            attempt = reco_ledger.next_attempt_number(eid)
            run_id = reco_ledger.start_reconciliation_run(
                eid,
                attempt_number=attempt,
                agent_name=capabilities.get("agent_name"),
                agent_version=capabilities.get("agent_version"),
                toolset_version=capabilities.get("toolset_version"),
            )
            run_ids[eid] = run_id
        except Exception:
            logger.exception("failed to open reconciliation run for event %s", eid)
    return run_ids


def _claim_events_for_processing(
    reco_ledger: ReconciliationLedgerPort,
    event_ids: list[str],
) -> None:
    """Persist queued -> processing so DB refetches agree with live streams."""
    for eid in event_ids:
        try:
            reco_ledger.claim_event_for_processing(eid)
        except Exception:
            logger.exception("failed to mark event %s processing", eid)


def _flush_outcome_trace(
    reco_ledger: ReconciliationLedgerPort,
    *,
    run_ids: dict[str, str],
    outcome: BatchAgentOutcome,
    completed_event_ids: set[str],
) -> None:
    """Persist the agent's trace + finalize each event run.

    Work events are duplicated across every event's run in the batch — the
    batch ran once, but each event needs its own work-event log so it
    renders independently in the UI.
    """
    if not run_ids:
        return
    records = build_work_events(
        prompt=outcome.prompt,
        messages_json=outcome.agent_messages_json,
        final_response=outcome.final_response,
        error=None if outcome.ok else outcome.error,
    )
    for eid, run_id in run_ids.items():
        _append_records(reco_ledger, run_id, records)
        try:
            if outcome.ok and eid in completed_event_ids:
                reco_ledger.record_run_success(run_id)
            else:
                reco_ledger.record_run_failure(
                    run_id,
                    outcome.error or "agent_returned_not_ok",
                )
        except Exception:
            logger.exception("failed to finalize reconciliation run %s", run_id)


def _flush_failure_trace(
    reco_ledger: ReconciliationLedgerPort,
    *,
    run_ids: dict[str, str],
    outcome: BatchAgentOutcome,
) -> None:
    if not run_ids:
        return
    records = build_work_events(
        prompt=outcome.prompt,
        messages_json=outcome.agent_messages_json,
        final_response=outcome.final_response,
        error=outcome.error,
    )
    for run_id in run_ids.values():
        _append_records(reco_ledger, run_id, records)
        try:
            reco_ledger.record_run_failure(run_id, outcome.error or "agent_crashed")
        except Exception:
            logger.exception("failed to mark reconciliation run %s failed", run_id)


def _append_records(
    reco_ledger: ReconciliationLedgerPort,
    run_id: str,
    records: list[WorkEventRecord],
) -> None:
    for rec in records:
        try:
            reco_ledger.record_run_work_event(
                run_id,
                event_kind=rec.event_kind,
                title=rec.title,
                body=rec.body,
                payload=rec.payload,
            )
        except Exception:
            logger.exception(
                "failed to append work event (kind=%s) to run %s",
                rec.event_kind,
                run_id,
            )


def _fmt_event_ids(event_ids: list[str], *, head: int = 10) -> str:
    """Compact id list for one log line: ``a,b,c (+N more)``."""
    if len(event_ids) <= head:
        return ",".join(event_ids)
    return f"{','.join(event_ids[:head])} (+{len(event_ids) - head} more)"


def _str_or_none(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _chunk_events(
    events: list[ContextEvent], chunk_size: int
) -> list[list[ContextEvent]]:
    """Split events into chunks of ``chunk_size``. Order is preserved.

    A single-chunk result is returned for batches at or below the limit,
    which preserves the legacy "one agent run per batch" code path
    unchanged.
    """
    if len(events) <= chunk_size:
        return [list(events)]
    return [list(events[i : i + chunk_size]) for i in range(0, len(events), chunk_size)]


def _safe_publish_status_one(
    stream: EventStreamPublisherPort,
    *,
    pot_id: str,
    event_id: str,
    status: str,
    stage: str | None = None,
    message: str | None = None,
) -> None:
    try:
        stream.publish_status(
            pot_id=pot_id,
            event_id=event_id,
            status=status,
            stage=stage,
            message=message,
        )
    except Exception:
        logger.warning(
            "stream publish_status failed for event %s", event_id, exc_info=True
        )


def _safe_publish_status(
    stream: EventStreamPublisherPort,
    *,
    pot_id: str,
    event_ids: list[str],
    status: str,
    stage: str | None = None,
    message: str | None = None,
) -> None:
    """Best-effort fan-out of a status transition to ``event_ids``.

    Each call is wrapped so a partial Redis outage can't poison the batch —
    streaming is liveness sugar, not correctness.
    """
    for eid in event_ids:
        try:
            stream.publish_status(
                pot_id=pot_id,
                event_id=eid,
                status=status,
                stage=stage,
                message=message,
            )
        except Exception:
            logger.warning(
                "stream publish_status failed for event %s", eid, exc_info=True
            )


def _safe_publish_end(
    stream: EventStreamPublisherPort,
    *,
    pot_id: str,
    event_id: str,
    status: str,
    error: str | None = None,
) -> None:
    try:
        stream.publish_end(pot_id=pot_id, event_id=event_id, status=status, error=error)
    except Exception:
        logger.warning(
            "stream publish_end failed for event %s", event_id, exc_info=True
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


def _primary_repo_name_for_pot(pots: PotResolutionPort, pot_id: str) -> str | None:
    resolved = pots.resolve_pot(pot_id)
    if resolved is None:
        return None
    primary = resolved.primary_repo()
    return primary.repo_name if primary else None
