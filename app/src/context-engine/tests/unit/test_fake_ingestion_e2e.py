"""CGT-6: fake-port end-to-end ingestion contract (no FastAPI/Celery/Neo4j).

Pins the canonical pipeline:
``IngestionSubmissionRequest`` → ``DefaultIngestionSubmissionService.submit``
→ ``admit_event`` → batch enqueue → ``handle_process_batch`` → fake agent
→ ``apply_plan_async``.

Assertions use durable port outcomes (duplicate flag, enqueue counts, batch
status, graph apply calls) — not private helpers or error message strings.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import pytest

from application.services.ingestion_submission_service import (
    DefaultIngestionSubmissionService,
)
from application.use_cases.context_graph_jobs import handle_process_batch
from application.use_cases.flush_windowed_batches import force_flush_pot
from domain.context_events import ContextEvent, EventScope
from domain.context_events import EventRef as ContextEventRef
from domain.graph_mutations import ProvenanceContext
from domain.ingestion_event_models import (
    CreateIngestionEventParams,
    EventListFilters,
    EventListPage,
    EventTransition,
    IngestionEvent,
    IngestionSubmissionRequest,
)
from domain.ingestion_kinds import INGESTION_KIND_AGENT_RECONCILIATION
from domain.ports.event_stream import NoOpEventStreamPublisher
from domain.ports.ingestion_config import (
    IngestionConfig,
    InMemoryIngestionConfig,
)
from domain.ports.pot_resolution import PotResolutionPort, single_github_repo_pot
from domain.ports.reconciliation_ledger import (
    ContextEventRow,
    ReconciliationLedgerPort,
    ReconciliationRunRow,
    ReconciliationWorkEventRow,
)
from domain.reconciliation import (
    EpisodeDraft,
    MutationSummary,
    ReconciliationPlan,
    ReconciliationResult,
)
from domain.reconciliation_batch import (
    BATCH_STATUS_CLAIMED,
    BATCH_STATUS_DONE,
    BATCH_STATUS_PENDING,
    BATCH_STATUS_RUNNING,
    BatchAgentContext,
    BatchAgentOutcome,
    BatchEventRef,
    ReconciliationBatch,
)

pytestmark = pytest.mark.unit


def _now() -> datetime:
    return datetime(2026, 5, 22, 12, 0, tzinfo=timezone.utc)


def _submission_request(**overrides: Any) -> IngestionSubmissionRequest:
    base = {
        "pot_id": "pot-1",
        "ingestion_kind": INGESTION_KIND_AGENT_RECONCILIATION,
        "source_channel": "test",
        "source_system": "github",
        "event_type": "pull_request",
        "action": "merged",
        "payload": {"title": "Merge widgets"},
        "source_id": "pr_42_merged",
        "occurred_at": _now(),
    }
    base.update(overrides)
    return IngestionSubmissionRequest(**base)


class _EnabledSettings:
    def is_enabled(self) -> bool:
        return True

    def neo4j_uri(self) -> str | None:
        return None

    def neo4j_user(self) -> str | None:
        return None

    def neo4j_password(self) -> str | None:
        return None

    def backfill_max_prs_per_run(self) -> int:
        return 50


@dataclass
class _ApplyCall:
    expected_pot_id: str
    event_id: str


class RecordingGraphPort:
    """Minimal ``ContextGraphPort`` that records ``apply_plan_async`` calls."""

    enabled = True

    def __init__(self) -> None:
        self.apply_calls: list[_ApplyCall] = []

    async def apply_plan_async(
        self,
        plan: ReconciliationPlan,
        *,
        expected_pot_id: str,
        provenance_context: ProvenanceContext | None = None,
    ) -> ReconciliationResult:
        del provenance_context
        self.apply_calls.append(
            _ApplyCall(
                expected_pot_id=expected_pot_id,
                event_id=plan.event_ref.event_id,
            )
        )
        return ReconciliationResult(
            ok=True,
            episode_uuids=["ep-fake"],
            mutation_summary=MutationSummary(episodes_written=1),
        )

    def query(self, request: Any) -> Any:
        raise NotImplementedError

    async def query_async(self, request: Any) -> Any:
        raise NotImplementedError

    def apply_plan(
        self,
        plan: ReconciliationPlan,
        *,
        expected_pot_id: str,
        provenance_context: ProvenanceContext | None = None,
    ) -> ReconciliationResult:
        return asyncio.run(
            self.apply_plan_async(
                plan,
                expected_pot_id=expected_pot_id,
                provenance_context=provenance_context,
            )
        )


class PlanApplyingFakeAgent:
    """Agent that applies a minimal plan per event via the graph port."""

    def __init__(self, graph: RecordingGraphPort) -> None:
        self._graph = graph

    def run_batch(
        self,
        ctx: BatchAgentContext,
        *,
        checkpoints: object | None = None,
        execution_log: object | None = None,
    ) -> BatchAgentOutcome:
        del checkpoints, execution_log
        completed: list[str] = []
        for ev in ctx.events:
            plan = ReconciliationPlan(
                event_ref=ContextEventRef(
                    event_id=ev.event_id,
                    source_system=ev.source_system,
                    pot_id=ev.pot_id,
                ),
                summary="fake e2e apply",
                episodes=[
                    EpisodeDraft(
                        name="episode",
                        episode_body="body",
                        source_description="test",
                        reference_time=ev.occurred_at or _now(),
                    )
                ],
            )
            asyncio.run(
                self._graph.apply_plan_async(
                    plan,
                    expected_pot_id=ctx.pot_id,
                    provenance_context=ProvenanceContext(
                        source_kind=ev.ingestion_kind,
                        created_by_agent="fake-e2e",
                    ),
                )
            )
            completed.append(ev.event_id)
        return BatchAgentOutcome(
            ok=True,
            completed_event_ids=completed,
            tool_call_count=len(completed),
            agent_name="fake-e2e",
            agent_version="1",
            toolset_version="none",
        )

    def capability_metadata(self) -> dict[str, Any]:
        return {
            "agent": "fake-e2e",
            "version": "1",
            "toolset_version": "none",
        }


def _dedupe_key(scope: EventScope, source_id: str) -> tuple[str, str, str, str, str]:
    return (
        scope.pot_id,
        scope.provider,
        scope.provider_host,
        scope.repo_name,
        source_id,
    )


def _row_from_event(event: ContextEvent, *, status: str = "queued") -> ContextEventRow:
    return ContextEventRow(
        id=event.event_id,
        pot_id=event.pot_id,
        provider=event.provider,
        provider_host=event.provider_host,
        repo_name=event.repo_name,
        source_system=event.source_system,
        event_type=event.event_type,
        action=event.action,
        source_id=event.source_id,
        source_event_id=event.source_event_id,
        payload=dict(event.payload),
        occurred_at=event.occurred_at,
        received_at=event.received_at or _now(),
        status=status,
        ingestion_kind=event.ingestion_kind,
        idempotency_key=event.idempotency_key,
        source_channel=event.source_channel,
        actor=event.actor,
    )


def _ingestion_from_row(row: ContextEventRow) -> IngestionEvent:
    status: str
    if row.status in ("reconciled", "done"):
        status = "done"
    elif row.status == "failed":
        status = "error"
    elif row.status == "processing":
        status = "processing"
    else:
        status = "queued"
    return IngestionEvent(
        event_id=row.id,
        pot_id=row.pot_id,
        ingestion_kind=row.ingestion_kind,
        source_channel=row.source_channel or "test",
        source_system=row.source_system,
        event_type=row.event_type,
        action=row.action,
        source_id=row.source_id,
        dedup_key=None,
        status=status,  # type: ignore[arg-type]
        stage=None,
        submitted_at=row.received_at,
        started_at=None,
        completed_at=None,
        error=None,
        payload=row.payload,
        provider=row.provider,
        provider_host=row.provider_host,
        repo_name=row.repo_name,
        raw_status=row.status,
    )


@dataclass
class InMemoryIngestionHarness:
    """In-memory ledger + batches + event store + job queue for E2E tests."""

    pots: PotResolutionPort = field(
        default_factory=lambda: _StaticPots(
            single_github_repo_pot("pot-1", "acme/widgets")
        )
    )
    dedupe_index: dict[tuple[str, str, str, str, str], str] = field(
        default_factory=dict
    )
    context_rows: dict[str, ContextEventRow] = field(default_factory=dict)
    ingestion_events: dict[str, IngestionEvent] = field(default_factory=dict)
    batches: dict[str, ReconciliationBatch] = field(default_factory=dict)
    batch_members: dict[str, list[BatchEventRef]] = field(default_factory=dict)
    enqueued_batch_ids: list[str] = field(default_factory=list)
    runs: dict[str, ReconciliationRunRow] = field(default_factory=dict)

    # --- ReconciliationLedgerPort ---

    def append_event(
        self, scope: EventScope, event: ContextEvent
    ) -> tuple[str, bool]:
        key = _dedupe_key(scope, event.source_id)
        if key in self.dedupe_index:
            return self.dedupe_index[key], False
        self.dedupe_index[key] = event.event_id
        row = _row_from_event(event, status="received")
        self.context_rows[event.event_id] = row
        params = CreateIngestionEventParams(
            event_id=event.event_id,
            pot_id=event.pot_id,
            ingestion_kind=event.ingestion_kind,
            source_channel=event.source_channel or "test",
            source_system=event.source_system,
            event_type=event.event_type,
            action=event.action,
            source_id=event.source_id,
            dedup_key=event.idempotency_key,
            status="queued",
            stage=None,
            payload=dict(event.payload),
            submitted_at=row.received_at,
            provider=event.provider,
            provider_host=event.provider_host,
            repo_name=event.repo_name,
            actor=event.actor,
        )
        self.ingestion_events[event.event_id] = self.create_event(params)
        return event.event_id, True

    def get_event_by_id(self, event_id: str) -> ContextEventRow | None:
        return self.context_rows.get(event_id)

    def mark_event_queued(self, event_id: str) -> None:
        row = self.context_rows.get(event_id)
        if row is not None:
            self.context_rows[event_id] = _row_from_event(
                _context_from_row(row), status="queued"
            )

    def claim_event_for_processing(self, event_id: str) -> bool:
        row = self.context_rows.get(event_id)
        if row is None or row.status in ("reconciled", "failed"):
            return False
        self.context_rows[event_id] = _row_from_event(
            _context_from_row(row), status="processing"
        )
        self._transition_ingestion(event_id, to_status="processing")
        return True

    def start_reconciliation_run(
        self,
        event_id: str,
        *,
        attempt_number: int,
        agent_name: str | None,
        agent_version: str | None,
        toolset_version: str | None,
    ) -> str:
        run_id = f"run-{event_id}-{attempt_number}"
        self.runs[run_id] = ReconciliationRunRow(
            id=run_id,
            event_id=event_id,
            attempt_number=attempt_number,
            status="running",
            agent_name=agent_name,
            agent_version=agent_version,
            toolset_version=toolset_version,
            plan_summary=None,
            episode_count=None,
            entity_mutation_count=None,
            edge_mutation_count=None,
            error=None,
            started_at=_now(),
            completed_at=None,
        )
        return run_id

    def record_plan_metadata(
        self,
        run_id: str,
        *,
        plan_summary: str,
        episode_count: int,
        entity_mutation_count: int,
        edge_mutation_count: int,
    ) -> None:
        del run_id, plan_summary, episode_count, entity_mutation_count, edge_mutation_count

    def record_run_plan_json(self, run_id: str, body: dict[str, Any]) -> None:
        del run_id, body

    def record_run_work_event(
        self,
        run_id: str,
        *,
        event_kind: str,
        title: str | None = None,
        body: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> str:
        del run_id, event_kind, title, body, payload
        return "work-1"

    def record_run_success(self, run_id: str) -> None:
        run = self.runs.get(run_id)
        if run is not None:
            self.runs[run_id] = ReconciliationRunRow(
                id=run.id,
                event_id=run.event_id,
                attempt_number=run.attempt_number,
                status="done",
                agent_name=run.agent_name,
                agent_version=run.agent_version,
                toolset_version=run.toolset_version,
                plan_summary=run.plan_summary,
                episode_count=run.episode_count,
                entity_mutation_count=run.entity_mutation_count,
                edge_mutation_count=run.edge_mutation_count,
                error=None,
                started_at=run.started_at,
                completed_at=_now(),
            )

    def record_run_failure(self, run_id: str, error: str) -> None:
        run = self.runs.get(run_id)
        if run is not None:
            self.runs[run_id] = ReconciliationRunRow(
                id=run.id,
                event_id=run.event_id,
                attempt_number=run.attempt_number,
                status="failed",
                agent_name=run.agent_name,
                agent_version=run.agent_version,
                toolset_version=run.toolset_version,
                plan_summary=run.plan_summary,
                episode_count=run.episode_count,
                entity_mutation_count=run.entity_mutation_count,
                edge_mutation_count=run.edge_mutation_count,
                error=error,
                started_at=run.started_at,
                completed_at=_now(),
            )

    def record_event_reconciled(self, event_id: str) -> None:
        self.record_events_reconciled([event_id])

    def record_events_reconciled(self, event_ids: list[str]) -> None:
        for eid in event_ids:
            row = self.context_rows.get(eid)
            if row is None:
                continue
            self.context_rows[eid] = _row_from_event(
                _context_from_row(row), status="reconciled"
            )
            self._transition_ingestion(
                eid,
                to_status="done",
                completed_at=_now(),
            )

    def record_event_failed(self, event_id: str, error: str) -> None:
        self.record_events_failed([event_id], error)

    def record_events_failed(self, event_ids: list[str], error: str) -> None:
        del error
        for eid in event_ids:
            row = self.context_rows.get(eid)
            if row is None:
                continue
            self.context_rows[eid] = _row_from_event(
                _context_from_row(row), status="failed"
            )
            self._transition_ingestion(eid, to_status="error")

    def fail_inflight_events(self, event_ids: list[str], error: str) -> int:
        self.record_events_failed(event_ids, error)
        return len(event_ids)

    def list_runs_for_event(self, event_id: str) -> list[ReconciliationRunRow]:
        return [r for r in self.runs.values() if r.event_id == event_id]

    def list_work_events_for_run(
        self, run_id: str
    ) -> list[ReconciliationWorkEventRow]:
        del run_id
        return []

    def next_attempt_number(self, event_id: str) -> int:
        del event_id
        return 1

    def mark_event_for_retry(self, event_id: str) -> None:
        row = self.context_rows.get(event_id)
        if row is not None:
            self.context_rows[event_id] = _row_from_event(
                _context_from_row(row), status="received"
            )

    def mark_events_for_retry(self, event_ids: list[str]) -> None:
        for eid in event_ids:
            self.mark_event_for_retry(eid)

    def mark_events_queued(self, event_ids: list[str]) -> None:
        for eid in event_ids:
            self.mark_event_queued(eid)

    def delete_all_for_pot(self, pot_id: str) -> int:
        removed = [
            eid
            for eid, row in self.context_rows.items()
            if row.pot_id == pot_id
        ]
        for eid in removed:
            del self.context_rows[eid]
            self.ingestion_events.pop(eid, None)
        return len(removed)

    def set_event_job_metadata(
        self,
        event_id: str,
        *,
        job_id: str | None = None,
        correlation_id: str | None = None,
    ) -> None:
        del job_id, correlation_id, event_id

    def summarize_pot_reconciliation(
        self, pot_id: str, *, recent_failure_limit: int = 5
    ) -> Any:
        del pot_id, recent_failure_limit
        from domain.context_status import ReconciliationLedgerHealth

        return ReconciliationLedgerHealth()

    # --- BatchRepositoryPort ---

    def upsert_open_batch_for_pot(self, pot_id: str, event_id: str) -> str:
        for batch_id, batch in self.batches.items():
            if batch.pot_id == pot_id and batch.status == BATCH_STATUS_PENDING:
                refs = self.batch_members.setdefault(batch_id, [])
                if not any(r.event_id == event_id for r in refs):
                    refs.append(BatchEventRef(event_id=event_id, added_at=_now()))
                return batch_id
        batch_id = f"batch-{uuid4().hex[:8]}"
        self.batches[batch_id] = ReconciliationBatch(
            id=batch_id,
            pot_id=pot_id,
            status=BATCH_STATUS_PENDING,
            attempt_count=1,
            created_at=_now(),
            claimed_at=None,
            completed_at=None,
            last_error=None,
        )
        self.batch_members[batch_id] = [
            BatchEventRef(event_id=event_id, added_at=_now())
        ]
        return batch_id

    def add_events_to_open_batch_for_pot(
        self, pot_id: str, event_ids: list[str]
    ) -> str:
        if not event_ids:
            raise ValueError("event_ids must be non-empty")
        batch_id = self.upsert_open_batch_for_pot(pot_id, event_ids[0])
        for eid in event_ids[1:]:
            self.upsert_open_batch_for_pot(pot_id, eid)
        return batch_id

    def claim_batch_by_id(self, batch_id: str) -> ReconciliationBatch | None:
        batch = self.batches.get(batch_id)
        if batch is None or batch.status != BATCH_STATUS_PENDING:
            return None
        claimed = ReconciliationBatch(
            id=batch.id,
            pot_id=batch.pot_id,
            status=BATCH_STATUS_CLAIMED,
            attempt_count=batch.attempt_count,
            created_at=batch.created_at,
            claimed_at=_now(),
            completed_at=None,
            last_error=None,
        )
        self.batches[batch_id] = claimed
        return claimed

    def get_batch(self, batch_id: str) -> ReconciliationBatch | None:
        return self.batches.get(batch_id)

    def list_stale_in_flight_batches(
        self, older_than_seconds: float
    ) -> list[ReconciliationBatch]:
        del older_than_seconds
        return []

    def list_events_for_batch(self, batch_id: str) -> list[BatchEventRef]:
        return list(self.batch_members.get(batch_id, []))

    def mark_batch_running(self, batch_id: str) -> None:
        batch = self.batches.get(batch_id)
        if batch is None:
            return
        self.batches[batch_id] = ReconciliationBatch(
            id=batch.id,
            pot_id=batch.pot_id,
            status=BATCH_STATUS_RUNNING,
            attempt_count=batch.attempt_count,
            created_at=batch.created_at,
            claimed_at=batch.claimed_at,
            completed_at=None,
            last_error=None,
        )

    def mark_batch_done(
        self,
        batch_id: str,
        *,
        completed_event_ids: list[str],
    ) -> None:
        batch = self.batches.get(batch_id)
        if batch is None:
            return
        self.batches[batch_id] = ReconciliationBatch(
            id=batch.id,
            pot_id=batch.pot_id,
            status=BATCH_STATUS_DONE,
            attempt_count=batch.attempt_count,
            created_at=batch.created_at,
            claimed_at=batch.claimed_at,
            completed_at=_now(),
            last_error=None,
        )
        done_at = _now()
        refs = self.batch_members.setdefault(batch_id, [])
        for ref in refs:
            if ref.event_id in completed_event_ids:
                idx = refs.index(ref)
                refs[idx] = BatchEventRef(
                    event_id=ref.event_id,
                    added_at=ref.added_at,
                    processed_at=done_at,
                )

    def mark_batch_failed(self, batch_id: str, error: str) -> None:
        batch = self.batches.get(batch_id)
        if batch is None:
            return
        self.batches[batch_id] = ReconciliationBatch(
            id=batch.id,
            pot_id=batch.pot_id,
            status="failed",
            attempt_count=batch.attempt_count,
            created_at=batch.created_at,
            claimed_at=batch.claimed_at,
            completed_at=_now(),
            last_error=error,
        )

    def get_open_batch_id_for_pot(self, pot_id: str) -> str | None:
        for batch_id, batch in self.batches.items():
            if batch.pot_id == pot_id and batch.status == BATCH_STATUS_PENDING:
                return batch_id
        return None

    def get_latest_batch_id_for_event(self, event_id: str) -> str | None:
        latest: tuple[datetime, str] | None = None
        for batch_id, refs in self.batch_members.items():
            for ref in refs:
                if ref.event_id != event_id:
                    continue
                if latest is None or ref.added_at > latest[0]:
                    latest = (ref.added_at, batch_id)
        return latest[1] if latest else None

    # --- IngestionEventStore ---

    def create_event(self, params: CreateIngestionEventParams) -> IngestionEvent:
        ev = IngestionEvent(
            event_id=params.event_id,
            pot_id=params.pot_id,
            ingestion_kind=params.ingestion_kind,
            source_channel=params.source_channel,
            source_system=params.source_system,
            event_type=params.event_type,
            action=params.action,
            source_id=params.source_id,
            dedup_key=params.dedup_key,
            status=params.status,
            stage=params.stage,
            submitted_at=params.submitted_at or _now(),
            started_at=None,
            completed_at=None,
            error=None,
            payload=dict(params.payload),
            metadata=dict(params.metadata),
            provider=params.provider,
            provider_host=params.provider_host,
            repo_name=params.repo_name,
            actor=params.actor,
        )
        self.ingestion_events[params.event_id] = ev
        return ev

    def get_event(self, event_id: str) -> IngestionEvent | None:
        row = self.context_rows.get(event_id)
        if row is None:
            return self.ingestion_events.get(event_id)
        return _ingestion_from_row(row)

    def find_duplicate(
        self, pot_id: str, dedup_key: str | None, ingestion_kind: str
    ) -> IngestionEvent | None:
        del pot_id, dedup_key, ingestion_kind
        return None

    def transition_event(
        self, event_id: str, transition: EventTransition
    ) -> IngestionEvent | None:
        ev = self.ingestion_events.get(event_id)
        if ev is None:
            return None
        updated = IngestionEvent(
            event_id=ev.event_id,
            pot_id=ev.pot_id,
            ingestion_kind=ev.ingestion_kind,
            source_channel=ev.source_channel,
            source_system=ev.source_system,
            event_type=ev.event_type,
            action=ev.action,
            source_id=ev.source_id,
            dedup_key=ev.dedup_key,
            status=transition.to_status or ev.status,
            stage=transition.to_stage if transition.to_stage is not None else ev.stage,
            submitted_at=ev.submitted_at,
            started_at=transition.started_at or ev.started_at,
            completed_at=transition.completed_at or ev.completed_at,
            error=transition.error if transition.error is not None else ev.error,
            payload=ev.payload,
            metadata=ev.metadata,
            provider=ev.provider,
            provider_host=ev.provider_host,
            repo_name=ev.repo_name,
            actor=ev.actor,
            raw_status=ev.raw_status,
        )
        self.ingestion_events[event_id] = updated
        return updated

    def list_events(
        self,
        pot_id: str,
        filters: EventListFilters | None,
        *,
        cursor: str | None,
        limit: int,
    ) -> EventListPage:
        del pot_id, filters, cursor, limit
        return EventListPage(items=(), next_cursor=None)

    def record_progress(
        self,
        event_id: str,
        *,
        step_total: int | None = None,
        step_done: int | None = None,
        step_error: int | None = None,
    ) -> None:
        del event_id, step_total, step_done, step_error

    # --- ContextGraphJobQueuePort ---

    def enqueue_batch(self, batch_id: str) -> None:
        self.enqueued_batch_ids.append(batch_id)

    def _transition_ingestion(
        self,
        event_id: str,
        *,
        to_status: str,
        completed_at: datetime | None = None,
    ) -> None:
        self.transition_event(
            event_id,
            EventTransition(
                to_status=to_status,  # type: ignore[arg-type]
                completed_at=completed_at,
            ),
        )


class _StaticPots:
    def __init__(self, resolved: Any) -> None:
        self._resolved = resolved

    def resolve_pot(self, pot_id: str) -> Any:
        if pot_id == self._resolved.pot_id:
            return self._resolved
        return None

    def list_pot_repos(self, pot_id: str) -> list[Any]:
        r = self.resolve_pot(pot_id)
        return list(r.repos) if r else []

    def find_repo(
        self,
        *,
        provider: str,
        provider_host: str,
        repo_name: str,
        user_id: str | None = None,
    ) -> Any:
        del provider, provider_host, repo_name, user_id
        return None


def _context_from_row(row: ContextEventRow) -> ContextEvent:
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
        payload=dict(row.payload),
        occurred_at=row.occurred_at,
        received_at=row.received_at,
        ingestion_kind=row.ingestion_kind,
        idempotency_key=row.idempotency_key,
        source_channel=row.source_channel,
        actor=row.actor,
    )


@dataclass
class _E2ETestContainer:
    """Minimal container duck-type for ``handle_process_batch``."""

    settings: _EnabledSettings
    episodic: RecordingGraphPort
    structural: RecordingGraphPort
    pots: PotResolutionPort
    reconciliation_agent: PlanApplyingFakeAgent
    context_graph: RecordingGraphPort
    harness: InMemoryIngestionHarness
    event_stream_publisher: NoOpEventStreamPublisher = field(
        default_factory=NoOpEventStreamPublisher
    )

    def policy(self) -> Any:
        from adapters.outbound.policy import DefaultPolicyAdapter

        return DefaultPolicyAdapter(
            settings=self.settings,
            pots=self.pots,
            reconciliation_agent_available=self.reconciliation_agent is not None,
            context_graph_available=self.context_graph is not None,
            episodic_available=getattr(self.episodic, "enabled", True),
        )

    def batch_repository(self, _session: object) -> InMemoryIngestionHarness:
        return self.harness

    def reconciliation_ledger(self, _session: object) -> ReconciliationLedgerPort:
        return self.harness

    def agent_checkpoint_store(self, _session: object) -> object:
        from unittest.mock import MagicMock

        store = MagicMock()
        store.load.return_value = None
        return store

    def agent_execution_log(self, _session: object) -> object:
        from domain.ports.agent_execution_log import NoOpAgentExecutionLog

        return NoOpAgentExecutionLog()


def _build_pipeline(
    *,
    ingestion_config: InMemoryIngestionConfig | None = None,
) -> tuple[
    DefaultIngestionSubmissionService,
    InMemoryIngestionHarness,
    RecordingGraphPort,
    PlanApplyingFakeAgent,
    _E2ETestContainer,
]:
    graph = RecordingGraphPort()
    agent = PlanApplyingFakeAgent(graph)
    harness = InMemoryIngestionHarness()
    service = DefaultIngestionSubmissionService(
        settings=_EnabledSettings(),
        pots=harness.pots,
        reconciliation_agent=agent,
        reco_ledger=harness,
        events=harness,
        batches=harness,
        jobs=harness,
        ingestion_config=ingestion_config,
    )
    container = _E2ETestContainer(
        settings=_EnabledSettings(),
        episodic=graph,
        structural=graph,
        pots=harness.pots,
        reconciliation_agent=agent,
        context_graph=graph,
        harness=harness,
    )
    return service, harness, graph, agent, container


def _process_enqueued(
    harness: InMemoryIngestionHarness, container: _E2ETestContainer
) -> dict[str, Any]:
    assert harness.enqueued_batch_ids, "expected at least one enqueued batch"
    batch_id = harness.enqueued_batch_ids[-1]
    return handle_process_batch(
        db=object(),
        batch_id=batch_id,
        build_container=lambda _db: container,
    )


class TestFakeIngestionE2E:
    def test_submit_through_process_applies_graph(self) -> None:
        service, harness, graph, _agent, container = _build_pipeline(
            ingestion_config=InMemoryIngestionConfig(
                {
                    "pot-1": IngestionConfig(
                        pot_id="pot-1",
                        mode="immediate",
                        window_minutes=5,
                        min_batch_size=None,
                    )
                }
            )
        )
        receipt = service.submit(_submission_request())
        assert receipt.duplicate is False
        assert receipt.status == "queued"
        assert len(harness.enqueued_batch_ids) == 1

        outcome = _process_enqueued(harness, container)
        assert outcome["status"] == "ok"
        assert len(outcome["completed_event_ids"]) == 1
        assert len(graph.apply_calls) == 1
        assert graph.apply_calls[0].expected_pot_id == "pot-1"
        batch = harness.batches[harness.enqueued_batch_ids[0]]
        assert batch.status == BATCH_STATUS_DONE

    def test_duplicate_submit_does_not_reenqueue_or_reapply(self) -> None:
        service, harness, graph, _agent, container = _build_pipeline(
            ingestion_config=InMemoryIngestionConfig(
                {
                    "pot-1": IngestionConfig(
                        pot_id="pot-1",
                        mode="immediate",
                        window_minutes=5,
                        min_batch_size=None,
                    )
                }
            )
        )
        first = service.submit(_submission_request())
        assert first.duplicate is False
        second = service.submit(_submission_request())
        assert second.duplicate is True
        assert second.event_id == first.event_id
        assert len(harness.enqueued_batch_ids) == 1

        _process_enqueued(harness, container)
        assert len(graph.apply_calls) == 1

        # Redundant worker dispatch (race) must not double-apply.
        batch_id = harness.enqueued_batch_ids[0]
        again = handle_process_batch(
            db=object(),
            batch_id=batch_id,
            build_container=lambda _db: container,
        )
        assert again["status"] == "skipped"
        assert len(graph.apply_calls) == 1

    def test_windowed_mode_defers_enqueue_until_force_flush(self) -> None:
        service, harness, graph, _agent, container = _build_pipeline(
            ingestion_config=InMemoryIngestionConfig(
                {
                    "pot-1": IngestionConfig(
                        pot_id="pot-1",
                        mode="windowed",
                        window_minutes=5,
                        min_batch_size=None,
                    )
                }
            )
        )
        receipt = service.submit(_submission_request())
        assert receipt.duplicate is False
        assert harness.enqueued_batch_ids == []
        assert harness.get_open_batch_id_for_pot("pot-1") is not None

        flushed = force_flush_pot(
            pot_id="pot-1",
            batches=harness,
            jobs=harness,
        )
        assert flushed is not None
        assert len(harness.enqueued_batch_ids) == 1

        outcome = _process_enqueued(harness, container)
        assert outcome["status"] == "ok"
        assert len(graph.apply_calls) == 1

    def test_immediate_mode_enqueues_on_admit(self) -> None:
        _service, harness, _graph, _agent, _container = _build_pipeline(
            ingestion_config=InMemoryIngestionConfig(
                {
                    "pot-1": IngestionConfig(
                        pot_id="pot-1",
                        mode="immediate",
                        window_minutes=5,
                        min_batch_size=None,
                    )
                }
            )
        )
        _service.submit(_submission_request())
        assert len(harness.enqueued_batch_ids) == 1
