"""Reconciliation lifecycle ledger (port)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional, Protocol

from potpie.context_engine.domain.actor import Actor
from potpie.context_engine.domain.context_events import ContextEvent, EventScope
from potpie.context_engine.domain.context_status import ReconciliationLedgerHealth


@dataclass(slots=True)
class ContextEventRow:
    id: str
    pot_id: str
    provider: str
    provider_host: str
    repo_name: str
    source_system: str
    event_type: str
    action: str
    source_id: str
    source_event_id: str | None
    payload: dict[str, Any]
    occurred_at: datetime | None
    received_at: datetime
    status: str
    ingestion_kind: str = "agent_reconciliation"
    job_id: str | None = None
    correlation_id: str | None = None
    idempotency_key: str | None = None
    source_channel: str | None = None
    actor: Actor | None = None


@dataclass(slots=True)
class ReconciliationRunRow:
    id: str
    event_id: str
    attempt_number: int
    status: str
    agent_name: str | None
    agent_version: str | None
    toolset_version: str | None
    plan_summary: str | None
    episode_count: int | None
    entity_mutation_count: int | None
    edge_mutation_count: int | None
    error: str | None
    started_at: datetime | None
    completed_at: datetime | None
    plan_json: dict[str, Any] | None = None


@dataclass(slots=True)
class ReconciliationWorkEventRow:
    id: str
    run_id: str
    event_id: str
    sequence: int
    event_kind: str
    title: str | None
    body: str | None
    payload: dict[str, Any]
    created_at: datetime


class ReconciliationLedgerPort(Protocol):
    def append_event(self, scope: EventScope, event: ContextEvent) -> tuple[str, bool]:
        """Insert canonical event. Returns ``(event_row_id, inserted)``; False if duplicate."""

    def get_event_by_id(self, event_id: str) -> Optional[ContextEventRow]: ...

    def claim_event_for_processing(self, event_id: str) -> bool:
        """Mark event as processing if still claimable; return False if already terminal."""

    def start_reconciliation_run(
        self,
        event_id: str,
        *,
        attempt_number: int,
        agent_name: str | None,
        agent_version: str | None,
        toolset_version: str | None,
    ) -> str:
        """Create a run row in ``running`` state; return run id."""

    def record_plan_metadata(
        self,
        run_id: str,
        *,
        plan_summary: str,
        episode_count: int,
        entity_mutation_count: int,
        edge_mutation_count: int,
    ) -> None: ...

    def record_run_plan_json(self, run_id: str, body: dict[str, Any]) -> None:
        """Persist full planner output JSON on the reconciliation run (audit / replay)."""

    def record_run_work_event(
        self,
        run_id: str,
        *,
        event_kind: str,
        title: str | None = None,
        body: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> str:
        """Append an ordered agent-working event for a reconciliation run."""

    def record_run_success(self, run_id: str) -> None: ...

    def record_run_failure(self, run_id: str, error: str) -> None: ...

    def record_event_reconciled(self, event_id: str) -> None: ...

    def record_events_reconciled(self, event_ids: list[str]) -> None:
        """Bulk ``record_event_reconciled``: ``→ reconciled`` for all ids in one
        statement.

        No-op on an empty list. All-or-nothing — a failure leaves none of the
        ids transitioned, so a bulk completion never half-applies. Prefer this
        over a per-event loop when closing a batch/chunk (the common path).
        """

    def record_event_failed(self, event_id: str, error: str) -> None: ...

    def record_events_failed(self, event_ids: list[str], error: str) -> None:
        """Bulk ``record_event_failed``: ``→ failed`` for all ids in one
        statement, sharing the batch/chunk-level ``error``.

        No-op on an empty list. All-or-nothing, same as
        ``record_events_reconciled``.
        """

    def fail_inflight_events(self, event_ids: list[str], error: str) -> int:
        """``→ failed`` for ids still in a non-terminal state, return the count.

        Unlike ``record_events_failed`` (an unconditional bulk set), this is
        status-guarded: it only flips events currently ``received`` /
        ``queued`` / ``processing``. Events a partially-completed batch
        already drove to ``reconciled`` are left untouched, so the reaper
        can fail a stuck batch's leftovers without clobbering work that
        actually finished. No-op on an empty list.
        """
        ...

    def list_runs_for_event(self, event_id: str) -> list[ReconciliationRunRow]: ...

    def list_work_events_for_run(
        self, run_id: str
    ) -> list[ReconciliationWorkEventRow]: ...

    def next_attempt_number(self, event_id: str) -> int:
        """Next ``attempt_number`` for a new reconciliation run (1-based)."""

    def mark_event_for_retry(self, event_id: str) -> None:
        """Set event status to ``received`` so ``claim_event_for_processing`` can run again."""

    def mark_events_for_retry(self, event_ids: list[str]) -> None:
        """Bulk ``mark_event_for_retry``: set every id to ``received`` in one statement.

        No-op on an empty list. All-or-nothing — a failure leaves none of
        the ids transitioned, so a bulk retry never half-applies.
        """

    def mark_events_queued(self, event_ids: list[str]) -> None:
        """Bulk ``mark_event_queued``: ``received`` → ``queued`` for all ids in one statement."""

    def delete_all_for_pot(self, pot_id: str) -> int:
        """Remove reconciliation rows for ``pot_id``; returns rows removed."""

    def mark_event_queued(self, event_id: str) -> None:
        """Transition ``received`` → ``queued`` (async ingestion)."""

    def set_event_job_metadata(
        self,
        event_id: str,
        *,
        job_id: str | None = None,
        correlation_id: str | None = None,
    ) -> None:
        """Attach broker / correlation ids after enqueue."""

    def summarize_pot_reconciliation(
        self,
        pot_id: str,
        *,
        recent_failure_limit: int = 5,
    ) -> ReconciliationLedgerHealth:
        """Per-pot reconciliation run health (for ``/status``)."""
        ...
