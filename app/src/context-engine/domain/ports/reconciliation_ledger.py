"""Reconciliation lifecycle ledger (port)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional, Protocol

from domain.context_events import ContextEvent, EventScope


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


@dataclass(slots=True)
class EpisodeStepRow:
    id: str
    pot_id: str
    event_id: str
    sequence: int
    step_kind: str
    step_json: dict[str, Any]
    status: str
    attempt_count: int
    applied_at: datetime | None
    error: str | None
    run_id: str | None


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


class ReconciliationLedgerPort(Protocol):
    def append_event(self, scope: EventScope, event: ContextEvent) -> tuple[str, bool]:
        """Insert canonical event. Returns ``(event_row_id, inserted)``; False if duplicate."""

    def get_event_by_id(self, event_id: str) -> Optional[ContextEventRow]:
        ...

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
    ) -> None:
        ...

    def record_run_plan_json(self, run_id: str, body: dict[str, Any]) -> None:
        """Persist full planner output JSON on the reconciliation run (audit / replay)."""

    def record_run_success(self, run_id: str) -> None:
        ...

    def record_run_failure(self, run_id: str, error: str) -> None:
        ...

    def record_event_reconciled(self, event_id: str) -> None:
        ...

    def record_event_failed(self, event_id: str, error: str) -> None:
        ...

    def list_runs_for_event(self, event_id: str) -> list[ReconciliationRunRow]:
        ...

    def next_attempt_number(self, event_id: str) -> int:
        """Next ``attempt_number`` for a new reconciliation run (1-based)."""

    def mark_event_for_retry(self, event_id: str) -> None:
        """Set event status to ``received`` so ``claim_event_for_processing`` can run again."""

    def delete_all_for_pot(self, pot_id: str) -> int:
        """Remove reconciliation rows for ``pot_id``; returns rows removed."""

    def mark_event_queued(self, event_id: str) -> None:
        """Transition ``received`` → ``queued`` (async ingestion)."""

    def mark_event_episodes_queued(self, event_id: str) -> None:
        """Planner finished persisting steps; apply workers will run."""

    def set_event_job_metadata(
        self,
        event_id: str,
        *,
        job_id: str | None = None,
        correlation_id: str | None = None,
    ) -> None:
        """Attach broker / correlation ids after enqueue."""

    def replace_episode_steps_for_event(
        self,
        pot_id: str,
        event_id: str,
        run_id: str | None,
        steps: list[tuple[int, str, dict[str, Any]]],
    ) -> None:
        """Replace all steps for an event: ``(sequence, step_kind, step_json)``."""

    def list_episode_steps(self, event_id: str) -> list[EpisodeStepRow]:
        ...

    def get_episode_step(self, event_id: str, sequence: int) -> EpisodeStepRow | None:
        ...

    def max_applied_sequence(self, event_id: str) -> int | None:
        """Highest ``sequence`` with status ``applied``, or ``None`` if none."""

    def update_episode_step_status(
        self,
        event_id: str,
        sequence: int,
        *,
        status: str,
        error: str | None = None,
        increment_attempt: bool = False,
    ) -> None:
        ...
