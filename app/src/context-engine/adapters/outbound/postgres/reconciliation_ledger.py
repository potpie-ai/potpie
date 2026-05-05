"""Postgres implementation of ``ReconciliationLedgerPort``."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from sqlalchemy import delete, func, or_, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from adapters.outbound.postgres.models import (
    ContextEpisodeStepModel,
    ContextEventModel,
    ContextReconciliationRun,
    ContextReconciliationWorkEvent,
)
from domain.context_events import ContextEvent, EventScope
from domain.context_status import ReconciliationLedgerHealth
from domain.ingestion_kinds import (
    EPISODE_STEP_APPLIED,
    EPISODE_STEP_APPLYING as EPISODE_STEP_APPLYING_VALUE,
    EPISODE_STEP_FAILED as EPISODE_STEP_FAILED_VALUE,
    EVENT_STATUS_EPISODES_QUEUED,
    INGESTION_KIND_RAW_EPISODE,
)
from domain.ports.reconciliation_ledger import (
    ContextEventRow,
    EpisodeStepRow,
    ReconciliationLedgerPort,
    ReconciliationRunRow,
    ReconciliationWorkEventRow,
)

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _ensure_aware(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


class SqlAlchemyReconciliationLedger(ReconciliationLedgerPort):
    def __init__(self, session: Session) -> None:
        self._db = session

    def append_event(self, scope: EventScope, event: ContextEvent) -> tuple[str, bool]:
        if (
            event.pot_id != scope.pot_id
            or event.provider != scope.provider
            or event.provider_host != scope.provider_host
            or event.repo_name != scope.repo_name
        ):
            raise ValueError("EventScope does not match ContextEvent partition fields")

        kind = event.ingestion_kind or "agent_reconciliation"
        if event.idempotency_key and kind == INGESTION_KIND_RAW_EPISODE:
            existing_id = self._db.scalar(
                select(ContextEventModel.id).where(
                    ContextEventModel.pot_id == event.pot_id,
                    ContextEventModel.ingestion_kind == INGESTION_KIND_RAW_EPISODE,
                    ContextEventModel.idempotency_key == event.idempotency_key,
                )
            )
            if existing_id:
                return str(existing_id), False

        row = ContextEventModel(
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
            payload=event.payload,
            occurred_at=event.occurred_at,
            received_at=event.received_at or _utcnow(),
            status="received",
            ingestion_kind=kind,
            idempotency_key=event.idempotency_key,
            source_channel=(event.source_channel or "unknown").strip() or "unknown",
            actor_user_id=event.actor.user_id if event.actor else None,
            actor_surface=event.actor.surface if event.actor else None,
            actor_client_name=event.actor.client_name if event.actor else None,
            actor_auth_method=event.actor.auth_method if event.actor else None,
        )
        self._db.add(row)
        try:
            self._db.commit()
            return event.event_id, True
        except IntegrityError:
            self._db.rollback()
            logger.info(
                "reconciliation_duplicate_event pot=%s repo=%s source_system=%s source_id=%s",
                event.pot_id,
                event.repo_name,
                event.source_system,
                event.source_id,
            )
            existing = self._db.scalar(
                select(ContextEventModel.id).where(
                    ContextEventModel.pot_id == event.pot_id,
                    ContextEventModel.provider == event.provider,
                    ContextEventModel.provider_host == event.provider_host,
                    ContextEventModel.repo_name == event.repo_name,
                    ContextEventModel.source_system == event.source_system,
                    ContextEventModel.source_id == event.source_id,
                )
            )
            return (str(existing) if existing else event.event_id, False)

    def get_event_by_id(self, event_id: str) -> ContextEventRow | None:
        row = self._db.scalar(
            select(ContextEventModel).where(ContextEventModel.id == event_id)
        )
        if row is None:
            return None
        return self._to_event_row(row)

    def claim_event_for_processing(self, event_id: str) -> bool:
        res = self._db.execute(
            update(ContextEventModel)
            .where(
                ContextEventModel.id == event_id,
                or_(
                    ContextEventModel.status == "received",
                    ContextEventModel.status == "queued",
                ),
            )
            .values(status="processing")
        )
        self._db.commit()
        return (res.rowcount or 0) > 0

    def start_reconciliation_run(
        self,
        event_id: str,
        *,
        attempt_number: int,
        agent_name: str | None,
        agent_version: str | None,
        toolset_version: str | None,
    ) -> str:
        run_id = str(uuid4())
        run = ContextReconciliationRun(
            id=run_id,
            event_id=event_id,
            attempt_number=attempt_number,
            status="running",
            agent_name=agent_name,
            agent_version=agent_version,
            toolset_version=toolset_version,
            started_at=_utcnow(),
        )
        self._db.add(run)
        self._db.commit()
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
        self._db.execute(
            update(ContextReconciliationRun)
            .where(ContextReconciliationRun.id == run_id)
            .values(
                plan_summary=plan_summary[:8000],
                episode_count=episode_count,
                entity_mutation_count=entity_mutation_count,
                edge_mutation_count=edge_mutation_count,
            )
        )
        self._db.commit()

    def record_run_plan_json(self, run_id: str, body: dict[str, Any]) -> None:
        self._db.execute(
            update(ContextReconciliationRun)
            .where(ContextReconciliationRun.id == run_id)
            .values(plan_json=dict(body))
        )
        self._db.commit()

    def record_run_work_event(
        self,
        run_id: str,
        *,
        event_kind: str,
        title: str | None = None,
        body: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> str:
        event_id = self._db.scalar(
            select(ContextReconciliationRun.event_id).where(
                ContextReconciliationRun.id == run_id
            )
        )
        if event_id is None:
            raise ValueError(f"unknown reconciliation run: {run_id}")
        current = self._db.scalar(
            select(func.max(ContextReconciliationWorkEvent.sequence)).where(
                ContextReconciliationWorkEvent.run_id == run_id
            )
        )
        row_id = str(uuid4())
        row = ContextReconciliationWorkEvent(
            id=row_id,
            run_id=run_id,
            event_id=str(event_id),
            sequence=int(current or 0) + 1,
            event_kind=event_kind[:64],
            title=title[:255] if title else None,
            body=body[:20000] if body else None,
            payload=dict(payload or {}),
            created_at=_utcnow(),
        )
        self._db.add(row)
        self._db.commit()
        return row_id

    def record_run_success(self, run_id: str) -> None:
        self._db.execute(
            update(ContextReconciliationRun)
            .where(ContextReconciliationRun.id == run_id)
            .values(status="succeeded", completed_at=_utcnow(), error=None)
        )
        self._db.commit()

    def record_run_failure(self, run_id: str, error: str) -> None:
        self._db.execute(
            update(ContextReconciliationRun)
            .where(ContextReconciliationRun.id == run_id)
            .values(status="failed", completed_at=_utcnow(), error=error[:8000])
        )
        self._db.commit()

    def record_event_reconciled(self, event_id: str) -> None:
        self._db.execute(
            update(ContextEventModel)
            .where(ContextEventModel.id == event_id)
            .values(status="reconciled")
        )
        self._db.commit()

    def record_event_failed(self, event_id: str, error: str) -> None:
        logger.warning(
            "context event %s reconciliation failed: %s", event_id, error[:2000]
        )
        self._db.execute(
            update(ContextEventModel)
            .where(ContextEventModel.id == event_id)
            .values(status="failed")
        )
        self._db.commit()

    def list_runs_for_event(self, event_id: str) -> list[ReconciliationRunRow]:
        rows = self._db.scalars(
            select(ContextReconciliationRun)
            .where(ContextReconciliationRun.event_id == event_id)
            .order_by(ContextReconciliationRun.attempt_number.asc())
        ).all()
        return [self._to_run_row(r) for r in rows]

    def list_work_events_for_run(self, run_id: str) -> list[ReconciliationWorkEventRow]:
        rows = self._db.scalars(
            select(ContextReconciliationWorkEvent)
            .where(ContextReconciliationWorkEvent.run_id == run_id)
            .order_by(ContextReconciliationWorkEvent.sequence.asc())
        ).all()
        return [self._to_work_event_row(r) for r in rows]

    def delete_all_for_pot(self, pot_id: str) -> int:
        """Delete reconciliation rows for ``pot_id`` (runs → events; episode steps cascade)."""
        event_ids = self._db.scalars(
            select(ContextEventModel.id).where(ContextEventModel.pot_id == pot_id)
        ).all()
        if not event_ids:
            return 0
        n_run = (
            self._db.execute(
                delete(ContextReconciliationRun).where(
                    ContextReconciliationRun.event_id.in_(event_ids)
                )
            ).rowcount
            or 0
        )
        n_ev = (
            self._db.execute(
                delete(ContextEventModel).where(ContextEventModel.pot_id == pot_id)
            ).rowcount
            or 0
        )
        self._db.commit()
        return int(n_run + n_ev)

    def next_attempt_number(self, event_id: str) -> int:
        """Return the next ``attempt_number`` for ``event_id`` (1-based)."""
        n = self._db.scalar(
            select(func.max(ContextReconciliationRun.attempt_number)).where(
                ContextReconciliationRun.event_id == event_id
            )
        )
        return int(n or 0) + 1

    def mark_event_for_retry(self, event_id: str) -> None:
        self._db.execute(
            update(ContextEventModel)
            .where(ContextEventModel.id == event_id)
            .values(status="received")
        )
        self._db.commit()

    def mark_event_queued(self, event_id: str) -> None:
        self._db.execute(
            update(ContextEventModel)
            .where(
                ContextEventModel.id == event_id, ContextEventModel.status == "received"
            )
            .values(status="queued")
        )
        self._db.commit()

    def mark_event_episodes_queued(self, event_id: str) -> None:
        self._db.execute(
            update(ContextEventModel)
            .where(ContextEventModel.id == event_id)
            .values(status=EVENT_STATUS_EPISODES_QUEUED)
        )
        self._db.commit()

    def set_event_job_metadata(
        self,
        event_id: str,
        *,
        job_id: str | None = None,
        correlation_id: str | None = None,
    ) -> None:
        vals: dict[str, Any] = {}
        if job_id is not None:
            vals["job_id"] = job_id
        if correlation_id is not None:
            vals["correlation_id"] = correlation_id
        if vals:
            self._db.execute(
                update(ContextEventModel)
                .where(ContextEventModel.id == event_id)
                .values(**vals)
            )
            self._db.commit()

    def replace_episode_steps_for_event(
        self,
        pot_id: str,
        event_id: str,
        run_id: str | None,
        steps: list[tuple[int, str, dict[str, Any]]],
    ) -> None:
        self._db.execute(
            delete(ContextEpisodeStepModel).where(
                ContextEpisodeStepModel.event_id == event_id
            )
        )
        for seq, step_kind, step_json in steps:
            self._db.add(
                ContextEpisodeStepModel(
                    id=str(uuid4()),
                    pot_id=pot_id,
                    event_id=event_id,
                    sequence=seq,
                    step_kind=step_kind,
                    step_json=step_json,
                    status="pending",
                    run_id=run_id,
                )
            )
        self._db.commit()

    def list_episode_steps(self, event_id: str) -> list[EpisodeStepRow]:
        rows = self._db.scalars(
            select(ContextEpisodeStepModel)
            .where(ContextEpisodeStepModel.event_id == event_id)
            .order_by(ContextEpisodeStepModel.sequence.asc())
        ).all()
        return [self._to_episode_row(r) for r in rows]

    def get_episode_step(self, event_id: str, sequence: int) -> EpisodeStepRow | None:
        row = self._db.scalar(
            select(ContextEpisodeStepModel).where(
                ContextEpisodeStepModel.event_id == event_id,
                ContextEpisodeStepModel.sequence == sequence,
            )
        )
        return self._to_episode_row(row) if row else None

    def max_applied_sequence(self, event_id: str) -> int | None:
        m = self._db.scalar(
            select(func.max(ContextEpisodeStepModel.sequence)).where(
                ContextEpisodeStepModel.event_id == event_id,
                ContextEpisodeStepModel.status == EPISODE_STEP_APPLIED,
            )
        )
        return int(m) if m is not None else None

    def update_episode_step_status(
        self,
        event_id: str,
        sequence: int,
        *,
        status: str,
        error: str | None = None,
        increment_attempt: bool = False,
    ) -> None:
        vals: dict[str, Any] = {"status": status}
        if error is not None:
            vals["error"] = error[:8000] if error else None
        if status == EPISODE_STEP_APPLIED:
            vals["applied_at"] = _utcnow()
        if increment_attempt:
            vals["attempt_count"] = ContextEpisodeStepModel.attempt_count + 1
        self._db.execute(
            update(ContextEpisodeStepModel)
            .where(
                ContextEpisodeStepModel.event_id == event_id,
                ContextEpisodeStepModel.sequence == sequence,
            )
            .values(**vals)
        )
        self._db.commit()

    def summarize_pot_reconciliation(
        self,
        pot_id: str,
        *,
        recent_failure_limit: int = 5,
        stuck_step_limit: int = 5,
    ) -> ReconciliationLedgerHealth:
        run_count_rows = self._db.execute(
            select(ContextReconciliationRun.status, func.count())
            .join(
                ContextEventModel,
                ContextEventModel.id == ContextReconciliationRun.event_id,
            )
            .where(ContextEventModel.pot_id == pot_id)
            .group_by(ContextReconciliationRun.status)
        ).all()
        run_counts: dict[str, int] = {}
        for status_value, n in run_count_rows:
            run_counts[str(status_value)] = int(n or 0)

        step_count_rows = self._db.execute(
            select(ContextEpisodeStepModel.status, func.count())
            .where(ContextEpisodeStepModel.pot_id == pot_id)
            .group_by(ContextEpisodeStepModel.status)
        ).all()
        step_counts: dict[str, int] = {}
        for status_value, n in step_count_rows:
            step_counts[str(status_value)] = int(n or 0)

        last_success = self._db.scalar(
            select(func.max(ContextReconciliationRun.completed_at))
            .join(
                ContextEventModel,
                ContextEventModel.id == ContextReconciliationRun.event_id,
            )
            .where(
                ContextEventModel.pot_id == pot_id,
                ContextReconciliationRun.status == "succeeded",
            )
        )
        last_failure = self._db.scalar(
            select(func.max(ContextReconciliationRun.completed_at))
            .join(
                ContextEventModel,
                ContextEventModel.id == ContextReconciliationRun.event_id,
            )
            .where(
                ContextEventModel.pot_id == pot_id,
                ContextReconciliationRun.status == "failed",
            )
        )

        failed_rows = self._db.execute(
            select(
                ContextReconciliationRun.id,
                ContextReconciliationRun.event_id,
                ContextReconciliationRun.attempt_number,
                ContextReconciliationRun.agent_name,
                ContextReconciliationRun.error,
                ContextReconciliationRun.completed_at,
                ContextReconciliationRun.started_at,
            )
            .join(
                ContextEventModel,
                ContextEventModel.id == ContextReconciliationRun.event_id,
            )
            .where(
                ContextEventModel.pot_id == pot_id,
                ContextReconciliationRun.status == "failed",
            )
            .order_by(
                func.coalesce(
                    ContextReconciliationRun.completed_at,
                    ContextReconciliationRun.started_at,
                ).desc()
            )
            .limit(max(recent_failure_limit, 0))
        ).all()
        recent_failed_runs = []
        for r in failed_rows:
            ts = r.completed_at or r.started_at
            recent_failed_runs.append(
                {
                    "run_id": r.id,
                    "event_id": r.event_id,
                    "attempt_number": r.attempt_number,
                    "agent_name": r.agent_name,
                    "error": r.error,
                    "at": _ensure_aware(ts).isoformat() if isinstance(ts, datetime) else None,
                }
            )

        stuck_rows = self._db.execute(
            select(
                ContextEpisodeStepModel.id,
                ContextEpisodeStepModel.event_id,
                ContextEpisodeStepModel.sequence,
                ContextEpisodeStepModel.step_kind,
                ContextEpisodeStepModel.status,
                ContextEpisodeStepModel.attempt_count,
                ContextEpisodeStepModel.error,
                ContextEpisodeStepModel.queued_at,
                ContextEpisodeStepModel.started_at,
            )
            .where(
                ContextEpisodeStepModel.pot_id == pot_id,
                or_(
                    ContextEpisodeStepModel.status == EPISODE_STEP_FAILED_VALUE,
                    ContextEpisodeStepModel.status == EPISODE_STEP_APPLYING_VALUE,
                ),
            )
            .order_by(
                func.coalesce(
                    ContextEpisodeStepModel.started_at,
                    ContextEpisodeStepModel.queued_at,
                ).desc()
            )
            .limit(max(stuck_step_limit, 0))
        ).all()
        stuck_step_samples = []
        for r in stuck_rows:
            ts = r.started_at or r.queued_at
            stuck_step_samples.append(
                {
                    "step_id": r.id,
                    "event_id": r.event_id,
                    "sequence": r.sequence,
                    "step_kind": r.step_kind,
                    "status": r.status,
                    "attempt_count": r.attempt_count,
                    "error": r.error,
                    "at": _ensure_aware(ts).isoformat() if isinstance(ts, datetime) else None,
                }
            )

        return ReconciliationLedgerHealth(
            run_counts=run_counts,
            step_counts=step_counts,
            last_run_success_at=_ensure_aware(last_success) if isinstance(last_success, datetime) else None,
            last_run_failure_at=_ensure_aware(last_failure) if isinstance(last_failure, datetime) else None,
            recent_failed_runs=recent_failed_runs,
            stuck_step_samples=stuck_step_samples,
        )

    @staticmethod
    def _to_episode_row(row: ContextEpisodeStepModel) -> EpisodeStepRow:
        return EpisodeStepRow(
            id=row.id,
            pot_id=row.pot_id,
            event_id=row.event_id,
            sequence=row.sequence,
            step_kind=row.step_kind,
            step_json=row.step_json,
            status=row.status,
            attempt_count=row.attempt_count,
            applied_at=row.applied_at,
            error=row.error,
            run_id=row.run_id,
        )

    @staticmethod
    def _to_event_row(row: ContextEventModel) -> ContextEventRow:
        from adapters.outbound.postgres.ingestion_event_store import _row_to_actor

        return ContextEventRow(
            id=row.id,
            pot_id=row.pot_id,
            provider=row.provider,
            provider_host=row.provider_host,
            repo_name=row.repo_name,
            source_system=row.source_system,
            event_type=row.event_type,
            action=row.action,
            source_id=row.source_id,
            source_event_id=row.source_event_id,
            payload=row.payload,
            occurred_at=row.occurred_at,
            received_at=row.received_at,
            status=row.status,
            ingestion_kind=getattr(row, "ingestion_kind", None)
            or "agent_reconciliation",
            job_id=getattr(row, "job_id", None),
            correlation_id=getattr(row, "correlation_id", None),
            idempotency_key=getattr(row, "idempotency_key", None),
            source_channel=getattr(row, "source_channel", None),
            actor=_row_to_actor(row),
        )

    @staticmethod
    def _to_run_row(row: ContextReconciliationRun) -> ReconciliationRunRow:
        pj = row.plan_json
        if pj is not None and not isinstance(pj, dict):
            pj = None
        return ReconciliationRunRow(
            id=row.id,
            event_id=row.event_id,
            attempt_number=row.attempt_number,
            status=row.status,
            agent_name=row.agent_name,
            agent_version=row.agent_version,
            toolset_version=row.toolset_version,
            plan_summary=row.plan_summary,
            episode_count=row.episode_count,
            entity_mutation_count=row.entity_mutation_count,
            edge_mutation_count=row.edge_mutation_count,
            error=row.error,
            started_at=row.started_at,
            completed_at=row.completed_at,
            plan_json=pj,
        )

    @staticmethod
    def _to_work_event_row(
        row: ContextReconciliationWorkEvent,
    ) -> ReconciliationWorkEventRow:
        payload = row.payload
        if not isinstance(payload, dict):
            payload = {}
        return ReconciliationWorkEventRow(
            id=row.id,
            run_id=row.run_id,
            event_id=row.event_id,
            sequence=row.sequence,
            event_kind=row.event_kind,
            title=row.title,
            body=row.body,
            payload=payload,
            created_at=row.created_at,
        )
