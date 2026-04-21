"""Default :class:`IngestionSubmissionService` — single enqueue/persist entry for producers."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from sqlalchemy.orm import Session

from adapters.outbound.postgres.ingestion_event_store import (
    SqlAlchemyIngestionEventStore,
)
from adapters.outbound.postgres.reconciliation_ledger import (
    SqlAlchemyReconciliationLedger,
)
from application.use_cases.event_reconciliation import (
    record_and_reconcile_context_event,
)
from application.use_cases.ingest_single_pr import run_merged_pr_ingest_core
from application.use_cases.record_context_event import record_context_event
from application.use_cases.wait_ingestion_event import wait_for_terminal_ingestion_event
from domain.context_events import (
    ContextEvent,
    EventScope,
    event_scope_from_resolved_pot,
)
from domain.ingestion_kinds import (
    INGESTION_KIND_AGENT_RECONCILIATION,
    INGESTION_KIND_GITHUB_MERGED_PR,
    INGESTION_KIND_RAW_EPISODE,
)
from domain.ingestion_event_models import EventReceipt, IngestionSubmissionRequest
from domain.ports.graph_mutation_applier import GraphMutationApplierPort
from domain.ports.pot_resolution import resolve_write_repo
from domain.ports.ingestion_submission import IngestionSubmissionService

if TYPE_CHECKING:
    from bootstrap.container import ContextEngineContainer


def _parse_iso_or_now(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if value is None:
        return datetime.now(timezone.utc)
    if isinstance(value, str):
        s = value.strip()
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return datetime.fromisoformat(s)
    raise ValueError("reference_time must be datetime or ISO-8601 string")


class DefaultIngestionSubmissionService(IngestionSubmissionService):
    """Persist via reconciliation ledger, enqueue via container job queue, optional terminal wait."""

    def __init__(
        self,
        container: "ContextEngineContainer",
        session: Session,
        *,
        mutation_applier: GraphMutationApplierPort | None = None,
    ) -> None:
        self._c = container
        self._session = session
        self._reco = SqlAlchemyReconciliationLedger(session)
        self._events = SqlAlchemyIngestionEventStore(session)
        self._mutation_applier = mutation_applier

    def submit(
        self,
        request: IngestionSubmissionRequest,
        *,
        sync: bool = False,
        wait: bool = False,
        timeout_seconds: float | None = None,
    ) -> EventReceipt:
        kind = request.ingestion_kind
        if kind == INGESTION_KIND_RAW_EPISODE:
            return self._submit_raw_episode(
                request, sync=sync, wait=wait, timeout_seconds=timeout_seconds
            )
        if kind == INGESTION_KIND_GITHUB_MERGED_PR:
            return self._submit_github_merged_pr(
                request, sync=sync, wait=wait, timeout_seconds=timeout_seconds
            )
        return self._submit_agent_reconciliation(
            request, sync=sync, wait=wait, timeout_seconds=timeout_seconds
        )

    def _submit_raw_episode(
        self,
        request: IngestionSubmissionRequest,
        *,
        sync: bool,
        wait: bool,
        timeout_seconds: float | None,
    ) -> EventReceipt:
        p = request.payload
        try:
            name = p["name"]
            episode_body = p["episode_body"]
            source_description = p["source_description"]
        except KeyError as e:
            raise ValueError(f"raw_episode payload missing field: {e.args[0]}") from e
        ref = _parse_iso_or_now(p.get("reference_time"))

        source_id = request.source_id or f"raw_episode_{uuid4()}"
        payload = dict(p)
        payload.update(
            {
                "name": name,
                "episode_body": episode_body,
                "source_description": source_description,
                "reference_time": ref.isoformat(),
            }
        )
        return self._submit_agent_reconciliation(
            IngestionSubmissionRequest(
                pot_id=request.pot_id,
                ingestion_kind=INGESTION_KIND_RAW_EPISODE,
                source_channel=request.source_channel,
                source_system=request.source_system or "context_engine_raw",
                event_type=request.event_type or "raw_episode",
                action=request.action or "ingest",
                payload=payload,
                metadata=request.metadata,
                idempotency_key=request.idempotency_key,
                dedup_key=request.dedup_key,
                event_id=request.event_id,
                source_id=source_id,
                provider=request.provider,
                provider_host=request.provider_host,
                repo_name=request.repo_name,
                source_event_id=request.source_event_id,
                artifact_refs=request.artifact_refs,
                occurred_at=request.occurred_at or ref,
            ),
            sync=sync,
            wait=wait,
            timeout_seconds=timeout_seconds,
        )

    def _submit_github_merged_pr(
        self,
        request: IngestionSubmissionRequest,
        *,
        sync: bool,
        wait: bool,
        timeout_seconds: float | None,
    ) -> EventReceipt:
        """Persist ``context_events`` row, then run merged-PR ingest (webhook / HTTP)."""
        p = request.payload
        try:
            pr_number = int(p["pr_number"])
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(
                "github_merged_pr payload requires integer pr_number"
            ) from e

        if not self._c.settings.is_enabled():
            raise ValueError("context_graph_disabled")

        resolved = self._c.pots.resolve_pot(request.pot_id)
        if resolved is None or not resolved.repos:
            raise ValueError("unknown_pot_id")
        explicit = (request.repo_name or p.get("repo_name") or "").strip() or None
        primary = resolve_write_repo(resolved, repo_name=explicit)
        if primary is None:
            raise ValueError(
                "ambiguous_or_unknown_repo" if not explicit else "repo_not_in_pot"
            )

        scope = EventScope(
            pot_id=request.pot_id,
            provider=primary.provider,
            provider_host=primary.provider_host,
            repo_name=primary.repo_name,
        )
        source_id = f"pr_{pr_number}_merged"
        eid = request.event_id or str(uuid4())
        event = ContextEvent(
            event_id=eid,
            source_system=request.source_system or "github",
            event_type=request.event_type or "pull_request",
            action=request.action or "merged",
            pot_id=request.pot_id,
            provider=primary.provider,
            provider_host=primary.provider_host,
            repo_name=primary.repo_name,
            source_id=source_id,
            source_event_id=request.source_event_id,
            occurred_at=request.occurred_at,
            received_at=None,
            payload=dict(p),
            ingestion_kind=INGESTION_KIND_GITHUB_MERGED_PR,
            source_channel=request.source_channel,
        )
        persisted_id, inserted = record_context_event(self._reco, scope, event)
        if not inserted:
            ev = self._events.get_event(persisted_id)
            return EventReceipt(
                event_id=persisted_id,
                status=ev.status if ev else "queued",
                terminal_event=ev,
                duplicate=True,
            )

        ledger = self._c.ledger(self._session)
        source = self._c.source_for_repo(primary.repo_name)
        out = run_merged_pr_ingest_core(
            self._c.settings,
            self._c.pots,
            source,
            ledger,
            self._c.episodic,
            self._c.structural,
            request.pot_id,
            pr_number,
            repo_name=explicit,
            is_live_bridge=bool(p.get("is_live_bridge", True)),
        )
        if out.get("status") == "success":
            self._reco.record_event_reconciled(eid)
            ev = self._events.get_event(eid)
            receipt = EventReceipt(
                event_id=eid,
                status=ev.status if ev else "done",
                terminal_event=ev,
                extras=out,
            )
        elif out.get("status") == "skipped":
            err = str(out.get("reason") or "skipped")
            self._reco.record_event_failed(eid, err)
            ev = self._events.get_event(eid)
            receipt = EventReceipt(
                event_id=eid,
                status="error",
                terminal_event=ev,
                error=err,
            )
        else:
            err = str(out.get("error") or out.get("reason") or "ingest_failed")
            self._reco.record_event_failed(eid, err)
            ev = self._events.get_event(eid)
            receipt = EventReceipt(
                event_id=eid,
                status="error",
                terminal_event=ev,
                error=err,
            )

        if wait:
            terminal = wait_for_terminal_ingestion_event(
                self._events,
                eid,
                timeout_seconds=timeout_seconds
                if timeout_seconds is not None
                else 300.0,
            )
            if terminal is not None:
                receipt = EventReceipt(
                    event_id=eid,
                    status=terminal.status,
                    terminal_event=terminal,
                    error=receipt.error,
                    duplicate=False,
                )
        return receipt

    def _submit_agent_reconciliation(
        self,
        request: IngestionSubmissionRequest,
        *,
        sync: bool,
        wait: bool,
        timeout_seconds: float | None,
    ) -> EventReceipt:
        if not request.source_id:
            raise ValueError(
                "source_id is required for agent reconciliation submissions"
            )
        if self._c.reconciliation_agent is None:
            raise ValueError("no_reconciliation_agent")

        resolved = self._c.pots.resolve_pot(request.pot_id)
        if resolved is None:
            raise ValueError("unknown_pot_id")
        explicit_repo = (request.repo_name or "").strip() or None
        primary = resolve_write_repo(resolved, repo_name=explicit_repo)
        if primary is None:
            if explicit_repo:
                raise ValueError("repo_not_in_pot")
            scope = event_scope_from_resolved_pot(request.pot_id, resolved)
            provider = request.provider or scope.provider
            provider_host = request.provider_host or scope.provider_host
            repo_name = scope.repo_name
        else:
            provider = request.provider or primary.provider
            provider_host = request.provider_host or primary.provider_host
            repo_name = request.repo_name or primary.repo_name

        eid = request.event_id or str(uuid4())
        kind = request.ingestion_kind or INGESTION_KIND_AGENT_RECONCILIATION
        event = ContextEvent(
            event_id=eid,
            source_system=request.source_system,
            event_type=request.event_type,
            action=request.action,
            pot_id=request.pot_id,
            provider=provider,
            provider_host=provider_host,
            repo_name=repo_name,
            source_id=request.source_id,
            source_event_id=request.source_event_id,
            artifact_refs=list(request.artifact_refs),
            occurred_at=request.occurred_at,
            received_at=None,
            payload=dict(request.payload),
            ingestion_kind=kind,
            idempotency_key=request.idempotency_key,
            source_channel=request.source_channel,
        )
        scope = EventScope(
            pot_id=request.pot_id,
            provider=provider,
            provider_host=provider_host,
            repo_name=repo_name,
        )

        out = record_and_reconcile_context_event(
            self._c.episodic,
            self._c.structural,
            self._c.reconciliation_agent,
            self._reco,
            scope,
            event,
            sync=sync,
            jobs=self._c.jobs,
            mutation_applier=self._mutation_applier,
        )

        if not out.inserted:
            ev = self._events.get_event(out.event_id)
            return EventReceipt(
                event_id=out.event_id,
                status=ev.status if ev else "queued",
                terminal_event=ev,
                duplicate=True,
            )

        if not sync:
            receipt = EventReceipt(
                event_id=out.event_id,
                status="queued",
                job_id=out.job_id,
            )
            if wait:
                terminal = wait_for_terminal_ingestion_event(
                    self._events,
                    out.event_id,
                    timeout_seconds=timeout_seconds
                    if timeout_seconds is not None
                    else 300.0,
                )
                if terminal is None:
                    return receipt
                return EventReceipt(
                    event_id=out.event_id,
                    status=terminal.status,
                    terminal_event=terminal,
                    job_id=out.job_id,
                )
            return receipt

        r = out.reconciliation
        ev = self._events.get_event(out.event_id)
        if r is None:
            return EventReceipt(
                event_id=out.event_id,
                status="error",
                error="reconciliation_missing",
                terminal_event=ev,
            )
        st = "error" if not r.ok else "done"
        return EventReceipt(
            event_id=out.event_id,
            status=st,
            error=r.error,
            terminal_event=ev,
            reconciliation=r,
        )
