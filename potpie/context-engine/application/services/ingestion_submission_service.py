"""Default :class:`IngestionSubmissionService` — single inbound for events.

Most events flow through the debounced batch queue and reconciliation agent.
Durable ``context_record`` submissions are the deliberate exception: they route
directly through ``GraphService.record`` so structured agent memories use the
same deterministic semantic-mutation path as local CLI/MCP writes and do not
require a reconciliation agent. The legacy ``connector_sync`` shortcut and its
source-specific dispatch are gone.
"""

from __future__ import annotations

import logging
from uuid import uuid4

from application.services.event_admission import admit_event
from application.services.ingestion_wait import wait_for_terminal_ingestion_event
from bootstrap import sentry_metrics_runtime
from bootstrap.observability_context import bind_correlation, correlation_scope
from bootstrap.observability_runtime import get_observability
from domain.ports.observability import SPAN_KIND_SERVER
from domain.context_events import (
    ContextEvent,
    EventScope,
    event_scope_from_resolved_pot,
)
from domain.ingestion_kinds import INGESTION_KIND_AGENT_RECONCILIATION
from domain.ingestion_event_models import EventReceipt, IngestionSubmissionRequest
from domain.ports.agent_context import RecordRequest
from domain.ports.batch_repository import BatchRepositoryPort
from domain.ports.context_graph_job_queue import ContextGraphJobQueuePort
from domain.ports.ingestion_config import IngestionConfigPort
from domain.ports.ingestion_event_store import IngestionEventStore
from domain.ports.ingestion_submission import IngestionSubmissionService
from domain.ports.pot_resolution import PotResolutionPort, resolve_write_repo
from domain.ports.reconciliation_agent import ReconciliationAgentPort
from domain.ports.reconciliation_ledger import ReconciliationLedgerPort
from domain.ports.services.graph_service import GraphService
from domain.ports.settings import ContextEngineSettingsPort

logger = logging.getLogger(__name__)


class DefaultIngestionSubmissionService(IngestionSubmissionService):
    """Persist context events into the debounced batch queue.

    The service is constructed with ports only. Hosts (or
    :func:`bootstrap.ingestion_server.IngestionServerContainer.ingestion_submission`)
    are responsible for binding sessions to the concrete adapters and
    handing them in.
    """

    def __init__(
        self,
        *,
        settings: ContextEngineSettingsPort,
        pots: PotResolutionPort,
        graph: GraphService | None = None,
        reconciliation_agent: ReconciliationAgentPort | None,
        reco_ledger: ReconciliationLedgerPort,
        events: IngestionEventStore,
        batches: BatchRepositoryPort,
        jobs: ContextGraphJobQueuePort,
        ingestion_config: IngestionConfigPort | None = None,
    ) -> None:
        self._settings = settings
        self._pots = pots
        self._graph = graph
        self._reconciliation_agent = reconciliation_agent
        self._reco = reco_ledger
        self._events = events
        self._batches = batches
        self._jobs = jobs
        self._ingestion_config = ingestion_config

    def submit(
        self,
        request: IngestionSubmissionRequest,
        *,
        sync: bool = False,
        wait: bool = False,
        timeout_seconds: float | None = None,
    ) -> EventReceipt:
        """Admit an event into the debounced batch queue.

        ``sync=True`` and ``wait=True`` both block until the event reaches a
        terminal state (``done`` / ``error``); they're aliases for ergonomic
        use by callers that want a synchronous response.

        Thin observability wrapper: opens the ``ingest.submit`` span and
        binds the pot to the correlation context so the (sync) ingress
        trace and every log line on this path carry the pot. The event id
        and the persisted trace correlation are bound inside ``_do_submit``
        once the id is known.
        """
        obs = get_observability()
        with correlation_scope(pot_id=request.pot_id):
            with obs.span(
                "ingest.submit",
                kind=SPAN_KIND_SERVER,
                attributes={
                    "pot_id": request.pot_id,
                    "ingest.source_system": request.source_system,
                    "ingest.event_type": request.event_type,
                },
            ) as span:
                receipt = self._do_submit(
                    request,
                    sync=sync,
                    wait=wait,
                    timeout_seconds=timeout_seconds,
                )
                span.set_attribute("ingest.event_id", receipt.event_id)
                span.set_attribute("ingest.duplicate", bool(receipt.duplicate))
                span.set_attribute("ingest.status", receipt.status)
                return receipt

    def _do_submit(
        self,
        request: IngestionSubmissionRequest,
        *,
        sync: bool = False,
        wait: bool = False,
        timeout_seconds: float | None = None,
    ) -> EventReceipt:
        if not self._settings.is_enabled():
            raise ValueError("context_graph_disabled")

        resolved = self._pots.resolve_pot(request.pot_id)
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

        if not request.source_id:
            raise ValueError("source_id is required for ingestion submissions")

        if _is_context_record_submission(request):
            return self._submit_context_record(request)

        if self._reconciliation_agent is None:
            raise ValueError("no_reconciliation_agent")

        eid = request.event_id or str(uuid4())
        # Bind the event id for the rest of this (sync) ingress path; the
        # enclosing correlation_scope in submit() resets it on exit.
        bind_correlation(event_id=eid, source=request.source_system)
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
            actor=request.actor,
        )
        scope = EventScope(
            pot_id=request.pot_id,
            provider=provider,
            provider_host=provider_host,
            repo_name=repo_name,
        )

        outcome = admit_event(
            self._reco,
            self._batches,
            self._jobs,
            scope,
            event,
            ingestion_config=self._ingestion_config,
        )

        obs = get_observability()
        metric_attrs = {"source": request.source_system}
        if outcome.inserted:
            # Resurrect the long-dead correlation_id column: persist the
            # ingress trace so the async batch run can span-link back to it
            # across the windowed delay.
            try:
                tp = obs.current_traceparent()
                if tp:
                    self._reco.set_event_job_metadata(
                        outcome.event_id, correlation_id=tp
                    )
            except Exception:  # noqa: BLE001 — observability never fails ingest
                logger.debug(
                    "observability: persist trace correlation failed",
                    exc_info=True,
                )
            obs.counter("ce.ingest.events_total", 1, attributes=metric_attrs)
            sentry_metrics_runtime.count(
                "ce.ingest.events_total",
                attributes={"result": "inserted"},
            )
            if outcome.batch_id:
                bind_correlation(batch_id=outcome.batch_id)
        else:
            obs.counter("ce.ingest.dedup_total", 1, attributes=metric_attrs)
            sentry_metrics_runtime.count(
                "ce.ingest.dedup_total",
                attributes={"result": "duplicate"},
            )

        if not outcome.inserted:
            ev = self._events.get_event(outcome.event_id)
            return EventReceipt(
                event_id=outcome.event_id,
                status=ev.status if ev else "queued",
                terminal_event=ev,
                duplicate=True,
            )

        receipt = EventReceipt(
            event_id=outcome.event_id,
            status="queued",
            job_id=outcome.batch_id,
        )
        if sync or wait:
            terminal = wait_for_terminal_ingestion_event(
                self._events,
                outcome.event_id,
                timeout_seconds=timeout_seconds
                if timeout_seconds is not None
                else 300.0,
            )
            if terminal is not None:
                return EventReceipt(
                    event_id=outcome.event_id,
                    status=terminal.status,
                    terminal_event=terminal,
                    job_id=outcome.batch_id,
                )
        return receipt

    def _submit_context_record(
        self, request: IngestionSubmissionRequest
    ) -> EventReceipt:
        if self._graph is None:
            raise ValueError("context_graph_disabled")

        payload = dict(request.payload or {})
        record = _mapping_payload(payload.get("record"), "record")
        scope = _mapping_payload(payload.get("scope"), "scope")
        record_type = _required_str(record.get("type") or request.action, "record.type")
        summary = _required_str(record.get("summary"), "record.summary")
        details = _mapping_payload(record.get("details") or {}, "record.details")
        source_refs = _string_tuple(record.get("source_refs") or request.artifact_refs)
        actor = request.actor
        metadata = {
            "surface": request.source_channel,
            "source_system": request.source_system,
            "source_id": request.source_id,
            **dict(request.metadata),
        }
        if actor is not None:
            metadata.update(
                {
                    "user": actor.user_id,
                    "surface": actor.surface,
                    "harness": actor.client_name,
                }
            )

        receipt = self._graph.record(
            RecordRequest(
                pot_id=request.pot_id,
                record_type=record_type,
                summary=summary,
                details=details,
                scope=scope,
                source_refs=source_refs,
                idempotency_key=request.idempotency_key or request.source_id,
                metadata=metadata,
            )
        )
        event_id = request.event_id or str(uuid4())
        return EventReceipt(
            event_id=event_id,
            status="done" if receipt.accepted else "error",
            error=None if receipt.accepted else receipt.detail or receipt.status,
            mutation_id=_optional_str(receipt.metadata.get("mutation_id")),
            extras={
                "record_id": receipt.record_id,
                "record_status": receipt.status,
                "mutations_applied": receipt.mutations_applied,
                "record_metadata": dict(receipt.metadata),
            },
        )


def _is_context_record_submission(request: IngestionSubmissionRequest) -> bool:
    return request.event_type == "context_record"


def _mapping_payload(value: object, name: str) -> dict[str, object]:
    if isinstance(value, dict):
        return dict(value)
    raise ValueError(f"{name} must be an object")


def _required_str(value: object, name: str) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    raise ValueError(f"{name} is required")


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _string_tuple(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,) if value else ()
    if not isinstance(value, (list, tuple)):
        raise ValueError("record.source_refs must be a list of strings")
    out: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError("record.source_refs must be a list of strings")
        if item:
            out.append(item)
    return tuple(out)
