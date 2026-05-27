"""Default :class:`IngestionSubmissionService` — single inbound for events.

After Phase 4 there is exactly one ingestion path: every event flows
through the debounced batch queue, gets debounced, and is processed by
the reconciliation agent. The legacy ``connector_sync`` shortcut and
its source-specific dispatch are gone — connectors contribute proposed
plans during agent batch processing, not during inbound submission.
"""

from __future__ import annotations

import logging

from observability import get_logger
from uuid import uuid4

from application.services.event_admission import admit_event
from application.services.ingestion_wait import wait_for_terminal_ingestion_event
from domain.context_events import (
    ContextEvent,
    EventScope,
    event_scope_from_resolved_pot,
)
from domain.ingestion_kinds import INGESTION_KIND_AGENT_RECONCILIATION
from domain.ingestion_event_models import EventReceipt, IngestionSubmissionRequest
from domain.ports.batch_repository import BatchRepositoryPort
from domain.ports.context_graph_job_queue import ContextGraphJobQueuePort
from domain.ports.ingestion_config import IngestionConfigPort
from domain.ports.ingestion_event_store import IngestionEventStore
from domain.ports.ingestion_submission import IngestionSubmissionService
from domain.ports.pot_resolution import PotResolutionPort, resolve_write_repo
from domain.ports.reconciliation_agent import ReconciliationAgentPort
from domain.ports.reconciliation_ledger import ReconciliationLedgerPort
from domain.ports.settings import ContextEngineSettingsPort

logger = get_logger(__name__)


class DefaultIngestionSubmissionService(IngestionSubmissionService):
    """Persist context events into the debounced batch queue.

    The service is constructed with ports only. Hosts (or
    :func:`bootstrap.container.ContextEngineContainer.ingestion_submission`)
    are responsible for binding sessions to the concrete adapters and
    handing them in.
    """

    def __init__(
        self,
        *,
        settings: ContextEngineSettingsPort,
        pots: PotResolutionPort,
        reconciliation_agent: ReconciliationAgentPort | None,
        reco_ledger: ReconciliationLedgerPort,
        events: IngestionEventStore,
        batches: BatchRepositoryPort,
        jobs: ContextGraphJobQueuePort,
        ingestion_config: IngestionConfigPort | None = None,
    ) -> None:
        self._settings = settings
        self._pots = pots
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
        """
        if not self._settings.is_enabled():
            raise ValueError("context_graph_disabled")
        if self._reconciliation_agent is None:
            raise ValueError("no_reconciliation_agent")

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
                timeout_seconds=timeout_seconds if timeout_seconds is not None else 300.0,
            )
            if terminal is not None:
                return EventReceipt(
                    event_id=outcome.event_id,
                    status=terminal.status,
                    terminal_event=terminal,
                    job_id=outcome.batch_id,
                )
        return receipt
