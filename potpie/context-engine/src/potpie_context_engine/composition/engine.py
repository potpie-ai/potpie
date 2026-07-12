"""Composition of engine domain services and injected outbound adapters."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from potpie_context_engine.adapters.outbound.graph.backends import build_backend
from potpie_context_engine.adapters.outbound.graph.inbox_stores import (
    LocalJsonGraphInboxStore,
)
from potpie_context_engine.adapters.outbound.graph.plan_stores import (
    LocalJsonGraphPlanStore,
)
from potpie_context_engine.adapters.outbound.ledger.cursor_store import (
    LocalLedgerCursorStore,
)
from potpie_context_engine.adapters.outbound.ledger.managed_client import (
    ManagedEventLedgerClient,
)
from potpie_context_engine.adapters.outbound.pots.local_pot_store import LocalPotStore
from potpie_context_engine.adapters.outbound.session.injection_ledger import (
    LocalInjectionLedger,
)
from potpie_context_engine.application.services.agent_context import AgentContextService
from potpie_context_engine.application.services.graph_service import DefaultGraphService
from potpie_context_engine.application.services.graph_workbench import (
    GraphWorkbenchService,
)
from potpie_context_engine.application.services.nudge_service import NudgeService
from potpie_context_engine.application.services.pot_management import (
    LocalPotManagementService,
)
from potpie_context_engine.bootstrap.logging_setup import configure_logging
from potpie_context_engine.bootstrap.observability_context import correlation_scope
from potpie_context_engine.bootstrap.observability_runtime import set_observability
from potpie_context_engine.bootstrap.observability_wiring import default_observability
from potpie_context_engine.composition.components import EngineComponents, LedgerFacade
from potpie_context_engine.domain.coherence import assert_runtime_coherence
from potpie_context_engine.domain.ports.graph.backend import GraphBackend
from potpie_context_engine.domain.ports.context_graph_job_queue import (
    ContextGraphJobQueuePort,
    NoOpContextGraphJobQueue,
)
from potpie_context_engine.domain.ports.ledger.client import EventLedgerClientPort
from potpie_context_engine.domain.ports.observability import ObservabilityPort


def default_backend_profile() -> str:
    for env_name in ("CONTEXT_ENGINE_BACKEND", "GRAPH_DB_BACKEND"):
        profile = (os.getenv(env_name) or "").strip().lower()
        if profile:
            return profile
    return "falkordb_lite"


def build_engine_components(
    *,
    backend: GraphBackend | None = None,
    profile: str = "local",
    ledger_client: EventLedgerClientPort | None = None,
    observability: ObservabilityPort | None = None,
    job_queue: ContextGraphJobQueuePort | None = None,
    settings: Any = None,
    data_dir: Path | None = None,
) -> EngineComponents:
    configure_logging()
    set_observability(observability or default_observability())
    with correlation_scope(source="context_engine"):
        backend = backend or build_backend(default_backend_profile(), settings=settings)
        pot_store = (
            LocalPotStore(home=data_dir) if data_dir is not None else LocalPotStore()
        )
        graph = DefaultGraphService(backend=backend)
        graph_workbench = GraphWorkbenchService(
            backend=backend,
            plan_store=(
                LocalJsonGraphPlanStore(home=data_dir)
                if data_dir is not None
                else LocalJsonGraphPlanStore()
            ),
            inbox_store=(
                LocalJsonGraphInboxStore(home=data_dir)
                if data_dir is not None
                else LocalJsonGraphInboxStore()
            ),
        )
        assert_runtime_coherence(reader_backed_includes=graph.backed_includes)
        pots = LocalPotManagementService(store=pot_store, backend=backend)
        agent_context = AgentContextService(graph=graph, pots=pots, profile=profile)
        ledger = LedgerFacade(
            client=ledger_client or ManagedEventLedgerClient(),
            cursors=(
                LocalLedgerCursorStore(home=data_dir)
                if data_dir is not None
                else LocalLedgerCursorStore()
            ),
        )
        nudge = NudgeService(
            graph=graph,
            ledger=(
                LocalInjectionLedger(path=data_dir / "nudge_sessions.json")
                if data_dir is not None
                else LocalInjectionLedger()
            ),
        )
        return EngineComponents(
            agent_context=agent_context,
            graph=graph,
            graph_workbench=graph_workbench,
            pots=pots,
            backend=backend,
            ledger=ledger,
            nudge=nudge,
            job_queue=job_queue or NoOpContextGraphJobQueue(),
            profile=profile,
        )


__all__ = ["build_engine_components", "default_backend_profile"]
