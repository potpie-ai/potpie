"""``build_host_shell`` — the skeleton composition root.

Wires the services + adapters behind every seam into one ``HostShell``. This is
the single composition root for the agent surface (CLI + MCP). The HTTP
ingestion subsystem has its own root, ``bootstrap/ingestion_server.py`` (Neo4j +
Postgres pipeline + connectors + reconciliation agent), which is being migrated
onto ``HostShell`` and is not imported on the CLI path.

Profile selection:
    backend profile defaults to ``falkordb_lite`` (embedded FalkorDBLite local
    stack). ``$CONTEXT_ENGINE_BACKEND`` overrides it; ``neo4j`` is the
    shape-first production target. The ledger defaults to an unbound dummy
    client; tests can inject a ``FixtureEventLedgerClient``.
"""

from __future__ import annotations

import os
from typing import Any

from potpie_context_engine.adapters.outbound.daemon_lifecycle import (
    InProcessDaemonLifecycle,
)
from potpie_context_engine.adapters.outbound.graph.backends import build_backend
from potpie_context_engine.adapters.outbound.graph.inbox_stores import (
    LocalJsonGraphInboxStore,
)
from potpie_context_engine.adapters.outbound.graph.plan_stores import (
    LocalJsonGraphPlanStore,
)
from potpie_context_engine.adapters.outbound.install.local_installer import (
    LocalInstaller,
)
from potpie_context_engine.adapters.outbound.ledger.cursor_store import (
    LocalLedgerCursorStore,
)
from potpie_context_engine.adapters.outbound.ledger.managed_client import (
    ManagedEventLedgerClient,
)
from potpie_context_engine.adapters.outbound.pots.flat_file_state_store import (
    FlatFileMigrator,
    FlatFileStateStore,
)
from potpie_context_engine.adapters.outbound.pots.local_pot_store import LocalPotStore
from potpie_context_engine.adapters.outbound.session.injection_ledger import (
    LocalInjectionLedger,
)
from potpie_context_engine.adapters.outbound.skills.claude_target import (
    ClaudeAgentTarget,
    CodexAgentTarget,
    CursorAgentTarget,
    OpenCodeAgentTarget,
)
from potpie_context_engine.adapters.outbound.skills.template_resources import (
    TemplateResourceProvider,
    resolve_template_resources,
)
from potpie_context_engine.application.services.agent_context import AgentContextService
from potpie_context_engine.application.services.auth_service import LocalAuthService
from potpie_context_engine.application.services.config_service import LocalConfigService
from potpie_context_engine.application.services.graph_service import DefaultGraphService
from potpie_context_engine.application.services.graph_workbench import (
    GraphWorkbenchService,
)
from potpie_context_engine.application.services.nudge_service import NudgeService
from potpie_context_engine.application.services.pot_management import (
    LocalPotManagementService,
)
from potpie_context_engine.application.services.setup_orchestrator import (
    DefaultSetupOrchestrator,
)
from potpie_context_engine.application.services.skill_manager import DefaultSkillManager
from potpie_context_engine.bootstrap.logging_setup import configure_logging
from potpie_context_engine.bootstrap.observability_context import correlation_scope
from potpie_context_engine.bootstrap.observability_runtime import set_observability
from potpie_context_engine.bootstrap.observability_wiring import default_observability
from potpie_context_engine.domain.coherence import assert_runtime_coherence
from potpie_context_engine.domain.ports.daemon.lifecycle import DaemonLifecyclePort
from potpie_context_engine.domain.ports.graph.backend import GraphBackend
from potpie_context_engine.domain.ports.ledger.client import EventLedgerClientPort
from potpie_context_engine.domain.ports.observability import ObservabilityPort
from potpie_context_engine.host.shell import HostShell, LedgerFacade


def default_backend_profile() -> str:
    # 'falkordb_lite' is the OSS local default: a graph-native, persistent
    # backend across CLI invocations.
    for env_name in ("CONTEXT_ENGINE_BACKEND", "GRAPH_DB_BACKEND"):
        profile = (os.getenv(env_name) or "").strip().lower()
        if profile:
            return profile
    return "falkordb_lite"


def default_host_mode() -> str:
    mode = (os.getenv("CONTEXT_ENGINE_HOST_MODE") or "daemon").strip().lower()
    if mode not in {"daemon", "in_process"}:
        raise ValueError(
            "invalid CONTEXT_ENGINE_HOST_MODE="
            f"{mode!r}; expected 'daemon' or 'in_process'"
        )
    return mode


def build_host_shell(
    *,
    backend: GraphBackend | None = None,
    profile: str = "local",
    ledger_client: EventLedgerClientPort | None = None,
    observability: ObservabilityPort | None = None,
    settings: Any = None,
    daemon_lifecycle: DaemonLifecyclePort | None = None,
    template_resources: TemplateResourceProvider | None = None,
) -> HostShell:
    """Compose a ``HostShell`` from the default local services + adapters.

    Pass ``backend`` to inject a specific ``GraphBackend`` (tests inject a shared
    ``InMemoryGraphBackend``); otherwise one is built from the configured
    profile. Pass ``ledger_client`` to inject a fixture ledger.
    """
    configure_logging()
    set_observability(observability or default_observability())
    with correlation_scope(source="host_shell"):
        template_resources = resolve_template_resources(template_resources)
        backend = backend or build_backend(default_backend_profile(), settings=settings)
        pot_store = LocalPotStore()

        graph = DefaultGraphService(backend=backend)
        graph_workbench = GraphWorkbenchService(
            backend=backend,
            plan_store=LocalJsonGraphPlanStore(),
            inbox_store=LocalJsonGraphInboxStore(),
        )
        assert_runtime_coherence(reader_backed_includes=graph.backed_includes)
        pots = LocalPotManagementService(store=pot_store, backend=backend)
        skills = DefaultSkillManager(
            targets={
                "claude": ClaudeAgentTarget(template_resources=template_resources),
                "codex": CodexAgentTarget(template_resources=template_resources),
                "cursor": CursorAgentTarget(template_resources=template_resources),
                "opencode": OpenCodeAgentTarget(template_resources=template_resources),
            },
            template_resources=template_resources,
        )
        agent_context = AgentContextService(
            graph=graph, pots=pots, skills=skills, profile=profile
        )

        ledger = LedgerFacade(
            client=ledger_client or ManagedEventLedgerClient(),
            cursors=LocalLedgerCursorStore(),
        )

        # The nudge brain reads through the graph service and dedups via a local
        # per-session injection ledger (both deterministic; no model on this path).
        nudge = NudgeService(graph=graph, ledger=LocalInjectionLedger())

        # Lifecycle components (each independently ownable; see the setup orchestrator).
        daemon = daemon_lifecycle or InProcessDaemonLifecycle()
        config = LocalConfigService()
        installer = LocalInstaller()
        auth = LocalAuthService()
        setup = DefaultSetupOrchestrator(
            config=config,
            installer=installer,
            backend=backend,
            pots=pots,
            # Relational state-store + migration seams (flat-file profile: skipped).
            state_store=FlatFileStateStore(),
            migrator=FlatFileMigrator(),
            daemon=daemon,
            auth=auth,
            skills=skills,
        )

        return HostShell(
            agent_context=agent_context,
            graph=graph,
            graph_workbench=graph_workbench,
            pots=pots,
            skills=skills,
            backend=backend,
            ledger=ledger,
            nudge=nudge,
            daemon=daemon,
            config=config,
            installer=installer,
            auth=auth,
            setup=setup,
            profile=profile,
        )


__all__ = ["build_host_shell", "default_backend_profile", "default_host_mode"]
