"""``build_host_shell`` — the skeleton composition root.

Wires the services + adapters behind every seam into one ``HostShell``. This is
the single composition root for the agent surface (CLI). The HTTP
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
from potpie_context_engine.adapters.outbound.resources import LocalResourceStore
from potpie_context_engine.adapters.outbound.resources.index import (
    ResourceIndexDrain,
    build_resource_index,
    default_resource_index_profile,
)
from potpie_context_engine.adapters.outbound.session.injection_ledger import (
    LocalInjectionLedger,
)
from potpie_context_engine.adapters.outbound.skills.claude_target import (
    ClaudeAgentTarget,
    ClaudePluginAgentTarget,
    CodexAgentTarget,
    CursorAgentTarget,
    OpenCodeAgentTarget,
)
from potpie_context_engine.application.services.agent_context import AgentContextService
from potpie_context_engine.application.services.auth_service import LocalAuthService
from potpie_context_engine.application.services.config_service import LocalConfigService
from potpie_context_core.api import build_graph_runtime
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
from potpie_context_core.coherence import assert_runtime_coherence
from potpie_context_core.reconciliation_config import ReconciliationConfig
from potpie_context_core.reconciliation_flags import reconciliation_config_from_env
from potpie_context_engine.domain.ports.ledger.client import EventLedgerClientPort
from potpie_context_engine.domain.ports.observability import ObservabilityPort
from potpie_context_engine.domain.ports.provisioning import ProvisionableGraphBackend
from potpie.daemon.lifecycle import Daemon
from potpie_context_engine.host.shell import HostShell, LedgerFacade, ResourceFacade


#: The graph-native local default, and the driver module it cannot run without.
_GRAPH_NATIVE_PROFILE = "falkordb_lite"
_GRAPH_NATIVE_DRIVER = "redislite"

#: Where a base install lands instead: a real, JSON-persisted backend with no
#: third-party dependencies at all.
_DEPENDENCY_FREE_PROFILE = "embedded"


def default_backend_profile() -> str:
    """The local backend to use when nobody has said which one.

    'falkordb_lite' is the OSS local default: a graph-native, persistent backend
    across CLI invocations. It needs the ``redislite`` driver, which the base
    ``potpie`` distribution — a remote-only client — does not install.

    When that driver is absent, fall back to ``embedded`` rather than returning
    a profile that is certain to fail on first use. ``embedded`` is a real,
    JSON-persisted backend with no third-party dependencies, so a base install
    keeps a working (if not graph-native) local graph instead of a crash. An
    explicit ``$CONTEXT_ENGINE_BACKEND`` is never second-guessed: if an operator
    named a profile, a loud failure about its missing driver is the right answer.
    """
    for env_name in ("CONTEXT_ENGINE_BACKEND", "GRAPH_DB_BACKEND"):
        profile = (os.getenv(env_name) or "").strip().lower()
        if profile:
            return profile
    if not graph_native_driver_available():
        return _DEPENDENCY_FREE_PROFILE
    return _GRAPH_NATIVE_PROFILE


def graph_native_driver_available() -> bool:
    """Whether the local graph-native backend's driver is installed.

    A spec probe, not an import: the question is only "did this installation
    take the ``potpie[local]`` extra", and answering it must not start an
    embedded server or pay the driver's import cost on every CLI invocation.
    """
    import importlib.util

    try:
        return importlib.util.find_spec(_GRAPH_NATIVE_DRIVER) is not None
    except (ImportError, ValueError):  # pragma: no cover - malformed installation
        return False


def build_skill_manager() -> DefaultSkillManager:
    """The skill manager on its own, with no host around it.

    Installing a skill copies template files into *this* machine's harness
    directories. It needs the bundle and a filesystem — not a graph, a backend,
    a pot, or a running process. Routing it through a full ``HostShell`` made it
    fail with "Potpie daemon is not running" on exactly the installs that most
    need skills to work: a fresh, remote-only one.

    Keeping the target registry here rather than in the CLI keeps one list:
    ``build_host_shell`` below composes the same function, so a harness added in
    one place cannot go missing in the other.
    """
    return DefaultSkillManager(
        targets={
            "claude": ClaudeAgentTarget(),
            # Registered so the harness is known rather than merely absent:
            # its bundle ships in the wheel and installs at project scope,
            # but leaving it out made every subcommand answer with a "Known:"
            # list that read as "unsupported".
            "claude-plugin": ClaudePluginAgentTarget(),
            "codex": CodexAgentTarget(),
            "cursor": CursorAgentTarget(),
            "opencode": OpenCodeAgentTarget(),
        }
    )


def default_host_mode() -> str:
    mode = (os.getenv("CONTEXT_ENGINE_HOST_MODE") or "daemon").strip().lower()
    if mode not in {"daemon", "in_process"}:
        raise ValueError(
            "invalid CONTEXT_ENGINE_HOST_MODE="
            f"{mode!r}; expected 'daemon' or 'in_process'"
        )
    return mode


def build_skill_manager() -> DefaultSkillManager:
    return DefaultSkillManager(
        targets={
            "claude": ClaudeAgentTarget(),
            "claude-plugin": ClaudePluginAgentTarget(),
            "codex": CodexAgentTarget(),
            "cursor": CursorAgentTarget(),
            "opencode": OpenCodeAgentTarget(),
        }
    )


def build_host_shell(
    *,
    backend: ProvisionableGraphBackend | None = None,
    profile: str = "local",
    ledger_client: EventLedgerClientPort | None = None,
    observability: ObservabilityPort | None = None,
    reconciliation_config: ReconciliationConfig | None = None,
    settings: Any = None,
) -> HostShell:
    """Compose a ``HostShell`` from the default local services + adapters.

    Pass ``backend`` to inject a provisionable graph backend (tests inject a
    shared ``InMemoryGraphBackend``); otherwise one is built from the configured
    profile. Runtime-only backends belong in :func:`build_graph_runtime`.
    Pass ``ledger_client`` to inject a fixture ledger.
    """
    configure_logging()
    set_observability(observability or default_observability())
    with correlation_scope(source="host_shell"):
        backend = backend or build_backend(default_backend_profile(), settings=settings)
        if not isinstance(backend, ProvisionableGraphBackend):
            raise TypeError(
                "host shell backend must implement deployment provisioning; "
                "use build_graph_runtime for runtime-only backends"
            )
        pot_store = LocalPotStore()
        reconciliation = reconciliation_config or reconciliation_config_from_env()

        # Document payloads live outside the graph; cloud swaps this store for
        # object storage without the facade or the CLI noticing. One store
        # instance is shared with pot management so ``pot reset`` / ``archive``
        # can purge the same tree ``resource import`` wrote (R8).
        resource_store = LocalResourceStore()
        # The retrieval index over those bytes. Built before the runtime
        # because the ``resources`` include family is answered by the read
        # trunk, which the runtime composes — the alternative, attaching it
        # afterwards, would leave a window where the family exists and answers
        # nothing.
        resource_index = build_resource_index(default_resource_index_profile())
        resource_drain = ResourceIndexDrain(index=resource_index)
        resource_drain.start()

        graph_runtime = build_graph_runtime(
            backend,
            LocalJsonGraphPlanStore(),
            LocalJsonGraphInboxStore(),
            reconciliation_config=reconciliation,
            resource_index=resource_index,
        )
        graph = graph_runtime.graph
        graph_workbench = graph_runtime.workbench
        assert_runtime_coherence(reader_backed_includes=graph.backed_includes)
        pots = LocalPotManagementService(
            store=pot_store, backend=backend, resources=resource_store
        )
        skills = build_skill_manager()
        agent_context = AgentContextService(
            graph=graph,
            pots=pots,
            skills=skills,
            profile=profile,
            # So `potpie status` reports the same graph health `graph quality`
            # does, instead of the backend's count-only projection.
            workbench=graph_workbench,
        )

        ledger = LedgerFacade(
            client=ledger_client or ManagedEventLedgerClient(),
            cursors=LocalLedgerCursorStore(),
        )

        # The graph service comes along because an import writes both halves —
        # bytes here, structure through the same write door every other
        # mutation uses. The claim query is the read side of the same join: what
        # else points at a section before it is removed, and did the write land.
        resources = ResourceFacade(
            store=resource_store,
            graph=graph,
            claims=backend.claim_query,
            # The same index the read trunk answers ``--include resources``
            # from. One instance, so an import is visible to the very next
            # search without a reload — two would be two databases.
            index=resource_index,
            drain=resource_drain,
        )

        # The nudge brain reads through the graph service and dedups via a local
        # per-session injection ledger (both deterministic; no model on this path).
        nudge = NudgeService(graph=graph, ledger=LocalInjectionLedger())

        # Lifecycle components (each independently ownable; see the setup orchestrator).
        daemon = Daemon(in_process=(default_host_mode() != "daemon"))
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
            resources=resources,
            nudge=nudge,
            daemon=daemon,
            config=config,
            installer=installer,
            auth=auth,
            setup=setup,
            profile=profile,
        )


__all__ = ["build_host_shell", "default_backend_profile", "default_host_mode"]
