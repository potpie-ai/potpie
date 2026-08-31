"""Explicit Potpie composition for root services and the local engine boundary."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from potpie.daemon.lifecycle import Daemon
from potpie.runtime.local_engine import (
    LocalEngineServices,
    LocalGraphMetadataOperationHandler,
)
from potpie.runtime.coordinator import OperationCoordinator
from potpie.runtime.root_services import (
    LedgerService,
    PotResourceService,
    RootRuntimeServices,
)
from potpie_context_engine.adapters.outbound.graph.backends import build_backend
from potpie_context_engine.adapters.outbound.graph.inbox_stores import (
    LocalJsonGraphInboxStore,
)
from potpie_context_engine.adapters.outbound.graph.plan_stores import (
    LocalJsonGraphPlanStore,
)
from potpie.setup.local_installer import (
    LocalInstaller,
)
from potpie_context_engine.adapters.outbound.ledger.cursor_store import (
    LocalLedgerCursorStore,
)
from potpie_context_engine.adapters.outbound.ledger.managed_client import (
    ManagedEventLedgerClient,
)
from potpie.setup.flat_file_state import (
    FlatFileMigrator,
    FlatFileStateStore,
)
from potpie.pots.local_store import LocalPotStore
from potpie_context_engine.adapters.outbound.session.injection_ledger import (
    LocalInjectionLedger,
)
from potpie.skills.targets import (
    ClaudeAgentTarget,
    CodexAgentTarget,
    CursorAgentTarget,
    OpenCodeAgentTarget,
)
from potpie.agent_context import AgentContextService
from potpie.auth.adapters.local_identity import LocalAuthService
from potpie.config.local import LocalConfigService
from potpie_context_engine.application.services.nudge_service import NudgeService
from potpie.pots.local_service import (
    LocalPotManagementService,
)
from potpie.setup.orchestrator import (
    DefaultSetupOrchestrator,
)
from potpie.skills.manager import DefaultSkillManager
from potpie_context_engine.bootstrap.logging_setup import configure_logging
from potpie_context_engine.bootstrap.observability_context import correlation_scope
from potpie_context_engine.bootstrap.observability_runtime import set_observability
from potpie_context_engine.bootstrap.observability_wiring import default_observability
from potpie_context_engine.core.runtime import build_graph_runtime
from potpie_context_engine.core.coherence import assert_runtime_coherence
from potpie_context_engine.core.reconciliation_config import ReconciliationConfig
from potpie_context_engine.core.reconciliation_flags import (
    reconciliation_config_from_env,
)
from potpie_context_engine.domain.ports.ledger.client import EventLedgerClientPort
from potpie_context_engine.domain.ports.observability import ObservabilityPort
from potpie_context_engine.domain.ports.provisioning import ProvisionableGraphBackend


@dataclass(frozen=True, slots=True)
class LocalRuntimeComposition:
    """One explicit composition with separate product and engine service groups."""

    root: RootRuntimeServices
    engine: LocalEngineServices
    coordinator: OperationCoordinator
    graph_metadata: LocalGraphMetadataOperationHandler


def default_backend_profile() -> str:
    """Env override, else the backend persisted at setup, else the OSS default.

    Reading ``config.json`` matters: without it a plain ``potpie daemon start``
    silently ran on ``falkordb_lite`` for a user whose setup had configured a
    different backend, and their graph looked empty rather than unreachable.
    """
    for env_name in ("CONTEXT_ENGINE_BACKEND", "GRAPH_DB_BACKEND"):
        profile = (os.getenv(env_name) or "").strip().lower()
        if profile:
            return profile
    return _configured_backend_profile() or "falkordb_lite"


def _configured_backend_profile() -> str | None:
    """Backend recorded in ``<home>/config.json`` by ``potpie setup``."""
    from potpie.config.local_paths import default_home

    try:
        with open(default_home() / "config.json", encoding="utf-8") as handle:
            document = json.load(handle)
    except (OSError, ValueError):
        return None
    # A hand-edited config can hold any JSON root; only an object has a
    # backend, and anything else must fall through to the default rather
    # than raising out of composition.
    if not isinstance(document, Mapping):
        return None
    configured = document.get("backend")
    if not isinstance(configured, str):
        return None
    return configured.strip().lower() or None


def default_host_mode() -> str:
    mode = (os.getenv("CONTEXT_ENGINE_HOST_MODE") or "daemon").strip().lower()
    if mode not in {"daemon", "in_process"}:
        raise ValueError(
            "invalid CONTEXT_ENGINE_HOST_MODE="
            f"{mode!r}; expected 'daemon' or 'in_process'"
        )
    return mode


def build_local_runtime(
    *,
    backend: ProvisionableGraphBackend | None = None,
    profile: str = "local",
    ledger_client: EventLedgerClientPort | None = None,
    observability: ObservabilityPort | None = None,
    reconciliation_config: ReconciliationConfig | None = None,
    settings: Any = None,
) -> LocalRuntimeComposition:
    """Compose root product services and context services without a host façade."""

    configure_logging()
    set_observability(observability or default_observability())
    with correlation_scope(source="local_runtime"):
        selected_backend = backend or build_backend(
            default_backend_profile(), settings=settings
        )
        if not isinstance(selected_backend, ProvisionableGraphBackend):
            raise TypeError(
                "local runtime backend must implement deployment provisioning; "
                "use build_graph_runtime for runtime-only backends"
            )
        pot_store = LocalPotStore()
        reconciliation = reconciliation_config or reconciliation_config_from_env()
        graph_runtime = build_graph_runtime(
            selected_backend,
            LocalJsonGraphPlanStore(),
            LocalJsonGraphInboxStore(),
            reconciliation_config=reconciliation,
        )
        graph = graph_runtime.graph
        graph_workbench = graph_runtime.workbench
        assert_runtime_coherence(reader_backed_includes=graph.backed_includes)

        pots = LocalPotManagementService(store=pot_store, backend=selected_backend)
        skills = DefaultSkillManager(
            targets={
                "claude": ClaudeAgentTarget(),
                "codex": CodexAgentTarget(),
                "cursor": CursorAgentTarget(),
                "opencode": OpenCodeAgentTarget(),
            }
        )
        agent_context = AgentContextService(
            graph=graph,
            pots=pots,
            skills=skills,
            profile=profile,
        )
        nudge = NudgeService(graph=graph, ledger=LocalInjectionLedger())

        daemon = Daemon(in_process=(default_host_mode() != "daemon"))
        config = LocalConfigService()
        installer = LocalInstaller()
        auth = LocalAuthService()
        setup = DefaultSetupOrchestrator(
            config=config,
            installer=installer,
            backend=selected_backend,
            pots=pots,
            state_store=FlatFileStateStore(),
            migrator=FlatFileMigrator(),
            daemon=daemon,
            auth=auth,
            skills=skills,
        )
        ledger = LedgerService(
            client=ledger_client or ManagedEventLedgerClient(),
            cursors=LocalLedgerCursorStore(),
        )

        engine_services = LocalEngineServices(
            pots=pots,
            agent_context=agent_context,
            graph=graph,
            graph_workbench=graph_workbench,
            backend=selected_backend,
            nudge=nudge,
        )
        return LocalRuntimeComposition(
            root=RootRuntimeServices(
                pots=PotResourceService(pots),
                backend=selected_backend,
                auth=auth,
                config=config,
                daemon=daemon,
                installer=installer,
                ledger=ledger,
                setup=setup,
                skills=skills,
                profile=profile,
            ),
            engine=engine_services,
            coordinator=OperationCoordinator(),
            graph_metadata=LocalGraphMetadataOperationHandler(engine_services),
        )


__all__ = [
    "LocalRuntimeComposition",
    "build_local_runtime",
    "default_backend_profile",
    "default_host_mode",
]
