"""Explicit Potpie composition for root services and the local engine boundary."""

from __future__ import annotations

import os
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
from potpie_context_engine.adapters.outbound.graph.local_commit_mirror import (
    LocalCommitMirror,
)
from potpie_context_engine.adapters.outbound.graph.local_rollback_previews import (
    LocalRollbackPreviews,
)
from potpie_context_engine.adapters.outbound.graph.inbox_stores import (
    LocalJsonGraphInboxStore,
)
from potpie_context_engine.adapters.outbound.graph.plan_stores import (
    LocalJsonGraphPlanStore,
)
from potpie_context_engine.adapters.outbound.resources import LocalResourceStore
from potpie_context_engine.adapters.outbound.resources.index import (
    NullResourceIndex,
    ResourceIndexDrain,
    build_resource_index,
    default_resource_index_profile,
)
from potpie_context_engine.application.services.resource_facade import (
    ResourceFacade,
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
from potpie.skills.harnesses import HARNESS_LAYOUTS
from potpie.skills.targets import AgentTarget
from potpie.agent_context import AgentContextService
from potpie.config.local_paths import default_home
from potpie.runtime.commit_access import (
    authorize_local_commit,
    local_commit_actor,
    local_commit_host,
)
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
from potpie_context_engine.core.ports.resource_index import ResourceIndexError
from potpie_context_engine.core.reconciliation_config import ReconciliationConfig
from potpie_context_engine.core.reconciliation_flags import (
    reconciliation_config_from_env,
)
from potpie_context_engine.domain.ports.ledger.client import EventLedgerClientPort
from potpie_context_engine.domain.ports.observability import ObservabilityPort
from potpie_context_engine.domain.ports.provisioning import ProvisionableGraphBackend


@dataclass(frozen=True, slots=True)
class LocalRuntimeComposition:
    """One explicit composition with separate product and engine service groups.

    Building it starts no thread. A process that serves engine operations (the
    daemon, or an in-process engine client) calls :meth:`start_background_work`
    once, and :meth:`close` on its way out; a CLI that only uses root services
    never starts the resource-index drain at all.
    """

    root: RootRuntimeServices
    engine: LocalEngineServices
    coordinator: OperationCoordinator
    graph_metadata: LocalGraphMetadataOperationHandler

    def start_background_work(self) -> None:
        """Start the resource-index drain. Idempotent; a lexical index has none."""
        resources = self.engine.resources
        drain = getattr(resources, "drain", None)
        if drain is not None:
            drain.start()

    def close(self) -> None:
        """Stop the drain thread and close the index database. Idempotent."""
        resources = self.engine.resources
        if resources is None:
            return
        drain = getattr(resources, "drain", None)
        if drain is not None:
            drain.stop()
        close_index = getattr(resources.index, "close", None)
        if callable(close_index):
            close_index()


def default_backend_profile() -> str:
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


def _resource_index() -> Any:
    """The configured retrieval index, or a labelled ``none`` index on a typo.

    An unknown profile still never answers as the default: the ``none`` index
    reports ``ready=False`` with the refusal in ``resource index status``,
    ``doctor`` and every import. It just does not take down every other
    command, including the ``config set resource_index`` that would fix it.
    """
    try:
        return build_resource_index(default_resource_index_profile())
    except ResourceIndexError as exc:
        repair = (
            f" {exc.recommended_next_action}" if exc.recommended_next_action else ""
        )
        return NullResourceIndex(detail=f"{exc}.{repair}")


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
        # Document payloads live outside the graph, under the same home. The
        # index over them is built before the runtime because the read trunk
        # answers the ``resources`` include family from it; one instance, so an
        # import is visible to the very next search.
        resource_store = LocalResourceStore()
        resource_index = _resource_index()
        resource_drain = ResourceIndexDrain(index=resource_index)
        home = default_home()
        graph_runtime = build_graph_runtime(
            selected_backend,
            LocalJsonGraphPlanStore(),
            LocalJsonGraphInboxStore(),
            reconciliation_config=reconciliation,
            resource_index=resource_index,
            resource_store=resource_store,
            # Commit history: a rebuildable listing index and server-held
            # rollback previews beside the graph, both under this home. Who may
            # read or roll back is decided per typed operation (commit_access).
            commit_mirror=LocalCommitMirror(home / "graph_commits.sqlite"),
            preview_store=LocalRollbackPreviews(home / "rollback_previews.sqlite"),
            commit_host=local_commit_host(home),
            commit_actor=local_commit_actor,
            commit_authorize=authorize_local_commit,
        )
        graph = graph_runtime.graph
        graph_workbench = graph_runtime.workbench
        assert_runtime_coherence(reader_backed_includes=graph.backed_includes)
        # An import writes both halves -- bytes here, structure through the
        # graph's write door -- and reads claims back to see what landed.
        resources = ResourceFacade.from_runtime(
            graph_runtime,
            store=resource_store,
            index=resource_index,
            drain=resource_drain,
        )

        pots = LocalPotManagementService(store=pot_store, backend=selected_backend)
        skills = DefaultSkillManager(
            targets={agent: AgentTarget(agent=agent) for agent in HARNESS_LAYOUTS}
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
            resources=resources,
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
