"""Standalone asynchronous facade over context-engine application services."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from potpie_context_engine.adapters.outbound.graph.backends import build_backend
from potpie_context_engine.adapters.outbound.graph.backends.embedded_backend import (
    EmbeddedGraphBackend,
)
from potpie_context_engine.adapters.outbound.graph.backends.in_memory_backend import (
    InMemoryGraphBackend,
)
from potpie_context_engine.composition import EngineComponents, build_engine_components
from potpie_context_engine.config import EngineConfig
from potpie_context_engine.contracts import (
    AgentEnvelope,
    DataPlaneStatus,
    EmptyRequest,
    EngineStatusReport,
    EngineStatusRequest,
    GraphCatalogRequest,
    GraphCatalogResult,
    GraphBackendInfo,
    GraphBackendInfoRequest,
    GraphCommitRequest,
    GraphEntitySearchRequest,
    GraphEntitySearchResult,
    GraphDescribeRequest,
    GraphHistoryRequest,
    GraphHistoryResult,
    GraphInboxAddRequest,
    GraphInboxClaimRequest,
    GraphInboxCloseRequest,
    GraphInboxItemRequest,
    GraphInboxListRequest,
    GraphInboxResult,
    GraphMutationCommitResult,
    GraphMutationProposal,
    GraphNeighborhoodRequest,
    GraphNudgeRequest,
    GraphNudgeResult,
    GraphProposeRequest,
    GraphQualityRequest,
    GraphQualityResult,
    GraphReadRequest,
    GraphReadResult,
    GraphRepairRequest,
    GraphSlice,
    GraphSnapshotExportRequest,
    GraphSnapshotImportRequest,
    GraphStatusRequest,
    LedgerHealth,
    LedgerPage,
    LedgerPullRequest,
    LedgerQueryRequest,
    LedgerSourcesRequest,
    LedgerSourcesResult,
    LedgerStatusRequest,
    OperationResult,
    PotArchiveRequest,
    PotCreateRequest,
    PotInfo,
    PotInfoRequest,
    PotListResult,
    PotRenameRequest,
    PotResetRequest,
    PotUseRequest,
    RepoDefaultClearRequest,
    RepoDefaultClearResult,
    RepoDefaultGetRequest,
    RepoDefaultListResult,
    RepoDefaultResult,
    RepoDefaultSetRequest,
    ProvisionApplyRequest,
    ProvisionInspectRequest,
    ProvisionPlan,
    ProvisionReport,
    ProvisionStep,
    RecordReceipt,
    RepairReport,
    RecordRequest,
    ResolveRequest,
    SearchRequest,
    SourceAddRequest,
    SourceInfo,
    SourceListRequest,
    SourceListResult,
    SourceRemoveRequest,
    SourceStatusRequest,
    TimelineRecentRequest,
    SnapshotManifest,
)
from potpie_context_engine.dependencies import EngineDependencies


@dataclass(slots=True)
class _ContextOperations:
    engine: ContextEngine

    async def resolve(self, request: ResolveRequest) -> AgentEnvelope:
        return self.engine._components.agent_context.resolve(request)

    async def search(self, request: SearchRequest) -> AgentEnvelope:
        return self.engine._components.agent_context.search(request)

    async def record(self, request: RecordRequest) -> RecordReceipt:
        return self.engine._components.agent_context.record(request)

    async def status(self, request: EngineStatusRequest) -> EngineStatusReport:
        return self.engine._status(request)


@dataclass(slots=True)
class _PotOperations:
    engine: ContextEngine

    async def list(self, request: EmptyRequest) -> PotListResult:
        del request
        items = tuple(self.engine._components.pots.list_pots())
        return PotListResult(items=items, count=len(items))

    async def info(self, request: PotInfoRequest) -> PotInfo | None:
        pots = self.engine._components.pots
        if request.ref is None:
            return pots.active_pot()
        return next(
            (p for p in pots.list_pots() if request.ref in (p.pot_id, p.name)),
            None,
        )

    async def create(self, request: PotCreateRequest) -> PotInfo:
        return self.engine._components.pots.create_pot(
            name=request.name, repo=request.repo, use=request.use
        )

    async def use(self, request: PotUseRequest) -> PotInfo:
        return self.engine._components.pots.use_pot(ref=request.ref)

    async def rename(self, request: PotRenameRequest) -> PotInfo:
        return self.engine._components.pots.rename_pot(
            ref=request.ref, new_name=request.new_name
        )

    async def reset(self, request: PotResetRequest) -> PotInfo:
        return self.engine._components.pots.reset_pot(
            ref=request.ref, confirm=request.confirm
        )

    async def archive(self, request: PotArchiveRequest) -> PotInfo:
        return self.engine._components.pots.archive_pot(ref=request.ref)

    async def repo_default(self, request: RepoDefaultGetRequest) -> RepoDefaultResult:
        return RepoDefaultResult(
            pot_id=self.engine._components.pots.repo_default(repo=request.repo)
        )

    async def set_repo_default(self, request: RepoDefaultSetRequest) -> OperationResult:
        self.engine._components.pots.set_repo_default(
            repo=request.repo, pot_id=request.pot_id
        )
        return OperationResult()

    async def clear_repo_default(
        self, request: RepoDefaultClearRequest
    ) -> RepoDefaultClearResult:
        return RepoDefaultClearResult(
            cleared=self.engine._components.pots.clear_repo_default(repo=request.repo)
        )

    async def list_repo_defaults(self, request: EmptyRequest) -> RepoDefaultListResult:
        del request
        return RepoDefaultListResult(
            items=self.engine._components.pots.list_repo_defaults()
        )


@dataclass(slots=True)
class _SourceOperations:
    engine: ContextEngine

    async def add(self, request: SourceAddRequest) -> SourceInfo:
        return self.engine._components.pots.add_source(
            pot_id=request.pot_id,
            kind=request.kind,
            location=request.location,
            name=request.name,
        )

    async def list(self, request: SourceListRequest) -> SourceListResult:
        items = tuple(self.engine._components.pots.list_sources(pot_id=request.pot_id))
        return SourceListResult(items=items, count=len(items))

    async def status(self, request: SourceStatusRequest) -> SourceInfo:
        return self.engine._components.pots.source_status(
            pot_id=request.pot_id, source_id=request.source_id
        )

    async def remove(self, request: SourceRemoveRequest) -> OperationResult:
        self.engine._components.pots.remove_source(
            pot_id=request.pot_id, source_id=request.source_id
        )
        return OperationResult()


@dataclass(slots=True)
class _GraphOperations:
    engine: ContextEngine

    async def catalog(self, request: GraphCatalogRequest) -> GraphCatalogResult:
        return self.engine._components.graph.catalog(request)

    async def describe(self, request: GraphDescribeRequest) -> dict[str, Any]:
        return self.engine._components.graph.describe(request)

    async def read(self, request: GraphReadRequest) -> GraphReadResult:
        return self.engine._components.graph.read(request)

    async def search_entities(
        self, request: GraphEntitySearchRequest
    ) -> GraphEntitySearchResult:
        return self.engine._components.graph.search_entities(request)

    async def status(self, request: GraphStatusRequest) -> DataPlaneStatus:
        return self.engine._components.graph.data_plane_status(request.pot_id)

    async def propose(self, request: GraphProposeRequest) -> GraphMutationProposal:
        return self.engine._components.graph_workbench.propose(
            request.payload,
            pot_id=request.pot_id,
            ttl_seconds=request.ttl_seconds,
        )

    async def commit(self, request: GraphCommitRequest) -> GraphMutationCommitResult:
        return self.engine._components.graph_workbench.commit(
            request.plan_id,
            pot_id=request.pot_id,
            approved_by=request.approved_by,
            verify=request.verify,
        )

    async def history(self, request: GraphHistoryRequest) -> GraphHistoryResult:
        return self.engine._components.graph_workbench.history(
            pot_id=request.pot_id,
            entity_key=request.entity_key,
            claim_key=request.claim_key,
            subgraph=request.subgraph,
            plan_id=request.plan_id,
            mutation_id=request.mutation_id,
            since=request.since,
            until=request.until,
            limit=request.limit,
        )

    async def quality(self, request: GraphQualityRequest) -> GraphQualityResult:
        filters = dict(request.filters)
        return self.engine._components.graph_workbench.quality(
            pot_id=request.pot_id,
            report=request.report,
            subgraph=filters.get("subgraph"),
            limit=int(filters.get("limit", 50)),
            confidence_threshold=float(filters.get("confidence_threshold", 0.5)),
        )

    async def nudge(self, request: GraphNudgeRequest) -> GraphNudgeResult:
        return self.engine._components.nudge.nudge(request)

    async def neighborhood(self, request: GraphNeighborhoodRequest) -> GraphSlice:
        return self.engine._components.backend.inspection.neighborhood(
            pot_id=request.pot_id,
            entity_key=request.entity_key,
            depth=request.depth,
        )

    async def inbox_add(self, request: GraphInboxAddRequest) -> GraphInboxResult:
        return self.engine._components.graph_workbench.inbox_add(
            pot_id=request.pot_id,
            summary=request.summary,
            details=request.details,
            evidence=request.evidence,
            source_refs=request.source_refs,
            suspected_subgraphs=request.suspected_subgraphs,
            created_by=request.created_by,
        )

    async def inbox_list(self, request: GraphInboxListRequest) -> GraphInboxResult:
        return self.engine._components.graph_workbench.inbox_list(
            pot_id=request.pot_id,
            status=request.status,
            claimed_by=request.claimed_by,
            suspected_subgraph=request.suspected_subgraph,
            source_ref=request.source_ref,
            since=request.since,
            until=request.until,
            limit=request.limit,
        )

    async def inbox_show(self, request: GraphInboxItemRequest) -> GraphInboxResult:
        return self.engine._components.graph_workbench.inbox_show(
            pot_id=request.pot_id, item_id=request.item_id
        )

    async def inbox_claim(self, request: GraphInboxClaimRequest) -> GraphInboxResult:
        return self.engine._components.graph_workbench.inbox_claim(
            pot_id=request.pot_id,
            item_id=request.item_id,
            claimed_by=request.claimed_by,
        )

    async def inbox_close(self, request: GraphInboxCloseRequest) -> GraphInboxResult:
        if request.action == "mark-applied":
            return self.engine._components.graph_workbench.inbox_mark_applied(
                pot_id=request.pot_id,
                item_id=request.item_id,
                closed_by=request.closed_by,
                linked_plan_id=request.linked_plan_id,
                linked_mutation_id=request.linked_mutation_id,
            )
        if request.action == "mark-rejected":
            return self.engine._components.graph_workbench.inbox_mark_rejected(
                pot_id=request.pot_id,
                item_id=request.item_id,
                closed_by=request.closed_by,
                rejection_reason=request.rejection_reason or "",
            )
        return self.engine._components.graph_workbench.inbox_close(
            pot_id=request.pot_id,
            item_id=request.item_id,
            closed_by=request.closed_by,
            linked_plan_id=request.linked_plan_id,
            linked_mutation_id=request.linked_mutation_id,
            rejection_reason=request.rejection_reason,
        )

    async def snapshot_export(
        self, request: GraphSnapshotExportRequest
    ) -> SnapshotManifest:
        return self.engine._components.backend.snapshot.export(
            pot_id=request.pot_id, destination=request.destination
        )

    async def snapshot_import(
        self, request: GraphSnapshotImportRequest
    ) -> SnapshotManifest:
        return self.engine._components.backend.snapshot.import_(
            pot_id=request.pot_id, source=request.source
        )

    async def repair(self, request: GraphRepairRequest) -> RepairReport:
        return self.engine._components.backend.analytics.repair(
            request.pot_id, targets=request.targets
        )

    async def backend_info(self, request: GraphBackendInfoRequest) -> GraphBackendInfo:
        del request
        capabilities = self.engine._components.backend.capabilities()
        return GraphBackendInfo(
            profile=self.engine._components.backend.profile,
            capabilities=capabilities.implemented(),
        )


@dataclass(slots=True)
class _LedgerOperations:
    engine: ContextEngine

    async def status(self, request: LedgerStatusRequest) -> LedgerHealth:
        del request
        return self.engine._components.ledger.status()

    async def sources(self, request: LedgerSourcesRequest) -> LedgerSourcesResult:
        items = tuple(self.engine._components.ledger.sources(pot_id=request.pot_id))
        return LedgerSourcesResult(items=items, count=len(items))

    async def query(self, request: LedgerQueryRequest) -> LedgerPage:
        return self.engine._components.ledger.query(
            pot_id=request.pot_id,
            source_id=request.source_id,
            kind=request.kind,
            since=request.since,
            until=request.until,
            limit=request.limit,
        )

    async def pull(self, request: LedgerPullRequest) -> LedgerPage:
        return self.engine._components.ledger.pull(
            pot_id=request.pot_id,
            source_id=request.source_id,
            limit=request.limit,
        )


@dataclass(slots=True)
class _TimelineOperations:
    engine: ContextEngine

    async def recent(self, request: TimelineRecentRequest) -> GraphReadResult:
        return self.engine._components.graph.read(
            GraphReadRequest(
                pot_id=request.pot_id,
                subgraph="recent_changes",
                view="timeline",
                query=request.query,
                scope=request.scope,
                limit=request.limit,
                since=request.since,
                until=request.until,
            )
        )


@dataclass(slots=True)
class _ProvisionOperations:
    engine: ContextEngine

    async def inspect(self, request: ProvisionInspectRequest) -> ProvisionPlan:
        del request
        config = self.engine.config
        return ProvisionPlan(
            backend=self.engine._components.backend.profile,
            data_dir=str(config.data_dir) if config.data_dir is not None else None,
            steps=(
                ProvisionStep(
                    name="storage.prepare",
                    required=config.storage_mode == "persistent",
                    state="needed"
                    if config.storage_mode == "persistent"
                    else "skipped",
                ),
                ProvisionStep(name="backend.provision", required=True, state="needed"),
            ),
        )

    async def apply(self, request: ProvisionApplyRequest) -> ProvisionReport:
        config = self.engine.config
        steps: list[ProvisionStep] = []
        if config.data_dir is not None:
            config.data_dir.mkdir(parents=True, exist_ok=True)
            steps.append(
                ProvisionStep(
                    name="storage.prepare",
                    required=True,
                    state="done",
                    detail=str(config.data_dir),
                )
            )
        else:
            steps.append(
                ProvisionStep(name="storage.prepare", required=False, state="skipped")
            )
        del request
        result = self.engine._components.backend.provision()
        steps.append(
            ProvisionStep(
                name=result.step,
                required=result.hard,
                state=result.state,
                detail=result.detail,
            )
        )
        return ProvisionReport(
            ok=all(step.state in {"done", "skipped", "planned"} for step in steps),
            backend=self.engine._components.backend.profile,
            steps=tuple(steps),
        )


@dataclass(slots=True)
class ContextEngine:
    """In-process implementation of the public asynchronous ``EngineClient``."""

    config: EngineConfig
    _components: EngineComponents
    _temporary_home: TemporaryDirectory[str] | None = None
    _http_application_factory: Any = None
    context: _ContextOperations = field(init=False)
    pots: _PotOperations = field(init=False)
    sources: _SourceOperations = field(init=False)
    graph: _GraphOperations = field(init=False)
    ledger: _LedgerOperations = field(init=False)
    timeline: _TimelineOperations = field(init=False)
    provision: _ProvisionOperations = field(init=False)
    _closed: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self.context = _ContextOperations(self)
        self.pots = _PotOperations(self)
        self.sources = _SourceOperations(self)
        self.graph = _GraphOperations(self)
        self.ledger = _LedgerOperations(self)
        self.timeline = _TimelineOperations(self)
        self.provision = _ProvisionOperations(self)

    def create_http_application(self) -> Any:
        if self._http_application_factory is None:
            raise RuntimeError("no HTTP application factory was injected")
        return self._http_application_factory(self)

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        close = getattr(self._components.backend, "aclose", None)
        if close is not None:
            result = close()
            if hasattr(result, "__await__"):
                await result
        if self._temporary_home is not None:
            self._temporary_home.cleanup()

    async def __aenter__(self) -> ContextEngine:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.aclose()

    def _status(self, request: EngineStatusRequest) -> EngineStatusReport:
        pots = self._components.pots
        active = pots.active_pot()
        pot_id = request.pot_id or (active.pot_id if active else None)
        pot = next(
            (item for item in pots.list_pots() if item.pot_id == pot_id),
            active if active and active.pot_id == pot_id else None,
        )
        sources = pots.list_sources(pot_id=pot_id) if pot_id else []
        readiness = (
            self._components.backend.mutation.readiness(pot_id) if pot_id else None
        )
        ready = bool(readiness and readiness.ready)
        reasons: list[str] = []
        if pot_id is None:
            reasons.append("no pot selected")
        if readiness is not None and not readiness.ready:
            reasons.append(readiness.detail or "backend is not ready")
        return EngineStatusReport(
            schema_version="1",
            pot_id=pot_id,
            pot_name=pot.name if pot else None,
            backend=self._components.backend.profile,
            backend_ready=ready,
            storage_ready=ready,
            ingestion_ready=ready,
            source_count=len(sources),
            degraded_reasons=tuple(reasons),
        )


def create_engine(
    config: EngineConfig,
    dependencies: EngineDependencies | None = None,
) -> ContextEngine:
    """Construct a standalone engine without product settings or path defaults."""

    dependencies = dependencies or EngineDependencies()
    temporary_home: TemporaryDirectory[str] | None = None
    if config.storage_mode == "in_memory":
        temporary_home = TemporaryDirectory(prefix="potpie-context-engine-")
        data_dir = Path(temporary_home.name)
    else:
        assert config.data_dir is not None
        data_dir = config.data_dir

    backend = dependencies.backend
    if backend is None:
        if config.backend == "in_memory":
            backend = InMemoryGraphBackend()
        elif config.backend == "embedded":
            backend = EmbeddedGraphBackend(home=data_dir)
        else:
            backend = build_backend(config.backend)

    components = build_engine_components(
        backend=backend,
        profile=config.profile,
        ledger_client=dependencies.ledger_client,
        observability=dependencies.observability,
        job_queue=dependencies.job_queue,
        data_dir=data_dir,
    )
    return ContextEngine(
        config=config,
        _components=components,
        _temporary_home=temporary_home,
        _http_application_factory=dependencies.http_application_factory,
    )


__all__ = ["ContextEngine", "create_engine"]
