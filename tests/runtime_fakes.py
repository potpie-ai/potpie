"""Typed async runtime adapters for synchronous unit-test service doubles."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from potpie.auth.services import AccountAuthService, IntegrationAuthService
from potpie.config import ProductConfigService
from potpie.daemon.lifecycle import Daemon
from potpie.install import LocalInstaller
from potpie.runtime import PotpieRuntime, ProductSettings
from potpie.skills import create_skill_service
from potpie_context_engine import EngineClient
from potpie_context_engine.client import (
    ContextClient,
    GraphClient,
    LedgerClient,
    PotsClient,
    ProvisionClient,
    SourcesClient,
    TimelineClient,
)
from potpie.runtime.contracts import (
    AgentEnvelope,
    CapabilityNotImplemented,
    DataPlaneStatus,
    EmptyRequest,
    EngineStatusReport,
    EngineStatusRequest,
    GraphBackendInfo,
    GraphBackendInfoRequest,
    GraphCatalogRequest,
    GraphCatalogResult,
    GraphCommitRequest,
    GraphDescribeRequest,
    GraphEntitySearchRequest,
    GraphEntitySearchResult,
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
    RecordReceipt,
    RecordRequest,
    RepairReport,
    RepoDefaultClearResult,
    RepoDefaultClearRequest,
    RepoDefaultGetRequest,
    RepoDefaultListResult,
    RepoDefaultResult,
    RepoDefaultSetRequest,
    RegisterRepoSourceRequest,
    RegisterRepoSourceResult,
    ResolveRequest,
    SearchRequest,
    SnapshotManifest,
    SourceAddRequest,
    SourceInfo,
    SourceListRequest,
    SourceListResult,
    SourceRemoveRequest,
    SourceStatusRequest,
)
from tests._auth_fakes import InMemoryCredentialStore


def _missing(capability: str) -> CapabilityNotImplemented:
    return CapabilityNotImplemented(capability)


class PotsClientAdapter:
    def __init__(self, service: Any) -> None:
        self.service = service

    async def list(self, request: EmptyRequest) -> PotListResult:
        del request
        items = tuple(self.service.list_pots())
        return PotListResult(items=items, count=len(items))

    async def info(self, request: PotInfoRequest) -> PotInfo | None:
        if request.ref:
            for pot in self.service.list_pots():
                if request.ref in (pot.pot_id, pot.name):
                    return pot
            return None
        return self.service.active_pot()

    async def create(self, request: PotCreateRequest) -> PotInfo:
        return self.service.create_pot(
            name=request.name, repo=request.repo, use=request.use
        )

    async def use(self, request: PotUseRequest) -> PotInfo:
        return self.service.use_pot(ref=request.ref)

    async def rename(self, request: PotRenameRequest) -> PotInfo:
        return self.service.rename_pot(ref=request.ref, new_name=request.new_name)

    async def reset(self, request: PotResetRequest) -> PotInfo:
        return self.service.reset_pot(ref=request.ref, confirm=request.confirm)

    async def archive(self, request: PotArchiveRequest) -> PotInfo:
        return self.service.archive_pot(ref=request.ref)

    async def repo_default(self, request: RepoDefaultGetRequest) -> RepoDefaultResult:
        method = getattr(self.service, "repo_default", None)
        if not callable(method):
            raise _missing("engine.pots.repo_default")
        return RepoDefaultResult(pot_id=method(repo=request.repo))

    async def set_repo_default(self, request: RepoDefaultSetRequest) -> OperationResult:
        method = getattr(self.service, "set_repo_default", None)
        if not callable(method):
            raise _missing("engine.pots.set_repo_default")
        method(repo=request.repo, pot_id=request.pot_id)
        return OperationResult()

    async def clear_repo_default(
        self, request: RepoDefaultClearRequest
    ) -> RepoDefaultClearResult:
        return RepoDefaultClearResult(
            cleared=bool(self.service.clear_repo_default(repo=request.repo))
        )

    async def list_repo_defaults(self, request: EmptyRequest) -> RepoDefaultListResult:
        del request
        return RepoDefaultListResult(items=self.service.list_repo_defaults())


class SourcesClientAdapter:
    def __init__(self, service: Any) -> None:
        self.service = service

    async def add(self, request: SourceAddRequest) -> SourceInfo:
        return self.service.add_source(
            pot_id=request.pot_id,
            kind=request.kind,
            location=request.location,
            name=request.name,
        )

    async def register_repo(
        self, request: RegisterRepoSourceRequest
    ) -> RegisterRepoSourceResult:
        from potpie.cli.repo_location import repo_identity_key

        repo_identity = repo_identity_key(request.location)
        if not repo_identity:
            raise ValueError("Could not resolve the repository identity.")
        default_method = getattr(self.service, "set_repo_default", None)
        if request.make_default and not callable(default_method):
            raise _missing("engine.sources.register_repo")
        list_sources = getattr(self.service, "list_sources", None)
        sources = list_sources(pot_id=request.pot_id) if callable(list_sources) else ()
        existing = next(
            (
                source
                for source in sources
                if source.kind == "repo"
                and repo_identity_key(source.location or source.name) == repo_identity
            ),
            None,
        )
        created = existing is None
        source = existing or self.service.add_source(
            pot_id=request.pot_id,
            kind="repo",
            location=request.location,
            name=request.name,
        )
        if request.make_default:
            assert callable(default_method)
            default_method(repo=repo_identity, pot_id=request.pot_id)
        return RegisterRepoSourceResult(
            source=source,
            repo_identity=repo_identity,
            created=created,
            default_bound=request.make_default,
        )

    async def list(self, request: SourceListRequest) -> SourceListResult:
        items = tuple(self.service.list_sources(pot_id=request.pot_id))
        return SourceListResult(items=items, count=len(items))

    async def status(self, request: SourceStatusRequest) -> SourceInfo:
        return self.service.source_status(
            pot_id=request.pot_id, source_id=request.source_id
        )

    async def remove(self, request: SourceRemoveRequest) -> OperationResult:
        self.service.remove_source(pot_id=request.pot_id, source_id=request.source_id)
        return OperationResult()


class ContextClientAdapter:
    def __init__(self, service: Any) -> None:
        self.service = service

    async def resolve(self, request: ResolveRequest) -> AgentEnvelope:
        return self.service.resolve(request)

    async def search(self, request: SearchRequest) -> AgentEnvelope:
        return self.service.search(request)

    async def record(self, request: RecordRequest) -> RecordReceipt:
        return self.service.record(request)

    async def status(self, request: EngineStatusRequest) -> EngineStatusReport:
        return self.service.status(request)


class GraphClientAdapter:
    def __init__(
        self,
        *,
        graph: Any = None,
        workbench: Any = None,
        backend: Any = None,
        nudge: Any = None,
    ) -> None:
        self.graph = graph
        self.workbench = workbench
        self.backend = backend
        self.nudge_service = nudge

    async def catalog(self, request: GraphCatalogRequest) -> GraphCatalogResult:
        return self.graph.catalog(request)

    async def describe(self, request: GraphDescribeRequest) -> dict[str, Any]:
        return self.graph.describe(request)

    async def read(self, request: GraphReadRequest) -> GraphReadResult:
        return self.graph.read(request)

    async def search_entities(
        self, request: GraphEntitySearchRequest
    ) -> GraphEntitySearchResult:
        return self.graph.search_entities(request)

    async def status(self, request: GraphStatusRequest) -> DataPlaneStatus:
        return self.graph.data_plane_status(request.pot_id)

    async def propose(self, request: GraphProposeRequest) -> GraphMutationProposal:
        return self.workbench.propose(
            request.payload,
            pot_id=request.pot_id,
            ttl_seconds=request.ttl_seconds,
        )

    async def commit(self, request: GraphCommitRequest) -> GraphMutationCommitResult:
        return self.workbench.commit(
            request.plan_id,
            pot_id=request.pot_id,
            approved_by=request.approved_by,
            verify=request.verify,
        )

    async def history(self, request: GraphHistoryRequest) -> GraphHistoryResult:
        return self.workbench.history(
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
        return self.workbench.quality(
            pot_id=request.pot_id, report=request.report, **dict(request.filters)
        )

    async def nudge(self, request: GraphNudgeRequest) -> GraphNudgeResult:
        return self.nudge_service.nudge(request)

    async def neighborhood(self, request: GraphNeighborhoodRequest) -> GraphSlice:
        return self.backend.inspection.neighborhood(
            pot_id=request.pot_id,
            entity_key=request.entity_key,
            depth=request.depth,
        )

    async def inbox_add(self, request: GraphInboxAddRequest) -> GraphInboxResult:
        return self.workbench.inbox_add(
            pot_id=request.pot_id,
            summary=request.summary,
            details=request.details,
            evidence=request.evidence,
            source_refs=request.source_refs,
            suspected_subgraphs=request.suspected_subgraphs,
            created_by=request.created_by,
        )

    async def inbox_list(self, request: GraphInboxListRequest) -> GraphInboxResult:
        return self.workbench.inbox_list(
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
        return self.workbench.inbox_show(pot_id=request.pot_id, item_id=request.item_id)

    async def inbox_claim(self, request: GraphInboxClaimRequest) -> GraphInboxResult:
        return self.workbench.inbox_claim(
            pot_id=request.pot_id,
            item_id=request.item_id,
            claimed_by=request.claimed_by,
        )

    async def inbox_close(self, request: GraphInboxCloseRequest) -> GraphInboxResult:
        method_name = {
            "mark-applied": "inbox_mark_applied",
            "mark-rejected": "inbox_mark_rejected",
            "close": "inbox_close",
        }[request.action]
        values: dict[str, Any] = {
            "pot_id": request.pot_id,
            "item_id": request.item_id,
            "closed_by": request.closed_by,
        }
        if request.action == "mark-applied":
            values.update(
                linked_plan_id=request.linked_plan_id,
                linked_mutation_id=request.linked_mutation_id,
            )
        elif request.action == "mark-rejected":
            values["rejection_reason"] = request.rejection_reason or ""
        else:
            values.update(
                linked_plan_id=request.linked_plan_id,
                linked_mutation_id=request.linked_mutation_id,
                rejection_reason=request.rejection_reason,
            )
        return getattr(self.workbench, method_name)(**values)

    async def snapshot_export(
        self, request: GraphSnapshotExportRequest
    ) -> SnapshotManifest:
        return self.backend.snapshot.export(
            pot_id=request.pot_id, destination=request.destination
        )

    async def snapshot_import(
        self, request: GraphSnapshotImportRequest
    ) -> SnapshotManifest:
        return self.backend.snapshot.import_(
            pot_id=request.pot_id, source=request.source
        )

    async def repair(self, request: GraphRepairRequest) -> RepairReport:
        return self.backend.analytics.repair(request.pot_id, targets=request.targets)

    async def backend_info(self, request: GraphBackendInfoRequest) -> GraphBackendInfo:
        del request
        if self.backend is None:
            return GraphBackendInfo(profile="memory", capabilities=())
        caps = self.backend.capabilities()
        return GraphBackendInfo(
            profile=self.backend.profile,
            capabilities=tuple(caps.implemented()),
        )


class LedgerClientAdapter:
    def __init__(self, service: Any) -> None:
        self.service = service

    async def status(self, request: LedgerStatusRequest) -> LedgerHealth:
        del request
        return self.service.status()

    async def sources(self, request: LedgerSourcesRequest) -> LedgerSourcesResult:
        items = tuple(self.service.sources(pot_id=request.pot_id))
        return LedgerSourcesResult(items=items, count=len(items))

    async def query(self, request: LedgerQueryRequest) -> LedgerPage:
        return self.service.query(
            pot_id=request.pot_id,
            source_id=request.source_id,
            kind=request.kind,
            since=request.since,
            until=request.until,
            limit=request.limit,
        )

    async def pull(self, request: LedgerPullRequest) -> LedgerPage:
        return self.service.pull(
            pot_id=request.pot_id,
            source_id=request.source_id,
            limit=request.limit,
        )


class TestEngineClient:
    def __init__(
        self,
        *,
        context: ContextClient | None = None,
        pots: PotsClient | None = None,
        sources: SourcesClient | None = None,
        graph: GraphClient | None = None,
        ledger: LedgerClient | None = None,
        timeline: TimelineClient | None = None,
        provision: ProvisionClient | None = None,
    ) -> None:
        missing = _MissingCapability()
        self.context = context or cast(ContextClient, missing)
        self.pots = pots or cast(PotsClient, missing)
        self.sources = sources or cast(SourcesClient, missing)
        self.graph = graph or cast(GraphClient, missing)
        self.ledger = ledger or cast(LedgerClient, missing)
        self.timeline = timeline or cast(TimelineClient, missing)
        self.provision = provision or cast(ProvisionClient, missing)

    async def aclose(self) -> None:
        return None


class _MissingCapability:
    def __getattr__(self, name: str) -> Any:
        raise _missing(f"test_engine.{name}")


def runtime_from_services(
    *,
    pots: Any = None,
    graph: Any = None,
    graph_workbench: Any = None,
    backend: Any = None,
    nudge: Any = None,
    context: Any = None,
    ledger: Any = None,
    daemon: Any = None,
    config: Any = None,
    data_dir: Path | None = None,
) -> PotpieRuntime:
    pots_client = PotsClientAdapter(pots) if pots is not None else None
    engine = TestEngineClient(
        context=ContextClientAdapter(context) if context is not None else None,
        pots=pots_client,
        sources=SourcesClientAdapter(pots) if pots is not None else None,
        graph=GraphClientAdapter(
            graph=graph,
            workbench=graph_workbench,
            backend=backend,
            nudge=nudge,
        )
        if any(value is not None for value in (graph, graph_workbench, backend, nudge))
        else None,
        ledger=LedgerClientAdapter(ledger) if ledger is not None else None,
    )
    root = data_dir or Path("/tmp/potpie-test-runtime")
    credentials = InMemoryCredentialStore()
    runtime = PotpieRuntime(
        settings=ProductSettings(
            data_dir=root,
            runtime_mode="in-process",
            backend="in_memory",
        ),
        engine=engine,
        auth=AccountAuthService(credentials),
        integrations=IntegrationAuthService(credentials),
        config=config or ProductConfigService(root),
        skills=create_skill_service(data_dir=root),
        installer=LocalInstaller(),
        daemon=cast(Daemon, daemon) if daemon is not None else Daemon(home=root),
    )
    structural_engine: EngineClient = engine
    assert structural_engine is runtime.engine
    return runtime


__all__ = ["runtime_from_services"]
