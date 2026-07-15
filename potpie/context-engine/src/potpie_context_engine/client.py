"""Asynchronous client protocol shared by local and daemon implementations."""

from __future__ import annotations

from typing import Any, Protocol

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
    RecordReceipt,
    RepairReport,
    RecordRequest,
    RegisterRepoSourceRequest,
    RegisterRepoSourceResult,
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


class ContextClient(Protocol):
    async def resolve(self, request: ResolveRequest) -> AgentEnvelope: ...

    async def search(self, request: SearchRequest) -> AgentEnvelope: ...

    async def record(self, request: RecordRequest) -> RecordReceipt: ...

    async def status(self, request: EngineStatusRequest) -> EngineStatusReport: ...


class PotsClient(Protocol):
    async def list(self, request: EmptyRequest) -> PotListResult: ...

    async def info(self, request: PotInfoRequest) -> PotInfo | None: ...

    async def create(self, request: PotCreateRequest) -> PotInfo: ...

    async def use(self, request: PotUseRequest) -> PotInfo: ...

    async def rename(self, request: PotRenameRequest) -> PotInfo: ...

    async def reset(self, request: PotResetRequest) -> PotInfo: ...

    async def archive(self, request: PotArchiveRequest) -> PotInfo: ...

    async def repo_default(
        self, request: RepoDefaultGetRequest
    ) -> RepoDefaultResult: ...

    async def set_repo_default(
        self, request: RepoDefaultSetRequest
    ) -> OperationResult: ...

    async def clear_repo_default(
        self, request: RepoDefaultClearRequest
    ) -> RepoDefaultClearResult: ...

    async def list_repo_defaults(
        self, request: EmptyRequest
    ) -> RepoDefaultListResult: ...


class SourcesClient(Protocol):
    async def add(self, request: SourceAddRequest) -> SourceInfo: ...

    async def register_repo(
        self, request: RegisterRepoSourceRequest
    ) -> RegisterRepoSourceResult: ...

    async def list(self, request: SourceListRequest) -> SourceListResult: ...

    async def status(self, request: SourceStatusRequest) -> SourceInfo: ...

    async def remove(self, request: SourceRemoveRequest) -> OperationResult: ...


class GraphClient(Protocol):
    async def catalog(self, request: GraphCatalogRequest) -> GraphCatalogResult: ...

    async def describe(self, request: GraphDescribeRequest) -> dict[str, Any]: ...

    async def read(self, request: GraphReadRequest) -> GraphReadResult: ...

    async def search_entities(
        self, request: GraphEntitySearchRequest
    ) -> GraphEntitySearchResult: ...

    async def status(self, request: GraphStatusRequest) -> DataPlaneStatus: ...

    async def propose(self, request: GraphProposeRequest) -> GraphMutationProposal: ...

    async def commit(
        self, request: GraphCommitRequest
    ) -> GraphMutationCommitResult: ...

    async def history(self, request: GraphHistoryRequest) -> GraphHistoryResult: ...

    async def quality(self, request: GraphQualityRequest) -> GraphQualityResult: ...

    async def nudge(self, request: GraphNudgeRequest) -> GraphNudgeResult: ...

    async def neighborhood(self, request: GraphNeighborhoodRequest) -> GraphSlice: ...

    async def inbox_add(self, request: GraphInboxAddRequest) -> GraphInboxResult: ...

    async def inbox_list(self, request: GraphInboxListRequest) -> GraphInboxResult: ...

    async def inbox_show(self, request: GraphInboxItemRequest) -> GraphInboxResult: ...

    async def inbox_claim(
        self, request: GraphInboxClaimRequest
    ) -> GraphInboxResult: ...

    async def inbox_close(
        self, request: GraphInboxCloseRequest
    ) -> GraphInboxResult: ...

    async def snapshot_export(
        self, request: GraphSnapshotExportRequest
    ) -> SnapshotManifest: ...

    async def snapshot_import(
        self, request: GraphSnapshotImportRequest
    ) -> SnapshotManifest: ...

    async def repair(self, request: GraphRepairRequest) -> RepairReport: ...

    async def backend_info(
        self, request: GraphBackendInfoRequest
    ) -> GraphBackendInfo: ...


class LedgerClient(Protocol):
    async def status(self, request: LedgerStatusRequest) -> LedgerHealth: ...

    async def sources(self, request: LedgerSourcesRequest) -> LedgerSourcesResult: ...

    async def query(self, request: LedgerQueryRequest) -> LedgerPage: ...

    async def pull(self, request: LedgerPullRequest) -> LedgerPage: ...


class TimelineClient(Protocol):
    async def recent(self, request: TimelineRecentRequest) -> GraphReadResult: ...


class ProvisionClient(Protocol):
    async def inspect(self, request: ProvisionInspectRequest) -> ProvisionPlan: ...

    async def apply(self, request: ProvisionApplyRequest) -> ProvisionReport: ...


class EngineClient(Protocol):
    @property
    def context(self) -> ContextClient: ...

    @property
    def pots(self) -> PotsClient: ...

    @property
    def sources(self) -> SourcesClient: ...

    @property
    def graph(self) -> GraphClient: ...

    @property
    def ledger(self) -> LedgerClient: ...

    @property
    def timeline(self) -> TimelineClient: ...

    @property
    def provision(self) -> ProvisionClient: ...

    async def aclose(self) -> None: ...


__all__ = [
    "ContextClient",
    "EngineClient",
    "GraphClient",
    "LedgerClient",
    "PotsClient",
    "ProvisionClient",
    "SourcesClient",
    "TimelineClient",
]
