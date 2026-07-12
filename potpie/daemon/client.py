"""Typed protocol-v1 engine client for the root-owned daemon transport."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx

from potpie_context_engine.contracts import (
    AgentEnvelope,
    BackendReadiness,
    EmptyRequest,
    EngineStatusRequest,
    EngineStatusReport,
    GraphCatalogResult,
    GraphCatalogRequest,
    GraphCommitRequest,
    GraphEntitySearchRequest,
    GraphHistoryRequest,
    GraphProposeRequest,
    GraphQualityRequest,
    GraphReadRequest,
    GraphStatusRequest,
    LedgerPullRequest,
    LedgerQueryRequest,
    LedgerSourcesRequest,
    LedgerStatusRequest,
    PotArchiveRequest,
    PotCreateRequest,
    PotInfoRequest,
    PotRenameRequest,
    PotResetRequest,
    PotUseRequest,
    ProvisionApplyRequest,
    ProvisionInspectRequest,
    RecordRequest,
    ResolveRequest,
    SearchRequest,
    SourceAddRequest,
    SourceListRequest,
    SourceRemoveRequest,
    SourceStatusRequest,
    TimelineRecentRequest,
)
from potpie_context_engine.contracts import (
    GraphEntitySearchResult,
    GraphHistoryResult,
    GraphMutationCommitResult,
    GraphMutationProposal,
    GraphQualityResult,
    GraphReadResult,
    LedgerHealth,
    LedgerPage,
    LedgerSourcesResult,
    OperationResult,
    PotInfo,
    PotListResult,
    ProvisionPlan,
    ProvisionReport,
    RecordReceipt,
    SourceInfo,
    SourceListResult,
)

from potpie.daemon.lifecycle import Daemon
from potpie.daemon.rpc import ENGINE_RPC_REGISTRY, RPC_PROTOCOL_VERSION
from potpie.runtime.errors import (
    DaemonRpcFailure,
    RpcProtocolMismatch,
    RuntimeDaemonUnavailable,
)


@dataclass(slots=True)
class DaemonRpcTransport:
    data_dir: Path
    timeout_s: float = 30.0
    daemon: Daemon = field(init=False)

    def __post_init__(self) -> None:
        self.daemon = Daemon(home=self.data_dir, in_process=False)

    async def call(self, method: str, request: Any) -> Any:
        discovery = self._discovery()
        request_id = str(uuid4())
        payload = {
            "protocol_version": RPC_PROTOCOL_VERSION,
            "request_id": request_id,
            "method": method,
            "params": ENGINE_RPC_REGISTRY.encode_request(method, request),
        }
        try:
            async with httpx.AsyncClient(timeout=self.timeout_s) as client:
                response = await client.post(
                    f"{discovery['base_url'].rstrip('/')}/rpc",
                    json=payload,
                    headers={"Authorization": f"Bearer {discovery['token']}"},
                )
        except httpx.RequestError as exc:
            raise RuntimeDaemonUnavailable(
                f"The Potpie daemon is not reachable: {exc}"
            ) from exc
        data = _response_json(response)
        response_version = data.get("protocol_version")
        raw_error = data.get("error")
        error: dict[str, Any] = raw_error if isinstance(raw_error, dict) else {}
        if (
            response_version != RPC_PROTOCOL_VERSION
            or error.get("code") == "RPC_PROTOCOL_MISMATCH"
        ):
            actual = None if response_version is None else str(response_version)
            raise RpcProtocolMismatch(actual=actual)
        if data.get("request_id") != request_id:
            raise DaemonRpcFailure(
                code="RPC_REQUEST_ID_MISMATCH",
                message="The daemon returned a response for another request.",
            )
        if response.status_code >= 400 or data.get("ok") is not True:
            raise DaemonRpcFailure(
                code=str(error.get("code") or "DAEMON_RPC_ERROR"),
                message=str(error.get("message") or "The daemon request failed."),
                details=error.get("details") or {},
                retryable=bool(error.get("retryable")),
            )
        return ENGINE_RPC_REGISTRY.decode_result(method, data.get("result"))

    async def health(self) -> dict[str, Any]:
        discovery = self._discovery()
        try:
            async with httpx.AsyncClient(timeout=self.timeout_s) as client:
                response = await client.get(
                    f"{discovery['base_url'].rstrip('/')}/healthz",
                    headers={"Authorization": f"Bearer {discovery['token']}"},
                )
        except httpx.RequestError as exc:
            raise RuntimeDaemonUnavailable() from exc
        return _response_json(response)

    def _discovery(self) -> dict[str, Any]:
        discovery = self.daemon.discovery()
        if discovery is None:
            raise RuntimeDaemonUnavailable()
        if not discovery.get("base_url") or not discovery.get("token"):
            raise RpcProtocolMismatch(actual=None)
        return dict(discovery)


@dataclass(slots=True)
class _ContextClient:
    rpc: DaemonRpcTransport

    async def resolve(self, request: ResolveRequest) -> AgentEnvelope:
        return await self.rpc.call("engine.context.resolve", request)

    async def search(self, request: SearchRequest) -> AgentEnvelope:
        return await self.rpc.call("engine.context.search", request)

    async def record(self, request: RecordRequest) -> RecordReceipt:
        return await self.rpc.call("engine.context.record", request)

    async def status(self, request: EngineStatusRequest) -> EngineStatusReport:
        return await self.rpc.call("engine.context.status", request)


@dataclass(slots=True)
class _PotsClient:
    rpc: DaemonRpcTransport

    async def list(self, request: EmptyRequest) -> PotListResult:
        return await self.rpc.call("engine.pots.list", request)

    async def info(self, request: PotInfoRequest) -> PotInfo | None:
        return await self.rpc.call("engine.pots.info", request)

    async def create(self, request: PotCreateRequest) -> PotInfo:
        return await self.rpc.call("engine.pots.create", request)

    async def use(self, request: PotUseRequest) -> PotInfo:
        return await self.rpc.call("engine.pots.use", request)

    async def rename(self, request: PotRenameRequest) -> PotInfo:
        return await self.rpc.call("engine.pots.rename", request)

    async def reset(self, request: PotResetRequest) -> PotInfo:
        return await self.rpc.call("engine.pots.reset", request)

    async def archive(self, request: PotArchiveRequest) -> PotInfo:
        return await self.rpc.call("engine.pots.archive", request)


@dataclass(slots=True)
class _SourcesClient:
    rpc: DaemonRpcTransport

    async def add(self, request: SourceAddRequest) -> SourceInfo:
        return await self.rpc.call("engine.sources.add", request)

    async def list(self, request: SourceListRequest) -> SourceListResult:
        return await self.rpc.call("engine.sources.list", request)

    async def status(self, request: SourceStatusRequest) -> SourceInfo:
        return await self.rpc.call("engine.sources.status", request)

    async def remove(self, request: SourceRemoveRequest) -> OperationResult:
        return await self.rpc.call("engine.sources.remove", request)


@dataclass(slots=True)
class _GraphClient:
    rpc: DaemonRpcTransport

    async def catalog(self, request: GraphCatalogRequest) -> GraphCatalogResult:
        return await self.rpc.call("engine.graph.catalog", request)

    async def read(self, request: GraphReadRequest) -> GraphReadResult:
        return await self.rpc.call("engine.graph.read", request)

    async def search_entities(
        self, request: GraphEntitySearchRequest
    ) -> GraphEntitySearchResult:
        return await self.rpc.call("engine.graph.search_entities", request)

    async def status(self, request: GraphStatusRequest) -> BackendReadiness:
        return await self.rpc.call("engine.graph.status", request)

    async def propose(self, request: GraphProposeRequest) -> GraphMutationProposal:
        return await self.rpc.call("engine.graph.propose", request)

    async def commit(self, request: GraphCommitRequest) -> GraphMutationCommitResult:
        return await self.rpc.call("engine.graph.commit", request)

    async def history(self, request: GraphHistoryRequest) -> GraphHistoryResult:
        return await self.rpc.call("engine.graph.history", request)

    async def quality(self, request: GraphQualityRequest) -> GraphQualityResult:
        return await self.rpc.call("engine.graph.quality", request)


@dataclass(slots=True)
class _LedgerClient:
    rpc: DaemonRpcTransport

    async def status(self, request: LedgerStatusRequest) -> LedgerHealth:
        return await self.rpc.call("engine.ledger.status", request)

    async def sources(self, request: LedgerSourcesRequest) -> LedgerSourcesResult:
        return await self.rpc.call("engine.ledger.sources", request)

    async def query(self, request: LedgerQueryRequest) -> LedgerPage:
        return await self.rpc.call("engine.ledger.query", request)

    async def pull(self, request: LedgerPullRequest) -> LedgerPage:
        return await self.rpc.call("engine.ledger.pull", request)


@dataclass(slots=True)
class _TimelineClient:
    rpc: DaemonRpcTransport

    async def recent(self, request: TimelineRecentRequest) -> GraphReadResult:
        return await self.rpc.call("engine.timeline.recent", request)


@dataclass(slots=True)
class _ProvisionClient:
    rpc: DaemonRpcTransport

    async def inspect(self, request: ProvisionInspectRequest) -> ProvisionPlan:
        return await self.rpc.call("engine.provision.inspect", request)

    async def apply(self, request: ProvisionApplyRequest) -> ProvisionReport:
        return await self.rpc.call("engine.provision.apply", request)


@dataclass(slots=True)
class DaemonEngineClient:
    rpc: DaemonRpcTransport
    context: _ContextClient = field(init=False)
    pots: _PotsClient = field(init=False)
    sources: _SourcesClient = field(init=False)
    graph: _GraphClient = field(init=False)
    ledger: _LedgerClient = field(init=False)
    timeline: _TimelineClient = field(init=False)
    provision: _ProvisionClient = field(init=False)

    def __post_init__(self) -> None:
        self.context = _ContextClient(self.rpc)
        self.pots = _PotsClient(self.rpc)
        self.sources = _SourcesClient(self.rpc)
        self.graph = _GraphClient(self.rpc)
        self.ledger = _LedgerClient(self.rpc)
        self.timeline = _TimelineClient(self.rpc)
        self.provision = _ProvisionClient(self.rpc)

    async def aclose(self) -> None:
        return None


def _response_json(response: httpx.Response) -> dict[str, Any]:
    try:
        data = response.json()
    except ValueError as exc:
        raise RuntimeDaemonUnavailable(
            f"The Potpie daemon returned non-JSON ({response.status_code})."
        ) from exc
    if not isinstance(data, dict):
        raise RuntimeDaemonUnavailable("The Potpie daemon returned invalid JSON.")
    return data


__all__ = ["DaemonEngineClient", "DaemonRpcTransport"]
