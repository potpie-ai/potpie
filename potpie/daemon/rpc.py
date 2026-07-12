"""Protocol-v1 typed, allowlisted RPC registry for engine operations only."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import TypeAdapter, ValidationError

from potpie_context_engine import EngineClient
from potpie_context_engine.contracts import (
    AgentEnvelope,
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
    RecordRequest,
    RepairReport,
    ResolveRequest,
    SearchRequest,
    SourceAddRequest,
    SourceInfo,
    SourceListRequest,
    SourceListResult,
    SourceRemoveRequest,
    SourceStatusRequest,
    SnapshotManifest,
    TimelineRecentRequest,
)
from potpie_context_engine.domain.errors import CapabilityNotImplemented, PotNotFound

RPC_PROTOCOL_VERSION = "1"


@dataclass(frozen=True, slots=True)
class RpcMethodSpec:
    method: str
    request_type: Any
    result_type: Any

    def __post_init__(self) -> None:
        if not self.method.startswith("engine."):
            raise ValueError("RPC methods must be namespaced under engine.*")

    @property
    def request_adapter(self) -> TypeAdapter[Any]:
        return TypeAdapter(self.request_type)

    @property
    def result_adapter(self) -> TypeAdapter[Any]:
        return TypeAdapter(self.result_type)

    async def invoke(self, engine: EngineClient, params: Any) -> Any:
        request = self.request_adapter.validate_python(params)
        _, surface, operation = self.method.split(".", 2)
        handler = getattr(getattr(engine, surface), operation)
        result = await handler(request)
        return self.result_adapter.dump_python(result, mode="json")


class EngineRpcRegistry:
    def __init__(self, specs: tuple[RpcMethodSpec, ...]) -> None:
        self._specs = {spec.method: spec for spec in specs}
        if len(self._specs) != len(specs):
            raise ValueError("duplicate engine RPC method")

    def get(self, method: str) -> RpcMethodSpec:
        try:
            return self._specs[method]
        except KeyError as exc:
            raise KeyError(f"unknown engine RPC method {method!r}") from exc

    def methods(self) -> tuple[str, ...]:
        return tuple(sorted(self._specs))

    def encode_request(self, method: str, request: Any) -> Any:
        return self.get(method).request_adapter.dump_python(request, mode="json")

    def decode_result(self, method: str, result: Any) -> Any:
        return self.get(method).result_adapter.validate_python(result)


ENGINE_RPC_REGISTRY = EngineRpcRegistry(
    (
        RpcMethodSpec("engine.context.resolve", ResolveRequest, AgentEnvelope),
        RpcMethodSpec("engine.context.search", SearchRequest, AgentEnvelope),
        RpcMethodSpec("engine.context.record", RecordRequest, RecordReceipt),
        RpcMethodSpec("engine.context.status", EngineStatusRequest, EngineStatusReport),
        RpcMethodSpec("engine.pots.list", EmptyRequest, PotListResult),
        RpcMethodSpec("engine.pots.info", PotInfoRequest, PotInfo | None),
        RpcMethodSpec("engine.pots.create", PotCreateRequest, PotInfo),
        RpcMethodSpec("engine.pots.use", PotUseRequest, PotInfo),
        RpcMethodSpec("engine.pots.rename", PotRenameRequest, PotInfo),
        RpcMethodSpec("engine.pots.reset", PotResetRequest, PotInfo),
        RpcMethodSpec("engine.pots.archive", PotArchiveRequest, PotInfo),
        RpcMethodSpec(
            "engine.pots.repo_default", RepoDefaultGetRequest, RepoDefaultResult
        ),
        RpcMethodSpec(
            "engine.pots.set_repo_default", RepoDefaultSetRequest, OperationResult
        ),
        RpcMethodSpec(
            "engine.pots.clear_repo_default",
            RepoDefaultClearRequest,
            RepoDefaultClearResult,
        ),
        RpcMethodSpec(
            "engine.pots.list_repo_defaults", EmptyRequest, RepoDefaultListResult
        ),
        RpcMethodSpec("engine.sources.add", SourceAddRequest, SourceInfo),
        RpcMethodSpec("engine.sources.list", SourceListRequest, SourceListResult),
        RpcMethodSpec("engine.sources.status", SourceStatusRequest, SourceInfo),
        RpcMethodSpec("engine.sources.remove", SourceRemoveRequest, OperationResult),
        RpcMethodSpec("engine.graph.catalog", GraphCatalogRequest, GraphCatalogResult),
        RpcMethodSpec("engine.graph.describe", GraphDescribeRequest, dict[str, Any]),
        RpcMethodSpec("engine.graph.read", GraphReadRequest, GraphReadResult),
        RpcMethodSpec(
            "engine.graph.search_entities",
            GraphEntitySearchRequest,
            GraphEntitySearchResult,
        ),
        RpcMethodSpec("engine.graph.status", GraphStatusRequest, DataPlaneStatus),
        RpcMethodSpec(
            "engine.graph.propose", GraphProposeRequest, GraphMutationProposal
        ),
        RpcMethodSpec(
            "engine.graph.commit", GraphCommitRequest, GraphMutationCommitResult
        ),
        RpcMethodSpec("engine.graph.history", GraphHistoryRequest, GraphHistoryResult),
        RpcMethodSpec("engine.graph.quality", GraphQualityRequest, GraphQualityResult),
        RpcMethodSpec("engine.graph.nudge", GraphNudgeRequest, GraphNudgeResult),
        RpcMethodSpec(
            "engine.graph.neighborhood", GraphNeighborhoodRequest, GraphSlice
        ),
        RpcMethodSpec("engine.graph.inbox_add", GraphInboxAddRequest, GraphInboxResult),
        RpcMethodSpec(
            "engine.graph.inbox_list", GraphInboxListRequest, GraphInboxResult
        ),
        RpcMethodSpec(
            "engine.graph.inbox_show", GraphInboxItemRequest, GraphInboxResult
        ),
        RpcMethodSpec(
            "engine.graph.inbox_claim", GraphInboxClaimRequest, GraphInboxResult
        ),
        RpcMethodSpec(
            "engine.graph.inbox_close", GraphInboxCloseRequest, GraphInboxResult
        ),
        RpcMethodSpec(
            "engine.graph.snapshot_export",
            GraphSnapshotExportRequest,
            SnapshotManifest,
        ),
        RpcMethodSpec(
            "engine.graph.snapshot_import",
            GraphSnapshotImportRequest,
            SnapshotManifest,
        ),
        RpcMethodSpec("engine.graph.repair", GraphRepairRequest, RepairReport),
        RpcMethodSpec(
            "engine.graph.backend_info", GraphBackendInfoRequest, GraphBackendInfo
        ),
        RpcMethodSpec("engine.ledger.status", LedgerStatusRequest, LedgerHealth),
        RpcMethodSpec(
            "engine.ledger.sources", LedgerSourcesRequest, LedgerSourcesResult
        ),
        RpcMethodSpec("engine.ledger.query", LedgerQueryRequest, LedgerPage),
        RpcMethodSpec("engine.ledger.pull", LedgerPullRequest, LedgerPage),
        RpcMethodSpec("engine.timeline.recent", TimelineRecentRequest, GraphReadResult),
        RpcMethodSpec(
            "engine.provision.inspect", ProvisionInspectRequest, ProvisionPlan
        ),
        RpcMethodSpec("engine.provision.apply", ProvisionApplyRequest, ProvisionReport),
    )
)


async def dispatch_rpc(engine: EngineClient, payload: Any) -> dict[str, Any]:
    request_id = _request_id(payload)
    if not isinstance(payload, dict):
        return rpc_failure(
            request_id,
            code="RPC_INVALID_REQUEST",
            message="RPC request must be a JSON object.",
        )
    version = payload.get("protocol_version")
    if version != RPC_PROTOCOL_VERSION:
        return rpc_failure(
            request_id,
            code="RPC_PROTOCOL_MISMATCH",
            message="The daemon RPC protocol is incompatible with this client.",
            details={"expected": RPC_PROTOCOL_VERSION, "actual": version},
            retryable=True,
        )
    method = payload.get("method")
    if not isinstance(method, str):
        return rpc_failure(
            request_id,
            code="RPC_INVALID_REQUEST",
            message="RPC method must be a string.",
        )
    try:
        spec = ENGINE_RPC_REGISTRY.get(method)
    except KeyError:
        return rpc_failure(
            request_id,
            code="RPC_METHOD_NOT_FOUND",
            message=f"Unknown engine RPC method {method!r}.",
        )
    try:
        result = await spec.invoke(engine, payload.get("params") or {})
    except ValidationError as exc:
        return rpc_failure(
            request_id,
            code="RPC_INVALID_PARAMS",
            message="RPC parameters failed schema validation.",
            details={"errors": exc.errors(include_url=False)},
        )
    except CapabilityNotImplemented as exc:
        return rpc_failure(
            request_id,
            code="ENGINE_CAPABILITY_NOT_IMPLEMENTED",
            message=str(exc),
            details={"capability": exc.capability, "detail": exc.detail},
        )
    except PotNotFound as exc:
        return rpc_failure(
            request_id,
            code="ENGINE_POT_NOT_FOUND",
            message=str(exc),
        )
    except ValueError as exc:
        return rpc_failure(
            request_id,
            code="ENGINE_VALIDATION_ERROR",
            message=str(exc),
            details={"detail": getattr(exc, "detail", None)},
        )
    except Exception as exc:
        _capture_unexpected_daemon_error(exc)
        return rpc_failure(
            request_id,
            code="ENGINE_INTERNAL_ERROR",
            message="An internal engine error occurred.",
        )
    return {
        "protocol_version": RPC_PROTOCOL_VERSION,
        "request_id": request_id,
        "ok": True,
        "result": result,
    }


def rpc_failure(
    request_id: str | None,
    *,
    code: str,
    message: str,
    details: dict[str, Any] | None = None,
    retryable: bool = False,
) -> dict[str, Any]:
    return {
        "protocol_version": RPC_PROTOCOL_VERSION,
        "request_id": request_id,
        "ok": False,
        "error": {
            "code": code,
            "message": message,
            "details": details or {},
            "retryable": retryable,
        },
    }


def _request_id(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return None
    value = payload.get("request_id")
    return value if isinstance(value, str) and value else None


def _capture_unexpected_daemon_error(exc: Exception) -> None:
    try:
        from potpie.daemon.telemetry.sentry_runtime import (
            capture_unexpected_daemon_error,
        )

        capture_unexpected_daemon_error(
            exc,
            error_code="ENGINE_INTERNAL_ERROR",
            error_kind="unexpected",
        )
    except Exception:
        return


__all__ = [
    "ENGINE_RPC_REGISTRY",
    "RPC_PROTOCOL_VERSION",
    "EngineRpcRegistry",
    "RpcMethodSpec",
    "dispatch_rpc",
    "rpc_failure",
]
