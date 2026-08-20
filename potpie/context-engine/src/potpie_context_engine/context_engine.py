"""Public, context-bound asynchronous Context Engine façade."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Literal, Protocol, TypeVar

from potpie_context_engine.outcomes import (
    EngineLifecycleError,
    Failure,
    Outcome,
    Success,
)
from potpie_context_engine.requests import (
    CatalogRequest,
    CommitRequest,
    DataPlaneStatusRequest,
    DescribeRequest,
    EngineRequest,
    ExportSnapshotRequest,
    HistoryRequest,
    ImportSnapshotRequest,
    InboxAddRequest,
    InboxClaimRequest,
    InboxCloseRequest,
    InboxListRequest,
    InboxMarkAppliedRequest,
    InboxMarkRejectedRequest,
    InboxShowRequest,
    InspectRequest,
    MutateRequest,
    NeighborhoodRequest,
    NudgeRequest,
    ProcessingStatusRequest,
    ProposeRequest,
    QualityRequest,
    ReadRequest,
    RecordRequest,
    RepairRequest,
    ResolveRequest,
    SearchEntitiesRequest,
    SearchRequest,
    SubmitArtifactRequest,
    SubmitEventRequest,
)


@dataclass(frozen=True, slots=True)
class ContextIdentity:
    """Opaque logical isolation identity permanently bound to one engine."""

    value: str

    def __post_init__(self) -> None:
        if not self.value.strip():
            raise ValueError("context identity must not be empty")


@dataclass(frozen=True, slots=True)
class EngineConfig:
    """Explicit engine-owned configuration supplied by a compatible host."""

    data_dir: str | None = None
    values: Mapping[str, object] = field(default_factory=dict)


ResourceOwnership = Literal["borrowed", "transferred"]


@dataclass(frozen=True, slots=True)
class EngineResource:
    """Explicit lifecycle declaration for one resource-bearing dependency."""

    name: str
    ownership: ResourceOwnership
    close: Callable[[], Awaitable[None]] | None = None

    def __post_init__(self) -> None:
        if self.ownership == "transferred" and self.close is None:
            raise ValueError("transferred engine resources require an async close hook")


class ContextOperations(Protocol):
    async def resolve(
        self, context: ContextIdentity, request: ResolveRequest
    ) -> object | Outcome[object]: ...

    async def search(
        self, context: ContextIdentity, request: SearchRequest
    ) -> object | Outcome[object]: ...

    async def record(
        self, context: ContextIdentity, request: RecordRequest
    ) -> object | Outcome[object]: ...

    async def data_plane_status(
        self, context: ContextIdentity, request: DataPlaneStatusRequest
    ) -> object | Outcome[object]: ...


class GraphOperations(Protocol):
    async def catalog(
        self, context: ContextIdentity, request: CatalogRequest
    ) -> object | Outcome[object]: ...

    async def describe(
        self, context: ContextIdentity, request: DescribeRequest
    ) -> object | Outcome[object]: ...

    async def read(
        self, context: ContextIdentity, request: ReadRequest
    ) -> object | Outcome[object]: ...

    async def search_entities(
        self, context: ContextIdentity, request: SearchEntitiesRequest
    ) -> object | Outcome[object]: ...

    async def mutate(
        self, context: ContextIdentity, request: MutateRequest
    ) -> object | Outcome[object]: ...

    async def neighborhood(
        self, context: ContextIdentity, request: NeighborhoodRequest
    ) -> object | Outcome[object]: ...

    async def inspect(
        self, context: ContextIdentity, request: InspectRequest
    ) -> object | Outcome[object]: ...

    async def export_snapshot(
        self, context: ContextIdentity, request: ExportSnapshotRequest
    ) -> object | Outcome[object]: ...

    async def import_snapshot(
        self, context: ContextIdentity, request: ImportSnapshotRequest
    ) -> object | Outcome[object]: ...

    async def repair(
        self, context: ContextIdentity, request: RepairRequest
    ) -> object | Outcome[object]: ...


class WorkbenchOperations(Protocol):
    async def propose(
        self, context: ContextIdentity, request: ProposeRequest
    ) -> object | Outcome[object]: ...

    async def commit(
        self, context: ContextIdentity, request: CommitRequest
    ) -> object | Outcome[object]: ...

    async def history(
        self, context: ContextIdentity, request: HistoryRequest
    ) -> object | Outcome[object]: ...

    async def quality(
        self, context: ContextIdentity, request: QualityRequest
    ) -> object | Outcome[object]: ...

    async def inbox_add(
        self, context: ContextIdentity, request: InboxAddRequest
    ) -> object | Outcome[object]: ...

    async def inbox_list(
        self, context: ContextIdentity, request: InboxListRequest
    ) -> object | Outcome[object]: ...

    async def inbox_show(
        self, context: ContextIdentity, request: InboxShowRequest
    ) -> object | Outcome[object]: ...

    async def inbox_claim(
        self, context: ContextIdentity, request: InboxClaimRequest
    ) -> object | Outcome[object]: ...

    async def inbox_mark_applied(
        self, context: ContextIdentity, request: InboxMarkAppliedRequest
    ) -> object | Outcome[object]: ...

    async def inbox_mark_rejected(
        self, context: ContextIdentity, request: InboxMarkRejectedRequest
    ) -> object | Outcome[object]: ...

    async def inbox_close(
        self, context: ContextIdentity, request: InboxCloseRequest
    ) -> object | Outcome[object]: ...


class IngestionOperations(Protocol):
    async def submit_event(
        self, context: ContextIdentity, request: SubmitEventRequest
    ) -> object | Outcome[object]: ...

    async def submit_artifact(
        self, context: ContextIdentity, request: SubmitArtifactRequest
    ) -> object | Outcome[object]: ...

    async def processing_status(
        self, context: ContextIdentity, request: ProcessingStatusRequest
    ) -> object | Outcome[object]: ...


class NudgeOperations(Protocol):
    async def nudge(
        self, context: ContextIdentity, request: NudgeRequest
    ) -> object | Outcome[object]: ...


@dataclass(frozen=True, slots=True)
class EngineDependencies:
    """Focused engine-owned operation groups and resource ownership metadata."""

    context: ContextOperations
    graph: GraphOperations
    workbench: WorkbenchOperations
    ingestion: IngestionOperations
    nudge: NudgeOperations
    resources: tuple[EngineResource, ...] = ()


RequestT = TypeVar("RequestT", bound=EngineRequest)
Operation = Callable[[ContextIdentity, RequestT], Awaitable[object | Outcome[object]]]

_FORBIDDEN_SELECTOR_FIELDS = frozenset({"pot_id", "active_pot", "context_selector"})


class ContextEngine:
    """Finite asynchronous façade bound to one immutable context identity."""

    def __init__(
        self,
        *,
        context: ContextIdentity,
        config: EngineConfig,
        dependencies: EngineDependencies,
    ) -> None:
        self._context = context
        self._config = config
        self._dependencies = dependencies
        self._closed = False
        self._close_lock = asyncio.Lock()

    @property
    def context(self) -> ContextIdentity:
        return self._context

    @property
    def config(self) -> EngineConfig:
        return self._config

    @property
    def is_closed(self) -> bool:
        return self._closed

    async def __aenter__(self) -> ContextEngine:
        if self._closed:
            raise RuntimeError("a closed ContextEngine cannot be re-entered")
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.close()

    async def close(self) -> Outcome[None]:
        async with self._close_lock:
            if self._closed:
                return Success(None)
            self._closed = True
            failures: list[dict[str, str]] = []
            for resource in reversed(self._dependencies.resources):
                if resource.ownership != "transferred" or resource.close is None:
                    continue
                try:
                    await resource.close()
                except Exception as exc:  # cleanup must continue for later resources
                    failures.append(
                        {"resource": resource.name, "error_type": type(exc).__name__}
                    )
            if failures:
                return Failure(
                    EngineLifecycleError(
                        code="engine_close_failed",
                        message="one or more engine-owned resources failed to close",
                        details={"failures": tuple(failures)},
                        recommended_next_action="inspect resource cleanup logs",
                        retry_posture="safe",
                    )
                )
            return Success(None)

    async def _invoke(
        self,
        operation: str,
        handler: Operation[RequestT],
        request: RequestT,
    ) -> Outcome[object]:
        if self._closed:
            return Failure(
                EngineLifecycleError(
                    code="engine_closed",
                    message="the ContextEngine is closed",
                )
            )
        forbidden = sorted(_FORBIDDEN_SELECTOR_FIELDS.intersection(request.payload))
        if forbidden:
            from potpie_context_engine.outcomes import DomainError

            return Failure(
                DomainError(
                    code="context_selector_forbidden",
                    message="operation requests cannot override the bound context",
                    details={"operation": operation, "fields": tuple(forbidden)},
                )
            )
        result = await handler(self._context, request)
        if isinstance(result, (Success, Failure)):
            return result
        return Success(result)

    async def resolve(self, request: ResolveRequest) -> Outcome[object]:
        return await self._invoke("resolve", self._dependencies.context.resolve, request)

    async def search(self, request: SearchRequest) -> Outcome[object]:
        return await self._invoke("search", self._dependencies.context.search, request)

    async def record(self, request: RecordRequest) -> Outcome[object]:
        return await self._invoke("record", self._dependencies.context.record, request)

    async def data_plane_status(
        self, request: DataPlaneStatusRequest
    ) -> Outcome[object]:
        return await self._invoke(
            "data_plane_status", self._dependencies.context.data_plane_status, request
        )

    async def catalog(self, request: CatalogRequest) -> Outcome[object]:
        return await self._invoke("catalog", self._dependencies.graph.catalog, request)

    async def describe(self, request: DescribeRequest) -> Outcome[object]:
        return await self._invoke("describe", self._dependencies.graph.describe, request)

    async def read(self, request: ReadRequest) -> Outcome[object]:
        return await self._invoke("read", self._dependencies.graph.read, request)

    async def search_entities(
        self, request: SearchEntitiesRequest
    ) -> Outcome[object]:
        return await self._invoke(
            "search_entities", self._dependencies.graph.search_entities, request
        )

    async def mutate(self, request: MutateRequest) -> Outcome[object]:
        return await self._invoke("mutate", self._dependencies.graph.mutate, request)

    async def neighborhood(self, request: NeighborhoodRequest) -> Outcome[object]:
        return await self._invoke(
            "neighborhood", self._dependencies.graph.neighborhood, request
        )

    async def inspect(self, request: InspectRequest) -> Outcome[object]:
        return await self._invoke("inspect", self._dependencies.graph.inspect, request)

    async def export_snapshot(
        self, request: ExportSnapshotRequest
    ) -> Outcome[object]:
        return await self._invoke(
            "export_snapshot", self._dependencies.graph.export_snapshot, request
        )

    async def import_snapshot(
        self, request: ImportSnapshotRequest
    ) -> Outcome[object]:
        return await self._invoke(
            "import_snapshot", self._dependencies.graph.import_snapshot, request
        )

    async def repair(self, request: RepairRequest) -> Outcome[object]:
        return await self._invoke("repair", self._dependencies.graph.repair, request)

    async def propose(self, request: ProposeRequest) -> Outcome[object]:
        return await self._invoke(
            "propose", self._dependencies.workbench.propose, request
        )

    async def commit(self, request: CommitRequest) -> Outcome[object]:
        return await self._invoke("commit", self._dependencies.workbench.commit, request)

    async def history(self, request: HistoryRequest) -> Outcome[object]:
        return await self._invoke(
            "history", self._dependencies.workbench.history, request
        )

    async def quality(self, request: QualityRequest) -> Outcome[object]:
        return await self._invoke(
            "quality", self._dependencies.workbench.quality, request
        )

    async def inbox_add(self, request: InboxAddRequest) -> Outcome[object]:
        return await self._invoke(
            "inbox_add", self._dependencies.workbench.inbox_add, request
        )

    async def inbox_list(self, request: InboxListRequest) -> Outcome[object]:
        return await self._invoke(
            "inbox_list", self._dependencies.workbench.inbox_list, request
        )

    async def inbox_show(self, request: InboxShowRequest) -> Outcome[object]:
        return await self._invoke(
            "inbox_show", self._dependencies.workbench.inbox_show, request
        )

    async def inbox_claim(self, request: InboxClaimRequest) -> Outcome[object]:
        return await self._invoke(
            "inbox_claim", self._dependencies.workbench.inbox_claim, request
        )

    async def inbox_mark_applied(
        self, request: InboxMarkAppliedRequest
    ) -> Outcome[object]:
        return await self._invoke(
            "inbox_mark_applied",
            self._dependencies.workbench.inbox_mark_applied,
            request,
        )

    async def inbox_mark_rejected(
        self, request: InboxMarkRejectedRequest
    ) -> Outcome[object]:
        return await self._invoke(
            "inbox_mark_rejected",
            self._dependencies.workbench.inbox_mark_rejected,
            request,
        )

    async def inbox_close(self, request: InboxCloseRequest) -> Outcome[object]:
        return await self._invoke(
            "inbox_close", self._dependencies.workbench.inbox_close, request
        )

    async def submit_event(self, request: SubmitEventRequest) -> Outcome[object]:
        return await self._invoke(
            "submit_event", self._dependencies.ingestion.submit_event, request
        )

    async def submit_artifact(
        self, request: SubmitArtifactRequest
    ) -> Outcome[object]:
        return await self._invoke(
            "submit_artifact", self._dependencies.ingestion.submit_artifact, request
        )

    async def processing_status(
        self, request: ProcessingStatusRequest
    ) -> Outcome[object]:
        return await self._invoke(
            "processing_status",
            self._dependencies.ingestion.processing_status,
            request,
        )

    async def nudge(self, request: NudgeRequest) -> Outcome[object]:
        return await self._invoke("nudge", self._dependencies.nudge.nudge, request)


async def create_engine(
    *,
    context: ContextIdentity,
    config: EngineConfig,
    dependencies: EngineDependencies,
) -> Outcome[ContextEngine]:
    """Validate explicit composition and create one context-bound engine."""

    missing = tuple(
        name
        for name in ("context", "graph", "workbench", "ingestion", "nudge")
        if getattr(dependencies, name) is None
    )
    if missing:
        return Failure(
            EngineLifecycleError(
                code="engine_dependencies_missing",
                message="required Context Engine dependencies are missing",
                details={"dependencies": missing},
            )
        )
    return Success(ContextEngine(context=context, config=config, dependencies=dependencies))


__all__ = [
    "ContextEngine",
    "ContextIdentity",
    "ContextOperations",
    "EngineConfig",
    "EngineDependencies",
    "EngineResource",
    "GraphOperations",
    "IngestionOperations",
    "NudgeOperations",
    "ResourceOwnership",
    "WorkbenchOperations",
    "create_engine",
]
