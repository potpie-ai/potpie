"""Synchronous CLI view over the asynchronous typed engine client."""

# mypy: disable-error-code=no-untyped-def

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from types import SimpleNamespace
from collections.abc import Coroutine
from typing import Any, TypeVar

from potpie.runtime.composition import PotpieRuntime
from potpie.runtime.contracts import (
    EmptyRequest,
    GraphCommitRequest,
    GraphBackendInfoRequest,
    GraphDescribeRequest,
    GraphHistoryRequest,
    GraphInboxAddRequest,
    GraphInboxClaimRequest,
    GraphInboxCloseRequest,
    GraphInboxItemRequest,
    GraphInboxListRequest,
    GraphNeighborhoodRequest,
    GraphNudgeRequest,
    GraphProposeRequest,
    GraphQualityRequest,
    GraphRepairRequest,
    GraphSnapshotExportRequest,
    GraphSnapshotImportRequest,
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
    RepoDefaultClearRequest,
    RepoDefaultGetRequest,
    RepoDefaultSetRequest,
    SourceAddRequest,
    SourceListRequest,
    SourceRemoveRequest,
    SourceStatusRequest,
)

T = TypeVar("T")


def await_engine(awaitable: Coroutine[Any, Any, T]) -> T:
    """Run one engine call from a synchronous Typer handler."""

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(awaitable)
    raise RuntimeError("synchronous CLI handlers cannot run inside an event loop")


@dataclass(slots=True)
class _ContextView:
    runtime: PotpieRuntime

    def resolve(self, request):
        return await_engine(self.runtime.engine.context.resolve(request))

    def search(self, request):
        return await_engine(self.runtime.engine.context.search(request))

    def record(self, request):
        return await_engine(self.runtime.engine.context.record(request))

    def status(self, request):
        return await_engine(self.runtime.engine.context.status(request))


@dataclass(slots=True)
class _PotsView:
    runtime: PotpieRuntime

    def list_pots(self):
        return list(await_engine(self.runtime.engine.pots.list(EmptyRequest())).items)

    def active_pot(self):
        return await_engine(self.runtime.engine.pots.info(PotInfoRequest()))

    def create_pot(self, *, name: str, repo: str | None = None, use: bool = False):
        return await_engine(
            self.runtime.engine.pots.create(
                PotCreateRequest(name=name, repo=repo, use=use)
            )
        )

    def use_pot(self, *, ref: str):
        return await_engine(self.runtime.engine.pots.use(PotUseRequest(ref=ref)))

    def rename_pot(self, *, ref: str, new_name: str):
        return await_engine(
            self.runtime.engine.pots.rename(
                PotRenameRequest(ref=ref, new_name=new_name)
            )
        )

    def reset_pot(self, *, ref: str, confirm: bool = False):
        return await_engine(
            self.runtime.engine.pots.reset(PotResetRequest(ref=ref, confirm=confirm))
        )

    def archive_pot(self, *, ref: str):
        return await_engine(
            self.runtime.engine.pots.archive(PotArchiveRequest(ref=ref))
        )

    def add_source(
        self, *, pot_id: str, kind: str, location: str, name: str | None = None
    ):
        return await_engine(
            self.runtime.engine.sources.add(
                SourceAddRequest(pot_id=pot_id, kind=kind, location=location, name=name)
            )
        )

    def list_sources(self, *, pot_id: str):
        return list(
            await_engine(
                self.runtime.engine.sources.list(SourceListRequest(pot_id=pot_id))
            ).items
        )

    def source_status(self, *, pot_id: str, source_id: str):
        return await_engine(
            self.runtime.engine.sources.status(
                SourceStatusRequest(pot_id=pot_id, source_id=source_id)
            )
        )

    def remove_source(self, *, pot_id: str, source_id: str) -> None:
        await_engine(
            self.runtime.engine.sources.remove(
                SourceRemoveRequest(pot_id=pot_id, source_id=source_id)
            )
        )

    def repo_default(self, *, repo: str) -> str | None:
        return await_engine(
            self.runtime.engine.pots.repo_default(RepoDefaultGetRequest(repo=repo))
        ).pot_id

    def set_repo_default(self, *, repo: str, pot_id: str) -> None:
        await_engine(
            self.runtime.engine.pots.set_repo_default(
                RepoDefaultSetRequest(repo=repo, pot_id=pot_id)
            )
        )

    def clear_repo_default(self, *, repo: str) -> bool:
        return await_engine(
            self.runtime.engine.pots.clear_repo_default(
                RepoDefaultClearRequest(repo=repo)
            )
        ).cleared

    def list_repo_defaults(self) -> dict[str, str]:
        return dict(
            await_engine(
                self.runtime.engine.pots.list_repo_defaults(EmptyRequest())
            ).items
        )


@dataclass(slots=True)
class _GraphView:
    runtime: PotpieRuntime

    def catalog(self, request):
        return await_engine(self.runtime.engine.graph.catalog(request))

    def describe(self, request: GraphDescribeRequest):
        return await_engine(self.runtime.engine.graph.describe(request))

    def read(self, request):
        return await_engine(self.runtime.engine.graph.read(request))

    def search_entities(self, request):
        return await_engine(self.runtime.engine.graph.search_entities(request))

    def data_plane_status(self, pot_id: str):
        return await_engine(
            self.runtime.engine.graph.status(GraphStatusRequest(pot_id=pot_id))
        )


@dataclass(slots=True)
class _GraphWorkbenchView:
    runtime: PotpieRuntime

    def propose(
        self, payload: dict[str, Any], *, pot_id: str, ttl_seconds: int | None = None
    ):
        return await_engine(
            self.runtime.engine.graph.propose(
                GraphProposeRequest(
                    pot_id=pot_id, payload=payload, ttl_seconds=ttl_seconds
                )
            )
        )

    def commit(
        self,
        plan_id: str,
        *,
        pot_id: str,
        approved_by: str | None = None,
        verify: bool = False,
    ):
        return await_engine(
            self.runtime.engine.graph.commit(
                GraphCommitRequest(
                    pot_id=pot_id,
                    plan_id=plan_id,
                    approved_by=approved_by,
                    verify=verify,
                )
            )
        )

    def history(self, **kwargs: Any):
        return await_engine(
            self.runtime.engine.graph.history(GraphHistoryRequest(**kwargs))
        )

    def quality(self, *, pot_id: str, report: str, **filters: Any):
        return await_engine(
            self.runtime.engine.graph.quality(
                GraphQualityRequest(pot_id=pot_id, report=report, filters=filters)
            )
        )

    def inbox_add(self, **kwargs: Any):
        return await_engine(
            self.runtime.engine.graph.inbox_add(GraphInboxAddRequest(**kwargs))
        )

    def inbox_list(self, **kwargs: Any):
        return await_engine(
            self.runtime.engine.graph.inbox_list(GraphInboxListRequest(**kwargs))
        )

    def inbox_show(self, **kwargs: Any):
        return await_engine(
            self.runtime.engine.graph.inbox_show(GraphInboxItemRequest(**kwargs))
        )

    def inbox_claim(self, **kwargs: Any):
        return await_engine(
            self.runtime.engine.graph.inbox_claim(GraphInboxClaimRequest(**kwargs))
        )

    def inbox_mark_applied(self, **kwargs: Any):
        return self._close(action="mark-applied", **kwargs)

    def inbox_mark_rejected(self, **kwargs: Any):
        return self._close(action="mark-rejected", **kwargs)

    def inbox_close(self, **kwargs: Any):
        return self._close(action="close", **kwargs)

    def _close(self, *, action: str, **kwargs: Any):
        return await_engine(
            self.runtime.engine.graph.inbox_close(
                GraphInboxCloseRequest(action=action, **kwargs)
            )
        )


@dataclass(slots=True)
class _NudgeView:
    runtime: PotpieRuntime

    def nudge(self, request: GraphNudgeRequest):
        return await_engine(self.runtime.engine.graph.nudge(request))


@dataclass(slots=True)
class _InspectionView:
    runtime: PotpieRuntime

    def neighborhood(self, *, pot_id: str, entity_key: str, depth: int = 1):
        return await_engine(
            self.runtime.engine.graph.neighborhood(
                GraphNeighborhoodRequest(
                    pot_id=pot_id, entity_key=entity_key, depth=depth
                )
            )
        )


@dataclass(slots=True)
class _SnapshotView:
    runtime: PotpieRuntime

    def export(self, *, pot_id: str, destination: str):
        return await_engine(
            self.runtime.engine.graph.snapshot_export(
                GraphSnapshotExportRequest(pot_id=pot_id, destination=destination)
            )
        )

    def import_(self, *, pot_id: str, source: str):
        return await_engine(
            self.runtime.engine.graph.snapshot_import(
                GraphSnapshotImportRequest(pot_id=pot_id, source=source)
            )
        )


@dataclass(slots=True)
class _AnalyticsView:
    runtime: PotpieRuntime

    def repair(self, pot_id: str, *, targets=()):
        return await_engine(
            self.runtime.engine.graph.repair(
                GraphRepairRequest(pot_id=pot_id, targets=tuple(targets))
            )
        )


@dataclass(slots=True)
class _MutationView:
    runtime: PotpieRuntime

    def readiness(self, pot_id: str):
        status = await_engine(
            self.runtime.engine.graph.status(GraphStatusRequest(pot_id=pot_id))
        )
        return SimpleNamespace(
            profile=status.backend_profile,
            ready=status.backend_ready,
            capability_ready={},
            detail=status.detail,
        )


@dataclass(slots=True)
class _BackendView:
    runtime: PotpieRuntime
    inspection: _InspectionView = None  # type: ignore[assignment]
    snapshot: _SnapshotView = None  # type: ignore[assignment]
    analytics: _AnalyticsView = None  # type: ignore[assignment]
    mutation: _MutationView = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.inspection = _InspectionView(self.runtime)
        self.snapshot = _SnapshotView(self.runtime)
        self.analytics = _AnalyticsView(self.runtime)
        self.mutation = _MutationView(self.runtime)

    @property
    def profile(self) -> str:
        return self._info().profile

    def capabilities(self):
        info = self._info()
        return SimpleNamespace(
            profile=info.profile,
            implemented=lambda: info.capabilities,
            inspection="inspection" in info.capabilities,
            snapshot="snapshot" in info.capabilities,
            analytics="analytics" in info.capabilities,
        )

    def _info(self):
        return await_engine(
            self.runtime.engine.graph.backend_info(GraphBackendInfoRequest())
        )


@dataclass(slots=True)
class _LedgerView:
    runtime: PotpieRuntime

    def status(self):
        return await_engine(self.runtime.engine.ledger.status(LedgerStatusRequest()))

    def sources(self, *, pot_id: str):
        return list(
            await_engine(
                self.runtime.engine.ledger.sources(LedgerSourcesRequest(pot_id=pot_id))
            ).items
        )

    def query(self, **kwargs: Any):
        return await_engine(
            self.runtime.engine.ledger.query(LedgerQueryRequest(**kwargs))
        )

    def pull(self, **kwargs: Any):
        return await_engine(
            self.runtime.engine.ledger.pull(LedgerPullRequest(**kwargs))
        )


@dataclass(slots=True)
class RuntimeEngineView:
    runtime: PotpieRuntime
    agent_context: _ContextView = None  # type: ignore[assignment]
    pots: _PotsView = None  # type: ignore[assignment]
    graph: _GraphView = None  # type: ignore[assignment]
    graph_workbench: _GraphWorkbenchView = None  # type: ignore[assignment]
    ledger: _LedgerView = None  # type: ignore[assignment]
    nudge: _NudgeView = None  # type: ignore[assignment]
    backend: _BackendView = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.agent_context = _ContextView(self.runtime)
        self.pots = _PotsView(self.runtime)
        self.graph = _GraphView(self.runtime)
        self.graph_workbench = _GraphWorkbenchView(self.runtime)
        self.ledger = _LedgerView(self.runtime)
        self.nudge = _NudgeView(self.runtime)
        self.backend = _BackendView(self.runtime)


def runtime_engine_view(runtime: PotpieRuntime) -> RuntimeEngineView:
    return RuntimeEngineView(runtime)


__all__ = ["RuntimeEngineView", "await_engine", "runtime_engine_view"]
