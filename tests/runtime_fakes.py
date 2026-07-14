"""Typed async runtime adapters for synchronous unit-test service doubles."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from potpie.runtime import PotpieRuntime, ProductSettings
from potpie.runtime.contracts import (
    CapabilityNotImplemented,
    GraphBackendInfo,
    OperationResult,
    PotListResult,
    RepoDefaultClearResult,
    RepoDefaultListResult,
    RepoDefaultResult,
    SourceListResult,
)


def _missing(capability: str) -> CapabilityNotImplemented:
    return CapabilityNotImplemented(capability)


class PotsClientAdapter:
    def __init__(self, service: Any) -> None:
        self.service = service

    async def list(self, request: Any) -> PotListResult:
        del request
        items = tuple(self.service.list_pots())
        return PotListResult(items=items, count=len(items))

    async def info(self, request: Any) -> Any:
        if request.ref:
            for pot in self.service.list_pots():
                if request.ref in (pot.pot_id, pot.name):
                    return pot
            return None
        return self.service.active_pot()

    async def create(self, request: Any) -> Any:
        return self.service.create_pot(
            name=request.name, repo=request.repo, use=request.use
        )

    async def use(self, request: Any) -> Any:
        return self.service.use_pot(ref=request.ref)

    async def rename(self, request: Any) -> Any:
        return self.service.rename_pot(ref=request.ref, new_name=request.new_name)

    async def reset(self, request: Any) -> Any:
        return self.service.reset_pot(ref=request.ref, confirm=request.confirm)

    async def archive(self, request: Any) -> Any:
        return self.service.archive_pot(ref=request.ref)

    async def repo_default(self, request: Any) -> RepoDefaultResult:
        method = getattr(self.service, "repo_default", None)
        if not callable(method):
            raise _missing("engine.pots.repo_default")
        return RepoDefaultResult(pot_id=method(repo=request.repo))

    async def set_repo_default(self, request: Any) -> OperationResult:
        method = getattr(self.service, "set_repo_default", None)
        if not callable(method):
            raise _missing("engine.pots.set_repo_default")
        method(repo=request.repo, pot_id=request.pot_id)
        return OperationResult()

    async def clear_repo_default(self, request: Any) -> RepoDefaultClearResult:
        return RepoDefaultClearResult(
            cleared=bool(self.service.clear_repo_default(repo=request.repo))
        )

    async def list_repo_defaults(self, request: Any) -> RepoDefaultListResult:
        del request
        return RepoDefaultListResult(items=self.service.list_repo_defaults())


class SourcesClientAdapter:
    def __init__(self, service: Any) -> None:
        self.service = service

    async def add(self, request: Any) -> Any:
        return self.service.add_source(
            pot_id=request.pot_id,
            kind=request.kind,
            location=request.location,
            name=request.name,
        )

    async def list(self, request: Any) -> SourceListResult:
        items = tuple(self.service.list_sources(pot_id=request.pot_id))
        return SourceListResult(items=items, count=len(items))

    async def status(self, request: Any) -> Any:
        return self.service.source_status(
            pot_id=request.pot_id, source_id=request.source_id
        )

    async def remove(self, request: Any) -> OperationResult:
        self.service.remove_source(pot_id=request.pot_id, source_id=request.source_id)
        return OperationResult()


class ContextClientAdapter:
    def __init__(self, service: Any) -> None:
        self.service = service

    async def resolve(self, request: Any) -> Any:
        return self.service.resolve(request)

    async def search(self, request: Any) -> Any:
        return self.service.search(request)

    async def record(self, request: Any) -> Any:
        return self.service.record(request)

    async def status(self, request: Any) -> Any:
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

    async def catalog(self, request: Any) -> Any:
        return self.graph.catalog(request)

    async def describe(self, request: Any) -> Any:
        return self.graph.describe(request)

    async def read(self, request: Any) -> Any:
        return self.graph.read(request)

    async def search_entities(self, request: Any) -> Any:
        return self.graph.search_entities(request)

    async def status(self, request: Any) -> Any:
        return self.graph.data_plane_status(request.pot_id)

    async def propose(self, request: Any) -> Any:
        return self.workbench.propose(
            request.payload,
            pot_id=request.pot_id,
            ttl_seconds=request.ttl_seconds,
        )

    async def commit(self, request: Any) -> Any:
        return self.workbench.commit(
            request.plan_id,
            pot_id=request.pot_id,
            approved_by=request.approved_by,
            verify=request.verify,
        )

    async def history(self, request: Any) -> Any:
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

    async def quality(self, request: Any) -> Any:
        return self.workbench.quality(
            pot_id=request.pot_id, report=request.report, **dict(request.filters)
        )

    async def nudge(self, request: Any) -> Any:
        return self.nudge_service.nudge(request)

    async def neighborhood(self, request: Any) -> Any:
        return self.backend.inspection.neighborhood(
            pot_id=request.pot_id,
            entity_key=request.entity_key,
            depth=request.depth,
        )

    async def inbox_add(self, request: Any) -> Any:
        return self.workbench.inbox_add(
            pot_id=request.pot_id,
            summary=request.summary,
            details=request.details,
            evidence=request.evidence,
            source_refs=request.source_refs,
            suspected_subgraphs=request.suspected_subgraphs,
            created_by=request.created_by,
        )

    async def inbox_list(self, request: Any) -> Any:
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

    async def inbox_show(self, request: Any) -> Any:
        return self.workbench.inbox_show(pot_id=request.pot_id, item_id=request.item_id)

    async def inbox_claim(self, request: Any) -> Any:
        return self.workbench.inbox_claim(
            pot_id=request.pot_id,
            item_id=request.item_id,
            claimed_by=request.claimed_by,
        )

    async def inbox_close(self, request: Any) -> Any:
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

    async def snapshot_export(self, request: Any) -> Any:
        return self.backend.snapshot.export(
            pot_id=request.pot_id, destination=request.destination
        )

    async def snapshot_import(self, request: Any) -> Any:
        return self.backend.snapshot.import_(
            pot_id=request.pot_id, source=request.source
        )

    async def repair(self, request: Any) -> Any:
        return self.backend.analytics.repair(request.pot_id, targets=request.targets)

    async def backend_info(self, request: Any) -> GraphBackendInfo:
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

    async def status(self, request: Any) -> Any:
        del request
        return self.service.status()

    async def sources(self, request: Any) -> Any:
        items = tuple(self.service.sources(pot_id=request.pot_id))
        return SimpleNamespace(items=items, count=len(items))

    async def query(self, request: Any) -> Any:
        return self.service.query(
            pot_id=request.pot_id,
            source_id=request.source_id,
            kind=request.kind,
            since=request.since,
            until=request.until,
            limit=request.limit,
        )

    async def pull(self, request: Any) -> Any:
        return self.service.pull(
            pot_id=request.pot_id,
            source_id=request.source_id,
            limit=request.limit,
        )


class TestEngineClient:
    def __init__(
        self,
        *,
        context: Any = None,
        pots: Any = None,
        sources: Any = None,
        graph: Any = None,
        ledger: Any = None,
        timeline: Any = None,
        provision: Any = None,
    ) -> None:
        missing = SimpleNamespace()
        self.context = context or missing
        self.pots = pots or missing
        self.sources = sources or missing
        self.graph = graph or missing
        self.ledger = ledger or missing
        self.timeline = timeline or missing
        self.provision = provision or missing

    async def aclose(self) -> None:
        return None


class _MemoryConfig:
    def __init__(self) -> None:
        self.values: dict[str, Any] = {}

    def set(self, key: str, value: Any) -> None:
        self.values[key] = value

    def get(self, key: str) -> Any:
        return self.values.get(key)


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
    return PotpieRuntime(
        settings=ProductSettings(
            data_dir=data_dir or Path("/tmp/potpie-test-runtime"),
            runtime_mode="in-process",
            backend="in_memory",
        ),
        engine=engine,
        auth=SimpleNamespace(),
        integrations=SimpleNamespace(),
        config=config or _MemoryConfig(),
        skills=SimpleNamespace(),
        installer=SimpleNamespace(),
        daemon=daemon or SimpleNamespace(),
    )


__all__ = ["runtime_from_services"]
