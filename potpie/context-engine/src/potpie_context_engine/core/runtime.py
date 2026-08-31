"""Public graph runtime composition and async-safe service facade."""

from __future__ import annotations

import asyncio
from contextvars import ContextVar
import importlib
import inspect
import logging
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, runtime_checkable

from potpie_context_engine.core.definition import (
    DEFAULT_GRAPH_DEFINITION,
    GraphDefinition,
)
from potpie_context_engine.core.mutation_policy import (
    DEFAULT_MUTATION_POLICY,
    GraphMutationPolicy,
)
from potpie_context_engine.core.ports.graph.backend import GraphBackend
from potpie_context_engine.core.ports.graph.inbox_store import GraphInboxStorePort
from potpie_context_engine.core.ports.graph.plan_store import GraphPlanStorePort
from potpie_context_engine.core.reconciliation_config import (
    ReconciliationConfig,
)
from potpie_context_engine.core.reconciliation_flags import (
    reconciliation_config_from_env,
)
from potpie_context_engine.core.workbench_service import GraphWorkbenchService


class RuntimeCompositionError(TypeError):
    """Runtime wiring does not implement the documented public contracts."""


_BRIDGE_LOOP: ContextVar[asyncio.AbstractEventLoop | None] = ContextVar(
    "graph_runtime_bridge_loop", default=None
)
_LOG = logging.getLogger(__name__)


class _SyncAsyncBridge:
    """Bridge one named call without hiding a facade's protocol members."""

    def __init__(self, target: Any) -> None:
        self._target = target

    def call(self, name: str, *args: Any, **kwargs: Any) -> Any:
        sync_method = getattr(self._target, name, None)
        if callable(sync_method):
            return sync_method(*args, **kwargs)
        async_method = getattr(self._target, f"{name}_async", None)
        if not callable(async_method):
            raise AttributeError(name)
        loop = _BRIDGE_LOOP.get()
        if loop is None or not loop.is_running():
            raise RuntimeCompositionError(
                f"async-only port method {name!r} requires the GraphRuntime "
                "async service path"
            )
        future = asyncio.run_coroutine_threadsafe(async_method(*args, **kwargs), loop)
        return future.result()

    async def call_async(self, name: str, *args: Any, **kwargs: Any) -> Any:
        return await self.call_async_named(name, f"{name}_async", *args, **kwargs)

    async def call_async_named(
        self,
        name: str,
        async_name: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        async_method = getattr(self._target, async_name, None)
        if callable(async_method):
            result = async_method(*args, **kwargs)
            return await result if inspect.isawaitable(result) else result
        sync_method = getattr(self._target, name, None)
        if not callable(sync_method):
            raise AttributeError(name)
        return await asyncio.to_thread(sync_method, *args, **kwargs)


class _MutationPortBridge:
    """Protocol-visible facade for a sync or async mutation port."""

    def __init__(self, target: Any) -> None:
        self._bridge = _SyncAsyncBridge(target)

    def apply(self, *args: Any, **kwargs: Any) -> Any:
        return self._bridge.call("apply", *args, **kwargs)

    async def apply_async(self, *args: Any, **kwargs: Any) -> Any:
        return await self._bridge.call_async("apply", *args, **kwargs)

    def lookup_execution(self, *args: Any, **kwargs: Any) -> Any:
        return self._bridge.call("lookup_execution", *args, **kwargs)

    async def lookup_execution_async(self, *args: Any, **kwargs: Any) -> Any:
        return await self._bridge.call_async("lookup_execution", *args, **kwargs)

    def invalidate(self, *args: Any, **kwargs: Any) -> Any:
        return self._bridge.call("invalidate", *args, **kwargs)

    async def invalidate_async(self, *args: Any, **kwargs: Any) -> Any:
        return await self._bridge.call_async("invalidate", *args, **kwargs)

    def reset_pot(self, *args: Any, **kwargs: Any) -> Any:
        return self._bridge.call("reset_pot", *args, **kwargs)

    async def reset_pot_async(self, *args: Any, **kwargs: Any) -> Any:
        return await self._bridge.call_async("reset_pot", *args, **kwargs)

    def readiness(self, *args: Any, **kwargs: Any) -> Any:
        return self._bridge.call("readiness", *args, **kwargs)

    async def readiness_async(self, *args: Any, **kwargs: Any) -> Any:
        return await self._bridge.call_async("readiness", *args, **kwargs)


class _ClaimQueryPortBridge:
    """Protocol-visible facade for a sync or async claim-query port."""

    def __init__(self, target: Any) -> None:
        self._target = target
        self._bridge = _SyncAsyncBridge(target)

    @property
    def match_mode(self) -> str:
        return getattr(self._target, "match_mode", "lexical")

    def find_claims(self, *args: Any, **kwargs: Any) -> Any:
        return self._bridge.call("find_claims", *args, **kwargs)

    async def find_claims_async(self, *args: Any, **kwargs: Any) -> Any:
        return await self._bridge.call_async("find_claims", *args, **kwargs)

    def entity_labels(self, *args: Any, **kwargs: Any) -> Any:
        return self._bridge.call("entity_labels", *args, **kwargs)

    async def entity_labels_async(self, *args: Any, **kwargs: Any) -> Any:
        return await self._bridge.call_async("entity_labels", *args, **kwargs)

    def entity_properties(self, *args: Any, **kwargs: Any) -> Any:
        return self._bridge.call("entity_properties", *args, **kwargs)

    async def entity_properties_async(self, *args: Any, **kwargs: Any) -> Any:
        return await self._bridge.call_async("entity_properties", *args, **kwargs)


class _SemanticPortBridge:
    def __init__(self, target: Any) -> None:
        self._bridge = _SyncAsyncBridge(target)

    def search(self, *args: Any, **kwargs: Any) -> Any:
        return self._bridge.call("search", *args, **kwargs)

    async def search_async(self, *args: Any, **kwargs: Any) -> Any:
        return await self._bridge.call_async("search", *args, **kwargs)


class _InspectionPortBridge:
    def __init__(self, target: Any) -> None:
        self._bridge = _SyncAsyncBridge(target)

    def neighborhood(self, *args: Any, **kwargs: Any) -> Any:
        return self._bridge.call("neighborhood", *args, **kwargs)

    async def neighborhood_async(self, *args: Any, **kwargs: Any) -> Any:
        return await self._bridge.call_async("neighborhood", *args, **kwargs)

    def path(self, *args: Any, **kwargs: Any) -> Any:
        return self._bridge.call("path", *args, **kwargs)

    async def path_async(self, *args: Any, **kwargs: Any) -> Any:
        return await self._bridge.call_async("path", *args, **kwargs)

    def labels(self, *args: Any, **kwargs: Any) -> Any:
        return self._bridge.call("labels", *args, **kwargs)

    async def labels_async(self, *args: Any, **kwargs: Any) -> Any:
        return await self._bridge.call_async("labels", *args, **kwargs)

    def slice(self, *args: Any, **kwargs: Any) -> Any:
        return self._bridge.call("slice", *args, **kwargs)

    async def slice_async(self, *args: Any, **kwargs: Any) -> Any:
        return await self._bridge.call_async("slice", *args, **kwargs)


class _AnalyticsPortBridge:
    def __init__(self, target: Any) -> None:
        self._bridge = _SyncAsyncBridge(target)

    def counts(self, *args: Any, **kwargs: Any) -> Any:
        return self._bridge.call("counts", *args, **kwargs)

    async def counts_async(self, *args: Any, **kwargs: Any) -> Any:
        return await self._bridge.call_async("counts", *args, **kwargs)

    def freshness(self, *args: Any, **kwargs: Any) -> Any:
        return self._bridge.call("freshness", *args, **kwargs)

    async def freshness_async(self, *args: Any, **kwargs: Any) -> Any:
        return await self._bridge.call_async("freshness", *args, **kwargs)

    def quality(self, *args: Any, **kwargs: Any) -> Any:
        return self._bridge.call("quality", *args, **kwargs)

    async def quality_async(self, *args: Any, **kwargs: Any) -> Any:
        return await self._bridge.call_async("quality", *args, **kwargs)

    def repair(self, *args: Any, **kwargs: Any) -> Any:
        return self._bridge.call("repair", *args, **kwargs)

    async def repair_async(self, *args: Any, **kwargs: Any) -> Any:
        return await self._bridge.call_async("repair", *args, **kwargs)


class _SnapshotPortBridge:
    def __init__(self, target: Any) -> None:
        self._bridge = _SyncAsyncBridge(target)

    def export(self, *args: Any, **kwargs: Any) -> Any:
        return self._bridge.call("export", *args, **kwargs)

    async def export_async(self, *args: Any, **kwargs: Any) -> Any:
        return await self._bridge.call_async("export", *args, **kwargs)

    def import_(self, *args: Any, **kwargs: Any) -> Any:
        return self._bridge.call("import_", *args, **kwargs)

    async def import_async(self, *args: Any, **kwargs: Any) -> Any:
        return await self._bridge.call_async_named(
            "import_", "import_async", *args, **kwargs
        )


class _StoreBridge:
    """Protocol-visible facade for a sync or async inbox store."""

    def __init__(self, target: Any) -> None:
        self._bridge = _SyncAsyncBridge(target)

    def save(self, *args: Any, **kwargs: Any) -> Any:
        return self._bridge.call("save", *args, **kwargs)

    async def save_async(self, *args: Any, **kwargs: Any) -> Any:
        return await self._bridge.call_async("save", *args, **kwargs)

    def get(self, *args: Any, **kwargs: Any) -> Any:
        return self._bridge.call("get", *args, **kwargs)

    async def get_async(self, *args: Any, **kwargs: Any) -> Any:
        return await self._bridge.call_async("get", *args, **kwargs)

    def list(self, *args: Any, **kwargs: Any) -> Any:
        return self._bridge.call("list", *args, **kwargs)

    async def list_async(self, *args: Any, **kwargs: Any) -> Any:
        return await self._bridge.call_async("list", *args, **kwargs)


class _PlanStoreBridge(_StoreBridge):
    """Plan-store facade including the atomic commit-state transition."""

    def compare_and_set(self, *args: Any, **kwargs: Any) -> Any:
        return self._bridge.call("compare_and_set", *args, **kwargs)

    async def compare_and_set_async(self, *args: Any, **kwargs: Any) -> Any:
        return await self._bridge.call_async("compare_and_set", *args, **kwargs)


class _BackendBridge:
    """Protocol-visible facade for a backend with async-capable nested ports."""

    def __init__(self, backend: Any) -> None:
        self._backend = backend
        self._mutation = _MutationPortBridge(backend.mutation)
        self._claim_query = _ClaimQueryPortBridge(backend.claim_query)
        self._semantic = _SemanticPortBridge(backend.semantic)
        self._inspection = _InspectionPortBridge(backend.inspection)
        self._analytics = _AnalyticsPortBridge(backend.analytics)
        self._snapshot = _SnapshotPortBridge(backend.snapshot)

    @property
    def profile(self) -> str:
        return self._backend.profile

    @property
    def match_mode(self) -> str:
        mode = getattr(self._backend, "match_mode", None)
        return mode if mode is not None else self._claim_query.match_mode

    @property
    def mutation(self) -> _MutationPortBridge:
        return self._mutation

    @property
    def claim_query(self) -> _ClaimQueryPortBridge:
        return self._claim_query

    @property
    def semantic(self) -> _SemanticPortBridge:
        return self._semantic

    @property
    def inspection(self) -> _InspectionPortBridge:
        return self._inspection

    @property
    def analytics(self) -> _AnalyticsPortBridge:
        return self._analytics

    @property
    def snapshot(self) -> _SnapshotPortBridge:
        return self._snapshot

    def capabilities(self) -> Any:
        return self._backend.capabilities()

    def bind_definition(self, definition: GraphDefinition) -> "_BackendBridge":
        return _BackendBridge(self._backend.bind_definition(definition))


@runtime_checkable
class GraphObserver(Protocol):
    def observe(self, event: str, fields: Mapping[str, Any]) -> None: ...


@dataclass(frozen=True, slots=True)
class NoOpGraphObserver:
    def observe(self, event: str, fields: Mapping[str, Any]) -> None:
        del event, fields


@dataclass(frozen=True, slots=True)
class GraphRuntime:
    """One fully wired graph runtime sharing one definition and policy."""

    backend: GraphBackend
    plan_store: GraphPlanStorePort
    inbox_store: GraphInboxStorePort | None
    definition: GraphDefinition
    policy: GraphMutationPolicy
    reconciliation_config: ReconciliationConfig
    observability: GraphObserver
    graph: Any
    workbench: GraphWorkbenchService

    def _notify(self, event: str, fields: Mapping[str, Any]) -> None:
        try:
            self.observability.observe(event, fields)
        except Exception:
            _LOG.warning("graph observer failed for %s", event, exc_info=True)

    def status(self, pot_id: str) -> dict[str, Any]:
        data_plane = self.graph.data_plane_status(pot_id)
        return {
            "pot_id": pot_id,
            "definition": self.definition.status_metadata(),
            "backend": {
                "profile": data_plane.backend_profile,
                "ready": data_plane.backend_ready,
                "detail": data_plane.detail,
                "match_mode": data_plane.match_mode,
                "counts": dict(data_plane.counts),
                "freshness": dict(data_plane.freshness),
                "quality": dict(data_plane.quality),
            },
            "readers": sorted(data_plane.reader_backed_includes),
        }

    def catalog(self, request):
        return self.graph.catalog(request)

    def resolve(self, request):
        return self.graph.resolve(request)

    def describe(self, request):
        return self.graph.describe(request)

    def read(self, request):
        result = self.graph.read(request)
        self._notify(
            "graph.read",
            {
                "pot_id": request.pot_id,
                "view": f"{request.subgraph}.{request.view}",
                "ok": result.ok,
            },
        )
        return result

    def search(self, request):
        return self.graph.search(request)

    def record(self, request):
        return self.graph.record(request)

    def search_entities(self, request):
        return self.graph.search_entities(request)

    def mutate(self, request):
        result = self.graph.mutate(request)
        self._notify(
            "graph.mutate",
            {
                "pot_id": request.pot_id,
                "status": result.status,
                "mutation_id": result.mutation_id,
                "ok": result.ok,
            },
        )
        return result

    def propose(self, payload, *, pot_id: str, ttl_seconds: int | None = None):
        result = self.workbench.propose(payload, pot_id=pot_id, ttl_seconds=ttl_seconds)
        self._notify(
            "graph.propose",
            {
                "pot_id": pot_id,
                "plan_id": result.plan_id,
                "status": result.status,
                "ok": result.ok,
            },
        )
        return result

    def commit(
        self,
        plan_id: str,
        *,
        pot_id: str,
        approved_by: str | None = None,
        verify: bool = False,
        recover_stale: bool = False,
    ):
        result = self.workbench.commit(
            plan_id,
            pot_id=pot_id,
            approved_by=approved_by,
            verify=verify,
            recover_stale=recover_stale,
        )
        self._notify(
            "graph.commit",
            {
                "pot_id": pot_id,
                "plan_id": plan_id,
                "mutation_id": result.mutation_id,
                "status": result.status,
                "ok": result.ok,
            },
        )
        return result

    def history(self, **kwargs):
        return self.workbench.history(**kwargs)

    def quality(self, **kwargs):
        return self.workbench.quality(**kwargs)

    def inbox_add(self, **kwargs):
        return self.workbench.inbox_add(**kwargs)

    def inbox_list(self, **kwargs):
        return self.workbench.inbox_list(**kwargs)

    def inbox_show(self, **kwargs):
        return self.workbench.inbox_show(**kwargs)

    def inbox_claim(self, **kwargs):
        return self.workbench.inbox_claim(**kwargs)

    def inbox_mark_applied(self, **kwargs):
        return self.workbench.inbox_mark_applied(**kwargs)

    def inbox_mark_rejected(self, **kwargs):
        return self.workbench.inbox_mark_rejected(**kwargs)

    def inbox_close(self, **kwargs):
        return self.workbench.inbox_close(**kwargs)

    async def status_async(self, pot_id: str) -> dict[str, Any]:
        return await _to_thread_with_bridge(self.status, pot_id)

    async def catalog_async(self, request):
        return await _async_call(self.graph, "catalog", request)

    async def resolve_async(self, request):
        return await _async_call(self.graph, "resolve", request)

    async def describe_async(self, request):
        return await _async_call(self.graph, "describe", request)

    async def read_async(self, request):
        result = await _async_call(self.graph, "read", request)
        self._notify(
            "graph.read",
            {
                "pot_id": request.pot_id,
                "view": f"{request.subgraph}.{request.view}",
                "ok": result.ok,
            },
        )
        return result

    async def search_async(self, request):
        return await _async_call(self.graph, "search", request)

    async def record_async(self, request):
        return await _async_call(self.graph, "record", request)

    async def search_entities_async(self, request):
        return await _async_call(self.graph, "search_entities", request)

    async def mutate_async(self, request):
        result = await _async_call(self.graph, "mutate", request)
        self._notify(
            "graph.mutate",
            {
                "pot_id": request.pot_id,
                "status": result.status,
                "mutation_id": result.mutation_id,
                "ok": result.ok,
            },
        )
        return result

    async def propose_async(
        self, payload, *, pot_id: str, ttl_seconds: int | None = None
    ):
        result = await _async_call(
            self.workbench,
            "propose",
            payload,
            pot_id=pot_id,
            ttl_seconds=ttl_seconds,
        )
        self._notify(
            "graph.propose",
            {
                "pot_id": pot_id,
                "plan_id": result.plan_id,
                "status": result.status,
                "ok": result.ok,
            },
        )
        return result

    async def commit_async(
        self,
        plan_id: str,
        *,
        pot_id: str,
        approved_by: str | None = None,
        verify: bool = False,
        recover_stale: bool = False,
    ):
        result = await _async_call(
            self.workbench,
            "commit",
            plan_id,
            pot_id=pot_id,
            approved_by=approved_by,
            verify=verify,
            recover_stale=recover_stale,
        )
        self._notify(
            "graph.commit",
            {
                "pot_id": pot_id,
                "plan_id": plan_id,
                "mutation_id": result.mutation_id,
                "status": result.status,
                "ok": result.ok,
            },
        )
        return result

    async def history_async(self, **kwargs):
        return await _async_call(self.workbench, "history", **kwargs)

    async def quality_async(self, **kwargs):
        return await _async_call(self.workbench, "quality", **kwargs)

    async def inbox_add_async(self, **kwargs):
        return await _async_call(self.workbench, "inbox_add", **kwargs)

    async def inbox_list_async(self, **kwargs):
        return await _async_call(self.workbench, "inbox_list", **kwargs)

    async def inbox_show_async(self, **kwargs):
        return await _async_call(self.workbench, "inbox_show", **kwargs)

    async def inbox_claim_async(self, **kwargs):
        return await _async_call(self.workbench, "inbox_claim", **kwargs)

    async def inbox_mark_applied_async(self, **kwargs):
        return await _async_call(self.workbench, "inbox_mark_applied", **kwargs)

    async def inbox_mark_rejected_async(self, **kwargs):
        return await _async_call(self.workbench, "inbox_mark_rejected", **kwargs)

    async def inbox_close_async(self, **kwargs):
        return await _async_call(self.workbench, "inbox_close", **kwargs)


async def _async_call(target: Any, method: str, *args: Any, **kwargs: Any) -> Any:
    token = _BRIDGE_LOOP.set(asyncio.get_running_loop())
    try:
        async_method = getattr(target, f"{method}_async", None)
        if callable(async_method):
            result = async_method(*args, **kwargs)
            return await result if inspect.isawaitable(result) else result
        return await asyncio.to_thread(getattr(target, method), *args, **kwargs)
    finally:
        _BRIDGE_LOOP.reset(token)


async def _to_thread_with_bridge(function: Any, *args: Any, **kwargs: Any) -> Any:
    token = _BRIDGE_LOOP.set(asyncio.get_running_loop())
    try:
        return await asyncio.to_thread(function, *args, **kwargs)
    finally:
        _BRIDGE_LOOP.reset(token)


def build_graph_runtime(
    backend: GraphBackend,
    plan_store: GraphPlanStorePort,
    inbox_store: GraphInboxStorePort | None = None,
    definition: GraphDefinition = DEFAULT_GRAPH_DEFINITION,
    policy: GraphMutationPolicy = DEFAULT_MUTATION_POLICY,
    observability: GraphObserver | None = None,
    reconciliation_config: ReconciliationConfig | None = None,
) -> GraphRuntime:
    """Validate composition and return the single supported graph runtime."""

    if not isinstance(definition, GraphDefinition):
        raise RuntimeCompositionError("definition must be a GraphDefinition")
    reconciliation = (
        reconciliation_config_from_env()
        if reconciliation_config is None
        else reconciliation_config
    )
    if not isinstance(reconciliation, ReconciliationConfig):
        raise RuntimeCompositionError(
            "reconciliation_config must be a ReconciliationConfig"
        )
    _require_methods(backend, "backend", ("bind_definition",))
    try:
        backend = backend.bind_definition(definition)
    except Exception as exc:
        raise RuntimeCompositionError(
            f"backend failed to bind graph definition: {exc}"
        ) from exc
    if backend is None:
        raise RuntimeCompositionError(
            "backend.bind_definition must return a definition-bound backend"
        )
    _require_methods(
        backend,
        "backend",
        (
            "bind_definition",
            "capabilities",
            "mutation",
            "claim_query",
            "semantic",
            "inspection",
            "analytics",
            "snapshot",
        ),
    )
    _require_sync_or_async_methods(
        backend.mutation,
        "backend.mutation",
        ("apply", "invalidate", "reset_pot", "readiness"),
    )
    _require_sync_or_async_methods(
        backend.claim_query,
        "backend.claim_query",
        ("find_claims", "entity_labels", "entity_properties"),
    )
    _require_sync_or_async_methods(
        plan_store,
        "plan_store",
        ("save", "get", "compare_and_set", "list"),
    )
    if inbox_store is not None:
        _require_sync_or_async_methods(
            inbox_store, "inbox_store", ("save", "get", "list")
        )
    observer = observability or NoOpGraphObserver()
    _require_methods(observer, "observability", ("observe",))

    try:
        composition = importlib.import_module("potpie_context_engine.composition")
    except ImportError as exc:
        raise RuntimeCompositionError(
            "potpie-context-engine is required to build the default graph "
            "implementation"
        ) from exc
    runtime_backend = _BackendBridge(backend)
    runtime_plan_store = _PlanStoreBridge(plan_store)
    runtime_inbox_store = _StoreBridge(inbox_store) if inbox_store is not None else None
    graph = composition.build_graph_service(
        backend=runtime_backend,
        definition=definition,
        policy=policy,
        reconciliation_config=reconciliation,
    )
    workbench = GraphWorkbenchService(
        backend=runtime_backend,
        plan_store=runtime_plan_store,
        inbox_store=runtime_inbox_store,
        definition=definition,
        policy=policy,
        reconciliation_config=reconciliation,
    )
    return GraphRuntime(
        backend=runtime_backend,
        plan_store=runtime_plan_store,
        inbox_store=runtime_inbox_store,
        definition=definition,
        policy=policy,
        reconciliation_config=reconciliation,
        observability=observer,
        graph=graph,
        workbench=workbench,
    )


def _require_methods(target: Any, name: str, methods: tuple[str, ...]) -> None:
    missing: list[str] = []
    for method in methods:
        try:
            value = getattr(target, method)
        except Exception:
            missing.append(method)
            continue
        if method in {
            "mutation",
            "claim_query",
            "semantic",
            "inspection",
            "analytics",
            "snapshot",
        }:
            if value is None:
                missing.append(method)
        elif not callable(value):
            missing.append(method)
    if missing:
        raise RuntimeCompositionError(
            f"{name} is missing required contract member(s): "
            + ", ".join(sorted(missing))
        )


def _require_sync_or_async_methods(
    target: Any, name: str, methods: tuple[str, ...]
) -> None:
    missing = [
        method
        for method in methods
        if not callable(getattr(target, method, None))
        and not callable(getattr(target, f"{method}_async", None))
    ]
    if missing:
        raise RuntimeCompositionError(
            f"{name} is missing required sync/async contract member(s): "
            + ", ".join(sorted(missing))
        )


__all__ = [
    "GraphObserver",
    "GraphRuntime",
    "NoOpGraphObserver",
    "RuntimeCompositionError",
    "build_graph_runtime",
]
