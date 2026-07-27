"""Public graph runtime composition and async-safe service facade."""

from __future__ import annotations

import asyncio
from contextvars import ContextVar
from dataclasses import dataclass
import importlib
import inspect
import logging
from time import perf_counter
from typing import Any, Awaitable, Callable, Mapping, Protocol, runtime_checkable

from potpie_context_core.definition import DEFAULT_GRAPH_DEFINITION, GraphDefinition
from potpie_context_core.mutation_policy import (
    DEFAULT_MUTATION_POLICY,
    GraphMutationPolicy,
)
from potpie_context_core.ports.graph.backend import GraphBackend
from potpie_context_core.ports.graph.inbox_store import GraphInboxStorePort
from potpie_context_core.ports.graph.plan_store import GraphPlanStorePort
from potpie_context_core.reconciliation_config import (
    DEFAULT_RECONCILIATION_CONFIG,
    ReconciliationConfig,
)
from potpie_context_core.workbench_service import GraphWorkbenchService


class RuntimeCompositionError(TypeError):
    """Runtime wiring does not implement the documented public contracts."""


_BRIDGE_LOOP: ContextVar[asyncio.AbstractEventLoop | None] = ContextVar(
    "graph_runtime_bridge_loop", default=None
)
_LOG = logging.getLogger(__name__)


class _AsyncPortBridge:
    """Expose sync-shaped calls to async-only ports from runtime worker threads."""

    def __init__(self, target: Any) -> None:
        self._target = target

    def __getattr__(self, name: str) -> Any:
        value = getattr(self._target, name, None)
        if value is not None:
            return value
        if name.endswith("_async"):
            sync_method = getattr(self._target, name.removesuffix("_async"), None)
            if callable(sync_method):

                async def call_async(*args: Any, **kwargs: Any) -> Any:
                    return await asyncio.to_thread(sync_method, *args, **kwargs)

                return call_async
        async_method = getattr(self._target, f"{name}_async", None)
        if not callable(async_method):
            raise AttributeError(name)

        def call(*args: Any, **kwargs: Any) -> Any:
            loop = _BRIDGE_LOOP.get()
            if loop is None or not loop.is_running():
                raise RuntimeCompositionError(
                    f"async-only port method {name!r} requires the GraphRuntime "
                    "async service path"
                )
            future = asyncio.run_coroutine_threadsafe(
                async_method(*args, **kwargs), loop
            )
            return future.result()

        return call


class _MutationPortBridge:
    """Protocol-visible facade for a sync or async mutation port."""

    def __init__(self, target: Any) -> None:
        self._bridge = _AsyncPortBridge(target)

    def apply(self, *args: Any, **kwargs: Any) -> Any:
        return self._bridge.apply(*args, **kwargs)

    async def apply_async(self, *args: Any, **kwargs: Any) -> Any:
        return await self._bridge.apply_async(*args, **kwargs)

    def invalidate(self, *args: Any, **kwargs: Any) -> Any:
        return self._bridge.invalidate(*args, **kwargs)

    def reset_pot(self, *args: Any, **kwargs: Any) -> Any:
        return self._bridge.reset_pot(*args, **kwargs)

    def readiness(self, *args: Any, **kwargs: Any) -> Any:
        return self._bridge.readiness(*args, **kwargs)


class _ClaimQueryPortBridge:
    """Protocol-visible facade for a sync or async claim-query port."""

    def __init__(self, target: Any) -> None:
        self._bridge = _AsyncPortBridge(target)

    def find_claims(self, *args: Any, **kwargs: Any) -> Any:
        return self._bridge.find_claims(*args, **kwargs)

    def entity_labels(self, *args: Any, **kwargs: Any) -> Any:
        return self._bridge.entity_labels(*args, **kwargs)

    def entity_properties(self, *args: Any, **kwargs: Any) -> Any:
        return self._bridge.entity_properties(*args, **kwargs)


class _StoreBridge:
    """Protocol-visible facade for a sync or async plan/inbox store."""

    def __init__(self, target: Any) -> None:
        self._bridge = _AsyncPortBridge(target)

    def save(self, *args: Any, **kwargs: Any) -> Any:
        return self._bridge.save(*args, **kwargs)

    def get(self, *args: Any, **kwargs: Any) -> Any:
        return self._bridge.get(*args, **kwargs)

    def compare_and_set(self, *args: Any, **kwargs: Any) -> Any:
        return self._bridge.compare_and_set(*args, **kwargs)

    def list(self, *args: Any, **kwargs: Any) -> Any:
        return self._bridge.list(*args, **kwargs)

    async def save_async(self, *args: Any, **kwargs: Any) -> Any:
        return await self._bridge.save_async(*args, **kwargs)

    async def get_async(self, *args: Any, **kwargs: Any) -> Any:
        return await self._bridge.get_async(*args, **kwargs)

    async def compare_and_set_async(self, *args: Any, **kwargs: Any) -> Any:
        return await self._bridge.compare_and_set_async(*args, **kwargs)

    async def list_async(self, *args: Any, **kwargs: Any) -> Any:
        return await self._bridge.list_async(*args, **kwargs)


class _BackendBridge:
    """Protocol-visible facade for a backend with async-capable nested ports."""

    def __init__(self, backend: Any) -> None:
        self._backend = backend
        self._mutation = _MutationPortBridge(backend.mutation)
        self._claim_query = _ClaimQueryPortBridge(backend.claim_query)
        self._semantic = _AsyncPortBridge(backend.semantic)
        self._inspection = _AsyncPortBridge(backend.inspection)
        self._analytics = _AsyncPortBridge(backend.analytics)
        self._snapshot = _AsyncPortBridge(backend.snapshot)

    @property
    def profile(self) -> str:
        return self._backend.profile

    @property
    def mutation(self) -> Any:
        return self._mutation

    @property
    def claim_query(self) -> Any:
        return self._claim_query

    @property
    def semantic(self) -> Any:
        return self._semantic

    @property
    def inspection(self) -> Any:
        return self._inspection

    @property
    def analytics(self) -> Any:
        return self._analytics

    @property
    def snapshot(self) -> Any:
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

    def _status_payload(self, pot_id: str) -> dict[str, Any]:
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

    def _notify(self, event: str, fields: Mapping[str, Any]) -> None:
        try:
            self.observability.observe(event, fields)
        except Exception:
            _LOG.warning("graph observer failed for %s", event, exc_info=True)

    def _run_observed(
        self,
        operation: str,
        fields: Mapping[str, Any],
        callback: Callable[[], Any],
    ) -> Any:
        event = f"graph.{operation}"
        started = perf_counter()
        self._notify(event, {**fields, "phase": "started"})
        try:
            result = callback()
        except Exception as exc:
            self._notify(
                event,
                {
                    **fields,
                    "phase": "failed",
                    "ok": False,
                    "error_type": type(exc).__name__,
                    "duration_ms": (perf_counter() - started) * 1000,
                },
            )
            raise
        self._notify(
            event,
            {
                **fields,
                **_result_observation(result),
                "phase": "completed",
                "duration_ms": (perf_counter() - started) * 1000,
            },
        )
        return result

    async def _run_observed_async(
        self,
        operation: str,
        fields: Mapping[str, Any],
        callback: Callable[[], Awaitable[Any]],
    ) -> Any:
        event = f"graph.{operation}"
        started = perf_counter()
        self._notify(event, {**fields, "phase": "started"})
        try:
            result = await callback()
        except Exception as exc:
            self._notify(
                event,
                {
                    **fields,
                    "phase": "failed",
                    "ok": False,
                    "error_type": type(exc).__name__,
                    "duration_ms": (perf_counter() - started) * 1000,
                },
            )
            raise
        self._notify(
            event,
            {
                **fields,
                **_result_observation(result),
                "phase": "completed",
                "duration_ms": (perf_counter() - started) * 1000,
            },
        )
        return result

    def status(self, pot_id: str) -> dict[str, Any]:
        return self._run_observed(
            "status", {"pot_id": pot_id}, lambda: self._status_payload(pot_id)
        )

    def catalog(self, request):
        return self._run_observed(
            "catalog", _request_fields(request), lambda: self.graph.catalog(request)
        )

    def resolve(self, request):
        return self._run_observed(
            "resolve", _request_fields(request), lambda: self.graph.resolve(request)
        )

    def describe(self, request):
        return self._run_observed(
            "describe", _request_fields(request), lambda: self.graph.describe(request)
        )

    def read(self, request):
        return self._run_observed(
            "read",
            {
                **_request_fields(request),
                "view": f"{request.subgraph}.{request.view}",
            },
            lambda: self.graph.read(request),
        )

    def search(self, request):
        return self._run_observed(
            "search", _request_fields(request), lambda: self.graph.search(request)
        )

    def record(self, request):
        return self._run_observed(
            "record", _request_fields(request), lambda: self.graph.record(request)
        )

    def search_entities(self, request):
        return self._run_observed(
            "search_entities",
            _request_fields(request),
            lambda: self.graph.search_entities(request),
        )

    def mutate(self, request):
        return self._run_observed(
            "mutate", _request_fields(request), lambda: self.graph.mutate(request)
        )

    def propose(self, payload, *, pot_id: str, ttl_seconds: int | None = None):
        return self._run_observed(
            "propose",
            {"pot_id": pot_id},
            lambda: self.workbench.propose(
                payload, pot_id=pot_id, ttl_seconds=ttl_seconds
            ),
        )

    def commit(
        self,
        plan_id: str,
        *,
        pot_id: str,
        approved_by: str | None = None,
        verify: bool = False,
    ):
        return self._run_observed(
            "commit",
            {"pot_id": pot_id, "plan_id": plan_id},
            lambda: self.workbench.commit(
                plan_id,
                pot_id=pot_id,
                approved_by=approved_by,
                verify=verify,
            ),
        )

    def history(self, **kwargs):
        return self._run_observed(
            "history", _keyword_fields(kwargs), lambda: self.workbench.history(**kwargs)
        )

    def quality(self, **kwargs):
        return self._run_observed(
            "quality", _keyword_fields(kwargs), lambda: self.workbench.quality(**kwargs)
        )

    def inbox_add(self, **kwargs):
        return self._run_observed(
            "inbox_add",
            _keyword_fields(kwargs),
            lambda: self.workbench.inbox_add(**kwargs),
        )

    def inbox_list(self, **kwargs):
        return self._run_observed(
            "inbox_list",
            _keyword_fields(kwargs),
            lambda: self.workbench.inbox_list(**kwargs),
        )

    def inbox_show(self, **kwargs):
        return self._run_observed(
            "inbox_show",
            _keyword_fields(kwargs),
            lambda: self.workbench.inbox_show(**kwargs),
        )

    def inbox_claim(self, **kwargs):
        return self._run_observed(
            "inbox_claim",
            _keyword_fields(kwargs),
            lambda: self.workbench.inbox_claim(**kwargs),
        )

    def inbox_mark_applied(self, **kwargs):
        return self._run_observed(
            "inbox_mark_applied",
            _keyword_fields(kwargs),
            lambda: self.workbench.inbox_mark_applied(**kwargs),
        )

    def inbox_mark_rejected(self, **kwargs):
        return self._run_observed(
            "inbox_mark_rejected",
            _keyword_fields(kwargs),
            lambda: self.workbench.inbox_mark_rejected(**kwargs),
        )

    def inbox_close(self, **kwargs):
        return self._run_observed(
            "inbox_close",
            _keyword_fields(kwargs),
            lambda: self.workbench.inbox_close(**kwargs),
        )

    async def status_async(self, pot_id: str) -> dict[str, Any]:
        return await self._run_observed_async(
            "status",
            {"pot_id": pot_id},
            lambda: _to_thread_with_bridge(self._status_payload, pot_id),
        )

    async def catalog_async(self, request):
        return await self._run_observed_async(
            "catalog",
            _request_fields(request),
            lambda: _async_call(self.graph, "catalog", request),
        )

    async def resolve_async(self, request):
        return await self._run_observed_async(
            "resolve",
            _request_fields(request),
            lambda: _async_call(self.graph, "resolve", request),
        )

    async def describe_async(self, request):
        return await self._run_observed_async(
            "describe",
            _request_fields(request),
            lambda: _async_call(self.graph, "describe", request),
        )

    async def read_async(self, request):
        return await self._run_observed_async(
            "read",
            {
                **_request_fields(request),
                "view": f"{request.subgraph}.{request.view}",
            },
            lambda: _async_call(self.graph, "read", request),
        )

    async def search_async(self, request):
        return await self._run_observed_async(
            "search",
            _request_fields(request),
            lambda: _async_call(self.graph, "search", request),
        )

    async def record_async(self, request):
        return await self._run_observed_async(
            "record",
            _request_fields(request),
            lambda: _async_call(self.graph, "record", request),
        )

    async def search_entities_async(self, request):
        return await self._run_observed_async(
            "search_entities",
            _request_fields(request),
            lambda: _async_call(self.graph, "search_entities", request),
        )

    async def mutate_async(self, request):
        return await self._run_observed_async(
            "mutate",
            _request_fields(request),
            lambda: _async_call(self.graph, "mutate", request),
        )

    async def propose_async(
        self, payload, *, pot_id: str, ttl_seconds: int | None = None
    ):
        return await self._run_observed_async(
            "propose",
            {"pot_id": pot_id},
            lambda: _async_call(
                self.workbench,
                "propose",
                payload,
                pot_id=pot_id,
                ttl_seconds=ttl_seconds,
            ),
        )

    async def commit_async(
        self,
        plan_id: str,
        *,
        pot_id: str,
        approved_by: str | None = None,
        verify: bool = False,
    ):
        return await self._run_observed_async(
            "commit",
            {"pot_id": pot_id, "plan_id": plan_id},
            lambda: _async_call(
                self.workbench,
                "commit",
                plan_id,
                pot_id=pot_id,
                approved_by=approved_by,
                verify=verify,
            ),
        )

    async def history_async(self, **kwargs):
        return await self._run_observed_async(
            "history",
            _keyword_fields(kwargs),
            lambda: _async_call(self.workbench, "history", **kwargs),
        )

    async def quality_async(self, **kwargs):
        return await self._run_observed_async(
            "quality",
            _keyword_fields(kwargs),
            lambda: _async_call(self.workbench, "quality", **kwargs),
        )

    async def inbox_add_async(self, **kwargs):
        return await self._run_observed_async(
            "inbox_add",
            _keyword_fields(kwargs),
            lambda: _async_call(self.workbench, "inbox_add", **kwargs),
        )

    async def inbox_list_async(self, **kwargs):
        return await self._run_observed_async(
            "inbox_list",
            _keyword_fields(kwargs),
            lambda: _async_call(self.workbench, "inbox_list", **kwargs),
        )

    async def inbox_show_async(self, **kwargs):
        return await self._run_observed_async(
            "inbox_show",
            _keyword_fields(kwargs),
            lambda: _async_call(self.workbench, "inbox_show", **kwargs),
        )

    async def inbox_claim_async(self, **kwargs):
        return await self._run_observed_async(
            "inbox_claim",
            _keyword_fields(kwargs),
            lambda: _async_call(self.workbench, "inbox_claim", **kwargs),
        )

    async def inbox_mark_applied_async(self, **kwargs):
        return await self._run_observed_async(
            "inbox_mark_applied",
            _keyword_fields(kwargs),
            lambda: _async_call(self.workbench, "inbox_mark_applied", **kwargs),
        )

    async def inbox_mark_rejected_async(self, **kwargs):
        return await self._run_observed_async(
            "inbox_mark_rejected",
            _keyword_fields(kwargs),
            lambda: _async_call(self.workbench, "inbox_mark_rejected", **kwargs),
        )

    async def inbox_close_async(self, **kwargs):
        return await self._run_observed_async(
            "inbox_close",
            _keyword_fields(kwargs),
            lambda: _async_call(self.workbench, "inbox_close", **kwargs),
        )


def _request_fields(request: Any) -> dict[str, Any]:
    pot_id = getattr(request, "pot_id", None)
    return {"pot_id": pot_id} if pot_id is not None else {}


def _keyword_fields(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: kwargs[key]
        for key in ("pot_id", "plan_id", "item_id")
        if kwargs.get(key) is not None
    }


def _result_observation(result: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for name in ("ok", "status", "plan_id", "mutation_id", "action"):
        value = getattr(result, name, None)
        if value is not None:
            fields[name] = value
    if "ok" not in fields:
        fields["ok"] = True
    return fields


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
    reconciliation_config: ReconciliationConfig = DEFAULT_RECONCILIATION_CONFIG,
    observability: GraphObserver | None = None,
) -> GraphRuntime:
    """Validate composition and return the single supported graph runtime."""

    if not isinstance(definition, GraphDefinition):
        raise RuntimeCompositionError("definition must be a GraphDefinition")
    if not isinstance(reconciliation_config, ReconciliationConfig):
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
            inbox_store,
            "inbox_store",
            ("save", "get", "compare_and_set", "list"),
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
    runtime_plan_store = _StoreBridge(plan_store)
    runtime_inbox_store = _StoreBridge(inbox_store) if inbox_store is not None else None
    graph = composition.build_graph_service(
        backend=runtime_backend,
        definition=definition,
        policy=policy,
        reconciliation_config=reconciliation_config,
    )
    workbench = GraphWorkbenchService(
        backend=runtime_backend,
        plan_store=runtime_plan_store,
        inbox_store=runtime_inbox_store,
        definition=definition,
        policy=policy,
        reconciliation_config=reconciliation_config,
    )
    return GraphRuntime(
        backend=runtime_backend,
        plan_store=runtime_plan_store,
        inbox_store=runtime_inbox_store,
        definition=definition,
        policy=policy,
        reconciliation_config=reconciliation_config,
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
