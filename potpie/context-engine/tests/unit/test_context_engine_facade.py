from __future__ import annotations

from collections.abc import Mapping

import pytest

from potpie_context_engine import (
    ContextEngine,
    ContextIdentity,
    DomainError,
    EngineConfig,
    EngineDependencies,
    EngineLifecycleError,
    EngineResource,
    Failure,
    Success,
    create_engine,
)
from potpie_context_engine.requests import (
    CatalogRequest,
    InboxAddRequest,
    NudgeRequest,
    ResolveRequest,
    SearchRequest,
    SubmitEventRequest,
)


class _Operations:
    def __init__(self) -> None:
        self.calls: list[tuple[str, ContextIdentity, Mapping[str, object]]] = []

    async def _record(self, name, context, request):
        self.calls.append((name, context, request.payload))
        return {"operation": name, "context": context.value}

    async def resolve(self, context, request):
        return await self._record("resolve", context, request)

    async def search(self, context, request):
        self.calls.append(("search", context, request.payload))
        return Failure(DomainError(code="not_found", message="not found"))

    async def catalog(self, context, request):
        return await self._record("catalog", context, request)

    async def inbox_add(self, context, request):
        return await self._record("inbox_add", context, request)

    async def submit_event(self, context, request):
        return await self._record("submit_event", context, request)

    async def nudge(self, context, request):
        return await self._record("nudge", context, request)


async def _engine(
    *, resources: tuple[EngineResource, ...] = ()
) -> tuple[ContextEngine, _Operations]:
    operations = _Operations()
    created = await create_engine(
        context=ContextIdentity("context-123"),
        config=EngineConfig(values={"profile": "test"}),
        dependencies=EngineDependencies(
            context=operations,
            graph=operations,
            workbench=operations,
            ingestion=operations,
            nudge=operations,
            resources=resources,
        ),
    )
    assert isinstance(created, Success)
    return created.value, operations


@pytest.mark.parametrize(
    ("method", "request_value"),
    [
        ("resolve", ResolveRequest({"task": "find the boundary"})),
        ("catalog", CatalogRequest()),
        ("inbox_add", InboxAddRequest({"summary": "review"})),
        ("submit_event", SubmitEventRequest({"kind": "change"})),
        ("nudge", NudgeRequest({"event": "prompt"})),
    ],
)
async def test_facade_delegates_with_the_bound_context(method, request_value) -> None:
    engine, operations = await _engine()

    outcome = await getattr(engine, method)(request_value)

    assert isinstance(outcome, Success)
    assert outcome.value == {"operation": method, "context": "context-123"}
    assert operations.calls[-1][1] == ContextIdentity("context-123")


async def test_facade_preserves_typed_failures() -> None:
    engine, _ = await _engine()

    outcome = await engine.search(SearchRequest({"query": "missing"}))

    assert isinstance(outcome, Failure)
    assert isinstance(outcome.error, DomainError)
    assert outcome.error.code == "not_found"


async def test_request_cannot_override_bound_context() -> None:
    engine, operations = await _engine()

    outcome = await engine.resolve(ResolveRequest({"pot_id": "other"}))

    assert isinstance(outcome, Failure)
    assert isinstance(outcome.error, DomainError)
    assert outcome.error.code == "context_selector_forbidden"
    assert operations.calls == []


async def test_close_is_idempotent_and_releases_transferred_resources_in_reverse() -> None:
    closed: list[str] = []

    async def close(name: str) -> None:
        closed.append(name)

    engine, _ = await _engine(
        resources=(
            EngineResource("borrowed", "borrowed"),
            EngineResource("first", "transferred", lambda: close("first")),
            EngineResource("second", "transferred", lambda: close("second")),
        )
    )

    first = await engine.close()
    second = await engine.close()

    assert isinstance(first, Success)
    assert isinstance(second, Success)
    assert closed == ["second", "first"]


async def test_close_continues_after_cleanup_failure() -> None:
    closed: list[str] = []

    async def fail() -> None:
        raise RuntimeError("secret cleanup detail")

    async def succeed() -> None:
        closed.append("succeeded")

    engine, _ = await _engine(
        resources=(
            EngineResource("succeeds", "transferred", succeed),
            EngineResource("fails", "transferred", fail),
        )
    )

    outcome = await engine.close()

    assert isinstance(outcome, Failure)
    assert isinstance(outcome.error, EngineLifecycleError)
    assert outcome.error.code == "engine_close_failed"
    assert closed == ["succeeded"]
    assert "secret cleanup detail" not in str(outcome.error.details)


async def test_closed_engine_returns_lifecycle_failure() -> None:
    engine, operations = await _engine()
    await engine.close()

    outcome = await engine.resolve(ResolveRequest())

    assert isinstance(outcome, Failure)
    assert isinstance(outcome.error, EngineLifecycleError)
    assert outcome.error.code == "engine_closed"
    assert operations.calls == []


def test_public_facade_has_the_accepted_flat_method_catalog() -> None:
    expected = {
        "resolve",
        "search",
        "record",
        "data_plane_status",
        "catalog",
        "describe",
        "read",
        "search_entities",
        "mutate",
        "neighborhood",
        "inspect",
        "export_snapshot",
        "import_snapshot",
        "repair",
        "propose",
        "commit",
        "history",
        "quality",
        "inbox_add",
        "inbox_list",
        "inbox_show",
        "inbox_claim",
        "inbox_mark_applied",
        "inbox_mark_rejected",
        "inbox_close",
        "submit_event",
        "submit_artifact",
        "processing_status",
        "nudge",
    }

    assert expected <= set(ContextEngine.__dict__)
    assert "execute" not in ContextEngine.__dict__
    assert "call" not in ContextEngine.__dict__
