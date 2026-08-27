from __future__ import annotations

import asyncio
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
    ResetContextRequest,
    SearchRequest,
    SubmitEventRequest,
)


class _Operations:
    def __init__(self) -> None:
        self.calls: list[tuple[str, ContextIdentity, Mapping[str, object]]] = []
        self.raise_on_resolve = False

    async def _record(self, name, context, request):
        if name == "resolve" and self.raise_on_resolve:
            raise RuntimeError("sensitive dependency detail")
        self.calls.append((name, context, request.to_payload()))
        return {"operation": name, "context": context.value}

    async def resolve(self, context, request):
        return await self._record("resolve", context, request)

    async def search(self, context, request):
        self.calls.append(("search", context, request.to_payload()))
        return Failure(DomainError(code="not_found", message="not found"))

    async def catalog(self, context, request):
        return await self._record("catalog", context, request)

    async def inbox_add(self, context, request):
        return await self._record("inbox_add", context, request)

    async def submit_event(self, context, request):
        return await self._record("submit_event", context, request)

    async def nudge(self, context, request):
        return await self._record("nudge", context, request)

    async def reset_context(self, context, request):
        return await self._record("reset_context", context, request)


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
        ("resolve", ResolveRequest(task="find the boundary")),
        ("catalog", CatalogRequest()),
        ("inbox_add", InboxAddRequest(summary="review")),
        (
            "submit_event",
            SubmitEventRequest(
                source_system="test",
                event_type="change",
                action="record",
                source_id="change-1",
            ),
        ),
        ("nudge", NudgeRequest(event="prompt", session_id="session-1")),
        ("reset_context", ResetContextRequest()),
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

    outcome = await engine.search(SearchRequest(query="missing"))

    assert isinstance(outcome, Failure)
    assert isinstance(outcome.error, DomainError)
    assert outcome.error.code == "not_found"


def test_request_types_do_not_accept_context_selectors() -> None:
    with pytest.raises(TypeError, match="pot_id"):
        ResolveRequest(pot_id="other")  # type: ignore[call-arg]


async def test_dependency_exceptions_become_redacted_typed_failures() -> None:
    engine, operations = await _engine()
    operations.raise_on_resolve = True

    outcome = await engine.resolve(ResolveRequest(task="fail"))

    assert isinstance(outcome, Failure)
    assert outcome.error.category == "dependency"
    assert outcome.error.code == "engine_dependency_failed"
    assert outcome.error.details == {
        "operation": "resolve",
        "error_type": "RuntimeError",
    }
    assert "sensitive dependency detail" not in str(outcome.error)


async def test_close_is_idempotent_and_releases_transferred_resources_in_reverse() -> (
    None
):
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


async def test_repeated_close_retries_only_pending_cleanup_failures() -> None:
    attempts = {"flaky": 0, "stable": 0}

    async def flaky() -> None:
        attempts["flaky"] += 1
        if attempts["flaky"] == 1:
            raise RuntimeError("first attempt fails")

    async def stable() -> None:
        attempts["stable"] += 1

    engine, _ = await _engine(
        resources=(
            EngineResource("stable", "transferred", stable),
            EngineResource("flaky", "transferred", flaky),
        )
    )

    first = await engine.close()
    second = await engine.close()
    third = await engine.close()

    assert isinstance(first, Failure)
    assert isinstance(second, Success)
    assert isinstance(third, Success)
    assert attempts == {"flaky": 2, "stable": 1}


async def test_close_drains_active_operations_before_releasing_resources() -> None:
    entered = asyncio.Event()
    proceed = asyncio.Event()
    events: list[str] = []

    class _BlockingOperations(_Operations):
        async def resolve(self, context, request):
            events.append("operation-entered")
            entered.set()
            await proceed.wait()
            events.append("operation-finished")
            return await super().resolve(context, request)

    async def close_resource() -> None:
        events.append("resource-closed")

    operations = _BlockingOperations()
    created = await create_engine(
        context=ContextIdentity("context-123"),
        config=EngineConfig(values={"profile": "test"}),
        dependencies=EngineDependencies(
            context=operations,
            graph=operations,
            workbench=operations,
            ingestion=operations,
            nudge=operations,
            resources=(EngineResource("owned", "transferred", close_resource),),
        ),
    )
    assert isinstance(created, Success)
    engine = created.value

    operation_task = asyncio.create_task(
        engine.resolve(ResolveRequest(task="finish before close"))
    )
    await entered.wait()
    close_task = asyncio.create_task(engine.close())
    await asyncio.sleep(0)

    rejected = await engine.resolve(ResolveRequest(task="too late"))
    assert isinstance(rejected, Failure)
    assert rejected.error.code == "engine_closed"
    assert not close_task.done()

    proceed.set()
    operation, closed = await asyncio.gather(operation_task, close_task)

    assert isinstance(operation, Success)
    assert isinstance(closed, Success)
    assert events == [
        "operation-entered",
        "operation-finished",
        "resource-closed",
    ]


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
        "reset_context",
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
