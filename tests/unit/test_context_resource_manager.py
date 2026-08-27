from __future__ import annotations

# ruff: noqa: S101 - pytest unit tests use assertions intentionally.

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, cast

import pytest

from potpie.runtime import (
    AcquisitionRequest,
    AuthenticatedActor,
    AuthenticationError,
    AuthorizationError,
    AuthorizationScope,
    CompositionFingerprint,
    ContextResourceManager,
    ContextSelector,
    DestructiveIntent,
    HostResource,
    ResourceComposition,
    ResourceLifecycleError,
    SelectionError,
)
from potpie_context_engine import (
    ContextEngine,
    ContextIdentity,
    EngineConfig,
    EngineDependencies,
    EngineLifecycleError,
    EngineResource,
    Failure,
    Outcome,
    Success,
)


Event = tuple[str, ...]


@dataclass
class _Resolver:
    events: list[Event]
    failure: SelectionError | None = None

    async def resolve(self, selector: ContextSelector):
        self.events.append(("resolve", selector.kind, selector.value or ""))
        if self.failure is not None:
            return Failure(self.failure)
        return Success(ContextIdentity(selector.value or "active-context"))


@dataclass
class _Authenticator:
    events: list[Event]
    failure: AuthenticationError | None = None

    async def authenticate(self, authentication: object):
        self.events.append(("authenticate", str(authentication)))
        if self.failure is not None:
            return Failure(self.failure)
        return Success(AuthenticatedActor(actor_id="actor-1"))


@dataclass
class _Authorizer:
    events: list[Event]
    failure: AuthorizationError | None = None
    scope_factory: (
        Callable[[AuthenticatedActor, str, ContextIdentity], AuthorizationScope] | None
    ) = None

    async def authorize(
        self,
        actor: AuthenticatedActor,
        operation: str,
        context: ContextIdentity,
    ):
        self.events.append(("authorize", actor.actor_id, operation, context.value))
        if self.failure is not None:
            return Failure(self.failure)
        if self.scope_factory is not None:
            return Success(self.scope_factory(actor, operation, context))
        return Success(
            AuthorizationScope(
                actor_id=actor.actor_id,
                operation=operation,
                context=context,
            )
        )


class _Composer:
    def __init__(
        self,
        events: list[Event],
        cleanup_events: list[str] | None = None,
        *,
        fingerprint_failure: ResourceLifecycleError | None = None,
        composition_failure: ResourceLifecycleError | None = None,
        fingerprint_for: Callable[[ContextIdentity], str] | None = None,
        composition_fingerprint_for: Callable[[ContextIdentity], str] | None = None,
        dependencies_for: Callable[[ContextIdentity], EngineDependencies] | None = None,
        host_resource_ownership: dict[str, Literal["retained", "transferred"]]
        | None = None,
        release_failures: set[str] | None = None,
        compose_gate: asyncio.Event | None = None,
        expected_concurrent_compositions: int = 0,
    ) -> None:
        self.events = events
        self.cleanup_events = cleanup_events if cleanup_events is not None else []
        self.fingerprint_failure = fingerprint_failure
        self.composition_failure = composition_failure
        self.fingerprint_for = fingerprint_for or (
            lambda context: f"fingerprint:{context.value}"
        )
        self.composition_fingerprint_for = (
            composition_fingerprint_for or self.fingerprint_for
        )
        self.dependencies_for = dependencies_for or (lambda _context: _dependencies())
        self.host_resource_ownership = host_resource_ownership or {}
        self.release_failures = release_failures or set()
        self.compose_gate = compose_gate
        self.compose_count = 0
        self.concurrent_compositions_reached = asyncio.Event()
        self.expected_concurrent_compositions = expected_concurrent_compositions

    async def fingerprint(self, context: ContextIdentity):
        self.events.append(("fingerprint", context.value))
        if self.fingerprint_failure is not None:
            return Failure(self.fingerprint_failure)
        return Success(CompositionFingerprint(self.fingerprint_for(context)))

    async def compose(
        self, context: ContextIdentity, fingerprint: CompositionFingerprint
    ):
        self.compose_count += 1
        self.events.append(("compose", context.value, fingerprint.value))
        if self.compose_count >= self.expected_concurrent_compositions > 0:
            self.concurrent_compositions_reached.set()
        if self.compose_gate is not None:
            await self.compose_gate.wait()
        if self.composition_failure is not None:
            return Failure(self.composition_failure)

        async def release_first() -> None:
            self.cleanup_events.append(f"release:{context.value}:first")
            if "first" in self.release_failures:
                raise RuntimeError("first cleanup failed")

        async def release_second() -> None:
            self.cleanup_events.append(f"release:{context.value}:second")
            if "second" in self.release_failures:
                raise RuntimeError("second cleanup failed")

        return Success(
            ResourceComposition(
                fingerprint=CompositionFingerprint(
                    self.composition_fingerprint_for(context)
                ),
                config=EngineConfig(values={"context": context.value}),
                dependencies=self.dependencies_for(context),
                resources=(
                    HostResource(
                        name="first",
                        release=release_first,
                        ownership=self.host_resource_ownership.get("first", "retained"),
                    ),
                    HostResource(
                        name="second",
                        release=release_second,
                        ownership=self.host_resource_ownership.get(
                            "second", "retained"
                        ),
                    ),
                ),
            )
        )


class _FakeEngine:
    def __init__(
        self,
        context: ContextIdentity,
        cleanup_events: list[str],
        resources: tuple[EngineResource, ...],
        close_failure: Exception | None = None,
    ) -> None:
        self.context = context
        self.cleanup_events = cleanup_events
        self.resources = resources
        self.close_failure = close_failure
        self.close_count = 0

    async def close(self):
        self.close_count += 1
        self.cleanup_events.append(f"close:{self.context.value}")
        if self.close_failure is not None:
            raise self.close_failure
        for resource in reversed(self.resources):
            if resource.ownership == "transferred" and resource.close is not None:
                await resource.close()
        return Success(None)


class _EngineFactory:
    def __init__(
        self,
        events: list[Event],
        cleanup_events: list[str] | None = None,
        *,
        failure: EngineLifecycleError | None = None,
        close_failure: Exception | None = None,
    ) -> None:
        self.events = events
        self.cleanup_events = cleanup_events if cleanup_events is not None else []
        self.failure = failure
        self.close_failure = close_failure
        self.engines: list[_FakeEngine] = []

    async def __call__(
        self,
        *,
        context: ContextIdentity,
        config: EngineConfig,
        dependencies: EngineDependencies,
    ) -> Outcome[ContextEngine]:
        self.events.append(("create_engine", context.value, str(config.values)))
        if self.failure is not None:
            return Failure(self.failure)
        engine = _FakeEngine(
            context,
            self.cleanup_events,
            dependencies.resources,
            self.close_failure,
        )
        self.engines.append(engine)
        return Success(cast(ContextEngine, engine))


def _dependencies() -> EngineDependencies:
    operations = cast(object, object())
    return EngineDependencies(
        context=cast("object", operations),
        graph=cast("object", operations),
        workbench=cast("object", operations),
        ingestion=cast("object", operations),
        nudge=cast("object", operations),
        resources=(EngineResource(name="backend", ownership="borrowed"),),
    )


def _dependencies_with_transferred_resource(
    cleanup_events: list[str],
) -> EngineDependencies:
    dependencies = _dependencies()

    async def close_second() -> None:
        cleanup_events.append("engine-release:second")

    return EngineDependencies(
        context=dependencies.context,
        graph=dependencies.graph,
        workbench=dependencies.workbench,
        ingestion=dependencies.ingestion,
        nudge=dependencies.nudge,
        resources=(
            EngineResource(
                name="second",
                ownership="transferred",
                close=close_second,
            ),
        ),
    )


def _request(
    context: str = "context-a",
    *,
    request_id: str = "request-1",
    operation: str = "search",
    destructive: bool = False,
    destructive_intent: DestructiveIntent | None = None,
) -> AcquisitionRequest:
    return AcquisitionRequest(
        request_id=request_id,
        selector=ContextSelector(kind="explicit", value=context),
        operation=operation,
        authentication="credential",
        destructive=destructive,
        destructive_intent=destructive_intent,
    )


def _manager(
    events: list[Event],
    *,
    cleanup_events: list[str] | None = None,
    resolver: _Resolver | None = None,
    authenticator: _Authenticator | None = None,
    authorizer: _Authorizer | None = None,
    composer: _Composer | None = None,
    factory: _EngineFactory | None = None,
) -> tuple[ContextResourceManager, _Composer, _EngineFactory]:
    resolved_composer = composer or _Composer(events, cleanup_events)
    resolved_factory = factory or _EngineFactory(events, cleanup_events)
    return (
        ContextResourceManager(
            resolver=resolver or _Resolver(events),
            authenticator=authenticator or _Authenticator(events),
            authorizer=authorizer or _Authorizer(events),
            composer=resolved_composer,
            engine_factory=resolved_factory,
        ),
        resolved_composer,
        resolved_factory,
    )


@pytest.mark.anyio
async def test_acquire_resolves_authenticates_authorizes_then_composes() -> None:
    events: list[Event] = []
    manager, _, _ = _manager(events)

    outcome = await manager.acquire(_request())

    assert isinstance(outcome, Success)
    assert outcome.value.context == ContextIdentity("context-a")
    assert outcome.value.scope == AuthorizationScope(
        actor_id="actor-1",
        operation="search",
        context=ContextIdentity("context-a"),
    )
    assert outcome.value.engine.context == ContextIdentity("context-a")
    assert outcome.value.ownership.engine_lifetime == "resource_manager_shutdown"
    assert outcome.value.ownership.release_closes_engine is False
    assert outcome.value.ownership.host_resources == ("first", "second")
    assert tuple(
        resource.name for resource in outcome.value.ownership.engine_resources
    ) == ("backend",)
    assert [event[0] for event in events] == [
        "resolve",
        "authenticate",
        "authorize",
        "fingerprint",
        "compose",
        "create_engine",
    ]
    await outcome.value.release()
    assert isinstance(await manager.shutdown(), Success)


@pytest.mark.anyio
async def test_compatible_engines_are_cached_until_shutdown() -> None:
    events: list[Event] = []
    manager, composer, factory = _manager(events)

    first = await manager.acquire(_request(request_id="request-1"))
    second = await manager.acquire(_request(request_id="request-2"))

    assert isinstance(first, Success)
    assert isinstance(second, Success)
    assert first.value.engine is second.value.engine
    assert composer.compose_count == 1
    assert len(factory.engines) == 1
    await first.value.release()
    await first.value.release()
    await second.value.release()
    assert factory.engines[0].close_count == 0
    assert isinstance(await manager.shutdown(), Success)
    assert factory.engines[0].close_count == 1


@pytest.mark.anyio
async def test_different_contexts_receive_different_cached_engines() -> None:
    events: list[Event] = []
    manager, composer, factory = _manager(events)

    first = await manager.acquire(_request("context-a", request_id="request-1"))
    second = await manager.acquire(_request("context-b", request_id="request-2"))

    assert isinstance(first, Success)
    assert isinstance(second, Success)
    assert first.value.engine is not second.value.engine
    assert composer.compose_count == 2
    assert len(factory.engines) == 2
    await first.value.release()
    await second.value.release()
    assert isinstance(await manager.shutdown(), Success)


@pytest.mark.anyio
@pytest.mark.parametrize(
    "failure_stage", ["selection", "authentication", "authorization"]
)
async def test_boundary_failure_stops_before_composition(failure_stage: str) -> None:
    events: list[Event] = []
    resolver = _Resolver(events)
    authenticator = _Authenticator(events)
    authorizer = _Authorizer(events)
    if failure_stage == "selection":
        resolver.failure = SelectionError(code="not_found", message="not found")
    elif failure_stage == "authentication":
        authenticator.failure = AuthenticationError(
            code="bad_credential", message="bad credential"
        )
    else:
        authorizer.failure = AuthorizationError(code="forbidden", message="forbidden")
    manager, composer, factory = _manager(
        events,
        resolver=resolver,
        authenticator=authenticator,
        authorizer=authorizer,
    )

    outcome = await manager.acquire(_request())

    assert isinstance(outcome, Failure)
    assert outcome.error.category == failure_stage
    assert composer.compose_count == 0
    assert factory.engines == []
    assert isinstance(await manager.shutdown(), Success)


@pytest.mark.anyio
async def test_rejects_authorization_scope_that_does_not_match_request() -> None:
    events: list[Event] = []
    authorizer = _Authorizer(
        events,
        scope_factory=lambda actor, operation, context: AuthorizationScope(
            actor_id=actor.actor_id,
            operation=f"other-{operation}",
            context=context,
        ),
    )
    manager, composer, _ = _manager(events, authorizer=authorizer)

    outcome = await manager.acquire(_request())

    assert isinstance(outcome, Failure)
    assert outcome.error.code == "authorization_scope_invalid"
    assert composer.compose_count == 0
    assert isinstance(await manager.shutdown(), Success)


@pytest.mark.anyio
async def test_destructive_intent_must_match_exact_authorized_request() -> None:
    events: list[Event] = []
    manager, composer, _ = _manager(events)
    request = _request(operation="repair", destructive=True)

    missing = await manager.acquire(request)
    mismatched = await manager.acquire(
        _request(
            request_id="request-2",
            operation="repair",
            destructive=True,
            destructive_intent=DestructiveIntent(
                confirmed=True,
                operation="repair",
                selector=ContextSelector(kind="explicit", value="context-a"),
                request_id="different-request",
            ),
        )
    )
    accepted_request = _request(
        request_id="request-3",
        operation="repair",
        destructive=True,
        destructive_intent=DestructiveIntent(
            confirmed=True,
            operation="repair",
            selector=ContextSelector(kind="explicit", value="context-a"),
            request_id="request-3",
        ),
    )
    accepted = await manager.acquire(accepted_request)

    assert isinstance(missing, Failure)
    assert missing.error.code == "destructive_intent_invalid"
    assert isinstance(mismatched, Failure)
    assert mismatched.error.code == "destructive_intent_invalid"
    assert isinstance(accepted, Success)
    assert composer.compose_count == 1
    await accepted.value.release()
    assert isinstance(await manager.shutdown(), Success)


@pytest.mark.anyio
async def test_failed_engine_construction_releases_resources_in_reverse_order() -> None:
    events: list[Event] = []
    cleanup_events: list[str] = []
    factory = _EngineFactory(
        events,
        cleanup_events,
        failure=EngineLifecycleError(
            code="bad_dependencies",
            message="secret dependency failure",
            retry_posture="unknown",
        ),
    )
    manager, _, _ = _manager(
        events,
        cleanup_events=cleanup_events,
        factory=factory,
    )

    outcome = await manager.acquire(_request())

    assert isinstance(outcome, Failure)
    assert outcome.error.code == "engine_construction_failed"
    assert outcome.error.details == {
        "engine_error_code": "bad_dependencies",
        "acquisition_failure": {
            "category": "engine_lifecycle",
            "code": "bad_dependencies",
            "retry_posture": "unknown",
        },
    }
    assert "secret" not in outcome.error.message
    assert cleanup_events == [
        "release:context-a:second",
        "release:context-a:first",
    ]
    assert manager.cached_engine_count == 0
    assert isinstance(await manager.shutdown(), Success)


@pytest.mark.anyio
async def test_engine_factory_exception_releases_composed_resources() -> None:
    events: list[Event] = []
    cleanup_events: list[str] = []

    async def raising_factory(**_kwargs: object) -> Outcome[ContextEngine]:
        raise RuntimeError("sensitive construction detail")

    manager = ContextResourceManager(
        resolver=_Resolver(events),
        authenticator=_Authenticator(events),
        authorizer=_Authorizer(events),
        composer=_Composer(events, cleanup_events),
        engine_factory=raising_factory,
    )

    outcome = await manager.acquire(_request())

    assert isinstance(outcome, Failure)
    assert outcome.error.code == "engine_construction_failed"
    assert outcome.error.details == {
        "error_type": "RuntimeError",
        "acquisition_failure": {
            "category": "resource_lifecycle",
            "code": "engine_construction_failed",
            "retry_posture": "unknown",
        },
    }
    assert "sensitive" not in str(outcome.error.details)
    assert cleanup_events == [
        "release:context-a:second",
        "release:context-a:first",
    ]
    assert manager.cached_engine_count == 0


@pytest.mark.anyio
async def test_fingerprint_mismatch_preserves_cleanup_failure_detail() -> None:
    events: list[Event] = []
    cleanup_events: list[str] = []
    composer = _Composer(
        events,
        cleanup_events,
        composition_fingerprint_for=lambda context: f"changed:{context.value}",
        release_failures={"second"},
    )
    manager, _, factory = _manager(
        events,
        cleanup_events=cleanup_events,
        composer=composer,
    )

    outcome = await manager.acquire(_request())

    assert isinstance(outcome, Failure)
    assert outcome.error.code == "composition_fingerprint_changed"
    assert outcome.error.details == {
        "acquisition_failure": {
            "category": "resource_lifecycle",
            "code": "composition_fingerprint_changed",
            "retry_posture": "safe",
        },
        "cleanup_failures": ({"resource": "second", "error_type": "RuntimeError"},),
    }
    assert cleanup_events == [
        "release:context-a:second",
        "release:context-a:first",
    ]
    assert factory.engines == []
    assert manager.cached_engine_count == 0
    assert isinstance(await manager.shutdown(), Success)


@pytest.mark.anyio
async def test_failed_engine_construction_preserves_acquisition_and_cleanup() -> None:
    events: list[Event] = []
    cleanup_events: list[str] = []
    composer = _Composer(
        events,
        cleanup_events,
        release_failures={"first"},
    )
    factory = _EngineFactory(
        events,
        cleanup_events,
        failure=EngineLifecycleError(
            code="bad_dependencies",
            message="sensitive dependency failure",
            retry_posture="unknown",
        ),
    )
    manager, _, _ = _manager(
        events,
        cleanup_events=cleanup_events,
        composer=composer,
        factory=factory,
    )

    outcome = await manager.acquire(_request())

    assert isinstance(outcome, Failure)
    assert outcome.error.details == {
        "engine_error_code": "bad_dependencies",
        "acquisition_failure": {
            "category": "engine_lifecycle",
            "code": "bad_dependencies",
            "retry_posture": "unknown",
        },
        "cleanup_failures": ({"resource": "first", "error_type": "RuntimeError"},),
    }
    assert "sensitive" not in str(outcome.error.details)
    assert cleanup_events == [
        "release:context-a:second",
        "release:context-a:first",
    ]
    assert isinstance(await manager.shutdown(), Success)


@pytest.mark.anyio
async def test_transferred_resource_has_one_terminal_cleanup_owner() -> None:
    events: list[Event] = []
    cleanup_events: list[str] = []
    composer = _Composer(
        events,
        cleanup_events,
        dependencies_for=lambda _context: _dependencies_with_transferred_resource(
            cleanup_events
        ),
        host_resource_ownership={"second": "transferred"},
    )
    manager, _, _ = _manager(
        events,
        cleanup_events=cleanup_events,
        composer=composer,
    )

    acquired = await manager.acquire(_request())

    assert isinstance(acquired, Success)
    assert acquired.value.ownership.host_resources == ("first",)
    assert tuple(
        resource.name for resource in acquired.value.ownership.engine_resources
    ) == ("second",)
    await acquired.value.release()
    assert isinstance(await manager.shutdown(), Success)
    assert cleanup_events == [
        "close:context-a",
        "engine-release:second",
        "release:context-a:first",
    ]


@pytest.mark.anyio
async def test_mismatched_transfer_declarations_fail_and_release_host_resources() -> (
    None
):
    events: list[Event] = []
    cleanup_events: list[str] = []
    composer = _Composer(
        events,
        cleanup_events,
        dependencies_for=lambda _context: _dependencies_with_transferred_resource(
            cleanup_events
        ),
    )
    manager, _, factory = _manager(
        events,
        cleanup_events=cleanup_events,
        composer=composer,
    )

    outcome = await manager.acquire(_request())

    assert isinstance(outcome, Failure)
    assert outcome.error.code == "composition_ownership_invalid"
    assert outcome.error.details == {
        "host_transfer_resources": (),
        "engine_transfer_resources": ("second",),
        "acquisition_failure": {
            "category": "resource_lifecycle",
            "code": "composition_ownership_invalid",
            "retry_posture": "unknown",
        },
    }
    assert cleanup_events == [
        "release:context-a:second",
        "release:context-a:first",
    ]
    assert factory.engines == []
    assert isinstance(await manager.shutdown(), Success)


@pytest.mark.anyio
async def test_shutdown_drains_leases_and_closes_in_reverse_acquisition_order() -> None:
    events: list[Event] = []
    cleanup_events: list[str] = []
    manager, _, _ = _manager(events, cleanup_events=cleanup_events)
    first = await manager.acquire(_request("context-a", request_id="request-1"))
    second = await manager.acquire(_request("context-b", request_id="request-2"))
    assert isinstance(first, Success)
    assert isinstance(second, Success)
    await first.value.release()

    shutdown_task = asyncio.create_task(manager.shutdown())
    await asyncio.sleep(0)
    rejected = await manager.acquire(_request("context-c", request_id="request-3"))

    assert not shutdown_task.done()
    assert isinstance(rejected, Failure)
    assert rejected.error.code == "resource_manager_draining"
    assert cleanup_events == []

    await second.value.release()
    shutdown = await asyncio.wait_for(shutdown_task, timeout=1)

    assert isinstance(shutdown, Success)
    assert cleanup_events == [
        "close:context-b",
        "release:context-b:second",
        "release:context-b:first",
        "close:context-a",
        "release:context-a:second",
        "release:context-a:first",
    ]
    assert isinstance(await manager.shutdown(), Success)


@pytest.mark.anyio
async def test_engine_cleanup_exception_does_not_skip_host_resource_release() -> None:
    events: list[Event] = []
    cleanup_events: list[str] = []
    factory = _EngineFactory(
        events,
        cleanup_events,
        close_failure=RuntimeError("sensitive engine cleanup failure"),
    )
    manager, _, _ = _manager(
        events,
        cleanup_events=cleanup_events,
        factory=factory,
    )
    acquired = await manager.acquire(_request())
    assert isinstance(acquired, Success)
    await acquired.value.release()

    shutdown = await manager.shutdown()

    assert isinstance(shutdown, Failure)
    assert shutdown.error.code == "resource_manager_shutdown_failed"
    assert shutdown.error.details == {
        "failures": (
            {
                "kind": "engine",
                "context": "context-a",
                "error_type": "RuntimeError",
            },
        )
    }
    assert "sensitive" not in str(shutdown.error.details)
    assert cleanup_events == [
        "close:context-a",
        "release:context-a:second",
        "release:context-a:first",
    ]
    assert manager.cached_engine_count == 1
    factory.engines[0].close_failure = None

    retried = await manager.shutdown()

    assert isinstance(retried, Success)
    assert manager.cached_engine_count == 0
    assert cleanup_events == [
        "close:context-a",
        "release:context-a:second",
        "release:context-a:first",
        "close:context-a",
    ]


@pytest.mark.anyio
async def test_shutdown_retries_only_failed_host_resources() -> None:
    events: list[Event] = []
    cleanup_events: list[str] = []
    composer = _Composer(
        events,
        cleanup_events,
        release_failures={"second"},
    )
    manager, _, _ = _manager(
        events,
        cleanup_events=cleanup_events,
        composer=composer,
    )
    acquired = await manager.acquire(_request())
    assert isinstance(acquired, Success)
    await acquired.value.release()

    first = await manager.shutdown()
    composer.release_failures.clear()
    second = await manager.shutdown()

    assert isinstance(first, Failure)
    assert isinstance(second, Success)
    assert cleanup_events == [
        "close:context-a",
        "release:context-a:second",
        "release:context-a:first",
        "release:context-a:second",
    ]


@pytest.mark.anyio
async def test_cancelled_release_finishes_lease_decrement_before_marking_released() -> (
    None
):
    events: list[Event] = []
    manager, _, _ = _manager(events)
    acquired = await manager.acquire(_request())
    assert isinstance(acquired, Success)
    lease = acquired.value

    async with manager._state:
        release_task = asyncio.create_task(lease.release())
        await asyncio.sleep(0)
        release_task.cancel()
        assert not lease.is_released

    with pytest.raises(asyncio.CancelledError):
        await release_task

    assert lease.is_released
    assert isinstance(await asyncio.wait_for(manager.shutdown(), timeout=1), Success)


@pytest.mark.anyio
async def test_unrelated_contexts_compose_concurrently() -> None:
    events: list[Event] = []
    gate = asyncio.Event()
    composer = _Composer(
        events,
        compose_gate=gate,
        expected_concurrent_compositions=2,
    )
    manager, _, _ = _manager(events, composer=composer)

    first_task = asyncio.create_task(
        manager.acquire(_request("context-a", request_id="request-1"))
    )
    second_task = asyncio.create_task(
        manager.acquire(_request("context-b", request_id="request-2"))
    )
    await asyncio.wait_for(composer.concurrent_compositions_reached.wait(), timeout=1)
    gate.set()
    first, second = await asyncio.gather(first_task, second_task)

    assert isinstance(first, Success)
    assert isinstance(second, Success)
    assert composer.compose_count == 2
    await first.value.release()
    await second.value.release()
    assert isinstance(await manager.shutdown(), Success)


@pytest.mark.anyio
async def test_same_composition_is_created_once_under_concurrent_acquisition() -> None:
    events: list[Event] = []
    gate = asyncio.Event()
    composer = _Composer(
        events,
        compose_gate=gate,
        expected_concurrent_compositions=1,
    )
    manager, _, _ = _manager(events, composer=composer)

    first_task = asyncio.create_task(manager.acquire(_request(request_id="request-1")))
    await asyncio.wait_for(composer.concurrent_compositions_reached.wait(), timeout=1)
    second_task = asyncio.create_task(manager.acquire(_request(request_id="request-2")))
    await asyncio.sleep(0)
    gate.set()
    first, second = await asyncio.gather(first_task, second_task)

    assert isinstance(first, Success)
    assert isinstance(second, Success)
    assert first.value.engine is second.value.engine
    assert composer.compose_count == 1
    await first.value.release()
    await second.value.release()
    assert isinstance(await manager.shutdown(), Success)
