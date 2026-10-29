"""Authorized context leases and daemon-lifetime engine ownership.

Every acquisition authenticates the caller first, then resolves the selector
for that actor, then authorizes the operation on the resolved context. An
unauthenticated caller therefore never learns whether a pot exists, and a
managed host can resolve the same selector differently for different actors.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Literal, Protocol, TypeAlias

from potpie_context_engine import (
    ContextEngine,
    ContextIdentity,
    EngineConfig,
    EngineDependencies,
    EngineResource,
    Failure,
    Outcome,
    Success,
    create_engine,
)
from potpie_context_engine.outcomes import RetryPosture


@dataclass(frozen=True, slots=True)
class SelectionError:
    code: str
    message: str
    details: Mapping[str, object] = field(default_factory=dict)
    recommended_next_action: str | None = None
    retry_posture: RetryPosture = "not_applicable"
    category: Literal["selection"] = "selection"


@dataclass(frozen=True, slots=True)
class AuthenticationError:
    code: str
    message: str
    details: Mapping[str, object] = field(default_factory=dict)
    recommended_next_action: str | None = None
    retry_posture: RetryPosture = "not_applicable"
    category: Literal["authentication"] = "authentication"


@dataclass(frozen=True, slots=True)
class AuthorizationError:
    code: str
    message: str
    details: Mapping[str, object] = field(default_factory=dict)
    recommended_next_action: str | None = None
    retry_posture: RetryPosture = "not_applicable"
    category: Literal["authorization"] = "authorization"


@dataclass(frozen=True, slots=True)
class ResourceLifecycleError:
    code: str
    message: str
    details: Mapping[str, object] = field(default_factory=dict)
    recommended_next_action: str | None = None
    retry_posture: RetryPosture = "unknown"
    category: Literal["resource_lifecycle"] = "resource_lifecycle"


ResourceManagerError: TypeAlias = (
    SelectionError | AuthenticationError | AuthorizationError | ResourceLifecycleError
)
LeaseOutcome: TypeAlias = (
    Success["AuthorizedContextLease"] | Failure[ResourceManagerError]
)
ShutdownOutcome: TypeAlias = Success[None] | Failure[ResourceLifecycleError]
CacheKey: TypeAlias = tuple[ContextIdentity, "CompositionFingerprint"]

logger = logging.getLogger(__name__)


SelectorKind = Literal["explicit", "active", "repository"]


@dataclass(frozen=True, slots=True)
class ContextSelector:
    """Exact Potpie context-selection input supplied by a caller."""

    kind: SelectorKind
    value: str | None = None

    def __post_init__(self) -> None:
        if self.kind in {"explicit", "repository"} and not (
            self.value and self.value.strip()
        ):
            raise ValueError(f"{self.kind} selectors require a non-empty value")
        if self.kind == "active" and self.value is not None:
            raise ValueError("active selectors cannot carry a value")


@dataclass(frozen=True, slots=True)
class AuthenticatedActor:
    actor_id: str
    attributes: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AuthorizationScope:
    actor_id: str
    operation: str
    context: ContextIdentity
    attributes: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DestructiveIntent:
    confirmed: bool
    operation: str
    selector: ContextSelector
    request_id: str


@dataclass(frozen=True, slots=True)
class AcquisitionRequest:
    request_id: str
    selector: ContextSelector
    operation: str
    authentication: object
    destructive: bool = False
    destructive_intent: DestructiveIntent | None = None

    def __post_init__(self) -> None:
        if not self.request_id.strip():
            raise ValueError("request_id must not be empty")
        if not self.operation.strip():
            raise ValueError("operation must not be empty")


@dataclass(frozen=True, slots=True)
class CompositionFingerprint:
    value: str

    def __post_init__(self) -> None:
        if not self.value.strip():
            raise ValueError("composition fingerprint must not be empty")


@dataclass(frozen=True, slots=True)
class HostResource:
    """Resource opened by Potpie and either retained or explicitly transferred."""

    name: str
    release: Callable[[], Awaitable[None]]
    ownership: Literal["retained", "transferred"] = "retained"


@dataclass(frozen=True, slots=True)
class ResourceComposition:
    fingerprint: CompositionFingerprint
    config: EngineConfig
    dependencies: EngineDependencies
    resources: tuple[HostResource, ...] = ()


@dataclass(frozen=True, slots=True)
class LeaseOwnership:
    """Expose dependency owners and the manager-retained engine lifetime."""

    host_resources: tuple[str, ...] = ()
    engine_resources: tuple[EngineResource, ...] = ()
    engine_lifetime: Literal["resource_manager_shutdown"] = "resource_manager_shutdown"
    release_closes_engine: Literal[False] = False


@dataclass(frozen=True, slots=True)
class AuthorizedContext:
    """An authenticated actor, its resolved context and the authorized scope."""

    actor: AuthenticatedActor
    context: ContextIdentity
    scope: AuthorizationScope


class ContextSelectorResolver(Protocol):
    async def resolve(
        self, selector: ContextSelector, *, actor: AuthenticatedActor
    ) -> Success[ContextIdentity] | Failure[SelectionError]: ...


class ActorAuthenticator(Protocol):
    async def authenticate(
        self, authentication: object
    ) -> Success[AuthenticatedActor] | Failure[AuthenticationError]: ...


class PassThroughAuthenticator:
    """Accept an actor a transport already authenticated, and nothing else.

    A serving endpoint authenticates each request's credential before any
    decoding and hands the Resource Manager the resulting actor; anything that
    is not an :class:`AuthenticatedActor` is refused rather than trusted.
    """

    async def authenticate(
        self, authentication: object
    ) -> Success[AuthenticatedActor] | Failure[AuthenticationError]:
        if isinstance(authentication, AuthenticatedActor):
            return Success(authentication)
        return Failure(
            AuthenticationError(
                code="actor_unauthenticated",
                message="the request carries no authenticated actor",
            )
        )


class OperationAuthorizer(Protocol):
    async def authorize(
        self,
        actor: AuthenticatedActor,
        operation: str,
        context: ContextIdentity,
    ) -> Success[AuthorizationScope] | Failure[AuthorizationError]: ...


class ContextResourceComposer(Protocol):
    async def fingerprint(
        self, context: ContextIdentity
    ) -> Success[CompositionFingerprint] | Failure[ResourceLifecycleError]: ...

    async def compose(
        self, context: ContextIdentity, fingerprint: CompositionFingerprint
    ) -> Success[ResourceComposition] | Failure[ResourceLifecycleError]: ...


@dataclass(slots=True)
class _CachedEngine:
    context: ContextIdentity
    fingerprint: CompositionFingerprint
    engine: ContextEngine
    resources: tuple[HostResource, ...]
    ownership: LeaseOwnership
    active_leases: int = 0
    engine_closed: bool = False


class AuthorizedContextLease:
    """Request scope over one authorized, context-bound cached engine."""

    def __init__(
        self,
        *,
        manager: ContextResourceManager,
        cache_key: CacheKey,
        context: ContextIdentity,
        scope: AuthorizationScope,
        engine: ContextEngine,
        ownership: LeaseOwnership,
    ) -> None:
        self._manager = manager
        self._cache_key = cache_key
        self.context = context
        self.scope = scope
        self.engine = engine
        self.ownership = ownership
        self._released = False
        self._release_lock = asyncio.Lock()

    @property
    def is_released(self) -> bool:
        return self._released

    async def __aenter__(self) -> AuthorizedContextLease:
        if self._released:
            raise RuntimeError("a released AuthorizedContextLease cannot be re-entered")
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.release()

    async def release(self) -> None:
        async with self._release_lock:
            if self._released:
                return
            release_task = asyncio.create_task(
                self._manager._release_lease(self._cache_key)
            )
            try:
                await asyncio.shield(release_task)
            except asyncio.CancelledError:
                await release_task
                self._released = True
                raise
            self._released = True


class ContextResourceManager:
    """Authenticate, resolve, authorize, compose, cache, lease, drain, close.

    ``max_idle_engines`` bounds how many cached engines may sit without a
    lease; past it the least recently used idle ones are closed. ``None``
    (the default) keeps every engine until shutdown, which suits a local
    runtime with a handful of pots; a host serving many pots sets a bound.
    """

    def __init__(
        self,
        *,
        resolver: ContextSelectorResolver,
        authenticator: ActorAuthenticator,
        authorizer: OperationAuthorizer,
        composer: ContextResourceComposer,
        engine_factory: Callable[
            ..., Awaitable[Outcome[ContextEngine]]
        ] = create_engine,
        max_idle_engines: int | None = None,
    ) -> None:
        if max_idle_engines is not None and max_idle_engines < 0:
            raise ValueError("max_idle_engines must not be negative")
        self._resolver = resolver
        self._authenticator = authenticator
        self._authorizer = authorizer
        self._composer = composer
        self._engine_factory = engine_factory
        self._max_idle_engines = max_idle_engines
        self._cache: dict[CacheKey, _CachedEngine] = {}
        self._creation_order: list[CacheKey] = []
        self._key_locks: dict[CacheKey, asyncio.Lock] = {}
        self._last_used: dict[CacheKey, int] = {}
        self._use_clock = 0
        self._retired: list[_CachedEngine] = []
        self._evictions_in_flight = 0
        self._state = asyncio.Condition()
        self._shutdown_lock = asyncio.Lock()
        self._accepting = True
        self._shutdown_complete = False
        self._inflight_acquisitions = 0

    @property
    def is_accepting(self) -> bool:
        return self._accepting

    @property
    def cached_engine_count(self) -> int:
        return len(self._cache)

    async def acquire(self, request: AcquisitionRequest) -> LeaseOutcome:
        async with self._state:
            if not self._accepting:
                return Failure(
                    ResourceLifecycleError(
                        code="resource_manager_draining",
                        message="the Context Resource Manager is draining",
                        retry_posture="safe",
                    )
                )
            self._inflight_acquisitions += 1
        try:
            return await self._acquire_started(request)
        finally:
            async with self._state:
                self._inflight_acquisitions -= 1
                self._state.notify_all()

    async def authenticate(
        self, authentication: object
    ) -> Success[AuthenticatedActor] | Failure[AuthenticationError]:
        """Authenticate a caller for an operation that selects no context."""

        return await self._authenticator.authenticate(authentication)

    async def authorize(
        self, request: AcquisitionRequest
    ) -> Success[AuthorizedContext] | Failure[ResourceManagerError]:
        """Authenticate, resolve and authorize a request without an engine.

        The same checks :meth:`acquire` runs before it composes anything, for
        operations an embedding host serves itself against a selected context.
        """

        if not self._accepting:
            return Failure(
                ResourceLifecycleError(
                    code="resource_manager_draining",
                    message="the Context Resource Manager is draining",
                    retry_posture="safe",
                )
            )
        return await self._authorize(request)

    async def _authorize(
        self, request: AcquisitionRequest
    ) -> Success[AuthorizedContext] | Failure[ResourceManagerError]:
        authentication = await self._authenticator.authenticate(request.authentication)
        if isinstance(authentication, Failure):
            return authentication
        actor = authentication.value

        selection = await self._resolver.resolve(request.selector, actor=actor)
        if isinstance(selection, Failure):
            return selection
        context = selection.value

        authorization = await self._authorizer.authorize(
            actor, request.operation, context
        )
        if isinstance(authorization, Failure):
            return authorization
        scope = authorization.value
        if (
            scope.actor_id != actor.actor_id
            or scope.operation != request.operation
            or scope.context != context
        ):
            return Failure(
                AuthorizationError(
                    code="authorization_scope_invalid",
                    message="authorization scope does not match the resolved request",
                )
            )

        intent_failure = self._validate_destructive_intent(request, scope)
        if intent_failure is not None:
            return Failure(intent_failure)
        return Success(AuthorizedContext(actor=actor, context=context, scope=scope))

    async def _acquire_started(self, request: AcquisitionRequest) -> LeaseOutcome:
        authorized = await self._authorize(request)
        if isinstance(authorized, Failure):
            return authorized
        context = authorized.value.context
        scope = authorized.value.scope

        fingerprint = await self._composer.fingerprint(context)
        if isinstance(fingerprint, Failure):
            return fingerprint
        cache_key = (context, fingerprint.value)
        key_lock = await self._lock_for(cache_key)

        async with key_lock:
            if not self._accepting:
                return Failure(
                    ResourceLifecycleError(
                        code="resource_manager_draining",
                        message="the Context Resource Manager began draining",
                        retry_posture="safe",
                    )
                )
            cached = self._cache.get(cache_key)
            if cached is None:
                created = await self._create_cached_engine(
                    context=context,
                    fingerprint=fingerprint.value,
                )
                if isinstance(created, Failure):
                    return created
                cached = created.value
                async with self._state:
                    self._cache[cache_key] = cached
                    self._creation_order.append(cache_key)
            async with self._state:
                cached.active_leases += 1
                self._touch(cache_key)

        return Success(
            AuthorizedContextLease(
                manager=self,
                cache_key=cache_key,
                context=context,
                scope=scope,
                engine=cached.engine,
                ownership=cached.ownership,
            )
        )

    async def shutdown(self) -> ShutdownOutcome:
        async with self._shutdown_lock:
            async with self._state:
                if self._shutdown_complete:
                    return Success(None)
                self._accepting = False
                await self._state.wait_for(
                    lambda: (
                        self._inflight_acquisitions == 0
                        and self._evictions_in_flight == 0
                        and all(
                            entry.active_leases == 0 for entry in self._cache.values()
                        )
                    )
                )

            failures: list[dict[str, str]] = []
            # Engines an idle eviction could not fully close are retried first.
            for cached in list(self._retired):
                failures.extend(await self._close_cached(cached))
            for cache_key in reversed(self._creation_order):
                failures.extend(await self._close_cached(self._cache[cache_key]))

            async with self._state:
                self._retired = [
                    cached
                    for cached in self._retired
                    if not (cached.engine_closed and not cached.resources)
                ]
                completed_keys = {
                    key
                    for key, cached in self._cache.items()
                    if cached.engine_closed and not cached.resources
                }
                for key in completed_keys:
                    del self._cache[key]
                    self._key_locks.pop(key, None)
                    self._last_used.pop(key, None)
                self._creation_order = [
                    key for key in self._creation_order if key not in completed_keys
                ]
                self._shutdown_complete = not self._cache and not self._retired
                self._state.notify_all()

            if failures:
                return Failure(
                    ResourceLifecycleError(
                        code="resource_manager_shutdown_failed",
                        message=(
                            "one or more cached engines or resources failed to close"
                        ),
                        details={"failures": tuple(failures)},
                        recommended_next_action="inspect resource cleanup logs",
                        retry_posture="safe",
                    )
                )
            return Success(None)

    @staticmethod
    async def _close_cached(cached: _CachedEngine) -> list[dict[str, str]]:
        """Close one cached engine, then release its retained host resources.

        Every resource gets an attempt even when the engine fails to close;
        what failed stays on the entry so a later attempt retries only that.
        """

        failures: list[dict[str, str]] = []
        if not cached.engine_closed:
            try:
                closed = await cached.engine.close()
            except Exception as exc:  # host cleanup must still be attempted
                failures.append(
                    {
                        "kind": "engine",
                        "context": cached.context.value,
                        "error_type": type(exc).__name__,
                    }
                )
            else:
                if isinstance(closed, Failure):
                    failures.append(
                        {
                            "kind": "engine",
                            "context": cached.context.value,
                            "code": closed.error.code,
                        }
                    )
                else:
                    cached.engine_closed = True
        pending_resources: list[HostResource] = []
        for resource in reversed(cached.resources):
            try:
                await resource.release()
            except Exception as exc:  # every resource still gets an attempt
                pending_resources.append(resource)
                failures.append(
                    {
                        "kind": "resource",
                        "resource": resource.name,
                        "error_type": type(exc).__name__,
                    }
                )
        cached.resources = tuple(reversed(pending_resources))
        return failures

    def _touch(self, cache_key: CacheKey) -> None:
        """Mark one cache entry as just used; callers hold ``_state``."""

        self._use_clock += 1
        self._last_used[cache_key] = self._use_clock

    async def _evict_idle_engines(self) -> None:
        """Close least recently used idle engines past ``max_idle_engines``."""

        if self._max_idle_engines is None:
            return
        while True:
            async with self._state:
                if not self._accepting:
                    return  # shutdown closes whatever is left
                idle = [
                    key
                    for key, cached in self._cache.items()
                    if cached.active_leases == 0
                ]
                if len(idle) <= self._max_idle_engines:
                    return
                victim = min(idle, key=lambda key: self._last_used.get(key, 0))
                self._evictions_in_flight += 1
            try:
                # Under the entry's own lock, so an acquisition of the same
                # context waits for the old engine to close before composing a
                # new one, and never picks up an engine being closed.
                async with await self._lock_for(victim):
                    async with self._state:
                        cached = self._cache.get(victim)
                        if cached is None or cached.active_leases:
                            continue
                        del self._cache[victim]
                        self._last_used.pop(victim, None)
                        self._creation_order.remove(victim)
                    failures = await self._close_cached(cached)
                    if failures:
                        logger.warning(
                            "idle engine for context %s did not close cleanly; "
                            "shutdown retries it",
                            cached.context.value,
                        )
                        async with self._state:
                            self._retired.append(cached)
            finally:
                async with self._state:
                    self._evictions_in_flight -= 1
                    self._state.notify_all()

    async def _lock_for(self, cache_key: CacheKey) -> asyncio.Lock:
        async with self._state:
            return self._key_locks.setdefault(cache_key, asyncio.Lock())

    async def _create_cached_engine(
        self,
        *,
        context: ContextIdentity,
        fingerprint: CompositionFingerprint,
    ) -> Success[_CachedEngine] | Failure[ResourceLifecycleError]:
        composed = await self._composer.compose(context, fingerprint)
        if isinstance(composed, Failure):
            return composed
        composition = composed.value
        if composition.fingerprint != fingerprint:
            acquisition_error = ResourceLifecycleError(
                code="composition_fingerprint_changed",
                message="resource composition changed during acquisition",
                retry_posture="safe",
            )
            return Failure(
                self._with_failed_acquisition_cleanup(
                    acquisition_error=acquisition_error,
                    cleanup_failures=await self._release_resources(
                        composition.resources
                    ),
                )
            )

        ownership = self._validate_composition_ownership(composition)
        if isinstance(ownership, Failure):
            return Failure(
                self._with_failed_acquisition_cleanup(
                    acquisition_error=ownership.error,
                    cleanup_failures=await self._release_resources(
                        composition.resources
                    ),
                )
            )
        retained_resources = ownership.value

        try:
            engine_outcome = await self._engine_factory(
                context=context,
                config=composition.config,
                dependencies=composition.dependencies,
            )
        except Exception as exc:
            acquisition_error = ResourceLifecycleError(
                code="engine_construction_failed",
                message="Context Engine construction failed",
                details={"error_type": type(exc).__name__},
                retry_posture="unknown",
            )
            return Failure(
                self._with_failed_acquisition_cleanup(
                    acquisition_error=acquisition_error,
                    cleanup_failures=await self._release_resources(
                        composition.resources
                    ),
                )
            )
        if isinstance(engine_outcome, Failure):
            cleanup_failures = await self._release_resources(composition.resources)
            acquisition_error = ResourceLifecycleError(
                code="engine_construction_failed",
                message="Context Engine construction failed",
                details={"engine_error_code": engine_outcome.error.code},
                retry_posture=engine_outcome.error.retry_posture,
            )
            return Failure(
                self._with_failed_acquisition_cleanup(
                    acquisition_error=acquisition_error,
                    cleanup_failures=cleanup_failures,
                    source_error=engine_outcome.error,
                )
            )

        return Success(
            _CachedEngine(
                context=context,
                fingerprint=fingerprint,
                engine=engine_outcome.value,
                resources=retained_resources,
                ownership=LeaseOwnership(
                    host_resources=tuple(
                        resource.name for resource in retained_resources
                    ),
                    engine_resources=composition.dependencies.resources,
                ),
            )
        )

    @staticmethod
    def _validate_composition_ownership(
        composition: ResourceComposition,
    ) -> Success[tuple[HostResource, ...]] | Failure[ResourceLifecycleError]:
        host_resources = composition.resources
        engine_resources = composition.dependencies.resources
        host_names = tuple(resource.name for resource in host_resources)
        engine_names = tuple(resource.name for resource in engine_resources)
        if len(set(host_names)) != len(host_names) or len(set(engine_names)) != len(
            engine_names
        ):
            return Failure(
                ResourceLifecycleError(
                    code="composition_ownership_invalid",
                    message="resource composition contains duplicate resource names",
                )
            )

        transferred_host_names = {
            resource.name
            for resource in host_resources
            if resource.ownership == "transferred"
        }
        transferred_engine_names = {
            resource.name
            for resource in engine_resources
            if resource.ownership == "transferred"
        }
        if transferred_host_names != transferred_engine_names:
            return Failure(
                ResourceLifecycleError(
                    code="composition_ownership_invalid",
                    message=(
                        "host-to-engine resource transfers must have matching "
                        "ownership declarations"
                    ),
                    details={
                        "host_transfer_resources": tuple(
                            sorted(transferred_host_names)
                        ),
                        "engine_transfer_resources": tuple(
                            sorted(transferred_engine_names)
                        ),
                    },
                )
            )

        retained_names = {
            resource.name
            for resource in host_resources
            if resource.ownership == "retained"
        }
        conflicting_names = retained_names.intersection(transferred_engine_names)
        if conflicting_names:
            return Failure(
                ResourceLifecycleError(
                    code="composition_ownership_invalid",
                    message=(
                        "a transferred Context Engine resource cannot retain host "
                        "cleanup ownership"
                    ),
                    details={"conflicting_resources": tuple(sorted(conflicting_names))},
                )
            )

        return Success(
            tuple(
                resource
                for resource in host_resources
                if resource.ownership == "retained"
            )
        )

    @staticmethod
    def _with_failed_acquisition_cleanup(
        *,
        acquisition_error: ResourceLifecycleError,
        cleanup_failures: tuple[dict[str, str], ...],
        source_error: object | None = None,
    ) -> ResourceLifecycleError:
        source = source_error or acquisition_error
        details = dict(acquisition_error.details)
        details["acquisition_failure"] = {
            "category": str(getattr(source, "category", "resource_lifecycle")),
            "code": str(getattr(source, "code", acquisition_error.code)),
            "retry_posture": str(
                getattr(source, "retry_posture", acquisition_error.retry_posture)
            ),
        }
        if cleanup_failures:
            details["cleanup_failures"] = cleanup_failures
        return ResourceLifecycleError(
            code=acquisition_error.code,
            message=acquisition_error.message,
            details=details,
            recommended_next_action=acquisition_error.recommended_next_action,
            retry_posture=acquisition_error.retry_posture,
        )

    async def _release_resources(
        self, resources: tuple[HostResource, ...]
    ) -> tuple[dict[str, str], ...]:
        failures: list[dict[str, str]] = []
        for resource in reversed(resources):
            try:
                await resource.release()
            except Exception as exc:
                failures.append(
                    {"resource": resource.name, "error_type": type(exc).__name__}
                )
        return tuple(failures)

    async def _release_lease(self, cache_key: CacheKey) -> None:
        async with self._state:
            cached = self._cache.get(cache_key)
            if cached is None or cached.active_leases == 0:
                return
            cached.active_leases -= 1
            self._touch(cache_key)
            self._state.notify_all()
        await self._evict_idle_engines()

    @staticmethod
    def _validate_destructive_intent(
        request: AcquisitionRequest, scope: AuthorizationScope
    ) -> AuthorizationError | None:
        if not request.destructive:
            return None
        intent = request.destructive_intent
        if (
            intent is None
            or not intent.confirmed
            or intent.operation != request.operation
            or intent.selector != request.selector
            or intent.request_id != request.request_id
            or scope.operation != request.operation
        ):
            return AuthorizationError(
                code="destructive_intent_invalid",
                message="destructive intent does not match the authorized request",
            )
        return None


__all__ = [
    "AcquisitionRequest",
    "ActorAuthenticator",
    "AuthenticatedActor",
    "AuthenticationError",
    "AuthorizationError",
    "AuthorizationScope",
    "AuthorizedContext",
    "AuthorizedContextLease",
    "CompositionFingerprint",
    "ContextResourceComposer",
    "ContextResourceManager",
    "ContextSelector",
    "ContextSelectorResolver",
    "DestructiveIntent",
    "HostResource",
    "LeaseOwnership",
    "LeaseOutcome",
    "OperationAuthorizer",
    "PassThroughAuthenticator",
    "ResourceComposition",
    "ResourceLifecycleError",
    "ResourceManagerError",
    "SelectionError",
    "ShutdownOutcome",
]
