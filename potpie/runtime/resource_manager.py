"""Authorized context leases and daemon-lifetime engine ownership."""

from __future__ import annotations

import asyncio
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


class ContextSelectorResolver(Protocol):
    async def resolve(
        self, selector: ContextSelector
    ) -> Success[ContextIdentity] | Failure[SelectionError]: ...


class ActorAuthenticator(Protocol):
    async def authenticate(
        self, authentication: object
    ) -> Success[AuthenticatedActor] | Failure[AuthenticationError]: ...


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
            self._released = True
            await self._manager._release_lease(self._cache_key)


class ContextResourceManager:
    """Resolve, authorize, compose, cache, lease, drain, and close engines."""

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
    ) -> None:
        self._resolver = resolver
        self._authenticator = authenticator
        self._authorizer = authorizer
        self._composer = composer
        self._engine_factory = engine_factory
        self._cache: dict[CacheKey, _CachedEngine] = {}
        self._creation_order: list[CacheKey] = []
        self._key_locks: dict[CacheKey, asyncio.Lock] = {}
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

    async def _acquire_started(self, request: AcquisitionRequest) -> LeaseOutcome:
        selection = await self._resolver.resolve(request.selector)
        if isinstance(selection, Failure):
            return selection
        context = selection.value

        authentication = await self._authenticator.authenticate(request.authentication)
        if isinstance(authentication, Failure):
            return authentication
        actor = authentication.value

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
                        and all(
                            entry.active_leases == 0 for entry in self._cache.values()
                        )
                    )
                )

            failures: list[dict[str, str]] = []
            for cache_key in reversed(self._creation_order):
                cached = self._cache[cache_key]
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
                for resource in reversed(cached.resources):
                    try:
                        await resource.release()
                    except Exception as exc:  # every resource still gets an attempt
                        failures.append(
                            {
                                "kind": "resource",
                                "resource": resource.name,
                                "error_type": type(exc).__name__,
                            }
                        )

            async with self._state:
                self._cache.clear()
                self._creation_order.clear()
                self._key_locks.clear()
                self._shutdown_complete = True
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

        engine_outcome = await self._engine_factory(
            context=context,
            config=composition.config,
            dependencies=composition.dependencies,
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
            self._state.notify_all()

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
    "ResourceComposition",
    "ResourceLifecycleError",
    "ResourceManagerError",
    "SelectionError",
    "ShutdownOutcome",
]
