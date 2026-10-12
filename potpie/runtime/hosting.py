"""The supported import surface for hosts that embed Potpie's typed protocol.

**Provisional.** Names may still move while the managed-host contract
settles. Import from here, not from the modules behind it, so a move is a
change in one place.

A managed host provides five things and gets the rest from this kit:

* a :class:`RequestAuthenticator` mapping each request's bearer credential to
  one :class:`AuthenticatedActor`;
* an :class:`ActorPotCatalog` (pots, active pot, repo defaults and repo
  sources per actor), wrapped in :class:`CatalogSelectorResolver`;
* an :class:`OperationAuthorizer` deciding by :func:`operation_spec` safety
  and destructiveness, and putting the actor's role in the scope's
  ``commit_access`` attribute;
* a :class:`HostOperationHandler` for the ``host.*`` catalog, and any
  :class:`HostExtension` operations of its own;
* :class:`EngineServices` over one graph runtime built with
  ``commit_actor=current_commit_actor`` and
  ``commit_authorize=authorize_scoped_commit``.

Then::

    manager = ContextResourceManager(
        resolver=CatalogSelectorResolver(catalog),
        authenticator=PassThroughAuthenticator(),
        authorizer=authorizer,
        composer=ServiceResourceComposer(services),
        max_idle_engines=64,
    )
    endpoint = OperationEndpoint(
        authenticator=keys,
        compatibility=PerRequestCompatibility(),
        engine_handler=TypedOperationHandler(
            manager,
            coordinator=OperationCoordinator(),
            context_free_handler=GraphMetadataOperationHandler(services),
        ),
        host_handler=host_handler,
        extensions=extensions,
        identity=HostIdentity(host_id="...", version="...", build=None),
        request_policy=RemoteRequestPolicy(),
    )
    response = await endpoint.handle(body, headers)  # from one POST route

The route is ``POST <base>/v1/operations``; reply with ``response.status``,
``response.body`` and ``response.headers``.
"""

from __future__ import annotations

from potpie.runtime.clients import (
    ClientOutcome,
    ContextFreeOperationHandler,
    TypedOperationHandler,
)
from potpie.runtime.commit_access import (
    COMMIT_ACCESS_BY_ROLE,
    authorization_scope,
    authorize_scoped_commit,
    commit_access_for,
    current_authorization_scope,
    current_commit_actor,
)
from potpie.runtime.coordinator import OperationCoordinator
from potpie.runtime.endpoint import (
    DEFAULT_MAX_REQUEST_BYTES,
    HOST_CONTROL_REFUSED,
    REQUEST_TOO_LARGE,
    STATELESS_TICKET,
    BearerTokenAuthenticator,
    CompatibilityPolicy,
    EndpointResponse,
    HostIdentity,
    OperationEndpoint,
    PerBootTicketCompatibility,
    PerRequestCompatibility,
    RequestAuthenticator,
    current_actor,
    current_actor_id,
)
from potpie.runtime.extensions import (
    EXTENSION_NAME,
    ExtensionHandler,
    HostExtension,
    operation_spec,
)
from potpie.runtime.host_operations import (
    HOST_OPERATION_CATALOG,
    GrantListRequest,
    GrantListResult,
    GrantRevokeRequest,
    GrantRevokeResult,
    GrantSetRequest,
    HostOperation,
    HostOperationHandler,
    PotCounts,
    PotCreateRequest,
    PotGrantInfo,
    PotListRequest,
    PotListResult,
    PotStatusRequest,
    PotUseRequest,
    RepoDefaultClearRequest,
    RepoDefaultClearResult,
    RepoDefaultResult,
    RepoDefaultSetRequest,
    SourceAddRequest,
    SourceListRequest,
    SourceListResult,
    WhoamiRequest,
    WhoamiResult,
)
from potpie.runtime.local_engine import (
    EngineServices,
    GraphMetadataOperationHandler,
    ServiceEngineOperations,
    ServiceResourceComposer,
)
from potpie.runtime.operations import (
    ENGINE_OPERATION_CATALOG,
    EngineOperation,
    OperationSpec,
    SafetyClass,
)
from potpie.runtime.protocol import DaemonBuild
from potpie.runtime.request_policy import (
    CALLER_ASSERTION_MISMATCH,
    HOST_PATH_REFUSED,
    RemoteRequestPolicy,
    RequestPolicy,
)
from potpie.runtime.resource_manager import (
    ActorAuthenticator,
    AuthenticatedActor,
    AuthenticationError,
    AuthorizationError,
    AuthorizationScope,
    AuthorizedContext,
    CompositionFingerprint,
    ContextResourceComposer,
    ContextResourceManager,
    ContextSelector,
    ContextSelectorResolver,
    HostResource,
    OperationAuthorizer,
    PassThroughAuthenticator,
    ResourceComposition,
    ResourceLifecycleError,
    SelectionError,
)
from potpie.runtime.selection import ActorPotCatalog, CatalogSelectorResolver
from potpie.pots.contracts import PotAggregateStatus, PotInfo, PotRepoSource, SourceInfo
from potpie_context_engine import (
    ContextIdentity,
    DependencyError,
    DomainError,
    Failure,
    Success,
)

__all__ = [
    "COMMIT_ACCESS_BY_ROLE",
    "CALLER_ASSERTION_MISMATCH",
    "DEFAULT_MAX_REQUEST_BYTES",
    "ENGINE_OPERATION_CATALOG",
    "EXTENSION_NAME",
    "HOST_CONTROL_REFUSED",
    "HOST_OPERATION_CATALOG",
    "HOST_PATH_REFUSED",
    "REQUEST_TOO_LARGE",
    "STATELESS_TICKET",
    "ActorAuthenticator",
    "ActorPotCatalog",
    "AuthenticatedActor",
    "AuthenticationError",
    "AuthorizationError",
    "AuthorizationScope",
    "AuthorizedContext",
    "BearerTokenAuthenticator",
    "CatalogSelectorResolver",
    "ClientOutcome",
    "CompatibilityPolicy",
    "CompositionFingerprint",
    "ContextFreeOperationHandler",
    "ContextIdentity",
    "ContextResourceComposer",
    "ContextResourceManager",
    "ContextSelector",
    "ContextSelectorResolver",
    "DaemonBuild",
    "DependencyError",
    "DomainError",
    "EndpointResponse",
    "EngineOperation",
    "EngineServices",
    "ExtensionHandler",
    "Failure",
    "GrantListRequest",
    "GrantListResult",
    "GrantRevokeRequest",
    "GrantRevokeResult",
    "GrantSetRequest",
    "GraphMetadataOperationHandler",
    "HostExtension",
    "HostIdentity",
    "HostOperation",
    "HostOperationHandler",
    "HostResource",
    "OperationAuthorizer",
    "OperationCoordinator",
    "OperationEndpoint",
    "OperationSpec",
    "PassThroughAuthenticator",
    "PerBootTicketCompatibility",
    "PerRequestCompatibility",
    "PotAggregateStatus",
    "PotCounts",
    "PotCreateRequest",
    "PotGrantInfo",
    "PotInfo",
    "PotListRequest",
    "PotListResult",
    "PotRepoSource",
    "PotStatusRequest",
    "PotUseRequest",
    "RemoteRequestPolicy",
    "RepoDefaultClearRequest",
    "RepoDefaultClearResult",
    "RepoDefaultResult",
    "RepoDefaultSetRequest",
    "RequestAuthenticator",
    "RequestPolicy",
    "ResourceComposition",
    "ResourceLifecycleError",
    "SafetyClass",
    "SelectionError",
    "ServiceEngineOperations",
    "ServiceResourceComposer",
    "SourceAddRequest",
    "SourceInfo",
    "SourceListRequest",
    "SourceListResult",
    "Success",
    "TypedOperationHandler",
    "WhoamiRequest",
    "WhoamiResult",
    "authorization_scope",
    "authorize_scoped_commit",
    "commit_access_for",
    "current_actor",
    "current_actor_id",
    "current_authorization_scope",
    "current_commit_actor",
    "operation_spec",
]
