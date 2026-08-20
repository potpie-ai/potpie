"""Potpie-owned runtime composition and resource-management boundaries."""

from potpie.runtime.resource_manager import (
    AcquisitionRequest,
    AuthenticatedActor,
    AuthenticationError,
    AuthorizationError,
    AuthorizationScope,
    AuthorizedContextLease,
    CompositionFingerprint,
    ContextResourceManager,
    ContextSelector,
    DestructiveIntent,
    HostResource,
    LeaseOwnership,
    ResourceComposition,
    ResourceLifecycleError,
    SelectionError,
)

__all__ = [
    "AcquisitionRequest",
    "AuthenticatedActor",
    "AuthenticationError",
    "AuthorizationError",
    "AuthorizationScope",
    "AuthorizedContextLease",
    "CompositionFingerprint",
    "ContextResourceManager",
    "ContextSelector",
    "DestructiveIntent",
    "HostResource",
    "LeaseOwnership",
    "ResourceComposition",
    "ResourceLifecycleError",
    "SelectionError",
]
