"""Temporary compatibility exports retained until the deletion gate."""

from __future__ import annotations

from typing import Any

from potpie.runtime.clients import ClientOutcome, LegacyEngineClientAdapter
from potpie.runtime.local_engine import (
    LocalCliAuthenticator,
    LocalCliAuthorizer,
    LocalContextResourceComposer,
    LocalContextSelectorResolver,
    LocalEngineOperations,
    build_local_resource_manager,
)
from potpie.runtime.operations import ENGINE_OPERATION_CATALOG
from potpie.runtime.protocol import EngineOperationRequest
from potpie.runtime.resource_manager import AuthorizationError, ContextSelector
from potpie_context_engine import Failure


HostContextSelectorResolver = LocalContextSelectorResolver
HostContextResourceComposer = LocalContextResourceComposer
HostShellEngineOperations = LocalEngineOperations


def build_legacy_engine_client(
    *, host: Any, selector: ContextSelector
) -> LegacyEngineClientAdapter:
    resolver = LocalContextSelectorResolver(host)
    authenticator = LocalCliAuthenticator()
    authorizer = LocalCliAuthorizer()
    operations = LocalEngineOperations(host)

    async def invoke(request: EngineOperationRequest) -> ClientOutcome:
        selection = await resolver.resolve(request.selector)
        if isinstance(selection, Failure):
            return selection
        authenticated = await authenticator.authenticate(authentication=None)
        if isinstance(authenticated, Failure):
            return authenticated
        authorized = await authorizer.authorize(
            authenticated.value,
            request.operation.value,
            selection.value,
        )
        if isinstance(authorized, Failure):
            return authorized
        destructive_failure = _validate_destructive_intent(request)
        if destructive_failure is not None:
            return Failure(destructive_failure)
        return await operations.invoke(
            request.operation,
            selection.value,
            request.payload,
        )

    return LegacyEngineClientAdapter(selector=selector, invoker=invoke)


def _validate_destructive_intent(
    request: EngineOperationRequest,
) -> AuthorizationError | None:
    if not ENGINE_OPERATION_CATALOG[request.operation].destructive:
        return None
    intent = request.destructive_intent
    if (
        intent is None
        or not intent.confirmed
        or intent.operation != request.operation.value
        or intent.selector != request.selector
        or intent.request_id != request.request_id
    ):
        return AuthorizationError(
            code="destructive_intent_invalid",
            message="destructive operation confirmation does not match the request",
        )
    return None


__all__ = [
    "HostContextResourceComposer",
    "HostContextSelectorResolver",
    "HostShellEngineOperations",
    "LocalCliAuthenticator",
    "LocalCliAuthorizer",
    "build_legacy_engine_client",
    "build_local_resource_manager",
]
