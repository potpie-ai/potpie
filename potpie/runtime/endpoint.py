"""Transport-neutral typed operation endpoint.

``OperationEndpoint.handle(body, headers)`` is the whole server side of the
typed operation protocol: the local daemon wraps it in aiohttp, and an
embedding host (a managed host) calls it from one route of its own web
framework. The endpoint owns, in this order:

1. the request size limit (413 ``request_too_large``);
2. authentication of every request from its headers alone; before it
   succeeds only ``protocol_version`` and ``request_id`` are read from the
   body, to correlate the 401;
3. decoding, against exactly the operations this endpoint serves;
4. compatibility (per-boot tickets on the local daemon, per-request checks on
   a managed host) and daemon control (handshake, status, shutdown);
5. lifecycle: refusing new operations with 503 ``daemon_draining`` while it
   drains, and counting the ones in flight;
6. the request policy (caller-asserted identities, host paths) and dispatch
   to the typed operation handler, with the authenticated actor bound for the
   handler's duration (:func:`current_actor`).
"""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import json
import os
import secrets
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Mapping
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Literal, Protocol, cast
from uuid import uuid4

from potpie.runtime.clients import ClientOutcome
from potpie.runtime.codec import decode_request, encode_response
from potpie.runtime.coordinator import OperationCoordinator
from potpie.runtime.extensions import HostExtension, extension_registry
from potpie.runtime.host_operations import HostOperation, HostOperationHandler
from potpie.runtime.operations import (
    DaemonControlOperation,
    EngineOperation,
    operation_catalog_fingerprint,
)
from potpie.runtime.protocol import (
    CONTROL_TICKET_OPERATIONS,
    PROTOCOL_MAX_VERSION,
    PROTOCOL_MIN_VERSION,
    PROTOCOL_VERSION,
    DaemonBuild,
    DaemonInternalError,
    DaemonStatusRequest,
    DaemonStatusResult,
    EngineOperationRequest,
    FailureResponse,
    HandshakePayload,
    HandshakeRequest,
    HandshakeResult,
    HostOperationRequest,
    LifecycleState,
    ProtocolError,
    ProtocolRequest,
    ProtocolResponse,
    ProtocolTransportError,
    RuntimeBoundaryError,
    ShutdownRequest,
    ShutdownResult,
    SuccessResponse,
)
from potpie.runtime.request_policy import RequestPolicy
from potpie.runtime.resource_manager import (
    AuthenticatedActor,
    AuthenticationError,
    AuthorizationError,
)
from potpie_context_engine import Failure, Success

#: The default request body limit, for the local daemon and managed hosts.
DEFAULT_MAX_REQUEST_BYTES = 64 * 1024 * 1024

#: The compatibility ticket a host that keeps no per-boot state answers with.
#: The handshake result requires a non-empty ticket; this one authorizes
#: nothing and is never checked.
STATELESS_TICKET = "stateless"

REQUEST_TOO_LARGE = "request_too_large"
HOST_CONTROL_REFUSED = "host_control_refused"

_ACTOR: ContextVar[AuthenticatedActor | None] = ContextVar(
    "potpie_endpoint_actor", default=None
)


def current_actor() -> AuthenticatedActor | None:
    """The authenticated actor of the request being handled, if any."""

    return _ACTOR.get()


def current_actor_id() -> str | None:
    actor = _ACTOR.get()
    return actor.actor_id if actor is not None else None


@dataclass(frozen=True, slots=True)
class HostIdentity:
    """What a host reports about itself in the handshake and in status.

    ``host_id`` is the handshake's ``instance_id``: per boot for the local
    daemon, a stable configured id for a managed host.
    """

    host_id: str
    version: str | None = None
    build: DaemonBuild | None = None
    backend_profile: str = "unknown"
    ui_url: str = "none"

    def __post_init__(self) -> None:
        for name in ("host_id", "backend_profile", "ui_url"):
            if not getattr(self, name).strip():
                raise ValueError(f"host identity {name} must not be empty")


@dataclass(frozen=True, slots=True)
class EndpointResponse:
    """One typed response: HTTP status, JSON body and response headers."""

    status: int
    body: bytes
    headers: Mapping[str, str] = field(default_factory=dict)

    def json(self) -> object:
        return json.loads(self.body)


class RequestAuthenticator(Protocol):
    """Authenticate one request from its headers, before anything is decoded.

    Header names are matched case-insensitively by the endpoint (it passes a
    lower-cased copy). Return the actor the credential maps to; the request's
    content never influences it.
    """

    async def authenticate(
        self, headers: Mapping[str, str]
    ) -> Success[AuthenticatedActor] | Failure[AuthenticationError]: ...


class BearerTokenAuthenticator:
    """One static bearer secret mapped to one actor (the local daemon's)."""

    def __init__(
        self,
        token: str,
        *,
        actor: AuthenticatedActor,
        failure_code: str = "authentication_failed",
    ) -> None:
        if len(token.encode()) < 32:
            raise ValueError("bearer token must contain at least 256 bits")
        self._expected = f"Bearer {token}".encode()
        self._actor = actor
        self._failure_code = failure_code

    async def authenticate(
        self, headers: Mapping[str, str]
    ) -> Success[AuthenticatedActor] | Failure[AuthenticationError]:
        supplied = headers.get("authorization", "")
        if hmac.compare_digest(supplied.encode(), self._expected):
            return Success(self._actor)
        return Failure(
            AuthenticationError(
                code=self._failure_code,
                message=(
                    "daemon authentication failed"
                    if self._failure_code.startswith("daemon_")
                    else "authentication failed"
                ),
            )
        )


class CompatibilityPolicy(Protocol):
    """How a host decides whether a client's request is compatible."""

    @property
    def serves_shutdown(self) -> bool:
        """Whether ``daemon.shutdown`` may stop this host."""
        ...

    def issue_ticket(self, payload: HandshakePayload) -> str:
        """The compatibility ticket a handshake answers with."""
        ...

    def check(
        self, request: ProtocolRequest
    ) -> tuple[RuntimeBoundaryError, int] | None:
        """A typed refusal and its HTTP status, or ``None`` to proceed."""
        ...


_TicketScope = Literal["full", "control"]
_TICKET_SCOPES: frozenset[str] = frozenset({"full", "control"})


class PerBootTicketCompatibility:
    """The local daemon's lock-step: a ticket per handshake, signed per boot.

    The ticket covers the whole catalog when the client's operation catalog
    fingerprint equals the daemon's, and only daemon status and shutdown when
    it differs, so a client from another build can still see and stop it.
    """

    serves_shutdown = True

    def __init__(
        self,
        *,
        instance_id: str,
        catalog_fingerprint: Callable[[], str] = operation_catalog_fingerprint,
    ) -> None:
        self._instance_id = instance_id
        self._catalog_fingerprint = catalog_fingerprint
        self._secret = secrets.token_bytes(32)

    def issue_ticket(self, payload: HandshakePayload) -> str:
        scope: _TicketScope = (
            "full"
            if payload.client_operation_catalog_fingerprint
            == self._catalog_fingerprint()
            else "control"
        )
        nonce = secrets.token_urlsafe(24)
        signature = hmac.new(
            self._secret, self._message(scope, nonce), "sha256"
        ).hexdigest()
        return f"{scope}.{nonce}.{signature}"

    def check(
        self, request: ProtocolRequest
    ) -> tuple[RuntimeBoundaryError, int] | None:
        scope = self.ticket_scope(getattr(request, "compatibility_ticket", None))
        if scope is None:
            return (
                ProtocolError(
                    code="compatibility_ticket_invalid",
                    message=(
                        "a ticket from a compatible handshake is required before "
                        "this operation"
                    ),
                    recommended_next_action="perform an authenticated handshake",
                    retry_posture="safe",
                ),
                409,
            )
        if scope == "full" or request.operation in CONTROL_TICKET_OPERATIONS:
            return None
        # A control-scoped ticket comes from a handshake whose operation
        # catalog differs from this daemon's; only status and shutdown share
        # wire semantics across that difference.
        return (
            ProtocolError(
                code="operation_catalog_mismatch",
                message=(
                    "this daemon serves a different operation catalog; "
                    "its ticket allows only daemon status and shutdown"
                ),
                details={"operation": str(request.operation)},
                recommended_next_action=(
                    "restart the daemon with 'potpie daemon restart'"
                ),
            ),
            409,
        )

    def ticket_scope(self, ticket: str | None) -> _TicketScope | None:
        """The scope a ticket this boot issued grants, or ``None`` if invalid."""

        if not ticket:
            return None
        try:
            claimed_scope, nonce, supplied_signature = ticket.split(".", 2)
        except ValueError:
            return None
        if claimed_scope not in _TICKET_SCOPES:
            return None
        scope = cast(_TicketScope, claimed_scope)
        expected = hmac.new(self._secret, self._message(scope, nonce), "sha256")
        if not hmac.compare_digest(supplied_signature, expected.hexdigest()):
            return None
        return scope

    def _message(self, scope: _TicketScope, nonce: str) -> bytes:
        return (
            f"{self._instance_id}:{PROTOCOL_VERSION}:"
            f"{self._catalog_fingerprint()}:{scope}:{nonce}"
        ).encode()


class PerRequestCompatibility:
    """A managed host's compatibility: decided per request, no tickets.

    The protocol version, the operation and its fields decide every request
    (the codec refuses what this host does not serve), so any replica can
    answer any request. The handshake answers :data:`STATELESS_TICKET`, and
    ``daemon.shutdown`` is refused: a protocol client never stops a host.
    """

    serves_shutdown = False

    def issue_ticket(self, payload: HandshakePayload) -> str:
        del payload
        return STATELESS_TICKET

    def check(
        self, request: ProtocolRequest
    ) -> tuple[RuntimeBoundaryError, int] | None:
        if isinstance(request, ShutdownRequest):
            return (
                AuthorizationError(
                    code=HOST_CONTROL_REFUSED,
                    message="this host is not stopped through the protocol",
                    details={"operation": request.operation.value},
                ),
                403,
            )
        return None


class OperationEndpoint:
    """Serve the typed operation protocol for one host, over any transport."""

    def __init__(
        self,
        *,
        authenticator: RequestAuthenticator,
        compatibility: CompatibilityPolicy,
        engine_handler: object,
        identity: HostIdentity,
        host_handler: HostOperationHandler | None = None,
        extensions: Iterable[HostExtension] = (),
        max_request_bytes: int = DEFAULT_MAX_REQUEST_BYTES,
        request_policy: RequestPolicy | None = None,
        engine_operations: Iterable[EngineOperation] | None = None,
        coordinator: OperationCoordinator | None = None,
        on_shutdown: Callable[[], Awaitable[object]] | None = None,
        lifecycle_state: LifecycleState = "ready",
        catalog_fingerprint: Callable[[], str] = operation_catalog_fingerprint,
    ) -> None:
        if max_request_bytes <= 0:
            raise ValueError("max_request_bytes must be positive")
        self._authenticator = authenticator
        self._compatibility = compatibility
        self._engine_handler = engine_handler
        self._host_handler = host_handler
        self._extensions = extension_registry(extensions)
        self._identity = identity
        self._max_request_bytes = max_request_bytes
        self._request_policy = request_policy
        self._engine_operations = (
            frozenset(engine_operations) if engine_operations is not None else None
        )
        self._coordinator = coordinator
        self._on_shutdown = on_shutdown
        self._catalog_fingerprint = catalog_fingerprint
        self._state: LifecycleState = lifecycle_state
        self._accepting = lifecycle_state == "ready"
        self._active_operations = 0
        self._condition = asyncio.Condition()

    # --- lifecycle --------------------------------------------------------

    @property
    def identity(self) -> HostIdentity:
        return self._identity

    @property
    def max_request_bytes(self) -> int:
        return self._max_request_bytes

    @property
    def lifecycle_state(self) -> LifecycleState:
        return self._state

    @property
    def active_operations(self) -> int:
        return self._active_operations

    async def mark_ready(self) -> None:
        async with self._condition:
            self._state = "ready"
            self._accepting = True
            self._condition.notify_all()

    async def begin_drain(self) -> bool:
        """Refuse new operations from now on; ``False`` if already finished."""

        async with self._condition:
            if self._state in {"stopped", "failed"}:
                return False
            self._state = "draining"
            self._accepting = False
            self._condition.notify_all()
            return True

    async def wait_idle(self) -> None:
        """Wait until no operation is in flight."""

        async with self._condition:
            await self._condition.wait_for(lambda: self._active_operations == 0)

    def set_lifecycle_state(self, state: LifecycleState) -> None:
        """Record a terminal or failed state the serving process reached."""

        self._state = state
        if state != "ready":
            self._accepting = False

    def capabilities(self) -> tuple[str, ...]:
        """Every operation this endpoint serves, as the handshake lists them."""

        engine = (
            set(EngineOperation)
            if self._engine_operations is None
            else set(self._engine_operations)
        )
        names = {operation.value for operation in engine}
        names |= {
            operation.value
            for operation in DaemonControlOperation
            if operation is not DaemonControlOperation.SHUTDOWN
            or self._compatibility.serves_shutdown
        }
        if self._host_handler is not None:
            names |= {operation.value for operation in HostOperation}
        names |= set(self._extensions)
        return tuple(sorted(names))

    # --- requests ---------------------------------------------------------

    def refuse_oversized(self, body: bytes | None = None) -> EndpointResponse:
        """The typed 413 for a body over the limit."""

        protocol_version, request_id = PROTOCOL_VERSION, str(uuid4())
        if body is not None and len(body) <= 4096:
            protocol_version, request_id = _correlation_identity(_parse(body))
        return self._respond(
            FailureResponse(
                protocol_version=protocol_version,
                request_id=request_id,
                outcome=Failure(
                    ProtocolError(
                        code=REQUEST_TOO_LARGE,
                        message="the request body exceeds this host's limit",
                        details={"max_bytes": self._max_request_bytes},
                        recommended_next_action=(
                            "send less per request (for example fewer files per "
                            "resource import)"
                        ),
                    )
                ),
            ),
            status=413,
        )

    async def handle(self, body: bytes, headers: Mapping[str, str]) -> EndpointResponse:
        """Answer one request body with one typed response."""

        if len(body) > self._max_request_bytes:
            return self.refuse_oversized()
        lowered = {str(key).lower(): str(value) for key, value in headers.items()}
        try:
            authenticated = await self._authenticator.authenticate(lowered)
        except Exception:
            authenticated = Failure(
                AuthenticationError(
                    code="authentication_unavailable",
                    message="the host could not authenticate the request",
                    retry_posture="safe",
                )
            )
        if isinstance(authenticated, Failure):
            # Only the correlation identity is read from an unauthenticated body.
            protocol_version, request_id = _correlation_identity(_parse(body))
            return self._respond(
                FailureResponse(
                    protocol_version=protocol_version,
                    request_id=request_id,
                    outcome=authenticated,
                ),
                status=401,
            )
        actor = authenticated.value

        document = _parse(body)
        protocol_version, request_id = _correlation_identity(document)
        decoded = decode_request(
            document,
            engine_operations=self._engine_operations,
            host_operations=self._host_handler is not None,
            extensions=self._extensions,
        )
        if isinstance(decoded, Failure):
            return self._respond(
                FailureResponse(
                    protocol_version=protocol_version,
                    request_id=request_id,
                    outcome=decoded,
                ),
                status=400,
            )
        request = decoded.value
        if not PROTOCOL_MIN_VERSION <= request.protocol_version <= PROTOCOL_MAX_VERSION:
            return self._respond(
                _failure_response(
                    request,
                    ProtocolError(
                        code="protocol_version_incompatible",
                        message="request protocol version is incompatible",
                        details={
                            "supported_min": PROTOCOL_MIN_VERSION,
                            "supported_max": PROTOCOL_MAX_VERSION,
                        },
                        recommended_next_action="install compatible Potpie versions",
                    ),
                ),
                status=409,
            )
        token = _ACTOR.set(actor)
        try:
            response, status = await self._execute(request, actor)
        except Exception:
            response = _failure_response(
                request,
                DaemonInternalError(
                    code="daemon_internal_failure",
                    message="the daemon failed to process the request",
                    details={"request_id": request.request_id},
                    recommended_next_action="inspect daemon logs",
                ),
            )
            status = 500
        finally:
            _ACTOR.reset(token)
        try:
            return self._respond(response, status=status)
        except Exception:
            return self._respond(
                _failure_response(
                    request,
                    DaemonInternalError(
                        code="daemon_response_encoding_failed",
                        message="the daemon could not encode the operation response",
                        details={"request_id": request.request_id},
                        recommended_next_action="inspect daemon logs",
                    ),
                ),
                status=500,
            )

    async def _execute(
        self, request: ProtocolRequest, actor: AuthenticatedActor
    ) -> tuple[ProtocolResponse, int]:
        if isinstance(request, HandshakeRequest):
            return self._handshake(request)
        refusal = self._compatibility.check(request)
        if refusal is not None:
            error, status = refusal
            return _failure_response(request, error), status
        if isinstance(request, ShutdownRequest):
            async with self._lifecycle_control():
                if self._on_shutdown is not None:
                    await self._on_shutdown()
                else:
                    await self.begin_drain()
                return _success(request, ShutdownResult(accepted=True)), 200
        if isinstance(request, DaemonStatusRequest):
            async with self._lifecycle_control():
                return _success(request, self._status()), 200
        async with self._condition:
            if not self._accepting:
                return (
                    _failure_response(
                        request,
                        ProtocolTransportError(
                            code="daemon_draining",
                            message="daemon is not accepting new operations",
                            retry_posture="safe",
                        ),
                    ),
                    503,
                )
            self._active_operations += 1
        try:
            outcome = await self._dispatch(request, actor)
        finally:
            async with self._condition:
                self._active_operations -= 1
                self._condition.notify_all()
        if isinstance(outcome, Success):
            return _success(request, outcome.value), 200
        return _failure_response(request, outcome.error), _status_for(outcome.error)

    async def _dispatch(
        self,
        request: EngineOperationRequest | HostOperationRequest,
        actor: AuthenticatedActor,
    ) -> ClientOutcome:
        handler = cast(_TypedHandler, self._engine_handler)
        if isinstance(request, EngineOperationRequest):
            if self._request_policy is not None:
                checked = self._request_policy.apply(request, actor=actor)
                if isinstance(checked, Failure):
                    return checked
                request = checked.value
            return await handler.handle(request, authentication=actor)
        if request.is_extension:
            return await handler.handle(
                request,
                authentication=actor,
                extension=self._extensions[str(request.operation)],
            )
        return await handler.handle(
            request, authentication=actor, host_handler=self._host_handler
        )

    def _handshake(self, request: HandshakeRequest) -> tuple[ProtocolResponse, int]:
        payload = request.payload
        if not (
            payload.client_protocol_min <= PROTOCOL_MAX_VERSION
            and PROTOCOL_MIN_VERSION <= payload.client_protocol_max
        ):
            return (
                _failure_response(
                    request,
                    ProtocolError(
                        code="protocol_version_incompatible",
                        message="client and daemon protocol ranges do not overlap",
                        recommended_next_action="install compatible Potpie versions",
                    ),
                ),
                409,
            )
        if (
            payload.expected_instance_id is not None
            and payload.expected_instance_id != self._identity.host_id
        ):
            return (
                _failure_response(
                    request,
                    ProtocolError(
                        code="daemon_instance_mismatch",
                        message="daemon instance identity does not match the request",
                        recommended_next_action="refresh discovery and reconnect",
                    ),
                ),
                409,
            )
        return (
            _success(
                request,
                HandshakeResult(
                    protocol_min=PROTOCOL_MIN_VERSION,
                    protocol_max=PROTOCOL_MAX_VERSION,
                    instance_id=self._identity.host_id,
                    lifecycle_state=self._state,
                    capabilities=self.capabilities(),
                    operation_catalog_fingerprint=self._catalog_fingerprint(),
                    compatibility_ticket=self._compatibility.issue_ticket(payload),
                ),
            ),
            200,
        )

    def _status(self) -> DaemonStatusResult:
        return DaemonStatusResult(
            instance_id=self._identity.host_id,
            pid=os.getpid(),
            lifecycle_state=self._state,
            backend_profile=self._identity.backend_profile,
            ui_url=self._identity.ui_url,
            version=self._identity.version,
            build=self._identity.build,
        )

    @contextlib.asynccontextmanager
    async def _lifecycle_control(self) -> AsyncIterator[None]:
        if self._coordinator is None:
            yield
            return
        async with self._coordinator.lifecycle_control():
            yield

    @staticmethod
    def _respond(response: ProtocolResponse, *, status: int) -> EndpointResponse:
        body = json.dumps(encode_response(response), separators=(",", ":")).encode()
        return EndpointResponse(
            status=status,
            body=body,
            headers={
                "Content-Type": "application/json",
                "X-Request-ID": response.request_id,
            },
        )


class _TypedHandler(Protocol):
    async def handle(
        self, request: object, *, authentication: object, **routing: object
    ) -> ClientOutcome: ...


def _parse(body: bytes) -> object:
    try:
        return json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return None


def _correlation_identity(document: object) -> tuple[int, str]:
    if not isinstance(document, Mapping):
        return PROTOCOL_VERSION, str(uuid4())
    protocol_version = document.get("protocol_version")
    if not isinstance(protocol_version, int) or isinstance(protocol_version, bool):
        protocol_version = PROTOCOL_VERSION
    request_id = document.get("request_id")
    if not isinstance(request_id, str) or not request_id.strip():
        request_id = str(uuid4())
    return protocol_version, request_id


def _success(request: ProtocolRequest, value: object) -> SuccessResponse[object]:
    return SuccessResponse(
        protocol_version=request.protocol_version,
        request_id=request.request_id,
        outcome=Success(value),
    )


def _failure_response(
    request: ProtocolRequest, error: RuntimeBoundaryError
) -> FailureResponse:
    return FailureResponse(
        protocol_version=request.protocol_version,
        request_id=request.request_id,
        outcome=Failure(error),
    )


def _status_for(error: RuntimeBoundaryError) -> int:
    return {
        "authentication": 401,
        "authorization": 403,
        "protocol": 400,
        "protocol_transport": 503,
        "daemon_internal": 500,
    }.get(error.category, 200)


__all__ = [
    "DEFAULT_MAX_REQUEST_BYTES",
    "HOST_CONTROL_REFUSED",
    "REQUEST_TOO_LARGE",
    "STATELESS_TICKET",
    "BearerTokenAuthenticator",
    "CompatibilityPolicy",
    "EndpointResponse",
    "HostIdentity",
    "OperationEndpoint",
    "PerBootTicketCompatibility",
    "PerRequestCompatibility",
    "RequestAuthenticator",
    "current_actor",
    "current_actor_id",
]
