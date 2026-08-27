"""Protocol-versioned request and response types for the local daemon."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic, Literal, Mapping, TypeAlias, TypeVar

from potpie.runtime.operations import (
    ENGINE_OPERATION_CATALOG,
    DaemonControlOperation,
    EngineOperation,
    operation_catalog_fingerprint,
)
from potpie.runtime.resource_manager import (
    AuthenticationError,
    AuthorizationError,
    ContextSelector,
    DestructiveIntent,
    ResourceLifecycleError,
    SelectionError,
)
from potpie_context_engine import EngineError, Failure, Success
from potpie_context_engine.outcomes import RetryPosture
from potpie_context_engine.requests import EngineRequest


PROTOCOL_VERSION = 2
PROTOCOL_MIN_VERSION = PROTOCOL_VERSION
PROTOCOL_MAX_VERSION = PROTOCOL_VERSION


@dataclass(frozen=True, slots=True)
class ProtocolError:
    code: str
    message: str
    details: Mapping[str, object] = field(default_factory=dict)
    recommended_next_action: str | None = None
    retry_posture: RetryPosture = "not_applicable"
    category: Literal["protocol"] = "protocol"


@dataclass(frozen=True, slots=True)
class ProtocolTransportError:
    code: str
    message: str
    details: Mapping[str, object] = field(default_factory=dict)
    recommended_next_action: str | None = None
    retry_posture: RetryPosture = "unknown"
    category: Literal["protocol_transport"] = "protocol_transport"


@dataclass(frozen=True, slots=True)
class DaemonInternalError:
    code: str
    message: str
    details: Mapping[str, object] = field(default_factory=dict)
    recommended_next_action: str | None = None
    retry_posture: RetryPosture = "unknown"
    category: Literal["daemon_internal"] = "daemon_internal"


RuntimeBoundaryError: TypeAlias = (
    AuthenticationError
    | AuthorizationError
    | SelectionError
    | ResourceLifecycleError
    | EngineError
    | ProtocolError
    | ProtocolTransportError
    | DaemonInternalError
)


@dataclass(frozen=True, slots=True)
class HandshakePayload:
    client_protocol_min: int = PROTOCOL_MIN_VERSION
    client_protocol_max: int = PROTOCOL_MAX_VERSION
    expected_instance_id: str | None = None
    client_operation_catalog_fingerprint: str = field(
        default_factory=operation_catalog_fingerprint
    )

    def __post_init__(self) -> None:
        if self.client_protocol_min <= 0:
            raise ValueError("client protocol minimum must be positive")
        if self.client_protocol_max < self.client_protocol_min:
            raise ValueError("client protocol range is invalid")


LifecycleState = Literal["starting", "ready", "draining", "failed", "stopped"]


@dataclass(frozen=True, slots=True)
class HandshakeResult:
    protocol_min: int
    protocol_max: int
    instance_id: str
    lifecycle_state: LifecycleState
    capabilities: tuple[str, ...]
    operation_catalog_fingerprint: str
    compatibility_ticket: str

    @property
    def ready(self) -> bool:
        return self.lifecycle_state == "ready"


@dataclass(frozen=True, slots=True)
class ShutdownPayload:
    reason: str = "client_requested"


@dataclass(frozen=True, slots=True)
class ShutdownResult:
    accepted: bool


@dataclass(frozen=True, slots=True)
class DaemonStatusPayload:
    pass


@dataclass(frozen=True, slots=True)
class DaemonStatusResult:
    instance_id: str
    pid: int
    lifecycle_state: LifecycleState
    backend_profile: str
    ui_url: str


@dataclass(frozen=True, slots=True)
class EngineOperationRequest:
    protocol_version: int
    request_id: str
    operation: EngineOperation
    selector: ContextSelector
    payload: EngineRequest
    destructive_intent: DestructiveIntent | None = None
    compatibility_ticket: str | None = None

    def __post_init__(self) -> None:
        _validate_request_identity(self.protocol_version, self.request_id)
        expected = ENGINE_OPERATION_CATALOG[self.operation].request_type
        if type(self.payload) is not expected:
            raise ValueError(
                f"{self.operation.value} requires payload {expected.__name__}"
            )


@dataclass(frozen=True, slots=True)
class HandshakeRequest:
    protocol_version: int
    request_id: str
    payload: HandshakePayload
    operation: Literal[DaemonControlOperation.HANDSHAKE] = (
        DaemonControlOperation.HANDSHAKE
    )

    def __post_init__(self) -> None:
        _validate_request_identity(self.protocol_version, self.request_id)


@dataclass(frozen=True, slots=True)
class ShutdownRequest:
    protocol_version: int
    request_id: str
    payload: ShutdownPayload
    compatibility_ticket: str | None = None
    operation: Literal[DaemonControlOperation.SHUTDOWN] = (
        DaemonControlOperation.SHUTDOWN
    )

    def __post_init__(self) -> None:
        _validate_request_identity(self.protocol_version, self.request_id)


@dataclass(frozen=True, slots=True)
class DaemonStatusRequest:
    protocol_version: int
    request_id: str
    payload: DaemonStatusPayload
    compatibility_ticket: str | None = None
    operation: Literal[DaemonControlOperation.STATUS] = DaemonControlOperation.STATUS

    def __post_init__(self) -> None:
        _validate_request_identity(self.protocol_version, self.request_id)


ProtocolRequest: TypeAlias = (
    EngineOperationRequest | HandshakeRequest | DaemonStatusRequest | ShutdownRequest
)

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class SuccessResponse(Generic[T]):
    protocol_version: int
    request_id: str
    outcome: Success[T]


@dataclass(frozen=True, slots=True)
class FailureResponse:
    protocol_version: int
    request_id: str
    outcome: Failure[RuntimeBoundaryError]


ProtocolResponse: TypeAlias = SuccessResponse[object] | FailureResponse


class TransportFailure(Exception):
    """Internal transport signal normalized by DaemonEngineClient."""

    def __init__(self, *, code: str, dispatched: bool) -> None:
        super().__init__(code)
        self.code = code
        self.dispatched = dispatched


def response_validation_error(
    request: ProtocolRequest, response: ProtocolResponse
) -> ProtocolError | None:
    if response.request_id != request.request_id:
        return ProtocolError(
            code="response_request_id_mismatch",
            message="daemon response request ID does not match the request",
            details={
                "expected_request_id": request.request_id,
                "received_request_id": response.request_id,
            },
            recommended_next_action="discard the response and reconnect",
        )
    if response.protocol_version != request.protocol_version:
        return ProtocolError(
            code="response_protocol_version_mismatch",
            message="daemon response protocol version does not match the request",
            details={
                "expected_protocol_version": request.protocol_version,
                "received_protocol_version": response.protocol_version,
            },
            recommended_next_action="restart with a compatible Potpie version",
        )
    return None


def _validate_request_identity(protocol_version: int, request_id: str) -> None:
    if protocol_version <= 0:
        raise ValueError("protocol version must be positive")
    if not request_id.strip():
        raise ValueError("request ID must not be empty")


__all__ = [
    "PROTOCOL_MAX_VERSION",
    "PROTOCOL_MIN_VERSION",
    "PROTOCOL_VERSION",
    "DaemonInternalError",
    "DaemonStatusPayload",
    "DaemonStatusRequest",
    "DaemonStatusResult",
    "EngineOperationRequest",
    "FailureResponse",
    "HandshakePayload",
    "HandshakeRequest",
    "HandshakeResult",
    "LifecycleState",
    "ProtocolError",
    "ProtocolRequest",
    "ProtocolResponse",
    "ProtocolTransportError",
    "RuntimeBoundaryError",
    "ShutdownPayload",
    "ShutdownRequest",
    "ShutdownResult",
    "SuccessResponse",
    "TransportFailure",
    "response_validation_error",
]
