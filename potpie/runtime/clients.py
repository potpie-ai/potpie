"""Named local and daemon Context Engine clients."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol, TypeAlias, cast
from uuid import uuid4

from potpie.runtime.coordinator import OperationCoordinator
from potpie.runtime.operations import (
    ENGINE_OPERATION_CATALOG,
    EngineOperation,
    SafetyClass,
    operation_capabilities,
    operation_catalog_fingerprint,
)
from potpie.runtime.protocol import (
    PROTOCOL_MAX_VERSION,
    PROTOCOL_MIN_VERSION,
    PROTOCOL_VERSION,
    DaemonInternalError,
    DaemonStatusPayload,
    DaemonStatusRequest,
    DaemonStatusResult,
    EngineOperationRequest,
    FailureResponse,
    HandshakePayload,
    HandshakeRequest,
    HandshakeResult,
    ProtocolError,
    ProtocolRequest,
    ProtocolResponse,
    ProtocolTransportError,
    RuntimeBoundaryError,
    ShutdownPayload,
    ShutdownRequest,
    ShutdownResult,
    SuccessResponse,
    TransportFailure,
    response_validation_error,
)
from potpie.runtime.resource_manager import (
    AcquisitionRequest,
    ContextResourceManager,
    ContextSelector,
    DestructiveIntent,
)
from potpie_context_engine import ContextEngine, Failure, Outcome, Success
from potpie_context_engine.requests import (
    CatalogRequest,
    CommitRequest,
    DataPlaneStatusRequest,
    DescribeRequest,
    EngineRequest,
    ExportSnapshotRequest,
    HistoryRequest,
    ImportSnapshotRequest,
    InboxAddRequest,
    InboxClaimRequest,
    InboxCloseRequest,
    InboxListRequest,
    InboxMarkAppliedRequest,
    InboxMarkRejectedRequest,
    InboxShowRequest,
    InspectRequest,
    MutateRequest,
    NeighborhoodRequest,
    NudgeRequest,
    ProcessingStatusRequest,
    ProposeRequest,
    QualityRequest,
    ReadRequest,
    RecordRequest,
    RepairRequest,
    ResolveRequest,
    SearchEntitiesRequest,
    SearchRequest,
    SubmitArtifactRequest,
    SubmitEventRequest,
)


ClientOutcome: TypeAlias = Success[object] | Failure[RuntimeBoundaryError]
RequestIdFactory: TypeAlias = Callable[[], str]


@dataclass(frozen=True, slots=True)
class DestructiveConfirmation:
    """One-call confirmation from which a bound client builds exact intent."""

    confirmed: bool


class DaemonTransport(Protocol):
    async def send(self, request: ProtocolRequest) -> ProtocolResponse: ...


class EngineClient(ABC):
    """Named client surface mirroring the public ContextEngine operation catalog."""

    @abstractmethod
    async def _dispatch(
        self,
        operation: EngineOperation,
        request: EngineRequest,
        confirmation: DestructiveConfirmation | None = None,
    ) -> ClientOutcome: ...

    async def resolve(self, request: ResolveRequest) -> ClientOutcome:
        return await self._dispatch(EngineOperation.RESOLVE, request)

    async def search(self, request: SearchRequest) -> ClientOutcome:
        return await self._dispatch(EngineOperation.SEARCH, request)

    async def record(self, request: RecordRequest) -> ClientOutcome:
        return await self._dispatch(EngineOperation.RECORD, request)

    async def data_plane_status(self, request: DataPlaneStatusRequest) -> ClientOutcome:
        return await self._dispatch(EngineOperation.DATA_PLANE_STATUS, request)

    async def catalog(self, request: CatalogRequest) -> ClientOutcome:
        return await self._dispatch(EngineOperation.CATALOG, request)

    async def describe(self, request: DescribeRequest) -> ClientOutcome:
        return await self._dispatch(EngineOperation.DESCRIBE, request)

    async def read(self, request: ReadRequest) -> ClientOutcome:
        return await self._dispatch(EngineOperation.READ, request)

    async def search_entities(self, request: SearchEntitiesRequest) -> ClientOutcome:
        return await self._dispatch(EngineOperation.SEARCH_ENTITIES, request)

    async def mutate(self, request: MutateRequest) -> ClientOutcome:
        return await self._dispatch(EngineOperation.MUTATE, request)

    async def neighborhood(self, request: NeighborhoodRequest) -> ClientOutcome:
        return await self._dispatch(EngineOperation.NEIGHBORHOOD, request)

    async def inspect(self, request: InspectRequest) -> ClientOutcome:
        return await self._dispatch(EngineOperation.INSPECT, request)

    async def export_snapshot(self, request: ExportSnapshotRequest) -> ClientOutcome:
        return await self._dispatch(EngineOperation.EXPORT_SNAPSHOT, request)

    async def import_snapshot(
        self,
        request: ImportSnapshotRequest,
        *,
        confirmation: DestructiveConfirmation | None = None,
    ) -> ClientOutcome:
        return await self._dispatch(
            EngineOperation.IMPORT_SNAPSHOT, request, confirmation
        )

    async def repair(
        self,
        request: RepairRequest,
        *,
        confirmation: DestructiveConfirmation | None = None,
    ) -> ClientOutcome:
        return await self._dispatch(EngineOperation.REPAIR, request, confirmation)

    async def propose(self, request: ProposeRequest) -> ClientOutcome:
        return await self._dispatch(EngineOperation.PROPOSE, request)

    async def commit(self, request: CommitRequest) -> ClientOutcome:
        return await self._dispatch(EngineOperation.COMMIT, request)

    async def history(self, request: HistoryRequest) -> ClientOutcome:
        return await self._dispatch(EngineOperation.HISTORY, request)

    async def quality(self, request: QualityRequest) -> ClientOutcome:
        return await self._dispatch(EngineOperation.QUALITY, request)

    async def inbox_add(self, request: InboxAddRequest) -> ClientOutcome:
        return await self._dispatch(EngineOperation.INBOX_ADD, request)

    async def inbox_list(self, request: InboxListRequest) -> ClientOutcome:
        return await self._dispatch(EngineOperation.INBOX_LIST, request)

    async def inbox_show(self, request: InboxShowRequest) -> ClientOutcome:
        return await self._dispatch(EngineOperation.INBOX_SHOW, request)

    async def inbox_claim(self, request: InboxClaimRequest) -> ClientOutcome:
        return await self._dispatch(EngineOperation.INBOX_CLAIM, request)

    async def inbox_mark_applied(
        self, request: InboxMarkAppliedRequest
    ) -> ClientOutcome:
        return await self._dispatch(EngineOperation.INBOX_MARK_APPLIED, request)

    async def inbox_mark_rejected(
        self, request: InboxMarkRejectedRequest
    ) -> ClientOutcome:
        return await self._dispatch(EngineOperation.INBOX_MARK_REJECTED, request)

    async def inbox_close(self, request: InboxCloseRequest) -> ClientOutcome:
        return await self._dispatch(EngineOperation.INBOX_CLOSE, request)

    async def submit_event(self, request: SubmitEventRequest) -> ClientOutcome:
        return await self._dispatch(EngineOperation.SUBMIT_EVENT, request)

    async def submit_artifact(self, request: SubmitArtifactRequest) -> ClientOutcome:
        return await self._dispatch(EngineOperation.SUBMIT_ARTIFACT, request)

    async def processing_status(
        self, request: ProcessingStatusRequest
    ) -> ClientOutcome:
        return await self._dispatch(EngineOperation.PROCESSING_STATUS, request)

    async def nudge(self, request: NudgeRequest) -> ClientOutcome:
        return await self._dispatch(EngineOperation.NUDGE, request)


EngineHandler: TypeAlias = Callable[
    [ContextEngine, EngineRequest], Awaitable[Outcome[object]]
]


async def _resolve(engine: ContextEngine, request: EngineRequest) -> Outcome[object]:
    return await engine.resolve(cast(ResolveRequest, request))


async def _search(engine: ContextEngine, request: EngineRequest) -> Outcome[object]:
    return await engine.search(cast(SearchRequest, request))


async def _record(engine: ContextEngine, request: EngineRequest) -> Outcome[object]:
    return await engine.record(cast(RecordRequest, request))


async def _data_plane_status(
    engine: ContextEngine, request: EngineRequest
) -> Outcome[object]:
    return await engine.data_plane_status(cast(DataPlaneStatusRequest, request))


async def _catalog(engine: ContextEngine, request: EngineRequest) -> Outcome[object]:
    return await engine.catalog(cast(CatalogRequest, request))


async def _describe(engine: ContextEngine, request: EngineRequest) -> Outcome[object]:
    return await engine.describe(cast(DescribeRequest, request))


async def _read(engine: ContextEngine, request: EngineRequest) -> Outcome[object]:
    return await engine.read(cast(ReadRequest, request))


async def _search_entities(
    engine: ContextEngine, request: EngineRequest
) -> Outcome[object]:
    return await engine.search_entities(cast(SearchEntitiesRequest, request))


async def _mutate(engine: ContextEngine, request: EngineRequest) -> Outcome[object]:
    return await engine.mutate(cast(MutateRequest, request))


async def _neighborhood(
    engine: ContextEngine, request: EngineRequest
) -> Outcome[object]:
    return await engine.neighborhood(cast(NeighborhoodRequest, request))


async def _inspect(engine: ContextEngine, request: EngineRequest) -> Outcome[object]:
    return await engine.inspect(cast(InspectRequest, request))


async def _export_snapshot(
    engine: ContextEngine, request: EngineRequest
) -> Outcome[object]:
    return await engine.export_snapshot(cast(ExportSnapshotRequest, request))


async def _import_snapshot(
    engine: ContextEngine, request: EngineRequest
) -> Outcome[object]:
    return await engine.import_snapshot(cast(ImportSnapshotRequest, request))


async def _repair(engine: ContextEngine, request: EngineRequest) -> Outcome[object]:
    return await engine.repair(cast(RepairRequest, request))


async def _propose(engine: ContextEngine, request: EngineRequest) -> Outcome[object]:
    return await engine.propose(cast(ProposeRequest, request))


async def _commit(engine: ContextEngine, request: EngineRequest) -> Outcome[object]:
    return await engine.commit(cast(CommitRequest, request))


async def _history(engine: ContextEngine, request: EngineRequest) -> Outcome[object]:
    return await engine.history(cast(HistoryRequest, request))


async def _quality(engine: ContextEngine, request: EngineRequest) -> Outcome[object]:
    return await engine.quality(cast(QualityRequest, request))


async def _inbox_add(engine: ContextEngine, request: EngineRequest) -> Outcome[object]:
    return await engine.inbox_add(cast(InboxAddRequest, request))


async def _inbox_list(engine: ContextEngine, request: EngineRequest) -> Outcome[object]:
    return await engine.inbox_list(cast(InboxListRequest, request))


async def _inbox_show(engine: ContextEngine, request: EngineRequest) -> Outcome[object]:
    return await engine.inbox_show(cast(InboxShowRequest, request))


async def _inbox_claim(
    engine: ContextEngine, request: EngineRequest
) -> Outcome[object]:
    return await engine.inbox_claim(cast(InboxClaimRequest, request))


async def _inbox_mark_applied(
    engine: ContextEngine, request: EngineRequest
) -> Outcome[object]:
    return await engine.inbox_mark_applied(cast(InboxMarkAppliedRequest, request))


async def _inbox_mark_rejected(
    engine: ContextEngine, request: EngineRequest
) -> Outcome[object]:
    return await engine.inbox_mark_rejected(cast(InboxMarkRejectedRequest, request))


async def _inbox_close(
    engine: ContextEngine, request: EngineRequest
) -> Outcome[object]:
    return await engine.inbox_close(cast(InboxCloseRequest, request))


async def _submit_event(
    engine: ContextEngine, request: EngineRequest
) -> Outcome[object]:
    return await engine.submit_event(cast(SubmitEventRequest, request))


async def _submit_artifact(
    engine: ContextEngine, request: EngineRequest
) -> Outcome[object]:
    return await engine.submit_artifact(cast(SubmitArtifactRequest, request))


async def _processing_status(
    engine: ContextEngine, request: EngineRequest
) -> Outcome[object]:
    return await engine.processing_status(cast(ProcessingStatusRequest, request))


async def _nudge(engine: ContextEngine, request: EngineRequest) -> Outcome[object]:
    return await engine.nudge(cast(NudgeRequest, request))


_ENGINE_HANDLERS: dict[EngineOperation, EngineHandler] = {
    EngineOperation.RESOLVE: _resolve,
    EngineOperation.SEARCH: _search,
    EngineOperation.RECORD: _record,
    EngineOperation.DATA_PLANE_STATUS: _data_plane_status,
    EngineOperation.CATALOG: _catalog,
    EngineOperation.DESCRIBE: _describe,
    EngineOperation.READ: _read,
    EngineOperation.SEARCH_ENTITIES: _search_entities,
    EngineOperation.MUTATE: _mutate,
    EngineOperation.NEIGHBORHOOD: _neighborhood,
    EngineOperation.INSPECT: _inspect,
    EngineOperation.EXPORT_SNAPSHOT: _export_snapshot,
    EngineOperation.IMPORT_SNAPSHOT: _import_snapshot,
    EngineOperation.REPAIR: _repair,
    EngineOperation.PROPOSE: _propose,
    EngineOperation.COMMIT: _commit,
    EngineOperation.HISTORY: _history,
    EngineOperation.QUALITY: _quality,
    EngineOperation.INBOX_ADD: _inbox_add,
    EngineOperation.INBOX_LIST: _inbox_list,
    EngineOperation.INBOX_SHOW: _inbox_show,
    EngineOperation.INBOX_CLAIM: _inbox_claim,
    EngineOperation.INBOX_MARK_APPLIED: _inbox_mark_applied,
    EngineOperation.INBOX_MARK_REJECTED: _inbox_mark_rejected,
    EngineOperation.INBOX_CLOSE: _inbox_close,
    EngineOperation.SUBMIT_EVENT: _submit_event,
    EngineOperation.SUBMIT_ARTIFACT: _submit_artifact,
    EngineOperation.PROCESSING_STATUS: _processing_status,
    EngineOperation.NUDGE: _nudge,
}

if set(_ENGINE_HANDLERS) != set(ENGINE_OPERATION_CATALOG):
    raise RuntimeError("each engine operation must have exactly one local handler")


class TypedEngineOperationHandler:
    """Shared transport-neutral lease-to-engine operation handler."""

    def __init__(
        self,
        resource_manager: ContextResourceManager,
        *,
        coordinator: OperationCoordinator,
    ) -> None:
        self._resource_manager = resource_manager
        self._coordinator = coordinator

    async def handle(
        self,
        request: EngineOperationRequest,
        *,
        authentication: object,
    ) -> ClientOutcome:
        spec = ENGINE_OPERATION_CATALOG[request.operation]
        try:
            acquisition = await self._resource_manager.acquire(
                AcquisitionRequest(
                    request_id=request.request_id,
                    selector=request.selector,
                    operation=request.operation.value,
                    authentication=authentication,
                    destructive=spec.destructive,
                    destructive_intent=request.destructive_intent,
                )
            )
        except Exception:
            return Failure(
                DaemonInternalError(
                    code="operation_acquisition_failed",
                    message="the typed operation could not acquire its context",
                    details={"request_id": request.request_id},
                    recommended_next_action="retry or inspect runtime logs",
                )
            )
        if isinstance(acquisition, Failure):
            return acquisition
        lease = acquisition.value
        try:
            try:
                async with self._coordinator.coordinate(
                    spec=spec,
                    context=lease.context,
                    request=request.payload,
                ):
                    return await _ENGINE_HANDLERS[request.operation](
                        lease.engine, request.payload
                    )
            except Exception:
                return Failure(
                    DaemonInternalError(
                        code="operation_handler_failed",
                        message="the typed operation handler failed",
                        details={"request_id": request.request_id},
                        recommended_next_action="retry or inspect runtime logs",
                    )
                )
        finally:
            await lease.release()


class LocalEngineClient(EngineClient):
    """Invoke explicit handlers through one ContextResourceManager policy."""

    def __init__(
        self,
        *,
        selector: ContextSelector,
        authentication: object,
        resource_manager: ContextResourceManager,
        coordinator: OperationCoordinator,
        request_id_factory: RequestIdFactory | None = None,
    ) -> None:
        self._selector = selector
        self._authentication = authentication
        self._handler = TypedEngineOperationHandler(
            resource_manager,
            coordinator=coordinator,
        )
        self._request_id_factory = request_id_factory or (lambda: str(uuid4()))

    async def _dispatch(
        self,
        operation: EngineOperation,
        request: EngineRequest,
        confirmation: DestructiveConfirmation | None = None,
    ) -> ClientOutcome:
        envelope_or_error = _build_operation_request(
            selector=self._selector,
            operation=operation,
            payload=request,
            confirmation=confirmation,
            request_id=self._request_id_factory(),
            compatibility_ticket=None,
        )
        if isinstance(envelope_or_error, Failure):
            return envelope_or_error
        return await self._handler.handle(
            envelope_or_error.value,
            authentication=self._authentication,
        )


class DaemonEngineClient(EngineClient):
    """Send typed Context Engine operations through one negotiated daemon protocol."""

    def __init__(
        self,
        *,
        selector: ContextSelector,
        transport: DaemonTransport,
        expected_instance_id: str | None,
        request_id_factory: RequestIdFactory | None = None,
    ) -> None:
        self._selector = selector
        self._transport = transport
        self._expected_instance_id = expected_instance_id
        self._request_id_factory = request_id_factory or (lambda: str(uuid4()))
        self._handshake_result: HandshakeResult | None = None

    @property
    def handshake_result(self) -> HandshakeResult | None:
        return self._handshake_result

    async def handshake(
        self,
    ) -> Success[HandshakeResult] | Failure[RuntimeBoundaryError]:
        request = HandshakeRequest(
            protocol_version=PROTOCOL_VERSION,
            request_id=self._request_id_factory(),
            payload=HandshakePayload(
                client_protocol_min=PROTOCOL_MIN_VERSION,
                client_protocol_max=PROTOCOL_MAX_VERSION,
                expected_instance_id=self._expected_instance_id,
                client_operation_catalog_fingerprint=(operation_catalog_fingerprint()),
            ),
        )
        response_or_error = await self._send(request, safety=None)
        if isinstance(response_or_error, Failure):
            return response_or_error
        response = response_or_error.value
        if isinstance(response, FailureResponse):
            return response.outcome
        result = response.outcome.value
        if not isinstance(result, HandshakeResult):
            return Failure(
                ProtocolError(
                    code="handshake_result_malformed",
                    message="daemon handshake returned an invalid result",
                )
            )
        validation_error = self._validate_handshake(result)
        if validation_error is not None:
            return Failure(validation_error)
        self._handshake_result = result
        return Success(result)

    async def _dispatch(
        self,
        operation: EngineOperation,
        request: EngineRequest,
        confirmation: DestructiveConfirmation | None = None,
    ) -> ClientOutcome:
        if self._handshake_result is None:
            return Failure(
                ProtocolError(
                    code="handshake_required",
                    message="a compatible daemon handshake is required",
                    recommended_next_action="perform an authenticated handshake",
                    retry_posture="safe",
                )
            )
        envelope_or_error = _build_operation_request(
            selector=self._selector,
            operation=operation,
            payload=request,
            confirmation=confirmation,
            request_id=self._request_id_factory(),
            compatibility_ticket=self._handshake_result.compatibility_ticket,
        )
        if isinstance(envelope_or_error, Failure):
            return envelope_or_error
        envelope = envelope_or_error.value
        response_or_error = await self._send(
            envelope,
            safety=ENGINE_OPERATION_CATALOG[operation].safety,
        )
        if isinstance(response_or_error, Failure):
            return response_or_error
        response = response_or_error.value
        return response.outcome

    async def _send(
        self,
        request: ProtocolRequest,
        *,
        safety: SafetyClass | None,
    ) -> Success[ProtocolResponse] | Failure[RuntimeBoundaryError]:
        return await _send_protocol_request(
            transport=self._transport,
            request=request,
            safety=safety,
        )

    def _validate_handshake(self, result: HandshakeResult) -> ProtocolError | None:
        return _validate_handshake_result(
            result,
            expected_instance_id=self._expected_instance_id,
        )


class DaemonControlClient:
    """Finite typed client for live daemon handshake and shutdown control."""

    def __init__(
        self,
        *,
        transport: DaemonTransport,
        expected_instance_id: str | None,
        request_id_factory: RequestIdFactory | None = None,
    ) -> None:
        self._transport = transport
        self._expected_instance_id = expected_instance_id
        self._request_id_factory = request_id_factory or (lambda: str(uuid4()))
        self._handshake_result: HandshakeResult | None = None

    @property
    def handshake_result(self) -> HandshakeResult | None:
        return self._handshake_result

    async def handshake(
        self,
    ) -> Success[HandshakeResult] | Failure[RuntimeBoundaryError]:
        request = HandshakeRequest(
            protocol_version=PROTOCOL_VERSION,
            request_id=self._request_id_factory(),
            payload=HandshakePayload(
                client_protocol_min=PROTOCOL_MIN_VERSION,
                client_protocol_max=PROTOCOL_MAX_VERSION,
                expected_instance_id=self._expected_instance_id,
                client_operation_catalog_fingerprint=(operation_catalog_fingerprint()),
            ),
        )
        response_or_error = await _send_protocol_request(
            transport=self._transport,
            request=request,
            safety=None,
        )
        if isinstance(response_or_error, Failure):
            return response_or_error
        response = response_or_error.value
        if isinstance(response, FailureResponse):
            return response.outcome
        result = response.outcome.value
        if not isinstance(result, HandshakeResult):
            return Failure(
                ProtocolError(
                    code="handshake_result_malformed",
                    message="daemon handshake returned an invalid result",
                )
            )
        validation_error = _validate_handshake_result(
            result,
            expected_instance_id=self._expected_instance_id,
        )
        if validation_error is not None:
            return Failure(validation_error)
        self._handshake_result = result
        return Success(result)

    async def shutdown(
        self, *, reason: str = "client_requested"
    ) -> Success[ShutdownResult] | Failure[RuntimeBoundaryError]:
        if self._handshake_result is None:
            return Failure(
                ProtocolError(
                    code="handshake_required",
                    message="a compatible daemon handshake is required",
                    recommended_next_action="perform an authenticated handshake",
                    retry_posture="safe",
                )
            )
        request = ShutdownRequest(
            protocol_version=PROTOCOL_VERSION,
            request_id=self._request_id_factory(),
            payload=ShutdownPayload(reason=reason),
            compatibility_ticket=self._handshake_result.compatibility_ticket,
        )
        response_or_error = await _send_protocol_request(
            transport=self._transport,
            request=request,
            safety=SafetyClass.DAEMON_LIFECYCLE_CONTROL,
        )
        if isinstance(response_or_error, Failure):
            return response_or_error
        response = response_or_error.value
        if isinstance(response, FailureResponse):
            return response.outcome
        result = response.outcome.value
        if not isinstance(result, ShutdownResult):
            return Failure(
                ProtocolError(
                    code="shutdown_result_malformed",
                    message="daemon shutdown returned an invalid result",
                )
            )
        return Success(result)

    async def status(
        self,
    ) -> Success[DaemonStatusResult] | Failure[RuntimeBoundaryError]:
        if self._handshake_result is None:
            return Failure(
                ProtocolError(
                    code="handshake_required",
                    message="a compatible daemon handshake is required",
                    recommended_next_action="perform an authenticated handshake",
                    retry_posture="safe",
                )
            )
        request = DaemonStatusRequest(
            protocol_version=PROTOCOL_VERSION,
            request_id=self._request_id_factory(),
            payload=DaemonStatusPayload(),
            compatibility_ticket=self._handshake_result.compatibility_ticket,
        )
        response_or_error = await _send_protocol_request(
            transport=self._transport,
            request=request,
            safety=SafetyClass.DAEMON_LIFECYCLE_CONTROL,
        )
        if isinstance(response_or_error, Failure):
            return response_or_error
        response = response_or_error.value
        if isinstance(response, FailureResponse):
            return response.outcome
        result = response.outcome.value
        if not isinstance(result, DaemonStatusResult):
            return Failure(
                ProtocolError(
                    code="daemon_status_result_malformed",
                    message="daemon status returned an invalid result",
                )
            )
        return Success(result)


def _build_operation_request(
    *,
    selector: ContextSelector,
    operation: EngineOperation,
    payload: EngineRequest,
    confirmation: DestructiveConfirmation | None,
    request_id: str,
    compatibility_ticket: str | None,
) -> Success[EngineOperationRequest] | Failure[ProtocolError]:
    spec = ENGINE_OPERATION_CATALOG[operation]
    if type(payload) is not spec.request_type:
        return Failure(
            ProtocolError(
                code="operation_payload_mismatch",
                message="operation payload does not match the operation catalog",
                details={
                    "operation": operation.value,
                    "expected_payload": spec.request_type.__name__,
                    "received_payload": type(payload).__name__,
                },
            )
        )
    if confirmation is not None and not spec.destructive:
        return Failure(
            ProtocolError(
                code="destructive_confirmation_unexpected",
                message="destructive confirmation is not valid for this operation",
                details={"operation": operation.value},
            )
        )
    intent = None
    if confirmation is not None:
        intent = DestructiveIntent(
            confirmed=confirmation.confirmed,
            operation=operation.value,
            selector=selector,
            request_id=request_id,
        )
    return Success(
        EngineOperationRequest(
            protocol_version=PROTOCOL_VERSION,
            request_id=request_id,
            operation=operation,
            selector=selector,
            payload=payload,
            destructive_intent=intent,
            compatibility_ticket=compatibility_ticket,
        )
    )


def _outcome_may_be_unknown(safety: SafetyClass | None, *, dispatched: bool) -> bool:
    return dispatched and safety in {
        SafetyClass.EXCLUSIVE_CONTEXT_MUTATION,
        SafetyClass.EXCLUSIVE_RESOURCE_MUTATION,
        SafetyClass.DAEMON_LIFECYCLE_CONTROL,
    }


async def _send_protocol_request(
    *,
    transport: DaemonTransport,
    request: ProtocolRequest,
    safety: SafetyClass | None,
) -> Success[ProtocolResponse] | Failure[RuntimeBoundaryError]:
    try:
        response = await transport.send(request)
    except TransportFailure as exc:
        outcome_unknown = _outcome_may_be_unknown(safety, dispatched=exc.dispatched)
        return Failure(
            ProtocolTransportError(
                code=exc.code,
                message="the daemon transport failed",
                details={
                    "dispatched": exc.dispatched,
                    "outcome_unknown": outcome_unknown,
                },
                recommended_next_action=(
                    "inspect operation state before retrying"
                    if outcome_unknown
                    else "retry after daemon readiness is restored"
                ),
                retry_posture="unknown" if outcome_unknown else "safe",
            )
        )
    except Exception:
        outcome_unknown = _outcome_may_be_unknown(safety, dispatched=True)
        return Failure(
            ProtocolTransportError(
                code="transport_defect",
                message="the daemon transport failed unexpectedly",
                details={
                    "dispatched": "unknown",
                    "outcome_unknown": outcome_unknown,
                },
                recommended_next_action=(
                    "inspect operation state and daemon client logs before retrying"
                    if outcome_unknown
                    else "inspect daemon client logs"
                ),
                retry_posture="unknown" if outcome_unknown else "safe",
            )
        )
    if not isinstance(response, (SuccessResponse, FailureResponse)):
        return Failure(
            ProtocolError(
                code="response_envelope_malformed",
                message="daemon returned an invalid response envelope",
            )
        )
    validation_error = response_validation_error(request, response)
    if validation_error is not None:
        return Failure(validation_error)
    return Success(response)


def _validate_handshake_result(
    result: HandshakeResult,
    *,
    expected_instance_id: str | None,
) -> ProtocolError | None:
    if expected_instance_id is not None and result.instance_id != expected_instance_id:
        return ProtocolError(
            code="daemon_instance_mismatch",
            message="daemon instance identity does not match discovery",
            recommended_next_action="refresh discovery and reconnect",
        )
    if not result.ready:
        return ProtocolError(
            code="daemon_not_ready",
            message="daemon handshake did not report ready state",
            details={"lifecycle_state": result.lifecycle_state},
            recommended_next_action="wait for daemon readiness",
            retry_posture="safe",
        )
    if not result.protocol_min <= PROTOCOL_VERSION <= result.protocol_max:
        return ProtocolError(
            code="protocol_version_incompatible",
            message="daemon and client protocol versions are incompatible",
            details={
                "client_version": PROTOCOL_VERSION,
                "daemon_min": result.protocol_min,
                "daemon_max": result.protocol_max,
            },
            recommended_next_action="install compatible Potpie versions",
        )
    if result.operation_catalog_fingerprint != operation_catalog_fingerprint():
        return ProtocolError(
            code="operation_catalog_mismatch",
            message="daemon and client operation catalogs do not match",
            recommended_next_action="restart with a compatible Potpie version",
        )
    if not result.compatibility_ticket:
        return ProtocolError(
            code="compatibility_ticket_missing",
            message="daemon handshake did not return a compatibility ticket",
            recommended_next_action="restart with a compatible Potpie version",
        )
    missing = sorted(set(operation_capabilities()) - set(result.capabilities))
    if missing:
        return ProtocolError(
            code="daemon_capability_missing",
            message="daemon does not support the required operation catalog",
            details={"missing_capabilities": tuple(missing)},
            recommended_next_action="install compatible Potpie versions",
        )
    return None


__all__ = [
    "ClientOutcome",
    "DaemonControlClient",
    "DaemonEngineClient",
    "DaemonTransport",
    "DestructiveConfirmation",
    "EngineClient",
    "LocalEngineClient",
    "RequestIdFactory",
    "TypedEngineOperationHandler",
]
