from __future__ import annotations

# ruff: noqa: S101 - pytest unit tests use assertions intentionally.

from collections.abc import Callable
from typing import cast

import pytest

from potpie.runtime import (
    ENGINE_OPERATION_CATALOG,
    PROTOCOL_VERSION,
    AcquisitionRequest,
    AuthorizationScope,
    ContextResourceManager,
    ContextSelector,
    DaemonEngineClient,
    DestructiveConfirmation,
    EngineClient,
    EngineOperation,
    EngineOperationRequest,
    FailureResponse,
    HandshakeRequest,
    HandshakeResult,
    LocalEngineClient,
    OperationCoordinator,
    ProtocolError,
    SafetyClass,
    SuccessResponse,
    TransportFailure,
    operation_capabilities,
    operation_catalog_fingerprint,
)
from potpie_context_engine import (
    ContextEngine,
    ContextIdentity,
    DomainError,
    Failure,
    Success,
)
from potpie_context_engine.requests import (
    RecordRequest,
    RepairRequest,
    ResetContextRequest,
    ResolveRequest,
    SearchRequest,
)


class _RecordingEngine:
    def __init__(self) -> None:
        self.context = ContextIdentity("context-a")
        self.calls: list[tuple[str, object]] = []
        self.raise_on_record = False

    async def resolve(self, request: ResolveRequest):
        self.calls.append(("resolve", request.to_payload()))
        return Success({"resolved": request.to_payload()})

    async def search(self, request: SearchRequest):
        self.calls.append(("search", request.to_payload()))
        return Success({"matches": (request.query,)})

    async def record(self, request: RecordRequest):
        self.calls.append(("record", request.to_payload()))
        if self.raise_on_record:
            raise RuntimeError("sensitive handler detail")
        return Success({"applied": 1})

    async def repair(self, request: RepairRequest):
        self.calls.append(("repair", request.to_payload()))
        return Success({"repaired": True})

    async def reset_context(self, request: ResetContextRequest):
        self.calls.append(("reset_context", request.to_payload()))
        return Success({"context_id": self.context.value, "reset": True})


class _Lease:
    def __init__(self, engine: _RecordingEngine) -> None:
        self.engine = cast(ContextEngine, engine)
        self.context = engine.context
        self.scope = AuthorizationScope(
            actor_id="actor-1",
            operation="search",
            context=engine.context,
        )
        self.release_count = 0

    async def release(self) -> None:
        self.release_count += 1


class _ResourceManager:
    def __init__(self, engine: _RecordingEngine) -> None:
        self.requests: list[AcquisitionRequest] = []
        self.leases: list[_Lease] = []
        self.engine = engine

    async def acquire(self, request: AcquisitionRequest):
        self.requests.append(request)
        lease = _Lease(self.engine)
        self.leases.append(lease)
        return Success(lease)


class _Transport:
    def __init__(self, responder: Callable[[object], object]) -> None:
        self.responder = responder
        self.requests: list[object] = []

    async def send(self, request):
        self.requests.append(request)
        result = self.responder(request)
        if isinstance(result, Exception):
            raise result
        return result


def _request_ids(*values: str):
    remaining = iter(values)
    return lambda: next(remaining)


def _handshake_result(**changes: object) -> HandshakeResult:
    values: dict[str, object] = {
        "protocol_min": PROTOCOL_VERSION,
        "protocol_max": PROTOCOL_VERSION,
        "instance_id": "instance-1",
        "lifecycle_state": "ready",
        "capabilities": operation_capabilities(),
        "operation_catalog_fingerprint": operation_catalog_fingerprint(),
        "compatibility_ticket": "test-ticket",
    }
    values.update(changes)
    return HandshakeResult(**values)  # type: ignore[arg-type]


def _success_response(request: object, value: object) -> SuccessResponse[object]:
    return SuccessResponse(
        protocol_version=cast("HandshakeRequest", request).protocol_version,
        request_id=cast("HandshakeRequest", request).request_id,
        outcome=Success(value),
    )


@pytest.mark.anyio
async def test_engine_client_surface_matches_finite_context_engine_catalog() -> None:
    expected = {operation.value for operation in EngineOperation}
    actual = {name for name in expected if callable(getattr(EngineClient, name, None))}

    assert actual == expected
    assert set(ENGINE_OPERATION_CATALOG) == set(EngineOperation)
    assert all(
        spec.safety
        in {
            SafetyClass.SHARED_CONTEXT_READ,
            SafetyClass.SHARED_CONTEXT_READ_EXCLUSIVE_RESOURCE_WRITE,
            SafetyClass.EXCLUSIVE_CONTEXT_MUTATION,
        }
        for spec in ENGINE_OPERATION_CATALOG.values()
    )
    assert len(operation_catalog_fingerprint()) == 64


def test_operation_envelope_rejects_mismatched_typed_payload() -> None:
    with pytest.raises(ValueError, match="requires payload ResolveRequest"):
        EngineOperationRequest(
            protocol_version=PROTOCOL_VERSION,
            request_id="request-1",
            operation=EngineOperation.RESOLVE,
            selector=ContextSelector(kind="explicit", value="context-a"),
            payload=SearchRequest(),
        )


@pytest.mark.anyio
async def test_local_client_acquires_lease_invokes_named_handler_and_releases() -> None:
    engine = _RecordingEngine()
    manager = _ResourceManager(engine)
    client = LocalEngineClient(
        selector=ContextSelector(kind="explicit", value="context-a"),
        authentication="credential",
        resource_manager=cast(ContextResourceManager, manager),
        coordinator=OperationCoordinator(),
        request_id_factory=_request_ids("request-1"),
    )

    outcome = await client.search(SearchRequest(query="typed"))

    assert outcome == Success({"matches": ("typed",)})
    assert engine.calls == [("search", SearchRequest(query="typed").to_payload())]
    assert manager.requests == [
        AcquisitionRequest(
            request_id="request-1",
            selector=ContextSelector(kind="explicit", value="context-a"),
            operation="search",
            authentication="credential",
        )
    ]
    assert manager.leases[0].release_count == 1


@pytest.mark.anyio
async def test_local_client_binds_destructive_confirmation_to_exact_request() -> None:
    engine = _RecordingEngine()
    manager = _ResourceManager(engine)
    selector = ContextSelector(kind="explicit", value="context-a")
    client = LocalEngineClient(
        selector=selector,
        authentication="credential",
        resource_manager=cast(ContextResourceManager, manager),
        coordinator=OperationCoordinator(),
        request_id_factory=_request_ids("repair-request"),
    )

    outcome = await client.repair(
        RepairRequest(targets=("labels",)),
        confirmation=DestructiveConfirmation(confirmed=True),
    )

    assert isinstance(outcome, Success)
    acquisition = manager.requests[0]
    assert acquisition.destructive is True
    assert acquisition.destructive_intent is not None
    assert acquisition.destructive_intent.confirmed is True
    assert acquisition.destructive_intent.operation == "repair"
    assert acquisition.destructive_intent.selector == selector
    assert acquisition.destructive_intent.request_id == "repair-request"


@pytest.mark.anyio
async def test_local_client_binds_reset_confirmation_to_exact_context() -> None:
    engine = _RecordingEngine()
    manager = _ResourceManager(engine)
    selector = ContextSelector(kind="explicit", value="context-a")
    client = LocalEngineClient(
        selector=selector,
        authentication="credential",
        resource_manager=cast(ContextResourceManager, manager),
        coordinator=OperationCoordinator(),
        request_id_factory=_request_ids("reset-request"),
    )

    outcome = await client.reset_context(
        ResetContextRequest(),
        confirmation=DestructiveConfirmation(confirmed=True),
    )

    assert isinstance(outcome, Success)
    acquisition = manager.requests[0]
    assert acquisition.operation == "reset_context"
    assert acquisition.destructive is True
    assert acquisition.destructive_intent is not None
    assert acquisition.destructive_intent.selector == selector
    assert acquisition.destructive_intent.request_id == "reset-request"


@pytest.mark.anyio
async def test_local_client_redacts_handler_defect_and_releases_lease() -> None:
    engine = _RecordingEngine()
    engine.raise_on_record = True
    manager = _ResourceManager(engine)
    client = LocalEngineClient(
        selector=ContextSelector(kind="explicit", value="context-a"),
        authentication="credential",
        resource_manager=cast(ContextResourceManager, manager),
        coordinator=OperationCoordinator(),
        request_id_factory=_request_ids("request-1"),
    )

    outcome = await client.record(RecordRequest())

    assert isinstance(outcome, Failure)
    assert outcome.error.code == "operation_handler_failed"
    assert "sensitive" not in outcome.error.message
    assert manager.leases[0].release_count == 1


@pytest.mark.anyio
async def test_daemon_client_requires_successful_handshake_before_domain_work() -> None:
    transport = _Transport(lambda request: _success_response(request, {}))
    client = DaemonEngineClient(
        selector=ContextSelector(kind="explicit", value="context-a"),
        transport=transport,
        expected_instance_id="instance-1",
    )

    outcome = await client.search(SearchRequest())

    assert isinstance(outcome, Failure)
    assert outcome.error.code == "handshake_required"
    assert transport.requests == []


@pytest.mark.anyio
async def test_daemon_client_handshake_then_typed_operation() -> None:
    def respond(request: object):
        if isinstance(request, HandshakeRequest):
            return _success_response(request, _handshake_result())
        assert isinstance(request, EngineOperationRequest)
        return _success_response(request, {"matches": (request.payload.query,)})

    transport = _Transport(respond)
    client = DaemonEngineClient(
        selector=ContextSelector(kind="explicit", value="context-a"),
        transport=transport,
        expected_instance_id="instance-1",
        request_id_factory=_request_ids("handshake-1", "search-1"),
    )

    handshake = await client.handshake()
    outcome = await client.search(SearchRequest(query="typed"))

    assert isinstance(handshake, Success)
    assert client.handshake_result == _handshake_result()
    assert outcome == Success({"matches": ("typed",)})
    sent = transport.requests[1]
    assert isinstance(sent, EngineOperationRequest)
    assert sent.protocol_version == PROTOCOL_VERSION
    assert sent.compatibility_ticket == "test-ticket"
    assert sent.request_id == "search-1"
    assert sent.operation is EngineOperation.SEARCH
    assert sent.selector == ContextSelector(kind="explicit", value="context-a")
    assert isinstance(sent.payload, SearchRequest)


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("result", "expected_code"),
    [
        (_handshake_result(instance_id="other"), "daemon_instance_mismatch"),
        (_handshake_result(lifecycle_state="starting"), "daemon_not_ready"),
        (
            _handshake_result(
                protocol_min=PROTOCOL_VERSION + 1,
                protocol_max=PROTOCOL_VERSION + 1,
            ),
            "protocol_version_incompatible",
        ),
        (
            _handshake_result(operation_catalog_fingerprint="different"),
            "operation_catalog_mismatch",
        ),
        (
            _handshake_result(capabilities=(EngineOperation.SEARCH.value,)),
            "daemon_capability_missing",
        ),
    ],
)
async def test_daemon_client_rejects_invalid_handshake(
    result: HandshakeResult, expected_code: str
) -> None:
    transport = _Transport(lambda request: _success_response(request, result))
    client = DaemonEngineClient(
        selector=ContextSelector(kind="explicit", value="context-a"),
        transport=transport,
        expected_instance_id="instance-1",
        request_id_factory=_request_ids("handshake-1"),
    )

    outcome = await client.handshake()

    assert isinstance(outcome, Failure)
    assert outcome.error.code == expected_code
    assert client.handshake_result is None


@pytest.mark.anyio
async def test_daemon_client_preserves_typed_failure_response() -> None:
    def respond(request: object):
        if isinstance(request, HandshakeRequest):
            return _success_response(request, _handshake_result())
        request = cast(EngineOperationRequest, request)
        return FailureResponse(
            protocol_version=request.protocol_version,
            request_id=request.request_id,
            outcome=Failure(DomainError(code="bad_query", message="bad query")),
        )

    transport = _Transport(respond)
    client = DaemonEngineClient(
        selector=ContextSelector(kind="explicit", value="context-a"),
        transport=transport,
        expected_instance_id="instance-1",
        request_id_factory=_request_ids("handshake-1", "search-1"),
    )
    assert isinstance(await client.handshake(), Success)

    outcome = await client.search(SearchRequest())

    assert outcome == Failure(DomainError(code="bad_query", message="bad query"))


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("operation", "expected_retry", "expected_unknown"),
    [("search", "safe", False), ("record", "unknown", True)],
)
async def test_daemon_client_never_replays_transport_failure(
    operation: str, expected_retry: str, expected_unknown: bool
) -> None:
    def respond(request: object):
        if isinstance(request, HandshakeRequest):
            return _success_response(request, _handshake_result())
        return TransportFailure(code="connection_lost", dispatched=True)

    transport = _Transport(respond)
    client = DaemonEngineClient(
        selector=ContextSelector(kind="explicit", value="context-a"),
        transport=transport,
        expected_instance_id="instance-1",
        request_id_factory=_request_ids("handshake-1", "operation-1"),
    )
    assert isinstance(await client.handshake(), Success)

    if operation == "search":
        outcome = await client.search(SearchRequest())
    else:
        outcome = await client.record(RecordRequest())

    assert isinstance(outcome, Failure)
    assert outcome.error.retry_posture == expected_retry
    assert outcome.error.details["outcome_unknown"] is expected_unknown
    assert len(transport.requests) == 2


@pytest.mark.anyio
async def test_response_correlation_mismatch_is_typed_protocol_failure() -> None:
    def respond(request: object):
        request = cast(HandshakeRequest, request)
        return SuccessResponse(
            protocol_version=request.protocol_version,
            request_id="different-request",
            outcome=Success(_handshake_result()),
        )

    client = DaemonEngineClient(
        selector=ContextSelector(kind="explicit", value="context-a"),
        transport=_Transport(respond),
        expected_instance_id="instance-1",
        request_id_factory=_request_ids("handshake-1"),
    )

    outcome = await client.handshake()

    assert isinstance(outcome, Failure)
    assert isinstance(outcome.error, ProtocolError)
    assert outcome.error.code == "response_request_id_mismatch"
