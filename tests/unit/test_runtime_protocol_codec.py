from __future__ import annotations

# ruff: noqa: S101 - pytest unit tests use assertions intentionally.

import pytest

from potpie.runtime import (
    ENGINE_OPERATION_CATALOG,
    PROTOCOL_VERSION,
    AuthenticationError,
    ContextSelector,
    DaemonInternalError,
    DaemonStatusPayload,
    DaemonStatusRequest,
    DaemonStatusResult,
    EngineOperation,
    EngineOperationRequest,
    FailureResponse,
    HandshakePayload,
    HandshakeRequest,
    HandshakeResult,
    ProtocolError,
    ShutdownPayload,
    ShutdownRequest,
    ShutdownResult,
    SuccessResponse,
    decode_request,
    decode_response,
    encode_request,
    encode_response,
    operation_capabilities,
    operation_catalog_fingerprint,
)
from potpie_context_engine import DomainError, Failure, Success
from potpie_context_engine.core.agent_envelope import (
    AgentEnvelope,
    CoverageReport,
    EvidenceItem,
)
from potpie_context_engine.requests import SearchRequest


@pytest.mark.parametrize("operation", list(EngineOperation))
def test_every_engine_operation_round_trips_without_python_class_identity(
    operation: EngineOperation,
) -> None:
    request_type = ENGINE_OPERATION_CATALOG[operation].request_type
    request = EngineOperationRequest(
        protocol_version=PROTOCOL_VERSION,
        request_id=f"request-{operation.value}",
        operation=operation,
        selector=ContextSelector(kind="explicit", value="context-a"),
        payload=request_type(),
    )

    document = encode_request(request)
    decoded = decode_request(document)

    assert "class" not in str(document).lower()
    assert "module" not in document
    assert isinstance(decoded, Success)
    assert decoded.value == request


@pytest.mark.parametrize(
    "protocol_request",
    [
        HandshakeRequest(
            protocol_version=PROTOCOL_VERSION,
            request_id="handshake-1",
            payload=HandshakePayload(expected_instance_id="instance-1"),
        ),
        ShutdownRequest(
            protocol_version=PROTOCOL_VERSION,
            request_id="shutdown-1",
            payload=ShutdownPayload(reason="test"),
        ),
        DaemonStatusRequest(
            protocol_version=PROTOCOL_VERSION,
            request_id="status-1",
            payload=DaemonStatusPayload(),
        ),
    ],
)
def test_control_requests_round_trip(protocol_request: object) -> None:
    document = encode_request(protocol_request)  # type: ignore[arg-type]
    decoded = decode_request(document)

    assert isinstance(decoded, Success)
    assert decoded.value == protocol_request


def test_engine_success_response_round_trips_to_the_catalog_result_type() -> None:
    request = EngineOperationRequest(
        protocol_version=PROTOCOL_VERSION,
        request_id="search-1",
        operation=EngineOperation.SEARCH,
        selector=ContextSelector(kind="explicit", value="context-a"),
        payload=SearchRequest(query="typed"),
    )
    result = AgentEnvelope(
        pot_id="context-a",
        intent="search",
        items=(
            EvidenceItem(
                include="raw_graph",
                candidate_key="claim-1",
                score=0.9,
                payload={"summary": "typed"},
                coverage_status="complete",
            ),
        ),
        coverage=(CoverageReport(include="raw_graph", status="complete"),),
    )
    response = SuccessResponse(
        protocol_version=PROTOCOL_VERSION,
        request_id=request.request_id,
        outcome=Success(result),
    )

    decoded = decode_response(encode_response(response), request=request)

    assert isinstance(decoded, Success)
    assert decoded.value == response
    assert isinstance(decoded.value.outcome.value, AgentEnvelope)


def test_handshake_success_response_round_trips_to_typed_result() -> None:
    request = HandshakeRequest(
        protocol_version=PROTOCOL_VERSION,
        request_id="handshake-1",
        payload=HandshakePayload(expected_instance_id="instance-1"),
    )
    result = HandshakeResult(
        protocol_min=1,
        protocol_max=1,
        instance_id="instance-1",
        lifecycle_state="ready",
        capabilities=operation_capabilities(),
        operation_catalog_fingerprint=operation_catalog_fingerprint(),
    )
    response = SuccessResponse(
        protocol_version=PROTOCOL_VERSION,
        request_id=request.request_id,
        outcome=Success(result),
    )

    decoded = decode_response(encode_response(response), request=request)

    assert isinstance(decoded, Success)
    assert decoded.value == response


def test_shutdown_success_response_round_trips_to_typed_result() -> None:
    request = ShutdownRequest(
        protocol_version=PROTOCOL_VERSION,
        request_id="shutdown-1",
        payload=ShutdownPayload(),
    )
    response = SuccessResponse(
        protocol_version=PROTOCOL_VERSION,
        request_id=request.request_id,
        outcome=Success(ShutdownResult(accepted=True)),
    )

    decoded = decode_response(encode_response(response), request=request)

    assert isinstance(decoded, Success)
    assert decoded.value == response


def test_daemon_status_success_response_round_trips_to_typed_result() -> None:
    request = DaemonStatusRequest(
        protocol_version=PROTOCOL_VERSION,
        request_id="status-1",
        payload=DaemonStatusPayload(),
    )
    response = SuccessResponse(
        protocol_version=PROTOCOL_VERSION,
        request_id=request.request_id,
        outcome=Success(
            DaemonStatusResult(
                instance_id="instance-1",
                pid=42,
                lifecycle_state="ready",
                backend_profile="embedded",
                ui_url="http://127.0.0.1:8765",
            )
        ),
    )

    decoded = decode_response(encode_response(response), request=request)

    assert isinstance(decoded, Success)
    assert decoded.value == response


@pytest.mark.parametrize(
    "error",
    [
        AuthenticationError(code="bad_token", message="bad token"),
        DomainError(code="bad_query", message="bad query"),
        ProtocolError(code="bad_envelope", message="bad envelope"),
        DaemonInternalError(code="defect", message="internal failure"),
    ],
)
def test_failure_response_preserves_structural_error_category(error: object) -> None:
    request = EngineOperationRequest(
        protocol_version=PROTOCOL_VERSION,
        request_id="search-1",
        operation=EngineOperation.SEARCH,
        selector=ContextSelector(kind="explicit", value="context-a"),
        payload=ENGINE_OPERATION_CATALOG[EngineOperation.SEARCH].request_type(),
    )
    response = FailureResponse(
        protocol_version=PROTOCOL_VERSION,
        request_id=request.request_id,
        outcome=Failure(error),  # type: ignore[arg-type]
    )

    decoded = decode_response(encode_response(response), request=request)

    assert isinstance(decoded, Success)
    assert decoded.value == response


@pytest.mark.parametrize(
    ("document", "code"),
    [
        ({}, "request_envelope_malformed"),
        (
            {
                "protocol_version": 1,
                "request_id": "request-1",
                "operation": "not.registered",
                "payload": {},
            },
            "operation_unknown",
        ),
        (
            {
                "protocol_version": 1,
                "request_id": "request-1",
                "operation": "search",
                "selector": {"kind": "explicit", "value": "context-a"},
                "payload": {},
                "unexpected": True,
            },
            "request_envelope_malformed",
        ),
    ],
)
def test_malformed_or_unknown_request_is_typed_protocol_failure(
    document: dict[str, object], code: str
) -> None:
    outcome = decode_request(document)

    assert isinstance(outcome, Failure)
    assert outcome.error.code == code


def test_response_requires_exactly_one_outcome_shape() -> None:
    request = HandshakeRequest(
        protocol_version=1,
        request_id="handshake-1",
        payload=HandshakePayload(),
    )
    malformed = {
        "protocol_version": 1,
        "request_id": "handshake-1",
        "outcome": {"ok": True, "value": {}, "error": {}},
    }

    outcome = decode_response(malformed, request=request)

    assert isinstance(outcome, Failure)
    assert outcome.error.code == "response_envelope_malformed"
