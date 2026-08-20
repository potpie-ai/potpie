"""Explicit JSON codec for the version-one typed daemon protocol."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from typing import TypeAlias, cast

from potpie.runtime.operations import (
    ENGINE_OPERATION_CATALOG,
    DaemonControlOperation,
    EngineOperation,
)
from potpie.runtime.protocol import (
    DaemonInternalError,
    EngineOperationRequest,
    FailureResponse,
    HandshakePayload,
    HandshakeRequest,
    HandshakeResult,
    LifecycleState,
    ProtocolError,
    ProtocolRequest,
    ProtocolResponse,
    ProtocolTransportError,
    RuntimeBoundaryError,
    ShutdownPayload,
    ShutdownRequest,
    ShutdownResult,
    SuccessResponse,
)
from potpie.runtime.resource_manager import (
    AuthenticationError,
    AuthorizationError,
    ContextSelector,
    DestructiveIntent,
    ResourceLifecycleError,
    SelectionError,
)
from potpie_context_engine import (
    DependencyError,
    DomainError,
    EngineLifecycleError,
    Failure,
    Success,
)
from potpie_context_engine.outcomes import RetryPosture


DecodeRequestOutcome: TypeAlias = Success[ProtocolRequest] | Failure[ProtocolError]
DecodeResponseOutcome: TypeAlias = Success[ProtocolResponse] | Failure[ProtocolError]

_RETRY_POSTURES = frozenset({"safe", "unsafe", "unknown", "not_applicable"})
_LIFECYCLE_STATES = frozenset(
    {"starting", "ready", "draining", "failed", "stopped"}
)


def encode_request(request: ProtocolRequest) -> dict[str, object]:
    common: dict[str, object] = {
        "protocol_version": request.protocol_version,
        "request_id": request.request_id,
        "operation": request.operation.value,
    }
    if isinstance(request, EngineOperationRequest):
        common.update(
            {
                "selector": {
                    "kind": request.selector.kind,
                    "value": request.selector.value,
                },
                "payload": _to_wire(request.payload.payload),
                "destructive_intent": (
                    _to_wire(request.destructive_intent)
                    if request.destructive_intent is not None
                    else None
                ),
            }
        )
        return common
    if isinstance(request, HandshakeRequest):
        common["payload"] = _to_wire(request.payload)
        return common
    if isinstance(request, ShutdownRequest):
        common["payload"] = _to_wire(request.payload)
        return common
    raise TypeError("unsupported protocol request")


def decode_request(document: object) -> DecodeRequestOutcome:
    if not isinstance(document, Mapping):
        return _protocol_failure("request_envelope_malformed", "request must be an object")
    common = _decode_common_envelope(document)
    if isinstance(common, Failure):
        return common
    protocol_version, request_id, operation_text = common.value

    try:
        engine_operation = EngineOperation(operation_text)
    except ValueError:
        engine_operation = None
    if engine_operation is not None:
        unexpected = _unexpected_keys(
            document,
            required={
                "protocol_version",
                "request_id",
                "operation",
                "selector",
                "payload",
            },
            optional={"destructive_intent"},
        )
        if unexpected is not None:
            return unexpected
        selector = _decode_selector(document["selector"])
        if isinstance(selector, Failure):
            return selector
        payload = document["payload"]
        if not isinstance(payload, Mapping):
            return _protocol_failure(
                "operation_payload_malformed", "operation payload must be an object"
            )
        intent = _decode_destructive_intent(document.get("destructive_intent"))
        if isinstance(intent, Failure):
            return intent
        request_type = ENGINE_OPERATION_CATALOG[engine_operation].request_type
        try:
            return Success(
                EngineOperationRequest(
                    protocol_version=protocol_version,
                    request_id=request_id,
                    operation=engine_operation,
                    selector=selector.value,
                    payload=request_type(payload=dict(payload)),
                    destructive_intent=intent.value,
                )
            )
        except (TypeError, ValueError):
            return _protocol_failure(
                "operation_payload_malformed",
                "operation payload does not match the typed request",
            )

    try:
        control_operation = DaemonControlOperation(operation_text)
    except ValueError:
        return _protocol_failure(
            "operation_unknown",
            "the requested operation is not registered",
            details={"operation": operation_text},
        )
    unexpected = _unexpected_keys(
        document,
        required={"protocol_version", "request_id", "operation", "payload"},
    )
    if unexpected is not None:
        return unexpected
    payload = document["payload"]
    if not isinstance(payload, Mapping):
        return _protocol_failure(
            "operation_payload_malformed", "control payload must be an object"
        )
    if control_operation is DaemonControlOperation.HANDSHAKE:
        if set(payload) != {
            "client_protocol_min",
            "client_protocol_max",
            "expected_instance_id",
        }:
            return _protocol_failure(
                "operation_payload_malformed", "handshake payload fields are invalid"
            )
        try:
            return Success(
                HandshakeRequest(
                    protocol_version=protocol_version,
                    request_id=request_id,
                    payload=HandshakePayload(
                        client_protocol_min=_required_int(
                            payload, "client_protocol_min"
                        ),
                        client_protocol_max=_required_int(
                            payload, "client_protocol_max"
                        ),
                        expected_instance_id=_optional_string(
                            payload, "expected_instance_id"
                        ),
                    ),
                )
            )
        except (TypeError, ValueError):
            return _protocol_failure(
                "operation_payload_malformed", "handshake payload is invalid"
            )
    if set(payload) != {"reason"} or not isinstance(payload.get("reason"), str):
        return _protocol_failure(
            "operation_payload_malformed", "shutdown payload is invalid"
        )
    return Success(
        ShutdownRequest(
            protocol_version=protocol_version,
            request_id=request_id,
            payload=ShutdownPayload(reason=cast(str, payload["reason"])),
        )
    )


def encode_response(response: ProtocolResponse) -> dict[str, object]:
    document: dict[str, object] = {
        "protocol_version": response.protocol_version,
        "request_id": response.request_id,
    }
    if isinstance(response, SuccessResponse):
        document["outcome"] = {
            "ok": True,
            "value": _to_wire(response.outcome.value),
        }
    else:
        document["outcome"] = {
            "ok": False,
            "error": _to_wire(response.outcome.error),
        }
    return document


def decode_response(
    document: object, *, request: ProtocolRequest
) -> DecodeResponseOutcome:
    if not isinstance(document, Mapping):
        return _protocol_failure(
            "response_envelope_malformed", "response must be an object"
        )
    if set(document) != {"protocol_version", "request_id", "outcome"}:
        return _protocol_failure(
            "response_envelope_malformed", "response envelope fields are invalid"
        )
    try:
        protocol_version = _required_int(document, "protocol_version")
        request_id = _required_string(document, "request_id")
    except (TypeError, ValueError):
        return _protocol_failure(
            "response_envelope_malformed", "response identity is invalid"
        )
    outcome = document["outcome"]
    if not isinstance(outcome, Mapping) or not isinstance(outcome.get("ok"), bool):
        return _protocol_failure(
            "response_envelope_malformed", "response outcome is invalid"
        )
    if outcome["ok"] is True:
        if set(outcome) != {"ok", "value"}:
            return _protocol_failure(
                "response_envelope_malformed", "success outcome fields are invalid"
            )
        result = _decode_result(outcome["value"], request)
        if isinstance(result, Failure):
            return result
        return Success(
            SuccessResponse(
                protocol_version=protocol_version,
                request_id=request_id,
                outcome=Success(result.value),
            )
        )
    if set(outcome) != {"ok", "error"}:
        return _protocol_failure(
            "response_envelope_malformed", "failure outcome fields are invalid"
        )
    error = _decode_error(outcome["error"])
    if isinstance(error, Failure):
        return error
    return Success(
        FailureResponse(
            protocol_version=protocol_version,
            request_id=request_id,
            outcome=Failure(error.value),
        )
    )


def _decode_result(
    value: object, request: ProtocolRequest
) -> Success[object] | Failure[ProtocolError]:
    if isinstance(request, HandshakeRequest):
        if not isinstance(value, Mapping) or set(value) != {
            "protocol_min",
            "protocol_max",
            "instance_id",
            "lifecycle_state",
            "capabilities",
            "operation_catalog_fingerprint",
        }:
            return _protocol_failure(
                "response_result_malformed", "handshake result is invalid"
            )
        try:
            lifecycle_state = _required_string(value, "lifecycle_state")
            if lifecycle_state not in _LIFECYCLE_STATES:
                raise ValueError("invalid lifecycle state")
            capabilities = value["capabilities"]
            if not isinstance(capabilities, list) or not all(
                isinstance(item, str) for item in capabilities
            ):
                raise TypeError("invalid capabilities")
            return Success(
                HandshakeResult(
                    protocol_min=_required_int(value, "protocol_min"),
                    protocol_max=_required_int(value, "protocol_max"),
                    instance_id=_required_string(value, "instance_id"),
                    lifecycle_state=cast(LifecycleState, lifecycle_state),
                    capabilities=tuple(capabilities),
                    operation_catalog_fingerprint=_required_string(
                        value, "operation_catalog_fingerprint"
                    ),
                )
            )
        except (TypeError, ValueError):
            return _protocol_failure(
                "response_result_malformed", "handshake result is invalid"
            )
    if isinstance(request, ShutdownRequest):
        if (
            not isinstance(value, Mapping)
            or set(value) != {"accepted"}
            or not isinstance(value["accepted"], bool)
        ):
            return _protocol_failure(
                "response_result_malformed", "shutdown result is invalid"
            )
        return Success(ShutdownResult(accepted=value["accepted"]))
    return Success(value)


def _decode_error(
    value: object,
) -> Success[RuntimeBoundaryError] | Failure[ProtocolError]:
    if not isinstance(value, Mapping) or set(value) != {
        "category",
        "code",
        "message",
        "details",
        "recommended_next_action",
        "retry_posture",
    }:
        return _protocol_failure(
            "response_error_malformed", "response error fields are invalid"
        )
    category = value.get("category")
    error_types = {
        "selection": SelectionError,
        "authentication": AuthenticationError,
        "authorization": AuthorizationError,
        "resource_lifecycle": ResourceLifecycleError,
        "domain": DomainError,
        "dependency": DependencyError,
        "engine_lifecycle": EngineLifecycleError,
        "protocol": ProtocolError,
        "protocol_transport": ProtocolTransportError,
        "daemon_internal": DaemonInternalError,
    }
    error_type = error_types.get(category)
    if error_type is None:
        return _protocol_failure(
            "response_error_malformed", "response error category is invalid"
        )
    try:
        details = value["details"]
        if not isinstance(details, Mapping):
            raise TypeError("details must be an object")
        retry_posture = _required_string(value, "retry_posture")
        if retry_posture not in _RETRY_POSTURES:
            raise ValueError("invalid retry posture")
        return Success(
            error_type(
                code=_required_string(value, "code"),
                message=_required_string(value, "message"),
                details=dict(details),
                recommended_next_action=_optional_string(
                    value, "recommended_next_action"
                ),
                retry_posture=cast(RetryPosture, retry_posture),
            )
        )
    except (TypeError, ValueError):
        return _protocol_failure(
            "response_error_malformed", "response error payload is invalid"
        )


def _decode_common_envelope(
    document: Mapping[object, object],
) -> Success[tuple[int, str, str]] | Failure[ProtocolError]:
    try:
        return Success(
            (
                _required_int(document, "protocol_version"),
                _required_string(document, "request_id"),
                _required_string(document, "operation"),
            )
        )
    except (KeyError, TypeError, ValueError):
        return _protocol_failure(
            "request_envelope_malformed", "request identity is invalid"
        )


def _decode_selector(
    value: object,
) -> Success[ContextSelector] | Failure[ProtocolError]:
    if not isinstance(value, Mapping) or set(value) != {"kind", "value"}:
        return _protocol_failure(
            "context_selector_malformed", "context selector fields are invalid"
        )
    try:
        kind = _required_string(value, "kind")
        if kind not in {"explicit", "active", "repository"}:
            raise ValueError("unknown selector kind")
        return Success(
            ContextSelector(
                kind=cast("object", kind),
                value=_optional_string(value, "value"),
            )
        )
    except (TypeError, ValueError):
        return _protocol_failure(
            "context_selector_malformed", "context selector is invalid"
        )


def _decode_destructive_intent(
    value: object,
) -> Success[DestructiveIntent | None] | Failure[ProtocolError]:
    if value is None:
        return Success(None)
    if not isinstance(value, Mapping) or set(value) != {
        "confirmed",
        "operation",
        "selector",
        "request_id",
    }:
        return _protocol_failure(
            "destructive_intent_malformed", "destructive intent fields are invalid"
        )
    selector = _decode_selector(value["selector"])
    if isinstance(selector, Failure):
        return selector
    if not isinstance(value["confirmed"], bool):
        return _protocol_failure(
            "destructive_intent_malformed", "destructive confirmation is invalid"
        )
    try:
        return Success(
            DestructiveIntent(
                confirmed=value["confirmed"],
                operation=_required_string(value, "operation"),
                selector=selector.value,
                request_id=_required_string(value, "request_id"),
            )
        )
    except (TypeError, ValueError):
        return _protocol_failure(
            "destructive_intent_malformed", "destructive intent is invalid"
        )


def _unexpected_keys(
    document: Mapping[object, object],
    *,
    required: set[str],
    optional: set[str] | None = None,
) -> Failure[ProtocolError] | None:
    optional = optional or set()
    keys = set(document)
    if not required.issubset(keys) or not keys.issubset(required | optional):
        return _protocol_failure(
            "request_envelope_malformed", "request envelope fields are invalid"
        )
    return None


def _required_int(document: Mapping[object, object], key: str) -> int:
    value = document[key]
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{key} must be an integer")
    return value


def _required_string(document: Mapping[object, object], key: str) -> str:
    value = document[key]
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{key} must be a non-empty string")
    return value


def _optional_string(document: Mapping[object, object], key: str) -> str | None:
    value = document[key]
    if value is not None and not isinstance(value, str):
        raise TypeError(f"{key} must be a string or null")
    return value


def _protocol_failure(
    code: str,
    message: str,
    *,
    details: Mapping[str, object] | None = None,
) -> Failure[ProtocolError]:
    return Failure(ProtocolError(code=code, message=message, details=details or {}))


def _to_wire(value: object) -> object:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("protocol mappings require string keys")
        return {str(key): _to_wire(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_wire(item) for item in value]
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _to_wire(getattr(value, item.name))
            for item in fields(value)
        }
    raise TypeError(f"unsupported protocol value: {type(value).__name__}")


__all__ = [
    "DecodeRequestOutcome",
    "DecodeResponseOutcome",
    "decode_request",
    "decode_response",
    "encode_request",
    "encode_response",
]
