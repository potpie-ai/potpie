"""Explicit JSON codec for the typed operation protocol.

Evolution rules the codec carries for every transport (local daemon and
managed hosts alike):

* Requests are compact: every field still at its default is left off the wire,
  recursively, so a call that does not use a newer field sends exactly what an
  older client would.
* A host refuses a request field it does not declare with
  ``request_fields_unsupported`` naming each field by its dotted path, and an
  operation it does not serve with ``operation_unknown``.
* Results decode leniently: a field this build does not know is ignored and a
  missing one takes its default. Requests stay strict.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping
from dataclasses import MISSING, Field, fields, is_dataclass
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import TypeAlias, cast
from uuid import UUID

from potpie.runtime.extensions import HostExtension
from potpie.runtime.host_operations import HOST_OPERATION_CATALOG, HostOperation
from potpie.runtime.operations import (
    ENGINE_OPERATION_CATALOG,
    DaemonControlOperation,
    EngineOperation,
)
from potpie.runtime.protocol import (
    DaemonBuild,
    DaemonInternalError,
    DaemonStatusPayload,
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
from potpie_context_engine.requests import request_from_payload
from potpie_context_engine.results import result_from_payload
from potpie_context_engine.typed_serialization import unknown_field_paths


DecodeRequestOutcome: TypeAlias = Success[ProtocolRequest] | Failure[ProtocolError]
DecodeResponseOutcome: TypeAlias = Success[ProtocolResponse] | Failure[ProtocolError]

_RETRY_POSTURES = frozenset({"safe", "unsafe", "unknown", "not_applicable"})
_LIFECYCLE_STATES = frozenset({"starting", "ready", "draining", "failed", "stopped"})
_DAEMON_STATUS_FIELDS = frozenset(
    {"instance_id", "pid", "lifecycle_state", "backend_profile", "ui_url"}
)
_DAEMON_STATUS_BUILD_FIELDS = frozenset({"version", "build"})
_DAEMON_BUILD_FIELDS = frozenset({"rev", "dirty", "built_at"})

#: Every error category a decoder accepts. Closed on purpose: a category an
#: older decoder does not know turns a typed refusal into a malformed
#: response, so new failure kinds are new codes inside these categories.
ERROR_CATEGORIES: frozenset[str] = frozenset(
    {
        "selection",
        "authentication",
        "authorization",
        "resource_lifecycle",
        "domain",
        "dependency",
        "engine_lifecycle",
        "protocol",
        "protocol_transport",
        "daemon_internal",
    }
)

#: The refusal for request fields a host does not declare. Its details name
#: the operation and every unknown field by dotted path.
REQUEST_FIELDS_UNSUPPORTED = "request_fields_unsupported"


def encode_request(request: ProtocolRequest) -> dict[str, object]:
    common: dict[str, object] = {
        "protocol_version": request.protocol_version,
        "request_id": request.request_id,
        "operation": str(request.operation),
    }
    if isinstance(request, (EngineOperationRequest, HostOperationRequest)):
        common.update(
            {
                "selector": {
                    "kind": request.selector.kind,
                    "value": request.selector.value,
                },
                "payload": compact_payload(request.payload),
                "destructive_intent": (
                    _to_wire(request.destructive_intent)
                    if request.destructive_intent is not None
                    else None
                ),
                "compatibility_ticket": request.compatibility_ticket,
            }
        )
        return common
    if isinstance(request, HandshakeRequest):
        common["payload"] = _to_wire(request.payload)
        return common
    if isinstance(request, DaemonStatusRequest):
        common["payload"] = {}
        common["compatibility_ticket"] = request.compatibility_ticket
        return common
    if isinstance(request, ShutdownRequest):
        common["payload"] = _to_wire(request.payload)
        common["compatibility_ticket"] = request.compatibility_ticket
        return common
    raise TypeError("unsupported protocol request")


def decode_request(
    document: object,
    *,
    engine_operations: Collection[EngineOperation] | None = None,
    host_operations: bool = False,
    extensions: Mapping[str, HostExtension] | None = None,
) -> DecodeRequestOutcome:
    """Decode one request envelope into its typed protocol request.

    By default only the engine and daemon-control catalogs are known, which
    is what the local daemon serves. A managed host names what it serves:
    ``engine_operations`` (a subset of the engine catalog; ``None`` is all of
    it), ``host_operations`` (the ``host.*`` catalog) and ``extensions`` (its
    ``ext.*`` operations by name). Anything else is ``operation_unknown``.
    """

    if not isinstance(document, Mapping):
        return _protocol_failure(
            "request_envelope_malformed", "request must be an object"
        )
    common = _decode_common_envelope(document)
    if isinstance(common, Failure):
        return common
    protocol_version, request_id, operation_text = common.value

    try:
        engine_operation: EngineOperation | None = EngineOperation(operation_text)
    except ValueError:
        engine_operation = None
    if engine_operation is not None and (
        engine_operations is None or engine_operation in engine_operations
    ):
        return _decode_operation_envelope(
            document,
            protocol_version=protocol_version,
            request_id=request_id,
            operation=engine_operation,
            request_type=ENGINE_OPERATION_CATALOG[engine_operation].request_type,
            build=EngineOperationRequest,
        )

    try:
        host_operation: HostOperation | None = HostOperation(operation_text)
    except ValueError:
        host_operation = None
    if host_operation is not None and host_operations:
        return _decode_operation_envelope(
            document,
            protocol_version=protocol_version,
            request_id=request_id,
            operation=host_operation,
            request_type=HOST_OPERATION_CATALOG[host_operation].request_type,
            build=HostOperationRequest,
        )

    extension = (extensions or {}).get(operation_text)
    if extension is not None:
        return _decode_operation_envelope(
            document,
            protocol_version=protocol_version,
            request_id=request_id,
            operation=extension.name,
            request_type=extension.request_type,
            build=HostOperationRequest,
        )
    if engine_operation is not None or host_operation is not None:
        return _operation_unknown(operation_text)

    try:
        control_operation = DaemonControlOperation(operation_text)
    except ValueError:
        return _operation_unknown(operation_text)
    required = {"protocol_version", "request_id", "operation", "payload"}
    if control_operation is not DaemonControlOperation.HANDSHAKE:
        required.add("compatibility_ticket")
    unexpected = _unexpected_keys(document, required=required)
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
            "client_operation_catalog_fingerprint",
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
                        client_operation_catalog_fingerprint=_required_string(
                            payload, "client_operation_catalog_fingerprint"
                        ),
                    ),
                )
            )
        except (TypeError, ValueError):
            return _protocol_failure(
                "operation_payload_malformed", "handshake payload is invalid"
            )
    if control_operation is DaemonControlOperation.STATUS:
        if payload:
            return _protocol_failure(
                "operation_payload_malformed", "status payload must be empty"
            )
        return Success(
            DaemonStatusRequest(
                protocol_version=protocol_version,
                request_id=request_id,
                payload=DaemonStatusPayload(),
                compatibility_ticket=_optional_string(document, "compatibility_ticket"),
            )
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
            compatibility_ticket=_optional_string(document, "compatibility_ticket"),
        )
    )


def _operation_unknown(operation: str) -> Failure[ProtocolError]:
    return _protocol_failure(
        "operation_unknown",
        "the requested operation is not registered",
        details={"operation": operation},
    )


def _decode_operation_envelope(
    document: Mapping[object, object],
    *,
    protocol_version: int,
    request_id: str,
    operation: object,
    request_type: type,
    build: Callable[..., ProtocolRequest],
) -> DecodeRequestOutcome:
    """Decode the selector, payload and intent every operation envelope carries."""

    unexpected = _unexpected_keys(
        document,
        required={
            "protocol_version",
            "request_id",
            "operation",
            "selector",
            "payload",
            "compatibility_ticket",
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
    typed_payload = decode_request_payload(str(operation), request_type, payload)
    if isinstance(typed_payload, Failure):
        return typed_payload
    try:
        return Success(
            build(
                protocol_version=protocol_version,
                request_id=request_id,
                operation=operation,
                selector=selector.value,
                payload=typed_payload.value,
                destructive_intent=intent.value,
                compatibility_ticket=_optional_string(document, "compatibility_ticket"),
            )
        )
    except (TypeError, ValueError):
        return _protocol_failure(
            "operation_payload_malformed",
            "operation payload does not match the typed request",
        )


def compact_payload(payload: object) -> dict[str, object]:
    """Wire form of one request payload with default-valued fields left off.

    Recursive through nested dataclasses. A default factory's field is left
    off when its value equals a fresh ``factory()``. When in doubt the field
    is sent: an unsent field the receiver needed is a silent behaviour change,
    an extra one is at worst a precise refusal.
    """

    if not is_dataclass(payload) or isinstance(payload, type):
        raise TypeError("request payloads must be dataclass instances")
    wire = _to_wire(payload, omit_defaults=True)
    assert isinstance(wire, dict)
    return wire


def decode_request_payload(
    operation: str, request_type: type, payload: Mapping[object, object]
) -> Success[object] | Failure[ProtocolError]:
    """Decode one operation payload strictly, naming any field the type lacks."""

    document = {str(key): value for key, value in payload.items()}
    try:
        unknown = unknown_field_paths(request_type, document)
    except (TypeError, ValueError):
        unknown = ()
    if unknown:
        return _protocol_failure(
            REQUEST_FIELDS_UNSUPPORTED,
            "this host does not support request fields: " + ", ".join(unknown),
            details={"operation": operation, "fields": unknown},
            recommended_next_action=(
                "update the host to a Potpie version that supports these fields, "
                "or retry without them"
            ),
        )
    try:
        return Success(request_from_payload(request_type, document))
    except (TypeError, ValueError):
        return _protocol_failure(
            "operation_payload_malformed",
            "operation payload does not match the typed request",
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
            "compatibility_ticket",
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
                    compatibility_ticket=_required_string(
                        value, "compatibility_ticket"
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
    if isinstance(request, DaemonStatusRequest):
        if not isinstance(value, Mapping) or set(value) not in (
            _DAEMON_STATUS_FIELDS,
            _DAEMON_STATUS_FIELDS | _DAEMON_STATUS_BUILD_FIELDS,
        ):
            return _protocol_failure(
                "response_result_malformed", "daemon status result is invalid"
            )
        try:
            lifecycle_state = _required_string(value, "lifecycle_state")
            if lifecycle_state not in _LIFECYCLE_STATES:
                raise ValueError("invalid lifecycle state")
            return Success(
                DaemonStatusResult(
                    instance_id=_required_string(value, "instance_id"),
                    pid=_required_int(value, "pid"),
                    lifecycle_state=cast(LifecycleState, lifecycle_state),
                    backend_profile=_required_string(value, "backend_profile"),
                    ui_url=_required_string(value, "ui_url"),
                    version=(
                        _optional_string(value, "version")
                        if "version" in value
                        else None
                    ),
                    build=_decode_daemon_build(value.get("build")),
                )
            )
        except (TypeError, ValueError):
            return _protocol_failure(
                "response_result_malformed", "daemon status result is invalid"
            )
    if isinstance(request, EngineOperationRequest):
        result_type = ENGINE_OPERATION_CATALOG[request.operation].result_type
    elif isinstance(request, HostOperationRequest) and not request.is_extension:
        result_type = HOST_OPERATION_CATALOG[
            HostOperation(request.operation)
        ].result_type
    else:
        # An extension's result type belongs to the caller that registered or
        # called it; it decodes the plain wire value itself.
        return Success(_from_wire(value))
    try:
        return Success(result_from_payload(result_type, value))
    except (TypeError, ValueError, KeyError):
        return _protocol_failure(
            "response_result_malformed",
            "operation result does not match the operation catalog",
            details={"operation": str(request.operation)},
        )


def _decode_daemon_build(value: object) -> DaemonBuild | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != _DAEMON_BUILD_FIELDS:
        raise ValueError("daemon build fields are invalid")
    dirty = value["dirty"]
    if dirty is not None and not isinstance(dirty, bool):
        raise TypeError("dirty must be a boolean or null")
    return DaemonBuild(
        rev=_optional_string(value, "rev"),
        dirty=dirty,
        built_at=_optional_string(value, "built_at"),
    )


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
    error_types: dict[str, type] = {
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
    recommended_next_action: str | None = None,
) -> Failure[ProtocolError]:
    return Failure(
        ProtocolError(
            code=code,
            message=message,
            details=details or {},
            recommended_next_action=recommended_next_action,
        )
    )


def _to_wire(value: object, *, omit_defaults: bool = False) -> object:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if is_dataclass(value) and not isinstance(value, type):
        wire: dict[str, object] = {}
        for item in fields(value):
            field_value = getattr(value, item.name)
            if omit_defaults and _is_default(item, field_value):
                continue
            wire[item.name] = _to_wire(field_value, omit_defaults=omit_defaults)
        return wire
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("protocol mappings require string keys")
        return {
            str(key): _to_wire(item, omit_defaults=omit_defaults)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_to_wire(item, omit_defaults=omit_defaults) for item in value]
    if isinstance(value, (set, frozenset)):
        return [
            _to_wire(item, omit_defaults=omit_defaults)
            for item in sorted(value, key=str)
        ]
    if isinstance(value, Enum):
        return _to_wire(value.value)
    if isinstance(value, (datetime, date, Path, UUID)):
        return str(value)
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _to_wire(model_dump(mode="json"))
    raise TypeError(f"unsupported protocol value: {type(value).__name__}")


def _is_default(item: Field[object], value: object) -> bool:
    """Is ``value`` this field's default? Unsure means no, so the field is sent."""

    if item.default is not MISSING:
        default = item.default
    elif item.default_factory is not MISSING:
        try:
            default = item.default_factory()
        except Exception:  # noqa: BLE001 - an odd factory only costs the field
            return False
    else:
        return False
    try:
        # The type check keeps ``0``/``0.0`` from passing for ``False``.
        return type(value) is type(default) and bool(value == default)
    except Exception:  # noqa: BLE001 - an odd __eq__ only costs sending the field
        return False


class _WireObject(dict[str, object]):
    """JSON object with mapping and attribute access for typed result consumers."""

    def __getattribute__(self, name: str) -> object:
        if not name.startswith("_") and dict.__contains__(self, name):
            return dict.__getitem__(self, name)
        return dict.__getattribute__(self, name)

    def to_dict(self) -> dict[str, object]:
        return {key: _plain_wire(value) for key, value in dict.items(self)}


def _from_wire(value: object) -> object:
    if isinstance(value, Mapping):
        return _WireObject({str(key): _from_wire(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_from_wire(item) for item in value]
    return value


def _plain_wire(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain_wire(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_plain_wire(item) for item in value]
    return value


__all__ = [
    "ERROR_CATEGORIES",
    "REQUEST_FIELDS_UNSUPPORTED",
    "DecodeRequestOutcome",
    "DecodeResponseOutcome",
    "compact_payload",
    "decode_request",
    "decode_request_payload",
    "decode_response",
    "encode_request",
    "encode_response",
]
