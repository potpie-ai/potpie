"""Finite typed operation catalog shared by local and daemon execution."""

from __future__ import annotations

import hashlib
import json
from dataclasses import MISSING, dataclass, fields, is_dataclass
from enum import Enum, StrEnum
from types import MappingProxyType
from typing import Any, Mapping, get_args, get_origin, get_type_hints

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
from potpie_context_engine.results import (
    CatalogResult,
    CommitResult,
    DataPlaneStatusResult,
    DescribeResult,
    ExportSnapshotResult,
    HistoryResult,
    ImportSnapshotResult,
    InboxAddResult,
    InboxClaimResult,
    InboxCloseResult,
    InboxListResult,
    InboxMarkAppliedResult,
    InboxMarkRejectedResult,
    InboxShowResult,
    InspectResult,
    MutateResult,
    NeighborhoodResult,
    NudgeResult,
    ProcessingStatusResult,
    ProposeResult,
    QualityResult,
    ReadResult,
    RecordResult,
    RepairResult,
    ResolveResult,
    SearchEntitiesResult,
    SearchResult,
    SubmitArtifactResult,
    SubmitEventResult,
)


class EngineOperation(StrEnum):
    RESOLVE = "resolve"
    SEARCH = "search"
    RECORD = "record"
    DATA_PLANE_STATUS = "data_plane_status"
    CATALOG = "catalog"
    DESCRIBE = "describe"
    READ = "read"
    SEARCH_ENTITIES = "search_entities"
    MUTATE = "mutate"
    NEIGHBORHOOD = "neighborhood"
    INSPECT = "inspect"
    EXPORT_SNAPSHOT = "export_snapshot"
    IMPORT_SNAPSHOT = "import_snapshot"
    REPAIR = "repair"
    PROPOSE = "propose"
    COMMIT = "commit"
    HISTORY = "history"
    QUALITY = "quality"
    INBOX_ADD = "inbox_add"
    INBOX_LIST = "inbox_list"
    INBOX_SHOW = "inbox_show"
    INBOX_CLAIM = "inbox_claim"
    INBOX_MARK_APPLIED = "inbox_mark_applied"
    INBOX_MARK_REJECTED = "inbox_mark_rejected"
    INBOX_CLOSE = "inbox_close"
    SUBMIT_EVENT = "submit_event"
    SUBMIT_ARTIFACT = "submit_artifact"
    PROCESSING_STATUS = "processing_status"
    NUDGE = "nudge"


class DaemonControlOperation(StrEnum):
    HANDSHAKE = "daemon.handshake"
    STATUS = "daemon.status"
    SHUTDOWN = "daemon.shutdown"


class SafetyClass(StrEnum):
    SHARED_CONTEXT_READ = "shared_context_read"
    EXCLUSIVE_CONTEXT_MUTATION = "exclusive_context_mutation"
    EXCLUSIVE_RESOURCE_MUTATION = "exclusive_resource_mutation"
    DAEMON_LIFECYCLE_CONTROL = "daemon_lifecycle_control"


@dataclass(frozen=True, slots=True)
class OperationSpec:
    operation: EngineOperation
    request_type: type[EngineRequest]
    result_type: type[object]
    safety: SafetyClass
    destructive: bool = False
    resource_type: str | None = None
    resource_identity_fields: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.safety is SafetyClass.EXCLUSIVE_RESOURCE_MUTATION:
            if self.resource_type is None or not self.resource_identity_fields:
                raise ValueError(
                    "exclusive resource mutations require resource conflict metadata"
                )
        elif self.resource_type is not None or self.resource_identity_fields:
            raise ValueError(
                "resource conflict metadata is valid only for resource mutations"
            )


_READ = SafetyClass.SHARED_CONTEXT_READ
_WRITE = SafetyClass.EXCLUSIVE_CONTEXT_MUTATION

_SPECS = (
    OperationSpec(EngineOperation.RESOLVE, ResolveRequest, ResolveResult, _READ),
    OperationSpec(EngineOperation.SEARCH, SearchRequest, SearchResult, _READ),
    OperationSpec(EngineOperation.RECORD, RecordRequest, RecordResult, _WRITE),
    OperationSpec(
        EngineOperation.DATA_PLANE_STATUS,
        DataPlaneStatusRequest,
        DataPlaneStatusResult,
        _READ,
    ),
    OperationSpec(EngineOperation.CATALOG, CatalogRequest, CatalogResult, _READ),
    OperationSpec(EngineOperation.DESCRIBE, DescribeRequest, DescribeResult, _READ),
    OperationSpec(EngineOperation.READ, ReadRequest, ReadResult, _READ),
    OperationSpec(
        EngineOperation.SEARCH_ENTITIES,
        SearchEntitiesRequest,
        SearchEntitiesResult,
        _READ,
    ),
    OperationSpec(EngineOperation.MUTATE, MutateRequest, MutateResult, _WRITE),
    OperationSpec(
        EngineOperation.NEIGHBORHOOD,
        NeighborhoodRequest,
        NeighborhoodResult,
        _READ,
    ),
    OperationSpec(EngineOperation.INSPECT, InspectRequest, InspectResult, _READ),
    OperationSpec(
        EngineOperation.EXPORT_SNAPSHOT,
        ExportSnapshotRequest,
        ExportSnapshotResult,
        _READ,
    ),
    OperationSpec(
        EngineOperation.IMPORT_SNAPSHOT,
        ImportSnapshotRequest,
        ImportSnapshotResult,
        _WRITE,
        destructive=True,
    ),
    OperationSpec(
        EngineOperation.REPAIR,
        RepairRequest,
        RepairResult,
        _WRITE,
        destructive=True,
    ),
    OperationSpec(EngineOperation.PROPOSE, ProposeRequest, ProposeResult, _WRITE),
    OperationSpec(EngineOperation.COMMIT, CommitRequest, CommitResult, _WRITE),
    OperationSpec(EngineOperation.HISTORY, HistoryRequest, HistoryResult, _READ),
    OperationSpec(EngineOperation.QUALITY, QualityRequest, QualityResult, _READ),
    OperationSpec(EngineOperation.INBOX_ADD, InboxAddRequest, InboxAddResult, _WRITE),
    OperationSpec(EngineOperation.INBOX_LIST, InboxListRequest, InboxListResult, _READ),
    OperationSpec(EngineOperation.INBOX_SHOW, InboxShowRequest, InboxShowResult, _READ),
    OperationSpec(
        EngineOperation.INBOX_CLAIM, InboxClaimRequest, InboxClaimResult, _WRITE
    ),
    OperationSpec(
        EngineOperation.INBOX_MARK_APPLIED,
        InboxMarkAppliedRequest,
        InboxMarkAppliedResult,
        _WRITE,
    ),
    OperationSpec(
        EngineOperation.INBOX_MARK_REJECTED,
        InboxMarkRejectedRequest,
        InboxMarkRejectedResult,
        _WRITE,
    ),
    OperationSpec(
        EngineOperation.INBOX_CLOSE, InboxCloseRequest, InboxCloseResult, _WRITE
    ),
    OperationSpec(
        EngineOperation.SUBMIT_EVENT,
        SubmitEventRequest,
        SubmitEventResult,
        _WRITE,
    ),
    OperationSpec(
        EngineOperation.SUBMIT_ARTIFACT,
        SubmitArtifactRequest,
        SubmitArtifactResult,
        _WRITE,
    ),
    OperationSpec(
        EngineOperation.PROCESSING_STATUS,
        ProcessingStatusRequest,
        ProcessingStatusResult,
        _READ,
    ),
    OperationSpec(EngineOperation.NUDGE, NudgeRequest, NudgeResult, _WRITE),
)

if len({spec.operation for spec in _SPECS}) != len(_SPECS):
    raise RuntimeError("each engine operation must have exactly one catalog entry")
if {spec.operation for spec in _SPECS} != set(EngineOperation):
    raise RuntimeError("the operation catalog must cover every engine operation")

ENGINE_OPERATION_CATALOG: Mapping[EngineOperation, OperationSpec] = MappingProxyType(
    {spec.operation: spec for spec in _SPECS}
)

DAEMON_CONTROL_SAFETY: Mapping[DaemonControlOperation, SafetyClass] = MappingProxyType(
    {
        DaemonControlOperation.HANDSHAKE: SafetyClass.DAEMON_LIFECYCLE_CONTROL,
        DaemonControlOperation.STATUS: SafetyClass.DAEMON_LIFECYCLE_CONTROL,
        DaemonControlOperation.SHUTDOWN: SafetyClass.DAEMON_LIFECYCLE_CONTROL,
    }
)


def operation_catalog_fingerprint() -> str:
    """Return a stable digest of protocol-visible operation semantics."""

    records = [
        {
            "kind": "engine",
            "operation": spec.operation.value,
            "request_type": spec.request_type.__name__,
            "result_type": spec.result_type.__name__,
            "request_schema": _type_schema(spec.request_type),
            "result_schema": _type_schema(spec.result_type),
            "safety": spec.safety.value,
            "destructive": spec.destructive,
            "resource_type": spec.resource_type,
            "resource_identity_fields": spec.resource_identity_fields,
        }
        for spec in sorted(_SPECS, key=lambda item: item.operation.value)
    ]
    records.extend(
        {
            "kind": "daemon_control",
            "operation": operation.value,
            "request_type": None,
            "result_type": None,
            "safety": safety.value,
            "destructive": False,
            "resource_type": None,
            "resource_identity_fields": (),
        }
        for operation, safety in sorted(
            DAEMON_CONTROL_SAFETY.items(), key=lambda item: item[0].value
        )
    )
    encoded = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _type_schema(annotation: object, *, seen: frozenset[str] = frozenset()) -> object:
    """Describe protocol-visible Python types without importing transport codecs."""

    origin = get_origin(annotation)
    if origin is not None:
        return {
            "origin": _qualified_name(origin),
            "arguments": tuple(
                _type_schema(argument, seen=seen) for argument in get_args(annotation)
            ),
        }
    if annotation is Any:
        return {"type": "typing.Any"}
    if isinstance(annotation, str):
        return {"forward_reference": annotation}
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return {
            "type": _qualified_name(annotation),
            "enum_members": tuple(
                (member.name, _stable_value(member.value)) for member in annotation
            ),
        }
    if isinstance(annotation, type) and is_dataclass(annotation):
        type_name = _qualified_name(annotation)
        if type_name in seen:
            return {"type": type_name, "recursive": True}
        try:
            hints = get_type_hints(annotation, include_extras=True)
        except (NameError, TypeError):
            hints = {field.name: field.type for field in fields(annotation)}
        field_schemas = []
        for field in fields(annotation):
            record: dict[str, object] = {
                "name": field.name,
                "type": _type_schema(
                    hints.get(field.name, field.type), seen=seen | {type_name}
                ),
            }
            if field.default is not MISSING:
                record["default"] = _stable_value(field.default)
            elif field.default_factory is not MISSING:
                record["default_factory"] = _qualified_name(field.default_factory)
            else:
                record["required"] = True
            field_schemas.append(record)
        return {"type": type_name, "fields": tuple(field_schemas)}
    return {"type": _qualified_name(annotation)}


def _qualified_name(value: object) -> str:
    module = getattr(value, "__module__", None)
    qualname = getattr(value, "__qualname__", None)
    if module and qualname:
        return f"{module}.{qualname}"
    return str(value)


def _stable_value(value: object) -> object:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Enum):
        return {"enum": _qualified_name(type(value)), "member": value.name}
    if isinstance(value, Mapping):
        return {
            str(key): _stable_value(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (tuple, list)):
        return tuple(_stable_value(item) for item in value)
    return {"type": _qualified_name(type(value)), "value": str(value)}


def operation_capabilities() -> tuple[str, ...]:
    """Return every finite protocol discriminator in deterministic order."""

    return tuple(
        sorted(
            [operation.value for operation in EngineOperation]
            + [operation.value for operation in DaemonControlOperation]
        )
    )


__all__ = [
    "DAEMON_CONTROL_SAFETY",
    "ENGINE_OPERATION_CATALOG",
    "DaemonControlOperation",
    "EngineOperation",
    "OperationSpec",
    "SafetyClass",
    "operation_capabilities",
    "operation_catalog_fingerprint",
]
