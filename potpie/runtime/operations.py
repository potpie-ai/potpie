"""Finite typed operation catalog shared by local and daemon execution."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Mapping

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
    OperationSpec(EngineOperation.RESOLVE, ResolveRequest, _READ),
    OperationSpec(EngineOperation.SEARCH, SearchRequest, _READ),
    OperationSpec(EngineOperation.RECORD, RecordRequest, _WRITE),
    OperationSpec(EngineOperation.DATA_PLANE_STATUS, DataPlaneStatusRequest, _READ),
    OperationSpec(EngineOperation.CATALOG, CatalogRequest, _READ),
    OperationSpec(EngineOperation.DESCRIBE, DescribeRequest, _READ),
    OperationSpec(EngineOperation.READ, ReadRequest, _READ),
    OperationSpec(EngineOperation.SEARCH_ENTITIES, SearchEntitiesRequest, _READ),
    OperationSpec(EngineOperation.MUTATE, MutateRequest, _WRITE),
    OperationSpec(EngineOperation.NEIGHBORHOOD, NeighborhoodRequest, _READ),
    OperationSpec(EngineOperation.INSPECT, InspectRequest, _READ),
    OperationSpec(EngineOperation.EXPORT_SNAPSHOT, ExportSnapshotRequest, _READ),
    OperationSpec(
        EngineOperation.IMPORT_SNAPSHOT,
        ImportSnapshotRequest,
        _WRITE,
        destructive=True,
    ),
    OperationSpec(
        EngineOperation.REPAIR,
        RepairRequest,
        _WRITE,
        destructive=True,
    ),
    OperationSpec(EngineOperation.PROPOSE, ProposeRequest, _WRITE),
    OperationSpec(EngineOperation.COMMIT, CommitRequest, _WRITE),
    OperationSpec(EngineOperation.HISTORY, HistoryRequest, _READ),
    OperationSpec(EngineOperation.QUALITY, QualityRequest, _READ),
    OperationSpec(EngineOperation.INBOX_ADD, InboxAddRequest, _WRITE),
    OperationSpec(EngineOperation.INBOX_LIST, InboxListRequest, _READ),
    OperationSpec(EngineOperation.INBOX_SHOW, InboxShowRequest, _READ),
    OperationSpec(EngineOperation.INBOX_CLAIM, InboxClaimRequest, _WRITE),
    OperationSpec(EngineOperation.INBOX_MARK_APPLIED, InboxMarkAppliedRequest, _WRITE),
    OperationSpec(
        EngineOperation.INBOX_MARK_REJECTED, InboxMarkRejectedRequest, _WRITE
    ),
    OperationSpec(EngineOperation.INBOX_CLOSE, InboxCloseRequest, _WRITE),
    OperationSpec(EngineOperation.SUBMIT_EVENT, SubmitEventRequest, _WRITE),
    OperationSpec(EngineOperation.SUBMIT_ARTIFACT, SubmitArtifactRequest, _WRITE),
    OperationSpec(EngineOperation.PROCESSING_STATUS, ProcessingStatusRequest, _READ),
    OperationSpec(EngineOperation.NUDGE, NudgeRequest, _WRITE),
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
