"""Finite typed operation catalog shared by local and daemon execution."""

from __future__ import annotations

import hashlib
import json
from collections import abc
from collections.abc import Iterable
from dataclasses import MISSING, dataclass, fields, is_dataclass, replace
from datetime import date, datetime
from enum import Enum, StrEnum
from pathlib import Path
from types import MappingProxyType, UnionType
from typing import (
    Any,
    Literal,
    Mapping,
    Union,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)
from uuid import UUID

from potpie_context_engine.requests import (
    ApplyPreviewRequest,
    CatalogRequest,
    CommitRequest,
    CommitShowRequest,
    CommitStatusRequest,
    CommitsRequest,
    DataPlaneStatusRequest,
    DescribeRequest,
    DisableRollbackRequest,
    EngineRequest,
    ExportSnapshotRequest,
    GraphSliceRequest,
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
    JournalStatusRequest,
    MutateRequest,
    NeighborhoodRequest,
    NudgeRequest,
    ProcessingStatusRequest,
    ProposeRequest,
    QualityRequest,
    ReadRequest,
    RebuildCommitsRequest,
    RecordRequest,
    RepairRequest,
    ResetContextRequest,
    ResolveRequest,
    ResourceGetRequest,
    ResourceImportRequest,
    ResourceIndexBuildRequest,
    ResourceIndexRebuildRequest,
    ResourceIndexStatusRequest,
    ResourceListRequest,
    ResourceRmRequest,
    ResourceStatusRequest,
    RevertPreviewRequest,
    RollbackPreviewRequest,
    SearchEntitiesRequest,
    SearchRequest,
    SubmitArtifactRequest,
    SubmitEventRequest,
    VerifyCommitRequest,
)
from potpie_context_engine.results import (
    ApplyPreviewResult,
    CatalogResult,
    CommitResult,
    CommitShowResult,
    CommitStatusResult,
    CommitsResult,
    DataPlaneStatusResult,
    DescribeResult,
    DisableRollbackResult,
    ExportSnapshotResult,
    GraphSliceResult,
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
    JournalStatusResult,
    MutateResult,
    NeighborhoodResult,
    NudgeResult,
    ProcessingStatusResult,
    ProposeResult,
    QualityResult,
    ReadResult,
    RebuildCommitsResult,
    RecordResult,
    RepairResult,
    ResetContextResult,
    ResolveResult,
    ResourceGetResult,
    ResourceImportResult,
    ResourceIndexBuildResult,
    ResourceIndexRebuildResult,
    ResourceIndexStatusResult,
    ResourceListResult,
    ResourceRmResult,
    ResourceStatusResult,
    RevertPreviewResult,
    RollbackPreviewResult,
    SearchEntitiesResult,
    SearchResult,
    SubmitArtifactResult,
    SubmitEventResult,
    VerifyCommitResult,
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
    GRAPH_SLICE = "graph_slice"
    EXPORT_SNAPSHOT = "export_snapshot"
    IMPORT_SNAPSHOT = "import_snapshot"
    REPAIR = "repair"
    RESET_CONTEXT = "reset_context"
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
    # Document resources: payload store + retrieval index.
    RESOURCE_IMPORT = "resource_import"
    RESOURCE_GET = "resource_get"
    RESOURCE_LIST = "resource_list"
    RESOURCE_RM = "resource_rm"
    RESOURCE_STATUS = "resource_status"
    RESOURCE_INDEX_STATUS = "resource_index_status"
    RESOURCE_INDEX_BUILD = "resource_index_build"
    RESOURCE_INDEX_REBUILD = "resource_index_rebuild"
    COMMIT_STATUS = "commit_status"
    VERIFY_COMMIT = "verify_commit"
    JOURNAL_STATUS = "journal_status"
    COMMITS = "commits"
    COMMIT_SHOW = "commit_show"
    REVERT_PREVIEW = "revert_preview"
    ROLLBACK_PREVIEW = "rollback_preview"
    APPLY_PREVIEW = "apply_preview"
    DISABLE_ROLLBACK = "disable_rollback"
    REBUILD_COMMITS = "rebuild_commits"


class DaemonControlOperation(StrEnum):
    HANDSHAKE = "daemon.handshake"
    STATUS = "daemon.status"
    SHUTDOWN = "daemon.shutdown"


class SafetyClass(StrEnum):
    SHARED_CONTEXT_READ = "shared_context_read"
    SHARED_CONTEXT_READ_EXCLUSIVE_RESOURCE_WRITE = (
        "shared_context_read_exclusive_resource_write"
    )
    EXCLUSIVE_CONTEXT_MUTATION = "exclusive_context_mutation"
    EXCLUSIVE_RESOURCE_MUTATION = "exclusive_resource_mutation"
    DAEMON_LIFECYCLE_CONTROL = "daemon_lifecycle_control"


@dataclass(frozen=True, slots=True)
class OperationSpec:
    """Protocol-visible semantics of one operation.

    ``operation`` is an :class:`EngineOperation`, a host operation
    (``host.*``) or an embedding host's extension name (``ext.<vendor>.<op>``).
    """

    operation: str
    request_type: type[object]
    result_type: type[object]
    safety: SafetyClass
    destructive: bool = False
    context_required: bool = True
    resource_type: str | None = None
    resource_identity_fields: tuple[str, ...] = ()
    request_policy: str | None = None

    def __post_init__(self) -> None:
        if self.safety in {
            SafetyClass.EXCLUSIVE_RESOURCE_MUTATION,
            SafetyClass.SHARED_CONTEXT_READ_EXCLUSIVE_RESOURCE_WRITE,
        }:
            if self.resource_type is None or not self.resource_identity_fields:
                raise ValueError(
                    "exclusive resource mutations require resource conflict metadata"
                )
        elif self.resource_type is not None or self.resource_identity_fields:
            raise ValueError(
                "resource conflict metadata is valid only for resource mutations"
            )

    @property
    def name(self) -> str:
        """The operation's wire name."""

        return str(self.operation)

    @property
    def family(self) -> str:
        """``engine``, ``host`` or ``extension``."""

        if isinstance(self.operation, EngineOperation):
            return "engine"
        if self.name.startswith("host."):
            return "host"
        if self.name.startswith("ext."):
            return "extension"
        return "unknown"


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
    OperationSpec(
        EngineOperation.DESCRIBE,
        DescribeRequest,
        DescribeResult,
        _READ,
        context_required=False,
    ),
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
        EngineOperation.GRAPH_SLICE, GraphSliceRequest, GraphSliceResult, _READ
    ),
    OperationSpec(
        EngineOperation.EXPORT_SNAPSHOT,
        ExportSnapshotRequest,
        ExportSnapshotResult,
        SafetyClass.SHARED_CONTEXT_READ_EXCLUSIVE_RESOURCE_WRITE,
        resource_type="snapshot_destination",
        resource_identity_fields=("destination",),
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
        request_policy="legacy_repair_v1",
    ),
    OperationSpec(
        EngineOperation.RESET_CONTEXT,
        ResetContextRequest,
        ResetContextResult,
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
    # --- document resources ----------------------------------------------
    # The store is pot-scoped and an import or removal also writes the graph
    # (structure claims, retractions), so both are context mutations. Index
    # build/rebuild touch only derived rows: they read the context (and so
    # wait out an import) while holding the index rows they rewrite.
    OperationSpec(
        EngineOperation.RESOURCE_IMPORT,
        ResourceImportRequest,
        ResourceImportResult,
        _WRITE,
    ),
    OperationSpec(
        EngineOperation.RESOURCE_GET, ResourceGetRequest, ResourceGetResult, _READ
    ),
    OperationSpec(
        EngineOperation.RESOURCE_LIST, ResourceListRequest, ResourceListResult, _READ
    ),
    OperationSpec(
        EngineOperation.RESOURCE_RM,
        ResourceRmRequest,
        ResourceRmResult,
        _WRITE,
        destructive=True,
    ),
    OperationSpec(
        EngineOperation.RESOURCE_STATUS,
        ResourceStatusRequest,
        ResourceStatusResult,
        _READ,
    ),
    OperationSpec(
        EngineOperation.RESOURCE_INDEX_STATUS,
        ResourceIndexStatusRequest,
        ResourceIndexStatusResult,
        _READ,
    ),
    OperationSpec(
        EngineOperation.RESOURCE_INDEX_BUILD,
        ResourceIndexBuildRequest,
        ResourceIndexBuildResult,
        SafetyClass.SHARED_CONTEXT_READ_EXCLUSIVE_RESOURCE_WRITE,
        resource_type="resource_index",
        resource_identity_fields=("doc",),
    ),
    OperationSpec(
        EngineOperation.RESOURCE_INDEX_REBUILD,
        ResourceIndexRebuildRequest,
        ResourceIndexRebuildResult,
        SafetyClass.SHARED_CONTEXT_READ_EXCLUSIVE_RESOURCE_WRITE,
        resource_type="resource_index",
        resource_identity_fields=("doc",),
    ),
    # Graph commit history, journal and rollback. Status reads never write, so
    # a caller that lost a commit's response may poll them without retrying it.
    OperationSpec(
        EngineOperation.COMMIT_STATUS,
        CommitStatusRequest,
        CommitStatusResult,
        _READ,
    ),
    OperationSpec(
        EngineOperation.VERIFY_COMMIT,
        VerifyCommitRequest,
        VerifyCommitResult,
        _READ,
    ),
    OperationSpec(
        EngineOperation.JOURNAL_STATUS,
        JournalStatusRequest,
        JournalStatusResult,
        _READ,
    ),
    OperationSpec(EngineOperation.COMMITS, CommitsRequest, CommitsResult, _READ),
    OperationSpec(
        EngineOperation.COMMIT_SHOW, CommitShowRequest, CommitShowResult, _READ
    ),
    # A preview plans an inverse against the current HEAD and stores it
    # server-side; it changes no graph state. It is exclusive so it is never
    # planned against a HEAD that a concurrent write is moving. Only applying a
    # preview changes the graph, so only that carries destructive intent.
    OperationSpec(
        EngineOperation.REVERT_PREVIEW,
        RevertPreviewRequest,
        RevertPreviewResult,
        _WRITE,
    ),
    OperationSpec(
        EngineOperation.ROLLBACK_PREVIEW,
        RollbackPreviewRequest,
        RollbackPreviewResult,
        _WRITE,
    ),
    OperationSpec(
        EngineOperation.APPLY_PREVIEW,
        ApplyPreviewRequest,
        ApplyPreviewResult,
        _WRITE,
        destructive=True,
    ),
    OperationSpec(
        EngineOperation.DISABLE_ROLLBACK,
        DisableRollbackRequest,
        DisableRollbackResult,
        _WRITE,
    ),
    OperationSpec(
        EngineOperation.REBUILD_COMMITS,
        RebuildCommitsRequest,
        RebuildCommitsResult,
        _WRITE,
    ),
)

if len({spec.operation for spec in _SPECS}) != len(_SPECS):
    raise RuntimeError("each engine operation must have exactly one catalog entry")
if {spec.operation for spec in _SPECS} != set(EngineOperation):
    raise RuntimeError("the operation catalog must cover every engine operation")

ENGINE_OPERATION_CATALOG: Mapping[EngineOperation, OperationSpec] = MappingProxyType(
    {spec.operation: spec for spec in _SPECS}
)


def _server_only_types() -> frozenset[type]:
    """Journal and restore types that only the engine may construct.

    A restore plan, its records, a stored preview and its request are rebuilt
    and checked against persisted hashes on the server. A wire payload that
    could carry one would let a caller choose what a rollback writes.
    """

    from potpie_context_engine.core.graph_journal import (
        CommitReceipt,
        JournalRecord,
        RecordChange,
        RollbackPreview,
        RollbackRequest,
    )
    from potpie_context_engine.core.graph_restore import RestorePlan, RestoreRecord

    return frozenset(
        {
            CommitReceipt,
            JournalRecord,
            RecordChange,
            RestorePlan,
            RestoreRecord,
            RollbackPreview,
            RollbackRequest,
        }
    )


def _referenced_types(annotation: object, seen: set[type]) -> None:
    for argument in get_args(annotation):
        _referenced_types(argument, seen)
    if isinstance(annotation, type) and annotation not in seen:
        seen.add(annotation)
        if is_dataclass(annotation):
            try:
                hints = get_type_hints(annotation)
            except (NameError, TypeError) as exc:
                # Fail closed: a field type nobody can resolve is a field type
                # nobody has checked.
                raise RuntimeError(
                    f"cannot resolve the field types of {annotation.__qualname__}"
                ) from exc
            for hint in hints.values():
                _referenced_types(hint, seen)


def request_wire_types(request_type: type[EngineRequest]) -> frozenset[type]:
    """Every type a decoded ``request_type`` payload can contain."""

    seen: set[type] = set()
    _referenced_types(request_type, seen)
    return frozenset(seen)


_SERVER_ONLY_INPUTS = {
    spec.operation.value: sorted(
        kind.__name__
        for kind in request_wire_types(spec.request_type) & _server_only_types()
    )
    for spec in _SPECS
}
if any(_SERVER_ONLY_INPUTS.values()):
    raise RuntimeError(
        "operation requests must not carry server-only restore types: "
        f"{ {op: kinds for op, kinds in _SERVER_ONLY_INPUTS.items() if kinds} }"
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
            "context_required": spec.context_required,
            "resource_type": spec.resource_type,
            "resource_identity_fields": spec.resource_identity_fields,
            **({"request_policy": spec.request_policy} if spec.request_policy else {}),
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


#: Version of the :func:`operation_schema_document` layout, not of the protocol.
OPERATION_SCHEMA_FORMAT = 1


def operation_schema(
    spec: OperationSpec, types: dict[str, dict[str, object]] | None = None
) -> dict[str, object]:
    """The protocol-visible contract of one operation, as plain JSON data.

    Unlike the catalog fingerprint this describes the wire, not the Python
    types: class names only key the ``types`` table, and the same wire shape
    reads the same on every supported Python version. Dataclasses are described
    once in ``types`` and referenced by name. It is what the committed
    evolution baseline records and checks additive-only.
    """

    table = types if types is not None else {}
    return {
        "family": _operation_family(spec),
        "safety": spec.safety.value,
        "destructive": spec.destructive,
        "context_required": spec.context_required,
        "resource_type": spec.resource_type,
        "resource_identity_fields": list(spec.resource_identity_fields),
        **({"request_policy": spec.request_policy} if spec.request_policy else {}),
        "request": wire_schema(spec.request_type, table),
        "result": wire_schema(spec.result_type, table),
    }


def effective_operation_spec(spec: OperationSpec, request: object) -> OperationSpec:
    """Resolve typed request policy once for confirmation, locking and retries."""
    if spec.request_policy == "legacy_repair_v1":
        if type(request) is not RepairRequest:
            raise TypeError("repair policy requires a typed RepairRequest")
        if request.legacy_mode == "dry_run":
            return replace(
                spec, safety=SafetyClass.SHARED_CONTEXT_READ, destructive=False
            )
    elif spec.request_policy is not None:
        raise ValueError("unsupported operation request policy")
    return spec


def operation_schema_document(
    specs: Iterable[OperationSpec] | None = None,
) -> dict[str, object]:
    """Every engine and host operation's wire contract plus the error categories.

    Extension operations belong to the embedding host that registers them and
    are not part of this document.
    """

    from potpie.runtime.codec import ERROR_CATEGORIES
    from potpie.runtime.host_operations import HOST_OPERATION_CATALOG
    from potpie.runtime.protocol import PROTOCOL_VERSION

    if specs is None:
        specs = (*_SPECS, *HOST_OPERATION_CATALOG.values())
    chosen = sorted(specs, key=_operation_name)
    types: dict[str, dict[str, object]] = {}
    operations = {
        _operation_name(spec): operation_schema(spec, types) for spec in chosen
    }
    return {
        "format": OPERATION_SCHEMA_FORMAT,
        "protocol_version": PROTOCOL_VERSION,
        "error_categories": sorted(ERROR_CATEGORIES),
        "operations": operations,
        "types": dict(sorted(types.items())),
    }


def wire_schema(
    annotation: object, types: dict[str, dict[str, object]]
) -> dict[str, object]:
    """Describe how values of ``annotation`` look on the wire.

    Nodes carry a ``kind``: ``ref`` (a dataclass, described once in ``types``
    as an ``object`` with ``fields``), ``array``, ``tuple``, ``map``,
    ``union``, ``literal``, ``enum``, a scalar kind, ``any``, or ``opaque`` (a
    type the codec passes through unchecked).
    """

    return _wire_schema(annotation, types)


_SCALAR_KINDS: Mapping[object, str] = MappingProxyType(
    {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        type(None): "null",
        bytes: "bytes",
    }
)


def _wire_schema(
    annotation: object, types: dict[str, dict[str, object]]
) -> dict[str, object]:
    if annotation is Any or annotation is object:
        return {"kind": "any"}
    if isinstance(annotation, str):
        return {"kind": "unresolved", "name": annotation}
    scalar = _SCALAR_KINDS.get(annotation)
    if scalar is not None:
        return {"kind": scalar}
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is Literal:
        return {"kind": "literal", "values": [_json_value(item) for item in args]}
    if origin is Union or origin is UnionType:
        return {"kind": "union", "options": [_wire_schema(i, types) for i in args]}
    if origin is tuple:
        if not args or (len(args) == 2 and args[1] is Ellipsis):
            item = args[0] if args else Any
            return {"kind": "array", "items": _wire_schema(item, types)}
        return {"kind": "tuple", "items": [_wire_schema(i, types) for i in args]}
    if origin in (list, set, frozenset, abc.Sequence, abc.Set):
        item = args[0] if args else Any
        return {"kind": "array", "items": _wire_schema(item, types)}
    if origin in (dict, abc.Mapping, abc.MutableMapping):
        key, value = (args[0], args[1]) if len(args) == 2 else (Any, Any)
        return {
            "kind": "map",
            "keys": _wire_schema(key, types),
            "values": _wire_schema(value, types),
        }
    if origin is not None:
        return {
            "kind": "opaque",
            "name": _qualified_name(origin),
            "arguments": [_wire_schema(item, types) for item in args],
        }
    if annotation in (datetime, date, Path, UUID):
        return {"kind": cast(type, annotation).__name__.lower()}
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return {
            "kind": "enum",
            "values": [_json_value(member.value) for member in annotation],
        }
    if isinstance(annotation, type) and is_dataclass(annotation):
        name = _type_key(annotation, types)
        if name not in types:
            # Placeholder first, so a type that refers to itself ends.
            types[name] = {
                "kind": "object",
                "python": _qualified_name(annotation),
                "fields": [],
            }
            try:
                hints = get_type_hints(annotation)
            except (NameError, TypeError):
                hints = {item.name: item.type for item in fields(annotation)}
            described: list[dict[str, object]] = []
            for item in fields(annotation):
                record: dict[str, object] = {
                    "name": item.name,
                    "schema": _wire_schema(hints.get(item.name, item.type), types),
                }
                if item.default is not MISSING:
                    record["default"] = _json_value(item.default)
                elif isinstance(item.default_factory, type):
                    # A container or dataclass factory: compare what it builds.
                    record["default"] = _json_value(item.default_factory())
                elif item.default_factory is not MISSING:
                    # A computed default (a timestamp, an id) has no fixed
                    # value; it is pinned by the function that computes it.
                    record["default_factory"] = _qualified_name(item.default_factory)
                else:
                    record["required"] = True
                described.append(record)
            types[name] = {
                "kind": "object",
                "python": _qualified_name(annotation),
                "fields": described,
            }
        return {"kind": "ref", "name": name}
    return {"kind": "opaque", "name": _qualified_name(annotation)}


def _type_key(annotation: type, types: dict[str, dict[str, object]]) -> str:
    """A short, stable key for one dataclass in the ``types`` table."""

    qualified = _qualified_name(annotation)
    name = annotation.__name__
    entry = types.get(name)
    if entry is None or entry.get("python") in (None, qualified):
        return name
    return qualified


def _json_value(value: object) -> object:
    """A default or literal value as the codec would put it on the wire."""

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Enum):
        return _json_value(value.value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_json_value(item) for item in sorted(value, key=str)]
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _json_value(getattr(value, item.name)) for item in fields(value)
        }
    return str(value)


def _operation_name(spec: OperationSpec) -> str:
    return spec.name


def _operation_family(spec: OperationSpec) -> str:
    return spec.family


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
    "OPERATION_SCHEMA_FORMAT",
    "DaemonControlOperation",
    "EngineOperation",
    "OperationSpec",
    "SafetyClass",
    "operation_capabilities",
    "operation_catalog_fingerprint",
    "operation_schema",
    "operation_schema_document",
    "request_wire_types",
    "wire_schema",
]
