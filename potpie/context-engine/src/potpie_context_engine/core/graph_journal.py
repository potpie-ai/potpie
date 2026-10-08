"""Versioned, storage-independent graph journal contracts.

Receipts describe effects, not submitted operations. Patch records deliberately
contain only changed fields; display context is a bounded historical annotation.
The JSON codec is closed (no import-by-name) and preserves missing versus null.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import datetime
from typing import Any

SCHEMA_VERSION = 1
FIELD_REGISTRY_VERSION = 1


class JournalError(ValueError):
    """A journal invariant failed before publication."""

    def __init__(self, message: str, *, code: str = "invalid_journal"):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class FieldValue:
    present: bool
    value: Any = None

    def __post_init__(self) -> None:
        if not self.present and self.value is not None:
            raise JournalError("absent fields cannot carry a value")


@dataclass(frozen=True, slots=True)
class FieldChange:
    path: tuple[str, ...]
    before: FieldValue
    after: FieldValue


@dataclass(frozen=True, slots=True)
class JournalRecord:
    record_id: str
    pot_id: str
    logical_key: str
    kind: str
    fields: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class RecordChange:
    record_id: str
    logical_key: str
    kind: str
    action: str
    fields: tuple[FieldChange, ...] = ()
    before_record: JournalRecord | None = None
    after_record: JournalRecord | None = None


@dataclass(frozen=True, slots=True)
class JournalLimits:
    max_records: int = 1_000
    max_bytes: int = 8_000_000
    max_detail_records: int = 200
    max_page_size: int = 200

    def __post_init__(self) -> None:
        if (
            min(
                self.max_records,
                self.max_bytes,
                self.max_detail_records,
                self.max_page_size,
            )
            < 1
        ):
            raise JournalError("journal limits must be positive")


@dataclass(frozen=True, slots=True)
class ResourceGuard:
    operation_id: str
    owner: str
    generation: int
    status: str = "incomplete"


@dataclass(frozen=True, slots=True)
class JournalState:
    pot_id: str
    generation: str
    sequence: int = 0
    head: str | None = None
    coverage_start: int = 1
    resource_generation: int = 0
    resource_guard: ResourceGuard | None = None
    rollback_enabled: bool = False
    limits: JournalLimits = field(default_factory=JournalLimits)


@dataclass(frozen=True, slots=True)
class JournalCapability:
    journal_supported: bool = False
    rollback_supported: bool = False
    durable: bool = False
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class CommitReceipt:
    commit_id: str
    pot_id: str
    journal_generation: str
    sequence: int
    parent_commit_id: str | None
    fingerprint: str
    committed_at: datetime
    actor: str
    message: str
    origin: str
    required_access: str
    rollback_supported: bool
    semantic_changes: tuple[RecordChange, ...] = ()
    audit_changes: tuple[RecordChange, ...] = ()
    display_context: tuple[Mapping[str, Any], ...] = ()
    plan_id: str | None = None
    resource_operation_id: str | None = None
    unsupported_reason: str | None = None
    reverts_commit_id: str | None = None
    rollback_target_commit_id: str | None = None
    schema_version: int = SCHEMA_VERSION
    field_registry_version: int = FIELD_REGISTRY_VERSION
    diff_complete: bool = True

    @property
    def affected_record_count(self) -> int:
        return len({c.record_id for c in (*self.semantic_changes, *self.audit_changes)})


@dataclass(frozen=True, slots=True)
class RollbackRequest:
    """Public target contract. Trusted restore operations live in another module."""

    target_commit_id: str
    expected_head: str
    mode: str = "revert"


@dataclass(frozen=True, slots=True)
class RollbackPreview:
    preview_id: str
    actor: str
    host: str
    pot_id: str
    journal_generation: str
    resource_generation: int
    expected_head: str
    inverse_hash: str
    target_commit_ids: tuple[str, ...]
    required_access: str
    expires_at: datetime
    conflicts: tuple[str, ...] = ()


_TYPES = {
    cls.__name__: cls
    for cls in (
        FieldValue,
        FieldChange,
        JournalRecord,
        RecordChange,
        JournalLimits,
        ResourceGuard,
        JournalState,
        JournalCapability,
        CommitReceipt,
        RollbackRequest,
        RollbackPreview,
    )
}


def encode_journal(value: Any) -> Any:
    """Canonical tagged values, including dictionaries with reserved-looking keys."""
    if is_dataclass(value) and not isinstance(value, type):
        name = type(value).__name__
        if _TYPES.get(name) is not type(value):
            raise JournalError(f"unsupported journal type: {name}")
        return {
            "type": name,
            "value": {
                f.name: encode_journal(getattr(value, f.name)) for f in fields(value)
            },
        }
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise JournalError("journal timestamps must be timezone aware")
        return {"type": "datetime", "value": value.isoformat()}
    if isinstance(value, Mapping):
        if any(not isinstance(k, str) for k in value):
            raise JournalError("journal map keys must be strings")
        return {
            "type": "map",
            "value": {k: encode_journal(v) for k, v in value.items()},
        }
    if isinstance(value, (list, tuple)):
        return {
            "type": "tuple" if isinstance(value, tuple) else "list",
            "value": [encode_journal(v) for v in value],
        }
    if isinstance(value, float) and not math.isfinite(value):
        raise JournalError("non-finite journal values are unsupported")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise JournalError(f"unsupported journal value: {type(value).__name__}")


def decode_journal(value: Any) -> Any:
    if not isinstance(value, dict):
        if value is None or isinstance(value, (str, int, float, bool)):
            encode_journal(value)
            return value
        raise JournalError("invalid journal value")
    if set(value) != {"type", "value"}:
        raise JournalError("invalid journal envelope")
    kind, raw = value["type"], value["value"]
    if kind == "datetime":
        result = datetime.fromisoformat(raw)
        encode_journal(result)
        return result
    if kind == "map":
        return {k: decode_journal(v) for k, v in raw.items()}
    if kind in {"tuple", "list"}:
        result = [decode_journal(v) for v in raw]
        return tuple(result) if kind == "tuple" else result
    cls = _TYPES.get(kind)
    if cls is None:
        raise JournalError(f"unsupported journal schema type: {kind}")
    result = cls(**{k: decode_journal(v) for k, v in raw.items()})
    if isinstance(result, CommitReceipt) and (
        result.schema_version != SCHEMA_VERSION
        or result.field_registry_version != FIELD_REGISTRY_VERSION
    ):
        raise JournalError("unsupported journal schema/field registry version")
    return result


def journal_json(value: Any) -> str:
    return json.dumps(
        encode_journal(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def journal_hash(value: Any) -> str:
    return hashlib.sha256(journal_json(value).encode()).hexdigest()


def strongest_access(*values: str) -> str:
    ranks = {"read": 0, "write": 1, "admin": 2}
    if any(v not in ranks for v in values):
        raise JournalError("unknown effect permission")
    return max(values, key=ranks.__getitem__, default="write")
