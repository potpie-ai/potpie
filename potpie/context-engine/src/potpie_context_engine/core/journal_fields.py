"""Field registry v1 and changed-field capture; provenance is classified by key."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace
from typing import Any

from potpie_context_engine.core.graph_journal import (
    CommitReceipt,
    FieldChange,
    FieldValue,
    JournalError,
    JournalRecord,
    RecordChange,
    journal_json,
)

AUDIT_FIELDS = frozenset(
    {
        "mutation_id",
        "prov_mutation_id",
        "prov_graph_updated_at",
        "graph_updated_at",
        "actor_user_id",
        "actor_surface",
        "actor_client_name",
        "actor_auth_method",
        "prov_created_by_agent",
        "prov_reconciliation_run_id",
        "revived_at",
        "expired_at",
        "invalidated_by",
        "deleted_by",
        "execution_observed_at",
        "execution_source_event_id",
        "execution_valid_from",
    }
)
IDENTITY_FIELDS = frozenset({"record_id", "uuid", "created_at", "pot_id", "group_id"})
DERIVED_FIELDS = frozenset(
    {
        "fact_embedding",
        "embedding",
        "embedding_model",
        "embedding_dimension",
        "embedding_dim",
        "semantic_similarity",
        "match_score",
    }
)


def field_group(path: tuple[str, ...]) -> str:
    # Only the immediate property name is special. A source's nested evidence
    # field named 'mutation_id' or 'observed_at' remains semantic.
    name = path[1] if path[0] == "properties" and len(path) > 1 else path[0]
    if name in IDENTITY_FIELDS:
        return "identity"
    if name in DERIVED_FIELDS:
        return "derived"
    if name in AUDIT_FIELDS:
        return "audit"
    return "semantic"


def leaf_values(value: Mapping[str, Any], prefix: tuple[str, ...] = ()) -> dict:
    result = {}
    for name, item in value.items():
        path = (*prefix, name)
        if isinstance(item, Mapping) and item:
            result.update(leaf_values(item, path))
        else:
            result[path] = item
    return result


def read_field(record: JournalRecord, path: tuple[str, ...]) -> FieldValue:
    current: Any = record.fields
    for part in path:
        if not isinstance(current, Mapping) or part not in current:
            return FieldValue(False)
        current = current[part]
    return FieldValue(True, deepcopy(current))


def set_field(
    record: JournalRecord, path: tuple[str, ...], value: FieldValue
) -> JournalRecord:
    if not path:
        raise JournalError("empty field path")
    body = deepcopy(dict(record.fields))
    current = body
    for part in path[:-1]:
        if part not in current:
            if not value.present:
                return record
            current[part] = {}
        if not isinstance(current[part], dict):
            raise JournalError("field path crosses a scalar")
        current = current[part]
    if value.present:
        current[path[-1]] = deepcopy(value.value)
    else:
        current.pop(path[-1], None)
    return replace(record, fields=body)


def semantic_fields(record: JournalRecord) -> dict:
    return {
        p: v
        for p, v in leaf_values(record.fields).items()
        if field_group(p) == "semantic" and not (p == ("properties",) and v == {})
    }


def capture_changes(
    before: Mapping[str, JournalRecord], after: Mapping[str, JournalRecord]
):
    semantic, audit = [], []
    for record_id in sorted(before.keys() | after.keys()):
        old, new = before.get(record_id), after.get(record_id)
        record = new or old
        assert record is not None
        if old is None or new is None:
            semantic.append(
                RecordChange(
                    record_id,
                    record.logical_key,
                    record.kind,
                    "create" if old is None else "retire",
                    before_record=semantic_record(old),
                    after_record=semantic_record(new),
                )
            )
            stamp_changes = tuple(
                FieldChange(
                    path,
                    FieldValue(old is not None, value if old else None),
                    FieldValue(new is not None, value if new else None),
                )
                for path, value in leaf_values(record.fields).items()
                if field_group(path) == "audit"
            )
            if stamp_changes:
                audit.append(
                    RecordChange(
                        record_id,
                        record.logical_key,
                        record.kind,
                        "patch",
                        stamp_changes,
                    )
                )
            continue
        if (old.pot_id, old.logical_key, old.kind) != (
            new.pot_id,
            new.logical_key,
            new.kind,
        ):
            raise JournalError("record identity changed")
        groups: dict[str, list[FieldChange]] = {"semantic": [], "audit": []}
        for path, left, right in changed_values(old.fields, new.fields):
            group = field_group(path)
            if group == "identity":
                raise JournalError("creation identity changed")
            if group in groups:
                groups[group].append(FieldChange(path, left, right))
        for group, target in (("semantic", semantic), ("audit", audit)):
            if groups[group]:
                action = "patch"
                if old.fields.get("active", True) != new.fields.get("active", True):
                    action = "restore" if new.fields.get("active", True) else "retire"
                target.append(
                    RecordChange(
                        record_id,
                        record.logical_key,
                        record.kind,
                        action,
                        tuple(groups[group]),
                    )
                )
    return tuple(semantic), tuple(audit)


def check_receipt_limits(receipt: CommitReceipt, limits) -> None:
    if receipt.affected_record_count > limits.max_records:
        raise JournalError("atomic record limit exceeded")
    if len(journal_json(receipt).encode()) > limits.max_bytes:
        raise JournalError("atomic journal byte limit exceeded")


def changed_values(before, after, prefix=()):
    for name in sorted(before.keys() | after.keys()):
        path = (*prefix, name)
        left, right = (
            FieldValue(name in before, before.get(name)),
            FieldValue(name in after, after.get(name)),
        )
        if journal_json(left) == journal_json(right):
            continue
        if (
            left.present
            and right.present
            and isinstance(left.value, Mapping)
            and isinstance(right.value, Mapping)
        ):
            yield from changed_values(left.value, right.value, path)
        else:
            yield path, left, right


def semantic_record(record):
    if record is None:
        return None
    result = deepcopy(record)
    for path in sorted(leaf_values(record.fields), reverse=True):
        if field_group(path) in {"derived", "audit"}:
            result = set_field(result, path, FieldValue(False))
    return result


def clear_derived(record: JournalRecord) -> JournalRecord:
    """Invalidate current retrieval caches after exact semantic restoration."""
    result = record
    for path in sorted(leaf_values(record.fields), reverse=True):
        if field_group(path) == "derived":
            result = set_field(result, path, FieldValue(False))
    return result
