"""Shared immutable effect receipts for memory, embedded and FalkorDB."""

import json
from collections.abc import Mapping
from datetime import datetime, timezone

from potpie_context_engine.core.graph_journal import CommitReceipt
from potpie_context_engine.core.journal_context import current_journal_context
from potpie_context_engine.core.journal_fields import (
    capture_changes,
    check_receipt_limits,
)


def _resource_record(record):
    if record is None:
        return False
    if record.logical_key.startswith(
        ("document:", "docsection:", "section:", "resource:")
    ):
        return True
    if {"Document", "DocumentSection", "Resource"} & set(
        record.fields.get("labels", ())
    ):
        return True

    def review_marker(value):
        if isinstance(value, Mapping):
            return any(
                "evidence_review" in str(key) or review_marker(child)
                for key, child in value.items()
            )
        if isinstance(value, (list, tuple)):
            return any(review_marker(child) for child in value)
        if isinstance(value, str) and value.startswith(("{", "[")):
            try:
                return review_marker(json.loads(value))
            except ValueError:
                return False
        return False

    return review_marker(record.fields)


def capture_receipt(
    state,
    before,
    after,
    *,
    commit_id,
    fingerprint,
    actor,
    message,
    origin=None,
    required_access=None,
    unsupported_reason=None,
    **extra,
):
    context = current_journal_context()
    semantic, audit = capture_changes(before, after)
    ids = {c.record_id for c in (*semantic, *audit)}
    # Effect classification includes creations, removals and unchanged resource
    # endpoints; a caller's origin label cannot make these effects reversible.
    display_ids = set(ids)
    for key in ids:
        for record in (before.get(key), after.get(key)):
            if record is not None:
                display_ids.update(
                    record.fields[name]
                    for name in ("subject_record_id", "object_record_id")
                    if name in record.fields
                )
    resource_owned = any(
        _resource_record(records.get(key))
        for records in (before, after)
        for key in display_ids
    )
    resource_operation_id = context.resource_operation_id
    if state.resource_guard:
        resource_operation_id = state.resource_guard.operation_id
    reason = unsupported_reason or (
        "resource graph/content cannot be restored"
        if resource_owned or resource_operation_id
        else None
    )
    # At most two endpoints per changed claim, bounded strings and receipt bytes.
    display = tuple(
        {
            "record_id": key,
            "logical_key": (after.get(key) or before[key]).logical_key[:256],
            "kind": (after.get(key) or before[key]).kind,
            "predicate": (after.get(key) or before[key]).fields.get("predicate"),
            "endpoint_record_ids": tuple(
                (after.get(key) or before[key]).fields[name]
                for name in ("subject_record_id", "object_record_id")
                if name in (after.get(key) or before[key]).fields
            ),
            "labels": tuple(
                str(label)[:80]
                for label in (after.get(key) or before[key]).fields.get("labels", ())[
                    :8
                ]
            ),
            "name": str(
                (after.get(key) or before[key])
                .fields.get("properties", {})
                .get("name", "")
            )[:160],
        }
        for key in sorted(display_ids)
        if key in before or key in after
    )
    receipt = CommitReceipt(
        commit_id,
        state.pot_id,
        state.generation,
        state.sequence + 1,
        state.head,
        fingerprint,
        datetime.now(timezone.utc),
        actor,
        message[:2000],
        origin
        or ("resource" if resource_operation_id or resource_owned else context.origin),
        required_access or context.required_access,
        reason is None,
        semantic,
        audit,
        display,
        plan_id=context.plan_id,
        resource_operation_id=resource_operation_id,
        unsupported_reason=reason,
        **extra,
    )
    check_receipt_limits(receipt, state.limits)
    return receipt
