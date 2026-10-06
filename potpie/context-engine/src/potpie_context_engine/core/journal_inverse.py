"""Pure inverse planning and final-state validation, shared by native stores."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone

from potpie_context_engine.core.definition import (
    DEFAULT_GRAPH_DEFINITION,
    GraphDefinition,
)
from potpie_context_engine.core.graph_journal import (
    CommitReceipt,
    JournalError,
    JournalRecord,
    JournalState,
    journal_hash,
    journal_json,
    strongest_access,
)
from potpie_context_engine.core.graph_mutations import EdgeUpsert, EntityUpsert
from potpie_context_engine.core.graph_restore import RestorePlan, RestoreRecord
from potpie_context_engine.core.journal_fields import (
    capture_changes,
    field_group,
    read_field,
    semantic_fields,
    set_field,
)
from potpie_context_engine.core.ontology import SYSTEM_EDGE_TYPES


def validate_state(
    records: Mapping[str, JournalRecord],
    *,
    pot_id: str,
    definition: GraphDefinition = DEFAULT_GRAPH_DEFINITION,
    resource_exists: Callable[[str], bool] | None = None,
) -> None:
    """Validate exact restored state without ordinary upsert normalization."""
    entities, claims = {}, []
    for record_id, record in records.items():
        if record_id != record.record_id or record.pot_id != pot_id:
            raise JournalError("record identity or pot scope mismatch")
        if not record.fields.get("active", True):
            continue
        if record.kind == "entity":
            if record.logical_key in entities:
                raise JournalError("ambiguous active entity incarnation")
            entities[record.logical_key] = record
            from potpie_context_engine.core.protocols import (
                PREFIXES,
                normalize_properties,
                property_evidence,
                protocol_entity_key,
            )

            for label in record.fields.get("labels", ()):
                if label in PREFIXES:
                    try:
                        key = protocol_entity_key(
                            label, record.fields.get("properties", {})
                        )
                    except ValueError as exc:
                        raise JournalError(
                            f"invalid restored protocol identity: {exc}"
                        ) from exc
                    if key != record.logical_key:
                        raise JournalError(
                            "restored protocol identity does not match its key"
                        )
                    properties = normalize_properties(
                        dict(record.fields.get("properties", {})), label
                    )
                    for evidence in property_evidence(properties):
                        ref = evidence.get("source_ref")
                        if (
                            isinstance(ref, str)
                            and ref.startswith("potpie://res/")
                            and (resource_exists is None or not resource_exists(ref))
                        ):
                            raise JournalError(
                                "protocol resource reference cannot be verified"
                            )
        elif record.kind == "claim":
            claims.append(record)
        else:
            raise JournalError("unsupported journal record kind")
    entity_ops = [
        EntityUpsert(
            key, tuple(r.fields.get("labels", ())), dict(r.fields.get("properties", {}))
        )
        for key, r in entities.items()
    ]
    edges, singletons = [], {}
    now = datetime.now(timezone.utc)
    for claim in claims:
        body = claim.fields
        source, target = (
            entities.get(body["subject_key"]),
            entities.get(body["object_key"]),
        )
        if source is None or target is None:
            raise JournalError("claim references an absent/retired endpoint")
        # Native adapters include these when identity is available. A logical
        # key that now points at another incarnation cannot satisfy a dependency.
        for name, endpoint in (
            ("subject_record_id", source),
            ("object_record_id", target),
        ):
            if name in body and body[name] != endpoint.record_id:
                raise JournalError("claim endpoint incarnation changed")
        properties = dict(body.get("properties", {}))
        properties.update(
            {
                k: v
                for k, v in body.items()
                if k not in {"properties", "active", "labels"}
            }
        )
        predicate = body["predicate"]
        invalid_at = body.get("invalid_at") or properties.get("invalid_at")
        valid_until = body.get("valid_until") or properties.get("valid_until")
        valid_at = body.get("valid_at") or properties.get("valid_at")

        def effective(value):
            if isinstance(value, str):
                value = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return value is not None and value <= now

        if (
            effective(invalid_at)
            or effective(valid_until)
            or (valid_at is not None and not effective(valid_at))
        ):
            continue
        # Native supersession writes bookkeeping relationships outside the
        # agent vocabulary. Their scope and endpoints still need validation.
        system_edge = predicate in SYSTEM_EDGE_TYPES
        if not system_edge:
            edges.append(
                EdgeUpsert(
                    predicate, source.logical_key, target.logical_key, properties
                )
            )
        from potpie_context_engine.core.protocols import PREFIXES

        protocol_edge = any(
            label in PREFIXES
            for endpoint in (source, target)
            for label in endpoint.fields.get("labels", ())
        )
        if protocol_edge and not system_edge:
            if (
                predicate not in {"DOCUMENTS", "AFFECTS"}
                and properties.get("subgraph") != "protocols"
            ):
                raise JournalError("restored protocol relation lacks protocols scope")
            parent_name = {
                "DEFINES_MESSAGE": "protocol_key",
                "HAS_FIELD": "message_key",
            }.get(predicate)
            if (
                parent_name
                and target.fields["properties"].get(parent_name) != source.logical_key
            ):
                raise JournalError(
                    "restored protocol relation contradicts its parent identity"
                )
        spec = definition.edge_types.get(predicate)
        from potpie_context_engine.core.graph_contract import (
            evidence_strength_for_truth,
        )

        if (
            spec is not None
            and spec.singleton
            and evidence_strength_for_truth(properties.get("truth")) == "deterministic"
        ):
            key = (source.record_id, predicate)
            prior = singletons.setdefault(key, target.record_id)
            if prior != target.record_id:
                raise JournalError("singleton cardinality conflict")
        refs = list(body.get("source_refs", properties.get("source_refs", ())))
        source_ref = body.get("source_ref", properties.get("source_ref"))
        if source_ref:
            refs.append(source_ref)
        evidence_values = body.get("evidence", properties.get("evidence", ()))
        if isinstance(evidence_values, str):
            import json

            evidence_values = json.loads(evidence_values)
        for evidence in evidence_values:
            if isinstance(evidence, Mapping) and evidence.get("source_ref"):
                refs.append(evidence["source_ref"])
                if protocol_edge:
                    metadata = evidence.get("metadata", {})
                    if {"source_ref", "authority"} & metadata.keys() or metadata.get(
                        "chunk_id", evidence["source_ref"]
                    ) != evidence["source_ref"]:
                        raise JournalError(
                            "restored protocol evidence overrides canonical identity"
                        )
        for ref in refs:
            if (
                isinstance(ref, str)
                and ref.startswith("potpie://res/")
                and (resource_exists is None or not resource_exists(ref))
            ):
                raise JournalError("resource reference cannot be verified")
    errors = definition.validate_structural_mutations(entity_ops, edges)
    if errors:
        raise JournalError(
            "restored state violates current definition: " + "; ".join(errors[:8])
        )


def plan_inverse(
    *,
    state: JournalState,
    current: Mapping[str, JournalRecord],
    receipts: Sequence[CommitReceipt],
    target_commit_id: str,
    expected_head: str,
    mode: str = "revert",
    validator: Callable[[Mapping[str, JournalRecord]], None],
) -> RestorePlan:
    if not state.rollback_enabled:
        raise JournalError("rollback capability is disabled", code="rollback_disabled")
    if state.resource_guard is not None:
        raise JournalError(
            "resource operation is active or incomplete", code="resource_incomplete"
        )
    if state.head != expected_head:
        raise JournalError("stale HEAD", code="preview_stale")
    if mode not in {"revert", "rollback"}:
        raise JournalError("unknown inverse mode")
    ordered = sorted(receipts, key=lambda r: r.sequence)
    target = next((r for r in ordered if r.commit_id == target_commit_id), None)
    if (
        target is None
        or target.journal_generation != state.generation
        or target.pot_id != state.pot_id
    ):
        raise JournalError(
            "target is outside journal coverage", code="outside_coverage"
        )
    selected = (
        [target]
        if mode == "revert"
        else [r for r in ordered if r.sequence > target.sequence]
    )
    if mode == "rollback":
        expected = list(range(target.sequence + 1, state.sequence + 1))
        if [r.sequence for r in selected] != expected:
            raise JournalError("non-contiguous journal coverage", code="coverage_gap")
        parent = target.commit_id
        for receipt in selected:
            if receipt.parent_commit_id != parent:
                raise JournalError("broken journal parent chain", code="coverage_gap")
            parent = receipt.commit_id
        if parent != state.head:
            raise JournalError("range does not reach HEAD", code="coverage_gap")
    virtual = deepcopy(dict(current))
    required_access = "write"
    for receipt in reversed(selected):
        if (
            receipt.pot_id != state.pot_id
            or receipt.journal_generation != state.generation
            or not receipt.rollback_supported
            or not receipt.diff_complete
        ):
            raise JournalError(
                f"non-revertible barrier: {receipt.commit_id}", code="rollback_barrier"
            )
        required_access = strongest_access(required_access, receipt.required_access)
        for change in receipt.semantic_changes:
            record = virtual.get(change.record_id)
            if record is None or (record.logical_key, record.kind) != (
                change.logical_key,
                change.kind,
            ):
                raise JournalError(
                    "missing or ambiguous record incarnation",
                    code="incarnation_conflict",
                )
            if change.before_record is None and change.after_record is not None:
                if semantic_fields(record) != semantic_fields(change.after_record):
                    raise JournalError(
                        f"creation/lifecycle conflict: {change.record_id}"
                    )
                virtual[record.record_id] = replace(
                    record, fields={**record.fields, "active": False}
                )
            elif change.before_record is not None and change.after_record is None:
                if record.fields.get("active", True):
                    raise JournalError("retired incarnation was already reactivated")
                virtual[record.record_id] = deepcopy(change.before_record)
            else:
                for delta in change.fields:
                    if field_group(delta.path) != "semantic":
                        raise JournalError("inverse contains non-semantic field")
                    if journal_json(read_field(record, delta.path)) != journal_json(
                        delta.after
                    ):
                        raise JournalError(
                            f"field conflict: {record.logical_key}:{'.'.join(delta.path)}"
                        )
                    record = set_field(record, delta.path, delta.before)
                virtual[record.record_id] = record
        # Validate after each virtual commit, rather than each constituent
        # record, so a jointly restored endpoint and claim can be consistent.
        validator(virtual)
    validator(virtual)
    semantic, _ = capture_changes(current, virtual)
    ids = {change.record_id for change in semantic}
    patches = tuple(RestoreRecord(current[key], virtual[key]) for key in sorted(ids))
    inverse_hash = journal_hash(tuple((p.before, p.after) for p in patches))
    if len(patches) > state.limits.max_records:
        raise JournalError("atomic rollback record limit exceeded")
    if (
        len(journal_json(tuple((p.before, p.after) for p in patches)).encode())
        > state.limits.max_bytes
    ):
        raise JournalError("atomic rollback byte limit exceeded")
    return RestorePlan(
        state.pot_id,
        state.generation,
        state.resource_generation,
        expected_head,
        patches,
        tuple(r.commit_id for r in selected),
        required_access,
        inverse_hash,
        mode,
        target_commit_id,
    )
