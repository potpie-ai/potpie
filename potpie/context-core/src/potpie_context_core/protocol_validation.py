"""Semantic protocol identity and extraction guards (not payload validation).

Used at the semantic write seam, with an optional claim port for detecting
same-identity extraction conflicts. Rejected plans remain reviewable in history.
"""

from __future__ import annotations

import json
import re
from typing import Any

from potpie_context_core.protocols import (
    IDENTITY_PROPERTIES,
    PREFIXES,
    normalize_properties,
    protocol_entity_key,
)
from potpie_context_core.semantic_mutations import SemanticMutationValidationIssue


def protocol_issues(request, claim_query=None) -> list[SemanticMutationValidationIssue]:
    issues = []
    pending: dict[str, dict[str, Any]] = {}
    labels = {prefix: label for label, prefix in PREFIXES.items()}
    for index, op in enumerate(request.operations):

        def error(message, code="invalid_protocol_identity", op_index=index):
            issues.append(
                SemanticMutationValidationIssue(
                    code=code, message=message, severity="error", op_index=op_index
                )
            )

        protocol_op = any(
            ref is not None
            and (ref.type in PREFIXES or ref.key.partition(":")[0] in labels)
            for ref in (op.subject, op.object)
        )
        if protocol_op:
            for evidence in op.evidence:
                if {"source_ref", "authority"} & evidence.metadata.keys():
                    error(
                        "Evidence metadata cannot override source_ref or authority",
                        "invalid_protocol_evidence",
                    )
                if (
                    evidence.metadata.get("chunk_id", evidence.source_ref)
                    != evidence.source_ref
                ):
                    error(
                        "Evidence chunk_id must equal its canonical source_ref",
                        "invalid_protocol_evidence",
                    )
        for ref in (op.subject, op.object):
            if ref is None:
                continue
            label = ref.type or labels.get(ref.key.partition(":")[0])
            if label not in PREFIXES:
                continue
            if not re.fullmatch(PREFIXES[label] + r":[0-9a-f]{64}", ref.key):
                error(f"{label} key must use the protocol typed tuple digest")
            # DOCUMENTS and AFFECTS may live in their owning subgraphs.
            if op.subgraph != "protocols" and op.predicate not in {
                "DOCUMENTS",
                "AFFECTS",
            }:
                error(
                    "protocol writes require explicit subgraph: protocols",
                    "missing_protocol_subgraph",
                )
            props = normalize_properties(ref.properties, label)
            before = pending.get(ref.key)
            if before is None:
                before = (
                    normalize_properties(
                        claim_query.entity_properties(
                            pot_id=request.pot_id, entity_key=ref.key
                        ),
                        label,
                    )
                    if claim_query is not None
                    else {}
                )
            identity = {**before, **props}
            try:
                if protocol_entity_key(label, identity) != ref.key:
                    error(
                        "protocol properties do not match the immutable identity tuple"
                    )
            except ValueError as exc:
                error(str(exc))
            conflicts = [
                k
                for k, v in props.items()
                if k in before and _encoded(before[k]) != _encoded(v)
            ]
            if conflicts:
                error(
                    f"{ref.key} conflicts with stored/pending extraction properties {sorted(conflicts)}; preserve it in inbox/history, or use an evidenced patch_entity correction",
                    "protocol_extraction_conflict",
                )
            pending[ref.key] = {**before, **props}
            if op.op == "patch_entity" and ref is op.subject:
                if not op.evidence:
                    error(
                        "protocol extraction corrections require evidence",
                        "missing_evidence",
                    )
                if IDENTITY_PROPERTIES[label] & op.patch.keys():
                    error(
                        "protocol identity cannot be patched; create the new revision instead"
                    )
                pending[ref.key].update(op.patch)
        if op.predicate == "DEFINES_MESSAGE" and op.object and op.subject:
            parent = pending.get(op.object.key, {}).get("protocol_key")
            if parent is not None and parent != op.subject.key:
                error("DEFINES_MESSAGE must agree with ProtocolMessage.protocol_key")
        if op.predicate == "HAS_FIELD" and op.object and op.subject:
            parent = pending.get(op.object.key, {}).get("message_key")
            if parent is not None and parent != op.subject.key:
                error("HAS_FIELD must agree with ProtocolField.message_key")
    return issues


def _encoded(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def protocol_content_readback(backend, *, pot_id, record) -> dict[str, Any]:
    """Compare public entity properties, including native/JSON containers."""
    batch = record.lowered_batch
    checked = []
    mismatches = []
    if batch is not None:
        for entity in batch.entity_upserts:
            label = next((label for label in entity.labels if label in PREFIXES), None)
            if label is None:
                continue
            expected = normalize_properties(entity.properties, label)
            actual = normalize_properties(
                backend.claim_query.entity_properties(
                    pot_id=pot_id, entity_key=entity.entity_key
                ),
                label,
            )
            differences = [
                key
                for key, value in expected.items()
                if key not in actual or _encoded(value) != _encoded(actual[key])
            ]
            checked.append(entity.entity_key)
            if differences:
                mismatches.append(
                    {"entity_key": entity.entity_key, "properties": differences}
                )
    return {"checked_entities": checked, "mismatches": mismatches}
