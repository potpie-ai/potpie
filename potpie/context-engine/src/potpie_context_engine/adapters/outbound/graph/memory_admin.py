"""Revisioned administrative operations for memory-backed graph capabilities."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, TYPE_CHECKING

from potpie_context_engine.core.graph_journal import JournalError
from potpie_context_engine.core.graph_snapshot import validate_snapshot_merge
from potpie_context_engine.core.graph_snapshot_mutation import prepare_snapshot_import
from potpie_context_engine.core.ports.graph.snapshot import SnapshotManifest
from potpie_context_engine.core.journal_context import (
    JournalWriteContext,
    journal_write_context,
)
from potpie_context_engine.core.ports.graph.analytics import RepairReport
from potpie_context_engine.core.reconciliation import (
    MutationResult,
    MutationSummary,
)

from .document_key_repair import document_key_finding, wants_document_key_repair


if TYPE_CHECKING:
    from .backends.embedded_backend import _EmbeddedMutation
    from .backends.in_memory_backend import _Analytics, _Mutation


def repair_entities(
    analytics: _Analytics, pot_id: str, targets: Sequence[str]
) -> RepairReport:
    """Select repair effects under the same guard used to execute their batch."""
    mutation = analytics.mutation
    with mutation.state_lock:
        store = analytics.store
        keys = {key for pid, key in store.entity_label_index if pid == pot_id}
        keys.update(key for pid, key in store.entity_property_index if pid == pot_id)
        keys.update(
            key
            for row in store.rows
            if row.pot_id == pot_id and not row.retired
            for key in (row.subject_key, row.object_key)
        )
        from .entity_repair import prepare_entity_repairs

        batch, repaired = prepare_entity_repairs(
            (
                {
                    "key": key,
                    "props": store.entity_property_index.get((pot_id, key), {}),
                    "labels": store.entity_label_index.get((pot_id, key), ()),
                }
                for key in sorted(keys)
            ),
            definition=analytics.definition,
            targets=targets,
        )
        expected_version = mutation.current_version(pot_id)
        findings = (
            (document_key_finding(keys),) if wants_document_key_repair(targets) else ()
        )
    if batch.entity_upserts:
        with journal_write_context(
            JournalWriteContext(origin="repair", required_access="admin")
        ):
            mutation.compare_and_apply(
                batch,
                expected_pot_id=pot_id,
                expected_version=expected_version,
            )
    parts = [
        f"repaired {count} {target.replace('_', ' ')}"
        for target, count in repaired.items()
    ]
    parts.extend(finding.detail for finding in findings if finding.detail)
    return RepairReport(
        pot_id=pot_id,
        targets=tuple(targets),
        repaired=repaired,
        detail=", ".join(parts)
        if parts
        else "in_memory projections are computed on read; nothing to rebuild",
        findings=findings,
    )


def _require_unjournaled(writer: _Mutation, pot_id: str, operation: str) -> None:
    if writer.journal_data.get("states", {}).get(pot_id) is not None:
        raise JournalError(
            f"{operation} is unsupported while journal capture is active"
        )


def apply_reset(writer: _Mutation, *, pot_id: str, mutation_id: str) -> MutationResult:
    """Run only on the isolated writer owned by ``apply_transaction``."""
    _require_unjournaled(writer, pot_id, "reset")
    before = len(writer.store.rows)
    writer.store.rows[:] = [row for row in writer.store.rows if row.pot_id != pot_id]
    for index in (writer.store.entity_label_index, writer.store.entity_property_index):
        for key in [key for key in index if key[0] == pot_id]:
            del index[key]
    return MutationResult(
        ok=True,
        mutation_id=mutation_id,
        mutation_summary=MutationSummary(),
        administrative_summary={"removed_claims": before - len(writer.store.rows)},
    )


def apply_snapshot_import(
    writer: _Mutation,
    payload: Mapping[str, Any],
    *,
    pot_id: str,
    mutation_id: str,
) -> MutationResult:
    """Validate exact merge effects before changing the isolated writer."""
    from .backends.in_memory_backend import _Snapshot, _row_from_dict

    _require_unjournaled(writer, pot_id, "snapshot import")
    existing = _Snapshot(writer.store, writer).export_data(pot_id=pot_id)
    validate_snapshot_merge(
        existing_entities=existing["entities"],
        existing_claims=existing["claims"],
        incoming=payload,
    )
    entity_keys = {row["key"] for row in existing["entities"]}
    claim_keys = {row["claim_key"] for row in existing["claims"]}
    entities = [row for row in payload["entities"] if row["key"] not in entity_keys]
    claims = [row for row in payload["claims"] if row["claim_key"] not in claim_keys]
    prepared_rows = [_row_from_dict(pot_id, row) for row in claims]
    for entity in entities:
        writer.store.set_entity_label(
            pot_id=pot_id, entity_key=entity["key"], labels=entity["labels"]
        )
        writer.store.set_entity_properties(
            pot_id=pot_id, entity_key=entity["key"], properties=entity["properties"]
        )
    writer.store.rows.extend(prepared_rows)
    return MutationResult(
        ok=True,
        mutation_id=mutation_id,
        mutation_summary=MutationSummary(),
        administrative_summary={
            "imported_entities": len(entities),
            "imported_claims": len(claims),
        },
    )


def import_snapshot(
    mutation: _Mutation | _EmbeddedMutation,
    *,
    pot_id: str,
    payload: Mapping[str, Any],
    mutation_id: str | None,
) -> SnapshotManifest:
    """Freeze portable content and execute it through the ordinary write port."""
    batch, provenance = prepare_snapshot_import(
        pot_id=pot_id,
        payload=payload,
        expected_version=mutation.current_version(pot_id),
        mutation_id=mutation_id,
    )
    result = mutation.apply(
        batch, expected_pot_id=pot_id, provenance_context=provenance
    )
    assert batch.snapshot_import is not None
    return SnapshotManifest(
        pot_id=pot_id,
        location="<inline>",
        format_version="2",
        entity_count=len(batch.snapshot_import["entities"]),
        claim_count=len(batch.snapshot_import["claims"]),
        metadata={
            "mutation_id": result.mutation_id,
            "execution": result.execution.to_dict()
            if result.execution is not None
            else None,
            **result.administrative_summary,
        },
    )
