"""Stage and publish one memory graph mutation and its authoritative receipt."""

from __future__ import annotations

from copy import deepcopy
from collections.abc import Sequence
from dataclasses import replace
from datetime import datetime, timezone
from typing import TYPE_CHECKING
import uuid

from potpie_context_engine.core.errors import GraphMutationVersionConflict
from potpie_context_engine.core.claim_corrections import (
    SupersessionBinding,
    prepare_claim_closures,
)
from potpie_context_engine.core.claim_preparation import prepare_assertions_from_rows
from potpie_context_engine.core.graph_mutations import ProvenanceContext
from potpie_context_engine.core.ports.graph.mutation import MutationExecutionState
from potpie_context_engine.core.ports.claim_query import ClaimRow
from potpie_context_engine.core.reconciliation import MutationBatch, MutationResult
from potpie_context_engine.core.reconciliation_config import ReconciliationConfig
from potpie_context_engine.core.reconciliation_validation import (
    validate_reconciliation_plan,
)

from ._mutation_execution import (
    CompletedMutationExecution,
    MutationExecutionAmbiguousError,
    MutationExecutionRegistry,
    complete_execution,
    mutation_batch_fingerprint,
    validate_execution_replay,
    validate_prepared_execution,
    validate_new_preparation,
)
from .memory_journal import MemoryJournal, adopt_store, clone_store

if TYPE_CHECKING:
    from .backends.in_memory_backend import _Mutation


def apply_transaction(
    owner: _Mutation,
    plan: MutationBatch,
    *,
    expected_pot_id: str,
    provenance: ProvenanceContext | None,
    reconciliation_config: ReconciliationConfig | None,
    expected_version: int | None = None,
) -> MutationResult:
    """No live graph, revision, journal or receipt changes before publication."""
    with owner.state_lock:
        original = deepcopy(plan)
        mutation_id = (
            provenance.mutation_id
            if provenance is not None and provenance.mutation_id
            else uuid.uuid4().hex
        )
        validate_prepared_execution(
            original,
            pot_id=expected_pot_id,
            mutation_id=mutation_id,
            provenance=provenance,
        )
        prior = owner.execution_registry.lookup(
            original, expected_pot_id=expected_pot_id, mutation_id=mutation_id
        )
        if prior.state == MutationExecutionState.completed.value:
            assert prior.result is not None
            validate_execution_replay(prior.result, provenance=provenance)
            return prior.result
        if prior.state != MutationExecutionState.absent.value:
            raise MutationExecutionAmbiguousError(
                prior.detail or f"mutation execution is {prior.state}"
            )
        validate_new_preparation(provenance)
        current = owner.revisions.get(expected_pot_id, 0)
        frozen_version = (
            provenance.prepared.expected_version
            if provenance and provenance.prepared
            else None
        )
        for required_version in (expected_version, frozen_version):
            if required_version is not None and current != required_version:
                raise GraphMutationVersionConflict(
                    expected=required_version, current=current
                )
        applied_at = datetime.now(timezone.utc)
        validated = deepcopy(original)
        validate_reconciliation_plan(
            validated,
            expected_pot_id,
            definition=owner.definition,
            config=reconciliation_config,
        )
        writer = replace(
            owner,
            store=clone_store(owner.store),
            on_change=None,
            revisions=dict(owner.revisions),
            journal_data=deepcopy(owner.journal_data),
            execution_registry=MutationExecutionRegistry(
                owner.execution_registry.completed()
            ),
        )
        _prepare_claim_effects(
            writer,
            validated,
            pot_id=expected_pot_id,
            mutation_id=mutation_id,
            provenance=provenance,
            applied_at=applied_at,
        )
        journal = MemoryJournal(writer)
        if journal.journal_state(expected_pot_id) is not None:
            result = journal.apply(
                validated,
                original_plan=original,
                pot_id=expected_pot_id,
                mutation_id=mutation_id,
                provenance=provenance,
                applied_at=applied_at,
            )
        else:
            result = writer._apply_uncached(
                validated,
                expected_pot_id=expected_pot_id,
                mutation_id=mutation_id,
                provenance=provenance,
                applied_at=applied_at,
            )
        result = complete_execution(
            result,
            original,
            pot_id=expected_pot_id,
            version_before=current,
            provenance=provenance,
            record_ids=_affected_claim_ids(
                owner.store.rows, writer.store.rows, expected_pot_id
            ),
            applied_at=applied_at,
        )
        assert result.execution is not None
        writer.revisions[expected_pot_id] = result.execution.version_after
        receipts = (
            *owner.execution_registry.completed(),
            CompletedMutationExecution(
                expected_pot_id,
                mutation_id,
                mutation_batch_fingerprint(original),
                result,
            ),
        )
        # Prepare independent public values before touching the live snapshot.
        returned = deepcopy(result)
        owner.execution_registry.replace_completed(receipts)
        adopt_store(owner.store, writer.store)
        owner.revisions.clear()
        owner.revisions.update(writer.revisions)
        owner.journal_data.clear()
        owner.journal_data.update(writer.journal_data)
    owner._notify()
    return returned


def _prepare_claim_effects(
    writer: _Mutation,
    plan: MutationBatch,
    *,
    pot_id: str,
    mutation_id: str,
    provenance: ProvenanceContext | None,
    applied_at: datetime,
) -> None:
    from .apply_plan import _build_provenance

    source = _build_provenance(
        plan,
        pot_id=pot_id,
        mutation_id=mutation_id,
        context=provenance,
        graph_updated_at=applied_at,
    )
    anchor = (
        provenance.prepared.prepared_at
        if provenance and provenance.prepared
        else applied_at
    )
    bindings: list[SupersessionBinding] = []
    rows = prepare_claim_closures(
        plan,
        (row for row in writer.store.rows if row.pot_id == pot_id),
        anchor=anchor,
        effective_at=source.event_occurred_at or source.valid_from or applied_at,
        bindings=bindings,
        singleton_predicates=writer.definition.singleton_predicates,
    )
    prepare_assertions_from_rows(
        plan,
        rows,
        pot_id=pot_id,
        mutation_id=mutation_id,
        anchor=anchor,
        definition=writer.definition,
        bindings=bindings,
        provenance=source,
    )


def _affected_claim_ids(
    before: Sequence[ClaimRow], after: Sequence[ClaimRow], pot_id: str
) -> tuple[str, ...]:
    old = {row.record_id: row for row in before if row.pot_id == pot_id}
    new = {row.record_id: row for row in after if row.pot_id == pot_id}
    return tuple(
        record_id
        for record_id in old.keys() | new.keys()
        if record_id and old.get(record_id) != new.get(record_id)
    )
