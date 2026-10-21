"""Frozen corrections retain exact identities, intervals and shared effects."""

from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone

import pytest

from potpie_context_engine.core.claim_corrections import (
    ClaimClosureResolution,
    prepare_claim_closures,
)
from potpie_context_engine.core.claim_lifecycle import ClaimLifecycleConflict
from potpie_context_engine.core.graph_mutations import EdgeDelete, InvalidationOp
from potpie_context_engine.core.graph_plans import (
    mutation_batch_to_dict,
    mutation_batch_from_dict,
    GraphMutationDiff,
)
from potpie_context_engine.core.ports.claim_query import ClaimRow
from potpie_context_engine.core.reconciliation import MutationBatch


def time(month):
    return datetime(2025, month, 1, tzinfo=timezone.utc)


def row(**changes):
    return replace(
        ClaimRow(
            pot_id="p",
            predicate="OWNED_BY",
            subject_key="service:a",
            object_key="team:a",
            claim_key="c",
            record_id="r",
            valid_at=time(1),
            invalid_at=time(2),
            valid_until=time(2),
            source_system="inventory",
            source_ref="stable",
            evidence_strength="deterministic",
            environment="prod",
            subgraph="infra_topology",
            properties={"source_identity_stable": True},
        ),
        **changes,
    )


def correction(identity="r", end=3):
    return InvalidationOp(
        None,
        None,
        "reviewed correction",
        valid_to=time(end).isoformat(),
        target_claim_record_ids=(identity,),
    )


def prepare(batch, rows):
    return prepare_claim_closures(
        batch, rows, anchor=time(5), bindings=[], singleton_predicates=("OWNED_BY",)
    )


def test_closed_interval_extension_changes_both_ends_and_round_trips():
    batch = MutationBatch(invalidations=[correction()])
    old = row()
    (corrected,) = prepare(batch, [old])
    assert corrected.valid_at == old.valid_at
    assert corrected.invalid_at == corrected.valid_until == time(3)
    (closure,) = batch.invalidations[0].closures
    assert ClaimClosureResolution.from_dict(closure.to_dict()) == closure
    assert mutation_batch_from_dict(mutation_batch_to_dict(batch)) == batch
    diff = GraphMutationDiff.from_batch(batch)
    assert GraphMutationDiff.from_dict(diff.to_dict()) == diff
    assert diff.claim_closures == (closure,)
    assert prepare_claim_closures(batch, [old], anchor=time(12), bindings=[]) == [
        corrected
    ]


@pytest.mark.parametrize(
    "changes",
    [
        {"record_id": "later"},
        {"claim_key": "other"},
        {"invalid_at": time(3)},
        {"retired": True},
    ],
)
def test_execution_does_not_reselect_or_rewrite_changed_target(changes):
    batch = MutationBatch(invalidations=[correction()])
    old = row()
    prepare(batch, [old])
    with pytest.raises(ClaimLifecycleConflict):
        prepare(batch, [replace(old, **changes)])


def test_frozen_empty_never_selects_later_rows_or_generates_effects():
    batch = MutationBatch(
        invalidations=[
            InvalidationOp(
                "team:a",
                None,
                "none",
                superseded_by_key="team:b",
                target_claim_record_ids=(),
            )
        ]
    )
    prepare(batch, [])
    assert batch.invalidations[0].closures == ()
    assert prepare(batch, [row()]) == [row()]
    assert not batch.entity_upserts and not batch.edge_upserts


@pytest.mark.parametrize(
    "changes", [{}, {"object_key": "team:b", "claim_key": "different"}]
)
def test_history_extension_cannot_overlap_logical_or_singleton_successor(changes):
    first = row()
    successor = row(
        record_id="next", valid_at=time(3), invalid_at=None, valid_until=None, **changes
    )
    with pytest.raises(ClaimLifecycleConflict, match="overlap"):
        prepare(MutationBatch(invalidations=[correction(end=4)]), [first, successor])


def test_joint_corrections_validate_final_history_not_transient_overlap():
    first = row(invalid_at=time(4), valid_until=time(4))
    second = row(
        record_id="next", valid_at=time(3), invalid_at=time(4), valid_until=time(4)
    )
    batch = MutationBatch(invalidations=[correction("next", 5), correction("r", 3)])
    final = prepare(batch, [first, second])
    assert final[0].invalid_at == final[1].valid_at == time(3)
    assert final[1].invalid_at == time(5)


@pytest.mark.parametrize(
    "scope",
    [
        {"environment": "stage"},
        {"source_ref": "other"},
        {"source_system": "other"},
        {"subgraph": "knowledge"},
    ],
)
def test_historical_overlap_in_another_scope_remains_visible(scope):
    first = row()
    other = row(
        record_id="other", valid_at=time(3), invalid_at=None, valid_until=None, **scope
    )
    final = prepare(MutationBatch(invalidations=[correction(end=4)]), [first, other])
    assert other in final


def test_duplicate_target_across_delete_and_correction_rejects():
    batch = MutationBatch(
        edge_deletes=[EdgeDelete("OWNED_BY", "service:a", "team:a")],
        invalidations=[correction()],
    )
    with pytest.raises(ClaimLifecycleConflict, match="duplicate"):
        prepare(batch, [row()])


def test_raw_delete_preserves_closed_history_and_does_not_add_properties():
    old = row()
    batch = MutationBatch(edge_deletes=[EdgeDelete("OWNED_BY", "service:a", "team:a")])
    assert prepare(batch, [old]) == [old]
    assert prepare(batch, [old])[0] is old


def test_frozen_target_must_stay_within_endpoint_axes():
    batch = MutationBatch(invalidations=[correction()])
    prepare(batch, [row()])
    changed = deepcopy(batch)
    changed.invalidations[0].target_entity_key = "service:other"
    with pytest.raises(ClaimLifecycleConflict):
        prepare(changed, [row()])


@pytest.mark.parametrize("keys", [("missing",), ("k", "k")])
def test_authored_claim_keys_cannot_silently_disappear(keys):
    batch = MutationBatch(
        invalidations=[InvalidationOp(None, None, "review", target_claim_keys=keys)]
    )
    with pytest.raises(ClaimLifecycleConflict):
        prepare(batch, [row()])


def test_keyless_incarnation_preparation_can_be_revalidated_unchanged():
    old = row(claim_key=None)
    batch = MutationBatch(invalidations=[correction()])
    first = prepare(batch, [old])
    assert batch.invalidations[0].target_claim_keys is None
    assert batch.invalidations[0].target_claim_record_ids == (old.record_id,)
    restored = mutation_batch_from_dict(mutation_batch_to_dict(batch))
    assert prepare(restored, [old]) == first


def test_key_and_incarnation_selectors_must_agree_after_intersection():
    batch = MutationBatch(
        invalidations=[
            InvalidationOp(
                None,
                None,
                "review",
                target_claim_keys=("c", "other"),
                target_claim_record_ids=("r",),
            )
        ]
    )
    with pytest.raises(ClaimLifecycleConflict, match="missing|scope"):
        prepare(
            batch, [row(), row(record_id="r2", claim_key="other", object_key="team:b")]
        )


def test_multiple_incarnations_keep_unique_derived_logical_selector():
    original = [
        row(),
        row(record_id="r2", valid_at=time(3), invalid_at=time(4), valid_until=time(4)),
    ]
    batch = MutationBatch(invalidations=[InvalidationOp("service:a", None, "review")])
    assert prepare(batch, original) == original
    assert batch.invalidations[0].target_claim_keys == ("c",)
    assert prepare(batch, original) == original


def test_derived_supersession_observation_uses_frozen_preparation_time():
    op = replace(correction(), superseded_by_key="team:b")
    batch = MutationBatch(invalidations=[op])
    original = row()
    prepare(batch, [original])
    effect = batch.edge_upserts[0]
    assert effect.properties["valid_at"] == time(3).isoformat()
    assert effect.properties["observed_at"] == time(5).isoformat()
    frozen = mutation_batch_to_dict(batch)
    prepare_claim_closures(batch, [original], anchor=time(12), bindings=[])
    assert mutation_batch_to_dict(batch) == frozen
