"""The shared lifecycle policy, independent of transport and storage mechanics."""

from dataclasses import replace
from datetime import datetime, timezone

import pytest

from potpie_context_engine.core.claim_lifecycle import (
    ClaimLifecycleConflict,
    ClaimWriteResolution,
    apply_claim_to_rows,
    close_claim_at,
    resolve_claim_assertion,
)
from potpie_context_engine.core.ports.claim_query import ClaimRow


def instant(month, day=1):
    return datetime(2025, month, day, tzinfo=timezone.utc)


def claim(**changes):
    return replace(
        ClaimRow(
            pot_id="p",
            predicate="OWNED_BY",
            subject_key="service:a",
            object_key="team:a",
            claim_key="claim:a",
            record_id="a",
            valid_at=instant(1),
            fact="same fact",
            truth="observed",
            evidence_strength="deterministic",
            subgraph="structure",
            environment="prod",
            source_system="inventory",
            source_ref="stable",
            properties={"source_identity_stable": True},
        ),
        **changes,
    )


def resolve(incoming, rows, *, anchor=instant(4), **kwargs):
    return resolve_claim_assertion(
        incoming, rows, singleton=True, anchor=anchor, **kwargs
    )


def test_reassertion_preserves_closed_history_and_freezes_new_identity():
    old = claim(invalid_at=instant(2))
    row, resolution = resolve(claim(record_id="b", valid_at=instant(3)), [old])
    assert row.record_id == "b"
    assert apply_claim_to_rows([old], row, resolution) == [old, row]
    assert row.valid_at == instant(3)
    assert ClaimWriteResolution.from_dict(resolution.to_dict()) == resolution


def test_active_refresh_preserves_start_and_scheduled_endpoint():
    old = claim(valid_until=instant(6))
    row, resolution = resolve(claim(record_id="b", valid_at=instant(3)), [old])
    assert (row.record_id, row.valid_at, row.valid_until) == (
        "a",
        instant(1),
        instant(6),
    )
    assert resolution.effective_at == instant(3)
    later, _ = resolve(claim(record_id="b", valid_at=instant(6)), [old])
    assert later.record_id == "b"


@pytest.mark.parametrize(
    "changes",
    [
        {"valid_at": instant(1, 15), "invalid_at": instant(2)},
        {"valid_at": instant(1, 15), "valid_until": instant(2)},
    ],
)
def test_assertion_cannot_rewrite_closed_interval(changes):
    old = claim(**changes)
    with pytest.raises(ClaimLifecycleConflict, match="closed history"):
        resolve(claim(record_id="b", valid_at=instant(1, 20)), [old])


def test_changed_content_and_explicit_endpoint_require_correction():
    for changes in (
        {"fact": "new fact"},
        {"truth": "agent_claim"},
        {"description": "changed"},
        {"valid_until": instant(6)},
    ):
        with pytest.raises(ClaimLifecycleConflict, match="correction"):
            resolve(claim(record_id="b", valid_at=instant(3), **changes), [claim()])


def test_refresh_omitted_content_cannot_erase_existing_truth():
    old = claim(confidence=0.8, description="original description")
    incoming = claim(
        record_id="b",
        valid_at=instant(3),
        claim_key=None,
        fact=None,
        confidence=None,
        description=None,
        truth=None,
    )
    row, _ = resolve(incoming, [old])
    assert (row.claim_key, row.fact, row.confidence, row.description, row.truth) == (
        old.claim_key,
        old.fact,
        old.confidence,
        old.description,
        old.truth,
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"environment": "staging"},
        {"environment": None},
        {"subgraph": "knowledge"},
        {"source_system": "support"},
        {"source_ref": "different"},
        {"properties": {}},
    ],
)
def test_singleton_scope_and_stability_preserve_competing_claims(changes):
    old = claim(**changes)
    incoming = claim(
        record_id="b", claim_key="claim:b", object_key="team:b", valid_at=instant(3)
    )
    row, decision = resolve(incoming, [old])
    assert decision.superseded_record_ids == ()
    assert apply_claim_to_rows([old], row, decision)[0] == old


def test_same_source_singleton_resolves_exact_predecessor():
    old = claim()
    incoming = claim(
        record_id="b", claim_key="claim:b", object_key="team:b", valid_at=instant(3)
    )
    row, decision = resolve(incoming, [old])
    assert decision.superseded_record_ids == ("a",)
    updated = apply_claim_to_rows([old], row, decision)
    assert updated[0].invalid_at == instant(3)
    with pytest.raises(ClaimLifecycleConflict, match="missing|ambiguous"):
        resolve(incoming, [replace(old, record_id="newer")], resolution=decision)


def test_future_overlap_is_rejected_but_touching_intervals_are_allowed():
    future = claim(valid_at=instant(6))
    incoming = claim(
        record_id="b", claim_key="claim:b", object_key="team:b", valid_at=instant(3)
    )
    with pytest.raises(ClaimLifecycleConflict, match="future"):
        resolve(incoming, [future])
    row, decision = resolve(replace(incoming, valid_until=instant(6)), [future])
    assert decision.superseded_record_ids == ()
    assert row.valid_until == instant(6)


def test_ambiguous_and_simultaneous_claims_fail_closed():
    with pytest.raises(ClaimLifecycleConflict, match="ambiguous"):
        resolve(
            claim(record_id="b", valid_at=instant(3)), [claim(), claim(record_id="c")]
        )
    with pytest.raises(ClaimLifecycleConflict, match="simultaneous"):
        resolve(
            claim(record_id="b", object_key="team:b", claim_key="claim:b"), [claim()]
        )


def test_metadata_patch_requires_exact_incarnation_and_preserves_observation():
    old = claim(invalid_at=instant(2), observed_at=instant(1))
    patch = claim(
        record_id="unused",
        valid_at=instant(3),
        observed_at=instant(3),
        properties={
            "source_identity_stable": True,
            "evidence_review_required": True,
            "other": "discard",
        },
    )
    row, decision = resolve(patch, [old], metadata_record_id="a")
    assert decision.metadata_review
    assert row.invalid_at == instant(2) and row.valid_at == instant(1)
    assert row.observed_at == instant(1)
    assert row.properties["evidence_review_required"]
    assert "other" not in row.properties


def test_closure_preserves_first_endpoint_and_rejects_negative_interval():
    old = claim(invalid_at=instant(2))
    assert close_claim_at(old, instant(3), anchor=instant(4)) == old
    with pytest.raises(ClaimLifecycleConflict, match="closed interval"):
        close_claim_at(old, instant(1, 15), anchor=instant(4))
    with pytest.raises(ClaimLifecycleConflict, match="after"):
        close_claim_at(claim(), instant(1), anchor=instant(4))
