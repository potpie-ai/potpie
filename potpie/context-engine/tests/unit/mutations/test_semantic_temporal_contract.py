"""Temporal preparation is frozen before execution and preserves time roles."""

from datetime import datetime, timezone

import pytest

from potpie_context_engine.core.semantic_mutations import SemanticMutationRequest
from potpie_context_engine.core.semantic_mutation_validator import (
    validate_semantic_request,
)
from potpie_context_engine.core.semantic_mutation_lowering import lower_semantic_request
from potpie_context_engine.core.graph_mutations import (
    EdgeUpsert,
    EntityUpsert,
    ProvenanceContext,
)
from potpie_context_engine.core.reconciliation import MutationBatch
from potpie_context_engine.core.reconciliation_config import ReconciliationConfig
from potpie_context_engine.adapters.outbound.graph.backends.in_memory_backend import (
    InMemoryGraphBackend,
)
from potpie_context_engine.core.ports.claim_query import ClaimQueryFilter

ANCHOR = datetime(2026, 8, 1, tzinfo=timezone.utc)
OCCURRED = "2026-01-01T05:30:00+05:30"
OBSERVED = "2026-02-01T00:00:00Z"


def request(operations):
    return SemanticMutationRequest.parse(
        {"pot_id": "p", "operations": operations},
        allow_review_required=True,
        approved_by="user:reviewer",
    )


def claim(**changes):
    return {
        "op": "link_entities",
        "subgraph": "infra_topology",
        "subject": {"key": "service:a", "type": "Service"},
        "predicate": "DEPENDS_ON",
        "object": {"key": "service:b", "type": "Service"},
        "truth": "source_observation",
        "evidence": [{"source_ref": "config:owner"}],
        **changes,
    }


def test_all_generated_claim_times_share_the_preparation_anchor():
    req = request([claim(), claim(object={"key": "service:c", "type": "Service"})])
    plan = validate_semantic_request(req)
    assert plan.ok
    plan.prepared_at = ANCHOR
    lower_semantic_request(req, plan)
    assert len(plan.batch.edge_upserts) == 2
    for edge in plan.batch.edge_upserts:
        for field in ("valid_at", "valid_from", "observed_at"):
            assert edge.properties[field] == ANCHOR.isoformat()


@pytest.mark.parametrize("value", ["2026-01-01", "2026-01-01T00:00:00", "invalid"])
def test_new_semantic_timestamps_require_explicit_offsets(value):
    plan = validate_semantic_request(request([claim(valid_from=value)]))
    assert not plan.ok
    assert any(issue.code == "bad_timestamp" for issue in plan.errors)


def test_both_authored_validity_aliases_are_checked_before_discarding_either():
    assert validate_semantic_request(
        request([claim(valid_from=OCCURRED, valid_at="2026-01-01T00:00:00Z")])
    ).ok
    assert not validate_semantic_request(
        request([claim(valid_from=OCCURRED, valid_at=OBSERVED)])
    ).ok


@pytest.mark.parametrize("kind", ["transition_state", "merge_duplicate_entities"])
def test_occurrence_and_observation_remain_separate_in_audit_projections(kind):
    op = {
        "op": kind,
        "reason": "reviewed",
        "occurred_at": OCCURRED,
        "observed_at": OBSERVED,
    }
    if kind == "transition_state":
        op.update(
            subject={"key": "decision:a", "type": "Decision"},
            from_state="proposed",
            to_state="accepted",
        )
    else:
        op.update(
            subject={"key": "service:a", "type": "Service"},
            object={"key": "service:b", "type": "Service"},
            external_ids={"losing": {"source": "a"}, "winning": {"source": "b"}},
        )
    req = request([op])
    plan = validate_semantic_request(req)
    assert plan.ok
    plan.prepared_at = ANCHOR
    lower_semantic_request(req, plan)
    edge = plan.batch.edge_upserts[0]
    assert edge.properties["occurred_at"] == "2026-01-01T00:00:00+00:00"
    assert edge.properties["observed_at"] == "2026-02-01T00:00:00+00:00"
    key = "state_transition_at" if kind == "transition_state" else "merge_recorded_at"
    projections = [
        entity.properties[key]
        for entity in plan.batch.entity_upserts
        if key in entity.properties
    ]
    assert projections == ["2026-01-01T00:00:00+00:00"]


def test_ontology_time_backfill_uses_event_occurrence_and_receipt_time():
    backend = InMemoryGraphBackend()
    batch = MutationBatch(
        entity_upserts=[
            EntityUpsert("person:a", ("Person",), {}),
            EntityUpsert("activity:a", ("Activity",), {}),
        ],
        edge_upserts=[EdgeUpsert("PERFORMED", "person:a", "activity:a")],
    )
    result = backend.mutation.apply(
        batch,
        expected_pot_id="p",
        provenance_context=ProvenanceContext(
            mutation_id="event",
            event_occurred_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            event_received_at=datetime(2026, 2, 1, tzinfo=timezone.utc),
        ),
        reconciliation_config=ReconciliationConfig(ontology_soft_fail=True),
    )
    assert result.ok
    row = backend.claim_query.find_claims(
        ClaimQueryFilter(pot_id="p", include_invalidated=True)
    )[0]
    assert row.valid_at == datetime(2026, 1, 1, tzinfo=timezone.utc)
    assert row.observed_at == datetime(2026, 2, 1, tzinfo=timezone.utc)


@pytest.mark.parametrize(
    "field", ["observed_at", "occurred_at", "valid_at", "invalid_at", "valid_until"]
)
def test_entity_temporal_fields_use_the_same_strict_write_contract(field):
    from potpie_context_engine.core.errors import MutationBatchValidationError

    backend = InMemoryGraphBackend()
    batch = MutationBatch(
        entity_upserts=[EntityUpsert("service:a", ("Service",), {field: "2026-01-01"})]
    )
    with pytest.raises(MutationBatchValidationError, match="offset"):
        backend.mutation.apply(
            batch,
            expected_pot_id="p",
            provenance_context=ProvenanceContext(mutation_id="bad-time"),
        )
    assert backend.mutation.current_version("p") == 0
    assert (
        backend.mutation.lookup_execution(
            batch, expected_pot_id="p", mutation_id="bad-time"
        ).state
        == "absent"
    )


@pytest.mark.parametrize("field", ["event_occurred_at", "event_received_at"])
def test_unused_provenance_time_still_requires_an_offset(field):
    from potpie_context_engine.core.errors import MutationBatchValidationError

    backend = InMemoryGraphBackend()
    batch = MutationBatch(entity_upserts=[EntityUpsert("service:a", ("Service",), {})])
    with pytest.raises(MutationBatchValidationError, match="offset"):
        backend.mutation.apply(
            batch,
            expected_pot_id="p",
            provenance_context=ProvenanceContext(
                mutation_id="bad-time", **{field: datetime(2026, 1, 1)}
            ),
        )
    assert backend.mutation.current_version("p") == 0


@pytest.mark.parametrize("end", ["2024-12-31T23:59:59Z", "2025-01-01T05:30:00+05:30"])
def test_authored_entity_valid_to_must_follow_start(end):
    from potpie_context_engine.core.claim_codec import normalize_entity_times

    with pytest.raises(ValueError, match="not positive"):
        normalize_entity_times({"valid_from": "2025-01-01T00:00:00Z", "valid_to": end})
