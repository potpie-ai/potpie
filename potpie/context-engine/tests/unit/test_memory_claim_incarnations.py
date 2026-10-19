"""Claim history and singleton scope agree across memory/JSON and journal modes."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone

import pytest

from potpie_context_engine.adapters.outbound.graph.backends.embedded_backend import (
    EmbeddedGraphBackend,
)
from potpie_context_engine.adapters.outbound.graph.backends.in_memory_backend import (
    InMemoryGraphBackend,
)
from potpie_context_engine.core.claim_lifecycle import ClaimLifecycleConflict
from potpie_context_engine.core.claim_preparation import prepare_claim_assertions
from potpie_context_engine.core.graph_contract import make_claim_key
from potpie_context_engine.core.graph_mutations import (
    EdgeDelete,
    EdgeUpsert,
    EntityUpsert,
    InvalidationOp,
    ProvenanceContext,
)
from potpie_context_engine.core.mutation_execution import (
    PreparedMutation,
    batch_fingerprint,
)
from potpie_context_engine.core.mutation_plan_store import InMemoryGraphPlanStore
from potpie_context_engine.core.ports.claim_query import ClaimQueryFilter
from potpie_context_engine.core.reconciliation import MutationBatch
from potpie_context_engine.core.workbench_service import GraphWorkbenchService

pytestmark = pytest.mark.unit
POT = "incarnation-pot"
JAN = datetime(2026, 1, 1, tzinfo=timezone.utc)
FEB = datetime(2026, 2, 1, tzinfo=timezone.utc)
MAR = datetime(2026, 3, 1, tzinfo=timezone.utc)
FUTURE = datetime(2027, 1, 1, tzinfo=timezone.utc)


@pytest.fixture(
    params=[("memory", False), ("memory", True), ("json", False), ("json", True)]
)
def backend(request, tmp_path):
    profile, journal = request.param
    result = (
        InMemoryGraphBackend()
        if profile == "memory"
        else EmbeddedGraphBackend(home=tmp_path)
    )
    if journal:
        result.journal.activate(pot_id=POT)
    return result


def _edge(*, at=JAN, target="team:a", key="claim:owner", **properties):
    return EdgeUpsert(
        "OWNED_BY",
        "service:api",
        target,
        {
            "claim_key": key,
            "subgraph": "code_topology",
            "source_system": "fixture",
            "source_ref": "fixture:inventory",
            "truth": "authoritative_fact",
            "fact": "API ownership",
            "description": "API ownership",
            "confidence": 1.0,
            "valid_from": at.isoformat(),
            **properties,
        },
    )


def _batch(edge):
    return MutationBatch(
        entity_upserts=[
            EntityUpsert("service:api", ("Entity", "Service")),
            EntityUpsert(edge.to_entity_key, ("Entity", "Team")),
        ],
        edge_upserts=[edge],
    )


def _apply(backend, mid, plan, *, at=None):
    return backend.mutation.apply(
        plan,
        expected_pot_id=POT,
        provenance_context=ProvenanceContext(mutation_id=mid, event_occurred_at=at),
    )


def _rows(backend, *, at=None, record_id=None):
    return backend.claim_query.find_claims(
        ClaimQueryFilter(
            pot_id=POT,
            as_of=at,
            include_invalidated=at is None,
            include_retired=True,
            record_id_in=(record_id,) if record_id else (),
        )
    )


def _retract(backend, *, at=FEB, mid="retract"):
    return _apply(
        backend,
        mid,
        MutationBatch(
            invalidations=[
                InvalidationOp(
                    None,
                    None,
                    "withdrawn",
                    valid_to=at.isoformat(),
                    target_claim_keys=("claim:owner",),
                )
            ]
        ),
    )


def _freeze(backend, plan, mid, *, anchor=MAR):
    prepare_claim_assertions(
        plan,
        pot_id=POT,
        mutation_id=mid,
        anchor=anchor,
        query=backend.claim_query,
        definition=backend.definition,
    )
    return ProvenanceContext(
        mutation_id=mid,
        prepared=PreparedMutation(
            pot_id=POT,
            mutation_id=mid,
            request_fingerprint="request:" + mid,
            batch_fingerprint=batch_fingerprint(plan),
            expected_version=backend.mutation.current_version(POT),
            prepared_at=anchor,
        ),
    )


def test_reassertion_keeps_closed_incarnation_and_gap_and_replays(backend):
    initial = _batch(_edge())
    _apply(backend, "january", initial)
    first = _rows(backend)[0]
    _retract(backend)
    march = _batch(_edge(at=MAR))
    receipt = _apply(backend, "march", march)
    history = _rows(backend)
    assert len(history) == 2
    old = next(row for row in history if row.record_id == first.record_id)
    new = next(row for row in history if row.record_id != first.record_id)
    assert (old.valid_at, old.invalid_at) == (JAN, FEB)
    assert (new.valid_at, new.invalid_at) == (MAR, None)
    assert _rows(backend, at=JAN) == [old]
    assert _rows(backend, at=FEB) == []
    assert _rows(backend, at=datetime(2026, 2, 15, tzinfo=timezone.utc)) == []
    assert _rows(backend, at=MAR) == [new]
    assert _rows(backend, record_id=old.record_id) == [old]
    assert receipt.execution.affected_claim_record_ids == (new.record_id,)
    before = backend.mutation.current_version(POT)
    assert _apply(backend, "march", march) == receipt
    assert backend.mutation.current_version(POT) == before
    assert _rows(backend) == history
    if isinstance(backend, EmbeddedGraphBackend):
        restarted = EmbeddedGraphBackend(home=backend.home)
        assert _rows(restarted) == history
        assert _apply(restarted, "march", march) == receipt


@pytest.mark.parametrize("end_field", ["valid_until", "invalid_at"])
def test_active_refresh_preserves_identity_start_and_scheduled_end(backend, end_field):
    _apply(backend, "initial", _batch(_edge(**{end_field: FUTURE.isoformat()})))
    before = _rows(backend)[0]
    _apply(backend, "refresh", _batch(_edge(at=MAR, observed_at=MAR.isoformat())))
    after = _rows(backend)[0]
    assert len(_rows(backend)) == 1
    assert (after.record_id, after.valid_at, after.invalid_at, after.valid_until) == (
        before.record_id,
        JAN,
        before.invalid_at,
        before.valid_until,
    )
    assert after.observed_at == MAR
    assert after.fact == before.fact


@pytest.mark.parametrize("end_field", ["valid_until", "invalid_at"])
def test_reassertion_at_scheduled_end_creates_new_incarnation(backend, end_field):
    _apply(backend, "initial", _batch(_edge(**{end_field: FUTURE.isoformat()})))
    old = _rows(backend)[0]
    _apply(backend, "at-end", _batch(_edge(at=FUTURE)))
    history = _rows(backend)
    assert len(history) == 2
    assert next(row for row in history if row.record_id == old.record_id) == old
    current = _rows(backend, at=FUTURE)
    assert len(current) == 1 and current[0].record_id != old.record_id
    assert current[0].valid_at == FUTURE


@pytest.mark.parametrize(
    "change",
    [
        {"environment": "prod"},
        {"source_ref": "fixture:other"},
        {"source_system": "other-system"},
        {"subgraph": "infra_topology"},
    ],
)
def test_singleton_keeps_competing_scopes_visible(backend, change):
    _apply(backend, "first", _batch(_edge()))
    original = _rows(backend)[0]
    _apply(
        backend,
        "other",
        _batch(_edge(at=MAR, target="team:b", key="claim:other", **change)),
    )
    rows = _rows(backend, at=MAR)
    assert len(rows) == 2
    assert next(row for row in rows if row.record_id == original.record_id) == original


def test_prod_and_stage_are_distinct_singleton_scopes(backend):
    _apply(backend, "prod", _batch(_edge(environment="prod")))
    _apply(
        backend, "stage", _batch(_edge(at=MAR, target="team:b", environment="stage"))
    )
    assert {row.environment for row in _rows(backend, at=MAR)} == {"prod", "stage"}


def test_same_stable_source_replacement_closes_exact_predecessor(backend):
    _apply(backend, "first", _batch(_edge()))
    first = _rows(backend)[0]
    _apply(
        backend,
        "competing",
        _batch(_edge(at=FEB, target="team:c", source_ref="fixture:competitor")),
    )
    competitor = next(row for row in _rows(backend) if row.object_key == "team:c")
    receipt = _apply(
        backend, "replacement", _batch(_edge(at=MAR, target="team:b", key="claim:new"))
    )
    assert _rows(backend, record_id=first.record_id)[0].invalid_at == MAR
    assert _rows(backend, record_id=competitor.record_id) == [competitor]
    assert {row.object_key for row in _rows(backend, at=MAR)} == {"team:b", "team:c"}
    assert first.record_id in receipt.execution.affected_claim_record_ids
    if backend.journal.journal_state(POT) is not None:
        commit = backend.journal.get_receipt(pot_id=POT, commit_id="replacement")
        assert {
            item.record_id for item in commit.semantic_changes if item.kind == "claim"
        } == set(receipt.execution.affected_claim_record_ids)


def test_unstable_source_cannot_automatically_replace(backend):
    _apply(backend, "first", _batch(_edge(source_identity_stable=False)))
    _apply(
        backend,
        "second",
        _batch(_edge(at=MAR, target="team:b", source_identity_stable=False)),
    )
    assert len(_rows(backend, at=MAR)) == 2


def test_generated_raw_source_does_not_become_stable(backend):
    edge = _edge()
    del edge.properties["source_system"]
    del edge.properties["source_ref"]
    _apply(backend, "generated", _batch(edge))
    row = _rows(backend)[0]
    assert row.source_ref.startswith("event:")
    assert row.properties["source_identity_stable"] is False


@pytest.mark.parametrize(
    "properties",
    [
        {"fact": "changed conclusion"},
        {"description": "changed wording"},
        {"truth": "agent_claim"},
        {"confidence": 0.5},
    ],
)
def test_truth_bearing_refresh_requires_correction_and_rolls_back(backend, properties):
    _apply(backend, "first", _batch(_edge()))
    original = _rows(backend)
    version = backend.mutation.current_version(POT)
    changed = _batch(_edge(at=MAR, **properties))
    changed.entity_upserts.append(
        EntityUpsert("service:unpublished", ("Entity", "Service"))
    )
    with pytest.raises(ClaimLifecycleConflict, match="explicit correction"):
        _apply(backend, "changed", changed)
    assert _rows(backend) == original
    assert backend.mutation.current_version(POT) == version
    assert (
        backend.mutation.lookup_execution(
            changed, expected_pot_id=POT, mutation_id="changed"
        ).state
        == "absent"
    )
    assert (
        backend.claim_query.entity_properties(
            pot_id=POT, entity_key="service:unpublished"
        )
        == {}
    )


def test_backdated_overlap_does_not_rewrite_closed_history(backend):
    _apply(backend, "first", _batch(_edge()))
    _retract(backend)
    before = _rows(backend)
    with pytest.raises(ClaimLifecycleConflict, match="closed history"):
        _apply(
            backend,
            "backdated",
            _batch(_edge(at=datetime(2026, 1, 15, tzinfo=timezone.utc))),
        )
    assert _rows(backend) == before


def test_metadata_review_targets_closed_physical_incarnation_only(backend):
    _apply(backend, "first", _batch(_edge(observed_at=JAN.isoformat())))
    _retract(backend)
    old = _rows(backend)[0]
    _apply(backend, "new", _batch(_edge(at=MAR)))
    new = next(row for row in _rows(backend) if row.record_id != old.record_id)
    edge = _edge(evidence_review_required=True, evidence_review_reason="source changed")
    edge.metadata_review_record_id = old.record_id
    receipt = _apply(backend, "review", _batch(edge))
    reviewed = _rows(backend, record_id=old.record_id)[0]
    assert (
        replace(reviewed, mutation_id=old.mutation_id, properties=old.properties) == old
    )
    assert reviewed.properties["evidence_review_required"] is True
    assert _rows(backend, record_id=new.record_id) == [new]
    assert receipt.execution.affected_claim_record_ids == (old.record_id,)


def test_frozen_resolution_is_consumed_without_reselection(backend, monkeypatch):
    _apply(backend, "first", _batch(_edge()))
    original = _rows(backend)[0]
    plan = _batch(_edge(at=MAR))
    context = _freeze(backend, plan, "frozen")
    chosen = plan.edge_upserts[0].resolution
    assert chosen.record_id == original.record_id and chosen.existing

    def refuse_selection(*args, **kwargs):
        raise AssertionError("execution reselected a frozen assertion")

    monkeypatch.setattr(
        "potpie_context_engine.core.claim_lifecycle._choose", refuse_selection
    )
    first = backend.mutation.apply(
        plan, expected_pot_id=POT, provenance_context=context
    )
    assert _rows(backend)[0].valid_at == JAN
    assert (
        backend.mutation.apply(plan, expected_pot_id=POT, provenance_context=context)
        == first
    )


@pytest.mark.parametrize("at", [JAN, datetime(2025, 12, 31, tzinfo=timezone.utc)])
def test_delete_cannot_create_nonpositive_interval(backend, at):
    _apply(backend, "first", _batch(_edge()))
    before = _rows(backend)
    plan = MutationBatch(edge_deletes=[EdgeDelete("OWNED_BY", "service:api", "team:a")])
    with pytest.raises(ClaimLifecycleConflict, match="after the incarnation start"):
        _apply(backend, "invalid-delete", plan, at=at)
    assert _rows(backend) == before


def test_delete_soft_closes_history_and_repeated_delete_keeps_endpoint(backend):
    _apply(backend, "first", _batch(_edge()))
    original = _rows(backend)[0]
    plan = MutationBatch(edge_deletes=[EdgeDelete("OWNED_BY", "service:api", "team:a")])
    first = _apply(backend, "delete", plan, at=FEB)
    closed = _rows(backend)[0]
    assert (closed.record_id, closed.invalid_at, closed.retired) == (
        original.record_id,
        FEB,
        False,
    )
    assert _rows(backend, at=JAN) == [closed]
    assert _rows(backend, at=FEB) == []
    assert first.mutation_summary.edge_deletes_applied == 1
    second = _apply(backend, "delete-again", plan, at=MAR)
    assert second.mutation_summary.edge_deletes_applied == 0
    assert _rows(backend) == [closed]
    assert _apply(backend, "delete", plan, at=FEB) == first


def test_semantic_preparation_and_raw_writer_share_active_refresh(backend):
    key = make_claim_key(
        pot_id=POT,
        subgraph="code_topology",
        subject_key="service:api",
        predicate="OWNED_BY",
        object_component="team:a",
        discriminator="fixture:inventory",
    )
    _apply(backend, "raw", _batch(_edge(key=key)))
    old = _rows(backend)[0]
    service = GraphWorkbenchService(
        backend=backend, plan_store=InMemoryGraphPlanStore()
    )
    proposal = service.propose(
        {
            "created_by": {"surface": "fixture"},
            "operations": [
                {
                    "op": "assert_claim",
                    "subgraph": "code_topology",
                    "predicate": "OWNED_BY",
                    "subject": {"key": "service:api", "type": "Service"},
                    "object": {"key": "team:a", "type": "Team"},
                    "truth": "authoritative_fact",
                    "description": "API ownership",
                    "valid_from": MAR.isoformat(),
                    "evidence": [{"source_ref": "fixture:inventory"}],
                }
            ],
        },
        pot_id=POT,
    )
    assert proposal.ok, proposal.detail
    result = service.commit(
        proposal.plan_id, pot_id=POT, approved_by="test", verify=False
    )
    assert result.ok, result.detail
    assert len(_rows(backend)) == 1
    current = _rows(backend)[0]
    assert (current.record_id, current.valid_at) == (old.record_id, JAN)


def test_snapshot_roundtrip_preserves_parallel_incarnations(backend, tmp_path):
    _apply(backend, "first", _batch(_edge()))
    _retract(backend)
    _apply(backend, "new", _batch(_edge(at=MAR)))
    history = _rows(backend)
    payload = backend.snapshot.export_data(pot_id=POT)
    assert len(payload["claims"]) == 2
    assert {row["record_id"] for row in payload["claims"]} == {
        row.record_id for row in history
    }
    fresh = (
        EmbeddedGraphBackend(home=tmp_path / "restored")
        if isinstance(backend, EmbeddedGraphBackend)
        else InMemoryGraphBackend()
    )
    imported = fresh.snapshot.import_data(pot_id=POT, payload=payload)
    restored = _rows(fresh)
    assert {row.record_id: row for row in restored} == {
        row.record_id: row for row in history
    }
    assert (
        fresh.snapshot.import_data(pot_id=POT, payload=payload).metadata
        == imported.metadata
    )
    assert fresh.mutation.current_version(POT) == 1
    assert _rows(fresh, at=FEB) == []


@pytest.mark.parametrize("journal", [False, True])
def test_failed_embedded_reassertion_preserves_closed_history(
    tmp_path, monkeypatch, journal
):
    backend = EmbeddedGraphBackend(home=tmp_path)
    if journal:
        backend.journal.activate(pot_id=POT)
    _apply(backend, "first", _batch(_edge()))
    _retract(backend)
    history = deepcopy(_rows(backend))
    before = (tmp_path / "graph.json").read_bytes()
    plan = _batch(_edge(at=MAR))

    def fail(*args, **kwargs):
        raise OSError("incarnation persistence failed")

    with monkeypatch.context() as patch:
        patch.setattr(EmbeddedGraphBackend, "_write_state", fail)
        with pytest.raises(OSError, match="incarnation persistence failed"):
            _apply(backend, "retry", plan)
    assert (tmp_path / "graph.json").read_bytes() == before
    restarted = EmbeddedGraphBackend(home=tmp_path)
    assert _rows(restarted) == history
    assert restarted.mutation.current_version(POT) == 2
    assert (
        restarted.mutation.lookup_execution(
            plan, expected_pot_id=POT, mutation_id="retry"
        ).state
        == "absent"
    )
    _apply(restarted, "retry", plan)
    assert len(_rows(restarted)) == 2
    assert restarted.mutation.current_version(POT) == 3


def test_raw_authored_context_preserves_source_and_event_times(backend):
    edge = _edge()
    for key in ("source_system", "source_ref", "valid_from"):
        del edge.properties[key]
    backend.mutation.apply(
        _batch(edge),
        expected_pot_id=POT,
        provenance_context=ProvenanceContext(
            mutation_id="authored-context",
            source_system="inventory",
            source_ref="inventory:api",
            event_occurred_at=JAN,
            event_received_at=FEB,
        ),
    )
    row = _rows(backend)[0]
    assert (row.source_system, row.source_ref, row.valid_at, row.observed_at) == (
        "inventory",
        "inventory:api",
        JAN,
        FEB,
    )
    assert row.properties["source_identity_stable"] is True


def test_review_flag_without_exact_target_cannot_edit_closed_history(backend):
    _apply(backend, "first", _batch(_edge()))
    _retract(backend)
    before = _rows(backend)
    with pytest.raises(ClaimLifecycleConflict, match="closed history"):
        _apply(
            backend, "ambiguous-review", _batch(_edge(evidence_review_required=True))
        )
    assert _rows(backend) == before


def test_backdated_repeat_delete_cannot_move_closed_endpoint(backend):
    _apply(backend, "first", _batch(_edge()))
    deletion = MutationBatch(
        edge_deletes=[EdgeDelete("OWNED_BY", "service:api", "team:a")]
    )
    _apply(backend, "delete", deletion, at=FEB)
    before = _rows(backend)
    with pytest.raises(ClaimLifecycleConflict, match="closed interval"):
        _apply(
            backend,
            "backdated-delete",
            deletion,
            at=datetime(2026, 1, 15, tzinfo=timezone.utc),
        )
    assert _rows(backend) == before


@pytest.mark.parametrize("existing", [(), ("service:api",), ("team:a",)])
def test_assertion_rejects_missing_endpoint_without_publishing(backend, existing):
    plan = MutationBatch(
        entity_upserts=[EntityUpsert(key, ("Entity",)) for key in existing],
        edge_upserts=[_edge()],
    )
    with pytest.raises(ClaimLifecycleConflict, match="endpoint is missing"):
        _apply(backend, "dangling", plan)
    assert _rows(backend) == []
    assert (
        backend.claim_query.entity_labels(
            pot_id=POT, entity_keys=("service:api", "team:a")
        )
        == {}
    )
    assert backend.mutation.current_version(POT) == 0
    assert (
        backend.mutation.lookup_execution(
            plan, expected_pot_id=POT, mutation_id="dangling"
        ).state
        == "absent"
    )
    state = backend.journal.journal_state(POT)
    if state is not None:
        assert state.sequence == 0


def test_assertion_uses_endpoints_already_present_without_upserting_them(backend):
    _apply(
        backend,
        "endpoints",
        MutationBatch(entity_upserts=_batch(_edge()).entity_upserts),
    )
    _apply(backend, "edge-only", MutationBatch(edge_upserts=[_edge()]))
    assert len(_rows(backend)) == 1


@pytest.mark.parametrize(
    "selector", ["missing", "duplicate-request", "duplicate-record", "empty"]
)
def test_exact_invalidation_validates_physical_identity_cardinality(backend, selector):
    _apply(backend, "initial", _batch(_edge()))
    original = _rows(backend)[0]
    identities = {
        "missing": ("missing",),
        "duplicate-request": (original.record_id, original.record_id),
        "duplicate-record": (original.record_id,),
        "empty": (),
    }[selector]
    if selector == "duplicate-record":
        duplicate = replace(
            original, properties={**original.properties, "corrupt_duplicate": True}
        )
        if isinstance(backend, EmbeddedGraphBackend):
            backend._transact(lambda inner: inner.store.add(duplicate))
        else:
            backend.store.add(duplicate)
    before = deepcopy(_rows(backend))
    version = backend.mutation.current_version(POT)
    plan = MutationBatch(
        invalidations=[
            InvalidationOp(
                None,
                None,
                "exact invalidation",
                valid_to=FEB.isoformat(),
                target_claim_keys=("claim:owner",),
                target_claim_record_ids=identities,
            )
        ]
    )
    if selector == "empty":
        receipt = _apply(backend, "exact", plan)
        assert receipt.mutation_summary.invalidations_applied == 0
        assert _rows(backend) == before
        return
    with pytest.raises(ClaimLifecycleConflict, match="missing or ambiguous"):
        _apply(backend, "exact", plan)
    assert _rows(backend) == before
    assert backend.mutation.current_version(POT) == version
    assert (
        backend.mutation.lookup_execution(
            plan, expected_pot_id=POT, mutation_id="exact"
        ).state
        == "absent"
    )


@pytest.mark.parametrize("mode", ["raw", "frozen"])
@pytest.mark.parametrize("entity_field", ["record_id", "uuid"])
def test_assertion_id_cannot_collide_with_entity_identity(
    backend, monkeypatch, mode, entity_field
):
    from uuid import UUID

    plan = _batch(_edge())
    if mode == "frozen":
        context = _freeze(backend, plan, "collision")
        collision = plan.edge_upserts[0].resolution.record_id
    else:
        collision = "a" * 32
        context = ProvenanceContext(mutation_id="collision")
        monkeypatch.setattr(
            "potpie_context_engine.adapters.outbound.graph.backends.in_memory_backend.uuid.uuid4",
            lambda: UUID(hex=collision),
        )
    plan.entity_upserts[0].properties[entity_field] = collision
    if context.prepared is not None:
        context = replace(
            context,
            prepared=replace(
                context.prepared, batch_fingerprint=batch_fingerprint(plan)
            ),
        )
    with pytest.raises(
        ClaimLifecycleConflict, match="collides with an entity identity"
    ):
        backend.mutation.apply(plan, expected_pot_id=POT, provenance_context=context)
    assert _rows(backend) == []
    assert backend.mutation.current_version(POT) == 0
    assert (
        backend.mutation.lookup_execution(
            plan, expected_pot_id=POT, mutation_id="collision"
        ).state
        == "absent"
    )


@pytest.mark.parametrize("frozen", [False, True])
def test_assertion_rejects_duplicate_id_outside_logical_candidate_scope(
    backend, frozen
):
    _apply(backend, "first", _batch(_edge()))
    old = _rows(backend)[0]
    plan = _batch(_edge(at=MAR))
    context = (
        _freeze(backend, plan, "duplicate")
        if frozen
        else ProvenanceContext(mutation_id="duplicate")
    )
    duplicate = replace(
        old,
        predicate="DEPENDS_ON",
        claim_key="claim:unrelated",
        source_ref="other-source",
    )
    if isinstance(backend, EmbeddedGraphBackend):
        backend._transact(lambda inner: inner.store.add(duplicate))
    else:
        backend.store.add(duplicate)
    before = deepcopy(_rows(backend))
    with pytest.raises(
        ClaimLifecycleConflict, match="missing, duplicated, or already exists"
    ):
        backend.mutation.apply(plan, expected_pot_id=POT, provenance_context=context)
    assert _rows(backend) == before
    assert backend.mutation.current_version(POT) == 1
    assert (
        backend.mutation.lookup_execution(
            plan, expected_pot_id=POT, mutation_id="duplicate"
        ).state
        == "absent"
    )


@pytest.mark.parametrize("reactivation", [False, None])
def test_retired_endpoint_rejects_assertion_until_same_batch_reactivation(
    backend, reactivation
):
    entities = _batch(_edge()).entity_upserts
    _apply(backend, "entities", MutationBatch(entity_upserts=entities))

    def retire(inner):
        inner.store.entity_property_index[(POT, "team:a")]["retired"] = True

    if isinstance(backend, EmbeddedGraphBackend):
        backend._transact(retire)
    else:
        retire(backend)
    plan = MutationBatch(edge_upserts=[_edge()])
    before = backend.mutation.current_version(POT)
    with pytest.raises(ClaimLifecycleConflict, match="missing or retired"):
        _apply(backend, "retired-endpoint", plan)
    assert _rows(backend) == []
    assert backend.mutation.current_version(POT) == before
    assert (
        backend.mutation.lookup_execution(
            plan, expected_pot_id=POT, mutation_id="retired-endpoint"
        ).state
        == "absent"
    )
    _apply(backend, "entity-observer", MutationBatch(entity_upserts=[entities[1]]))
    assert backend.claim_query.entity_properties(pot_id=POT, entity_key="team:a") == {}
    version = backend.mutation.current_version(POT)
    with pytest.raises(ClaimLifecycleConflict, match="missing or retired"):
        _apply(backend, "implicit-reactivation", _batch(_edge()))
    assert backend.mutation.current_version(POT) == version
    entities[1].properties["retired"] = reactivation
    _apply(
        backend,
        "reactivate",
        MutationBatch(entity_upserts=[entities[1]], edge_upserts=[_edge()]),
    )
    assert len(_rows(backend, at=MAR)) == 1
    assert not backend.claim_query.entity_properties(
        pot_id=POT, entity_key="team:a"
    ).get("retired", False)


@pytest.mark.parametrize("mode", ["raw", "frozen"])
def test_historical_entity_uuid_is_preserved_without_owning_record_identity(
    backend, monkeypatch, mode
):
    from uuid import UUID

    plan = _batch(_edge())
    if mode == "frozen":
        context = _freeze(backend, plan, "historical-alias")
        identity = plan.edge_upserts[0].resolution.record_id
    else:
        identity = "b" * 32
        context = ProvenanceContext(mutation_id="historical-alias")
        monkeypatch.setattr(
            "potpie_context_engine.adapters.outbound.graph.backends.in_memory_backend.uuid.uuid4",
            lambda: UUID(hex=identity),
        )
    plan.entity_upserts[0].properties.update(record_id="entity:source", uuid=identity)
    plan.entity_upserts[1].properties.update(record_id="entity:target")
    if context.prepared is not None:
        context = replace(
            context,
            prepared=replace(
                context.prepared, batch_fingerprint=batch_fingerprint(plan)
            ),
        )
    result = backend.mutation.apply(
        plan, expected_pot_id=POT, provenance_context=context
    )
    assert result.ok
    assert _rows(backend)[0].record_id == identity
    properties = backend.claim_query.entity_properties(
        pot_id=POT, entity_key="service:api"
    )
    assert properties["record_id"] == "entity:source"
    assert properties["uuid"] == identity
