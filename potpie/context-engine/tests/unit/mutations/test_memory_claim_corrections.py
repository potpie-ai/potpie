"""Exact corrections share one frozen transaction in memory/JSON journal modes."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone

import pytest

from potpie_context_engine.adapters.outbound.graph._mutation_execution import (
    MutationExecutionReuseError,
)
from potpie_context_engine.adapters.outbound.graph.backends.embedded_backend import (
    EmbeddedGraphBackend,
)
from potpie_context_engine.adapters.outbound.graph.backends.in_memory_backend import (
    InMemoryGraphBackend,
    _Mutation,
)
from potpie_context_engine.core.claim_lifecycle import ClaimLifecycleConflict
from potpie_context_engine.core.claim_preparation import prepare_claim_assertions
from potpie_context_engine.core.errors import GraphMutationVersionConflict
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
from potpie_context_engine.core.ports.claim_query import ClaimQueryFilter
from potpie_context_engine.core.reconciliation import MutationBatch

pytestmark = pytest.mark.unit
POT = "correction-pot"
JAN = datetime(2026, 1, 1, tzinfo=timezone.utc)
FEB = datetime(2026, 2, 1, tzinfo=timezone.utc)
MAR = datetime(2026, 3, 1, tzinfo=timezone.utc)
APR = datetime(2026, 4, 1, tzinfo=timezone.utc)
MAY = datetime(2026, 5, 1, tzinfo=timezone.utc)


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


def _rows(backend):
    return backend.claim_query.find_claims(
        ClaimQueryFilter(pot_id=POT, include_invalidated=True, include_retired=True)
    )


def _edge(*, start=JAN, end=None, target="team:a", key="claim:owner", **extra):
    return EdgeUpsert(
        "OWNED_BY",
        "service:api",
        target,
        {
            "claim_key": key,
            "subgraph": "code_topology",
            "source_system": "fixture",
            "source_ref": "fixture:inventory",
            "valid_at": start.isoformat(),
            "valid_until": end.isoformat() if end else None,
            "fact": "API ownership",
            "confidence": 1.0,
            **extra,
        },
    )


def _assertion(edge):
    return MutationBatch(
        entity_upserts=[
            EntityUpsert("service:api", ("Entity", "Service")),
            EntityUpsert(edge.to_entity_key, ("Entity", "Team")),
        ],
        edge_upserts=[edge],
    )


def _apply(backend, plan, mid, *, context=None):
    return backend.mutation.apply(
        plan,
        expected_pot_id=POT,
        provenance_context=context or ProvenanceContext(mutation_id=mid),
    )


def _freeze(backend, plan, mid, *, anchor=FEB):
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
            request_fingerprint=f"request:{mid}",
            batch_fingerprint=batch_fingerprint(plan),
            expected_version=backend.mutation.current_version(POT),
            prepared_at=anchor,
        ),
    )


def _invalidation(row, *, end=MAR, **extra):
    return InvalidationOp(
        None,
        None,
        "reviewed correction",
        valid_to=end.isoformat(),
        target_claim_keys=(row.claim_key,),
        target_claim_record_ids=(row.record_id,),
        **extra,
    )


@pytest.mark.parametrize("existing", [False, True])
def test_explicit_property_projection_requires_existing_entity_and_preserves_labels(
    backend, existing
):
    if existing:
        _apply(backend, _assertion(_edge()), "initial")
    version = backend.mutation.current_version(POT)
    plan = MutationBatch(
        entity_upserts=[
            EntityUpsert(
                "team:a", (), {"review_note": "reviewed"}, preserve_labels=True
            )
        ]
    )
    if existing:
        result = _apply(backend, plan, "projection")
        assert result.mutation_summary.entity_upserts_applied == 1
        assert backend.claim_query.entity_labels(pot_id=POT, entity_keys=["team:a"])[
            "team:a"
        ] == ("Entity", "Team")
        assert (
            backend.claim_query.entity_properties(pot_id=POT, entity_key="team:a")[
                "review_note"
            ]
            == "reviewed"
        )
        assert backend.mutation.current_version(POT) == version + 1
    else:
        with pytest.raises(
            ClaimLifecycleConflict, match="entity projection target is missing"
        ):
            _apply(backend, plan, "projection")
        assert (
            backend.claim_query.entity_labels(pot_id=POT, entity_keys=["team:a"]) == {}
        )
        assert backend.mutation.current_version(POT) == version
        assert (
            backend.mutation.lookup_execution(
                plan, expected_pot_id=POT, mutation_id="projection"
            ).state
            == "absent"
        )


def test_property_projection_preserves_unmentioned_authored_display_fields(backend):
    _apply(
        backend,
        MutationBatch(
            entity_upserts=[
                EntityUpsert(
                    "team:a",
                    ("Entity", "Team"),
                    {"summary": "old", "description": "authored", "name": "Alpha"},
                )
            ]
        ),
        "initial",
    )
    plan = MutationBatch(
        entity_upserts=[
            EntityUpsert("team:a", (), {"summary": "new"}, preserve_labels=True)
        ]
    )
    result = _apply(backend, plan, "display-patch")
    properties = backend.claim_query.entity_properties(pot_id=POT, entity_key="team:a")
    assert properties["summary"] == "new"
    assert properties["description"] == "authored"
    assert properties["name"] == "Alpha"
    assert backend.claim_query.entity_labels(pot_id=POT, entity_keys=["team:a"])[
        "team:a"
    ] == ("Entity", "Team")
    assert _apply(backend, plan, "display-patch") == result
    assert backend.mutation.current_version(POT) == 2
    if isinstance(backend, EmbeddedGraphBackend):
        restarted = EmbeddedGraphBackend(home=backend.home)
        assert (
            restarted.claim_query.entity_properties(pot_id=POT, entity_key="team:a")
            == properties
        )


def test_raw_provenance_defaults_are_persisted_from_shared_preparation(backend):
    edge = _edge()
    for name in ("valid_at", "source_system", "source_ref", "confidence"):
        edge.properties.pop(name)
    plan = _assertion(edge)
    plan.confidence = 0.7
    context = ProvenanceContext(
        mutation_id="provenance",
        source_system="inventory",
        source_ref="inventory:event",
        event_occurred_at=MAR,
        event_received_at=APR,
    )
    result = _apply(backend, plan, "provenance", context=context)
    row = _rows(backend)[0]
    assert (row.valid_at, row.observed_at, row.confidence) == (MAR, APR, 0.7)
    assert (row.source_system, row.source_ref) == ("inventory", "inventory:event")
    assert row.properties["source_identity_stable"] is True
    assert row.mutation_id == result.mutation_id == "provenance"
    assert edge.resolution is None
    assert "valid_at" not in edge.properties


@pytest.mark.parametrize("frozen", [False, True])
def test_exact_empty_correction_has_no_projection_or_supersession(backend, frozen):
    _apply(backend, _assertion(_edge()), "initial")
    before = deepcopy(_rows(backend))
    properties = backend.claim_query.entity_properties(pot_id=POT, entity_key="team:a")
    plan = MutationBatch(
        invalidations=[
            InvalidationOp(
                "team:a",
                None,
                "empty reviewed selection",
                superseded_by_key="team:missing",
                valid_to=MAR.isoformat(),
                target_claim_keys=(),
                target_claim_record_ids=(),
            )
        ]
    )
    context = _freeze(backend, plan, "empty") if frozen else None
    result = _apply(backend, plan, "empty", context=context)
    assert result.mutation_summary.invalidations_applied == 0
    assert result.mutation_summary.entity_upserts_applied == 0
    assert result.mutation_summary.edge_upserts_applied == 0
    assert result.execution.affected_claim_record_ids == ()
    assert _rows(backend) == before
    assert (
        backend.claim_query.entity_properties(pot_id=POT, entity_key="team:a")
        == properties
    )
    assert plan.entity_upserts == []
    assert plan.edge_upserts == []


@pytest.mark.parametrize("frozen", [False, True])
def test_closed_history_correction_changes_only_exact_incarnation(backend, frozen):
    _apply(backend, _assertion(_edge(end=FEB)), "january")
    first = _rows(backend)[0]
    _apply(backend, _assertion(_edge(start=APR)), "april")
    second = next(row for row in _rows(backend) if row.record_id != first.record_id)
    plan = MutationBatch(invalidations=[_invalidation(first)])
    context = _freeze(backend, plan, "correct") if frozen else None
    result = _apply(backend, plan, "correct", context=context)
    corrected = next(row for row in _rows(backend) if row.record_id == first.record_id)
    assert corrected == replace(first, invalid_at=MAR, valid_until=MAR)
    assert (
        next(row for row in _rows(backend) if row.record_id == second.record_id)
        == second
    )
    assert result.execution.affected_claim_record_ids == (first.record_id,)
    version = backend.mutation.current_version(POT)
    assert _apply(backend, plan, "correct", context=context) == result
    assert backend.mutation.current_version(POT) == version


def test_delayed_prepared_closure_uses_frozen_interval(backend, monkeypatch):
    _apply(backend, _assertion(_edge(end=APR)), "scheduled")
    original = _rows(backend)[0]
    plan = MutationBatch(invalidations=[_invalidation(original)])
    context = _freeze(backend, plan, "delayed", anchor=FEB)
    frozen = plan.invalidations[0].closures[0]
    assert frozen.before.valid_until == APR
    assert frozen.after.valid_until == MAR

    def forbid_reclosing(*args, **kwargs):
        raise AssertionError("frozen execution attempted to choose a closure endpoint")

    monkeypatch.setattr(
        "potpie_context_engine.core.claim_corrections.close_claim_at", forbid_reclosing
    )
    receipt = _apply(backend, plan, "delayed", context=context)
    assert _rows(backend) == [replace(original, invalid_at=MAR, valid_until=MAR)]
    assert receipt.execution.applied_at > MAY
    assert _apply(backend, plan, "delayed", context=context) == receipt


@pytest.mark.parametrize("frozen", [False, True])
def test_edge_deletion_closes_before_same_scope_replacement_assertion(backend, frozen):
    _apply(backend, _assertion(_edge()), "initial")
    original = _rows(backend)[0]
    plan = _assertion(_edge(start=MAR, target="team:b", key="claim:new"))
    plan.edge_deletes.append(EdgeDelete("OWNED_BY", "service:api", "team:a"))
    context = (
        _freeze(backend, plan, "delete-replace", anchor=MAR)
        if frozen
        else ProvenanceContext(mutation_id="delete-replace", event_occurred_at=MAR)
    )
    receipt = _apply(backend, plan, "delete-replace", context=context)
    history = _rows(backend)
    assert next(
        row for row in history if row.record_id == original.record_id
    ) == replace(original, invalid_at=MAR)
    new = next(row for row in history if row.record_id != original.record_id)
    assert (new.object_key, new.valid_at, new.invalid_at) == ("team:b", MAR, None)
    assert receipt.mutation_summary.edge_deletes_applied == 1
    assert receipt.mutation_summary.edge_upserts_applied == 1


def test_later_same_key_incarnation_cannot_replace_frozen_correction_target(backend):
    _apply(backend, _assertion(_edge()), "initial")
    original = _rows(backend)[0]
    plan = MutationBatch(invalidations=[_invalidation(original, end=FEB)])
    context = _freeze(backend, plan, "preview", anchor=FEB)
    _apply(
        backend,
        deepcopy(plan),
        "close",
        context=replace(
            context,
            mutation_id="close",
            prepared=replace(context.prepared, mutation_id="close"),
        ),
    )
    _apply(backend, _assertion(_edge(start=MAR)), "reassert")
    before = deepcopy(_rows(backend))
    with pytest.raises(GraphMutationVersionConflict):
        _apply(backend, plan, "preview", context=context)
    assert _rows(backend) == before
    assert backend.mutation.current_version(POT) == 3
    assert (
        backend.mutation.lookup_execution(
            plan, expected_pot_id=POT, mutation_id="preview"
        ).state
        == "absent"
    )


def test_frozen_correction_validates_original_interval_before_any_publication(backend):
    _apply(backend, _assertion(_edge()), "initial")
    original = _rows(backend)[0]
    plan = MutationBatch(invalidations=[_invalidation(original)])
    context = _freeze(backend, plan, "frozen")

    def alter_interval(inner):
        inner.store.rows[:] = [replace(row, invalid_at=APR) for row in inner.store.rows]

    if isinstance(backend, EmbeddedGraphBackend):
        backend._transact(alter_interval)
    else:
        alter_interval(backend)
    before = deepcopy(_rows(backend))
    with pytest.raises(
        ClaimLifecycleConflict, match="frozen correction interval changed"
    ):
        _apply(backend, plan, "frozen", context=context)
    assert _rows(backend) == before
    assert backend.mutation.current_version(POT) == 1
    assert (
        backend.mutation.lookup_execution(
            plan, expected_pot_id=POT, mutation_id="frozen"
        ).state
        == "absent"
    )


def _supersession(backend):
    _apply(backend, _assertion(_edge()), "initial")
    original = _rows(backend)[0]
    replacement = _edge(start=MAR, target="team:b", key="claim:replacement")
    return original, MutationBatch(
        entity_upserts=[EntityUpsert("team:b", ("Entity", "Team"))],
        edge_upserts=[replacement],
        invalidations=[
            InvalidationOp(
                "team:a",
                None,
                "ownership corrected",
                superseded_by_key="team:b",
                valid_to=MAR.isoformat(),
            )
        ],
    )


@pytest.mark.parametrize("frozen", [False, True])
def test_supersession_closes_before_assertion_and_preserves_projection_labels(
    backend, frozen
):
    original, plan = _supersession(backend)
    context = _freeze(backend, plan, "supersede", anchor=MAR) if frozen else None
    receipt = _apply(backend, plan, "supersede", context=context)
    history = _rows(backend)
    old = next(row for row in history if row.record_id == original.record_id)
    replacement = next(row for row in history if row.claim_key == "claim:replacement")
    system = next(row for row in history if row.predicate == "SUPERSEDES")
    assert old == replace(original, invalid_at=MAR)
    assert (replacement.valid_at, replacement.invalid_at, replacement.valid_until) == (
        MAR,
        None,
        None,
    )
    assert system.properties["supersedes_record_id"] == old.record_id
    assert system.properties["replacement_record_id"] == replacement.record_id
    assert (system.subject_key, system.object_key, system.valid_at) == (
        "team:b",
        "team:a",
        MAR,
    )
    assert backend.claim_query.entity_labels(pot_id=POT, entity_keys=["team:a"])[
        "team:a"
    ] == ("Entity", "Team")
    assert (
        backend.claim_query.entity_properties(pot_id=POT, entity_key="team:a")[
            "invalidated"
        ]
        is True
    )
    assert set(receipt.execution.affected_claim_record_ids) == {
        row.record_id for row in history
    }
    assert _apply(backend, plan, "supersede", context=context) == receipt
    if isinstance(backend, EmbeddedGraphBackend):
        restarted = EmbeddedGraphBackend(home=backend.home)
        assert _rows(restarted) == history
        assert _apply(restarted, plan, "supersede", context=context) == receipt


def test_failure_after_closure_projection_and_system_write_rolls_back(
    backend, monkeypatch
):
    original, plan = _supersession(backend)
    before = deepcopy(_rows(backend))
    properties = backend.claim_query.entity_properties(pot_id=POT, entity_key="team:a")
    disk = (
        (backend.home / "graph.json").read_bytes()
        if isinstance(backend, EmbeddedGraphBackend)
        else None
    )
    method = _Mutation._upsert_claim_row

    def fail_after_system(self, row, **kwargs):
        method(self, row, **kwargs)
        if row.predicate == "SUPERSEDES":
            raise RuntimeError("injected after correction effects")

    with monkeypatch.context() as patch:
        patch.setattr(_Mutation, "_upsert_claim_row", fail_after_system)
        with pytest.raises(RuntimeError, match="after correction effects"):
            _apply(backend, plan, "failure")
    assert _rows(backend) == before == [original]
    assert backend.mutation.current_version(POT) == 1
    assert (
        backend.claim_query.entity_properties(pot_id=POT, entity_key="team:a")
        == properties
    )
    assert backend.claim_query.entity_labels(pot_id=POT, entity_keys=["team:b"]) == {}
    assert (
        backend.mutation.lookup_execution(
            plan, expected_pot_id=POT, mutation_id="failure"
        ).state
        == "absent"
    )
    if disk is not None:
        assert (backend.home / "graph.json").read_bytes() == disk
        assert _rows(EmbeddedGraphBackend(home=backend.home)) == before


@pytest.mark.parametrize("journal", [False, True])
def test_correction_persistence_failure_keeps_original_intervals_and_receipt(
    tmp_path, monkeypatch, journal
):
    backend = EmbeddedGraphBackend(home=tmp_path)
    if journal:
        backend.journal.activate(pot_id=POT)
    original, plan = _supersession(backend)
    disk = (backend.home / "graph.json").read_bytes()
    context = _freeze(backend, plan, "disk-failure", anchor=MAR)

    def fail_persist(*args, **kwargs):
        raise OSError("injected correction persistence failure")

    with monkeypatch.context() as patch:
        patch.setattr(EmbeddedGraphBackend, "_write_state", fail_persist)
        with pytest.raises(OSError, match="correction persistence failure"):
            _apply(backend, plan, "disk-failure", context=context)
    assert (backend.home / "graph.json").read_bytes() == disk
    restarted = EmbeddedGraphBackend(home=backend.home)
    assert _rows(backend) == _rows(restarted) == [original]
    assert restarted.mutation.current_version(POT) == 1
    assert (
        restarted.mutation.lookup_execution(
            plan, expected_pot_id=POT, mutation_id="disk-failure"
        ).state
        == "absent"
    )
    assert restarted.claim_query.entity_labels(pot_id=POT, entity_keys=["team:b"]) == {}


@pytest.mark.parametrize("frozen", [False, True])
def test_exact_correction_preserves_other_source_and_environment_scopes(
    backend, frozen
):
    scopes = [
        (None, "inventory"),
        ("prod", "inventory"),
        ("stage", "inventory"),
        (None, "other"),
    ]
    for index, (environment, source) in enumerate(scopes):
        _apply(
            backend,
            _assertion(_edge(environment=environment, source_ref=f"fixture:{source}")),
            f"scope:{index}",
        )
    before = deepcopy(_rows(backend))
    selected = next(
        row
        for row in before
        if row.environment is None and row.source_ref == "fixture:inventory"
    )
    plan = MutationBatch(invalidations=[_invalidation(selected)])
    context = _freeze(backend, plan, "scope-correction") if frozen else None
    result = _apply(backend, plan, "scope-correction", context=context)
    after = {row.record_id: row for row in _rows(backend)}
    assert len(after) == 4
    for row in before:
        assert after[row.record_id] == (
            replace(row, invalid_at=MAR) if row == selected else row
        )
    assert result.execution.affected_claim_record_ids == (selected.record_id,)


@pytest.mark.parametrize("failure", ["duplicate-target", "wrong-key", "overlap"])
def test_invalid_correction_targets_fail_without_receipt(backend, failure):
    _apply(backend, _assertion(_edge(end=FEB)), "initial")
    original = _rows(backend)[0]
    _apply(backend, _assertion(_edge(start=APR)), "later")
    plan = MutationBatch(invalidations=[_invalidation(original)])
    if failure == "duplicate-target":
        plan.edge_deletes.append(EdgeDelete("OWNED_BY", "service:api", "team:a"))
        message = "duplicate correction incarnation target"
    elif failure == "wrong-key":
        plan.invalidations[0].target_claim_keys = ("claim:other",)
        message = "missing or outside its selected scope"
    else:
        plan.invalidations[0].valid_to = MAY.isoformat()
        message = "overlap"
    before = deepcopy(_rows(backend))
    with pytest.raises(ClaimLifecycleConflict, match=message):
        _apply(backend, plan, "invalid")
    assert _rows(backend) == before
    assert backend.mutation.current_version(POT) == 2
    assert (
        backend.mutation.lookup_execution(
            plan, expected_pot_id=POT, mutation_id="invalid"
        ).state
        == "absent"
    )


def test_raw_cannot_inject_frozen_empty_closures(backend):
    plan = MutationBatch(
        edge_deletes=[EdgeDelete("OWNED_BY", "service:api", "team:a", closures=())]
    )
    with pytest.raises(MutationExecutionReuseError, match="prepared"):
        _apply(backend, plan, "injection")
    assert backend.mutation.current_version(POT) == 0
    assert (
        backend.mutation.lookup_execution(
            plan, expected_pot_id=POT, mutation_id="injection"
        ).state
        == "absent"
    )
