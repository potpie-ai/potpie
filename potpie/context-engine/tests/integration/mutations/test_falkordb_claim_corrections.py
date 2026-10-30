# ruff: noqa: F811 - imported pytest fixtures are injected as test parameters.
"""Prepared3 exact corrections on real server/Lite ordinary/journal paths."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest
from redis.exceptions import ResponseError

from potpie_context_engine.adapters.outbound.graph import (
    falkordb_atomic,
    falkordb_journal,
)
from potpie_context_engine.adapters.outbound.graph._mutation_execution import (
    MutationExecutionReuseError,
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
from potpie_context_engine.core.reconciliation import (
    MutationBatch,
)

from tests.integration.mutations.test_falkordb_claim_lifecycle import (
    JAN,
    FEB,
    MAR,
    APR,
    apply,
    batch,
    close,
    edge,
    history,
    raw,
    seed_endpoints,
    store as store,
)
from tests.integration.backends.test_falkordb_reliability import (
    graph as graph,
    snapshot,
)

pytestmark = pytest.mark.integration


def prepared(store, plan, mid, *, anchor=APR):
    prepare_claim_assertions(
        plan,
        pot_id="p",
        mutation_id=mid,
        anchor=anchor,
        query=store.claim_query,
        definition=store.definition,
    )
    return ProvenanceContext(
        mutation_id=mid,
        prepared=PreparedMutation(
            "p",
            mid,
            "correction-request",
            batch_fingerprint(plan),
            store.mutation.current_version("p"),
            anchor,
        ),
    )


def execute(store, plan, context):
    return store.mutation.compare_and_apply(
        plan,
        expected_pot_id="p",
        expected_version=context.prepared.expected_version,
        provenance_context=context,
    )


def retract(row, point, **properties):
    return InvalidationOp(
        None,
        None,
        "source correction",
        valid_to=point.isoformat(),
        target_claim_keys=(row.claim_key,),
        target_claim_record_ids=(row.record_id,),
        **properties,
    )


@pytest.mark.parametrize("frozen", (False, True), ids=("raw", "prepared"))
def test_exact_empty_selection_has_no_projection_or_system_edge(store, graph, frozen):
    apply(store, batch(edge()), "seed")
    apply(
        store,
        MutationBatch(entity_upserts=[EntityUpsert("team:blue", ("Team",))]),
        "replacement-entity",
    )
    plan = MutationBatch(
        invalidations=[
            InvalidationOp(
                "team:red",
                None,
                "empty selection",
                valid_to=MAR.isoformat(),
                target_claim_keys=(),
                superseded_by_key="team:blue",
            )
        ]
    )
    context = (
        prepared(store, plan, "empty")
        if frozen
        else ProvenanceContext(mutation_id="empty")
    )
    if frozen:
        assert plan.invalidations[0].closures == ()
    before = snapshot(graph)[:2]
    fingerprint = batch_fingerprint(plan)
    result = (
        execute(store, plan, context)
        if frozen
        else store.mutation.apply(plan, expected_pot_id="p", provenance_context=context)
    )
    assert snapshot(graph)[:2] == before
    assert not result.execution.affected_claim_record_ids
    assert result.mutation_summary.entity_upserts_applied == 0
    assert result.mutation_summary.edge_upserts_applied == 0
    assert result.mutation_summary.invalidations_applied == 0
    assert result.execution.batch_fingerprint == fingerprint == batch_fingerprint(plan)
    assert (
        len(
            store.claim_query.find_claims(
                ClaimQueryFilter(pot_id="p", include_invalidated=True)
            )
        )
        == 1
    )


def test_exact_physical_selection_corrects_only_one_same_key_incarnation(store, graph):
    apply(store, batch(edge()), "old")
    old = history(store, include_invalidated=True)[0]
    close(store, old)
    apply(store, batch(edge(MAR)), "new")
    current = next(
        row
        for row in history(store, include_invalidated=True)
        if row.record_id != old.record_id
    )
    preserved = raw(graph)[current.record_id]
    point = JAN + timedelta(days=14)
    plan = MutationBatch(invalidations=[retract(old, point)])
    context = prepared(store, plan, "correct-old")
    closure = plan.invalidations[0].closures[0]
    assert closure.target.record_id == old.record_id
    assert (
        closure.before.valid_at,
        closure.before.invalid_at,
        closure.after.invalid_at,
    ) == (JAN, FEB, point)
    result = execute(store, plan, context)
    assert raw(graph)[current.record_id] == preserved
    assert result.execution.affected_claim_record_ids == (old.record_id,)
    assert history(store, as_of=JAN + timedelta(days=20)) == []
    assert [row.record_id for row in history(store, as_of=MAR)] == [current.record_id]
    before = snapshot(graph)
    assert execute(store, plan, context) == result
    assert snapshot(graph) == before


def test_historical_correction_extends_both_existing_endpoints(store):
    apply(store, batch(edge(valid_until=FEB.isoformat())), "scheduled-history")
    original = history(store, include_invalidated=True)[0]
    endpoint = FEB + timedelta(days=14)
    plan = MutationBatch(invalidations=[retract(original, endpoint)])
    context = prepared(store, plan, "extend-history")
    result = execute(store, plan, context)
    row = history(store, include_invalidated=True)[0]
    assert (row.record_id, row.valid_at, row.invalid_at, row.valid_until) == (
        original.record_id,
        JAN,
        endpoint,
        endpoint,
    )
    assert [
        item.record_id for item in history(store, as_of=FEB + timedelta(days=7))
    ] == [original.record_id]
    assert not history(store, as_of=endpoint)
    assert result.execution.affected_claim_record_ids == (original.record_id,)


def test_delayed_execution_uses_frozen_correction_after_original_endpoint(
    store, monkeypatch
):
    anchor = datetime.now(timezone.utc)
    ending, desired, later = (
        anchor + timedelta(days=2),
        anchor + timedelta(days=1),
        anchor + timedelta(days=3),
    )
    apply(store, batch(edge(valid_until=ending.isoformat())), "scheduled")
    original = history(store, include_invalidated=True)[0]
    plan = MutationBatch(invalidations=[retract(original, desired)])
    context = prepared(store, plan, "delayed", anchor=anchor)

    class Later(datetime):
        @classmethod
        def now(cls, tz=None):
            return later if tz else later.replace(tzinfo=None)

    monkeypatch.setattr(falkordb_atomic, "datetime", Later)
    result = execute(store, plan, context)
    row = history(store, include_invalidated=True)[0]
    assert (row.invalid_at, row.valid_until) == (desired, desired)
    assert result.execution.applied_at == later


@pytest.mark.parametrize("frozen", (False, True), ids=("raw", "prepared"))
def test_supersede_consumes_shared_ordinary_effects_and_keeps_replacement_live(
    store, graph, frozen
):
    apply(store, batch(edge()), "old")
    old = history(store, include_invalidated=True)[0]
    plan = batch(edge(MAR, "team:blue"))
    plan.invalidations.append(retract(old, MAR, superseded_by_key="team:blue"))
    context = (
        prepared(store, plan, "supersede")
        if frozen
        else ProvenanceContext(mutation_id="supersede")
    )
    fingerprint = batch_fingerprint(plan)
    result = (
        execute(store, plan, context)
        if frozen
        else store.mutation.apply(plan, expected_pot_id="p", provenance_context=context)
    )
    assert batch_fingerprint(plan) == fingerprint == result.execution.batch_fingerprint
    rows = {row.record_id: row for row in history(store, include_invalidated=True)}
    current = next(row for row in rows.values() if row.object_key == "team:blue")
    assert rows[old.record_id].invalid_at == MAR
    assert (current.valid_at, current.invalid_at, current.valid_until) == (
        MAR,
        None,
        None,
    )
    assert [row.record_id for row in history(store, as_of=APR)] == [current.record_id]
    system = store.claim_query.find_claims(
        ClaimQueryFilter(
            pot_id="p", predicate_in=("SUPERSEDES",), include_invalidated=True
        )
    )
    assert len(system) == 1
    assert system[0].properties["supersedes_record_id"] == old.record_id
    assert system[0].properties["replacement_record_id"] == current.record_id
    assert set(result.execution.affected_claim_record_ids) == {
        old.record_id,
        current.record_id,
        system[0].record_id,
    }
    before = snapshot(graph)
    replay = (
        execute(store, plan, context)
        if frozen
        else store.mutation.apply(plan, expected_pot_id="p", provenance_context=context)
    )
    assert replay == result and snapshot(graph) == before


def test_keyless_raw_supersession_binds_the_exact_replacement_record(store, graph):
    apply(store, batch(edge()), "old")
    old = history(store, include_invalidated=True)[0]
    replacement = EdgeUpsert(
        "OWNED_BY", "service:api", "team:blue", {"valid_at": MAR.isoformat()}
    )
    plan = batch(replacement)
    plan.invalidations.append(retract(old, MAR, superseded_by_key="team:blue"))
    fingerprint = batch_fingerprint(plan)
    result = apply(store, plan, "keyless-supersede")
    current = next(row for row in history(store) if row.object_key == "team:blue")
    system = store.claim_query.find_claims(
        ClaimQueryFilter(pot_id="p", predicate_in=("SUPERSEDES",))
    )
    assert len(system) == 1
    assert system[0].properties["supersedes_record_id"] == old.record_id
    assert system[0].properties["replacement_record_id"] == current.record_id
    assert current.record_id in result.execution.affected_claim_record_ids
    assert "claim_key" not in replacement.properties and replacement.resolution is None
    assert batch_fingerprint(plan) == fingerprint == result.execution.batch_fingerprint
    before = snapshot(graph)
    assert apply(store, plan, "keyless-supersede") == result
    assert snapshot(graph) == before


def test_failure_after_closure_projection_and_system_writes_rolls_back(
    store, graph, monkeypatch
):
    apply(store, batch(edge()), "old")
    plan = batch(edge(MAR, "team:blue"))
    plan.invalidations.append(
        InvalidationOp(
            "team:red",
            None,
            "replace structural entity",
            valid_to=MAR.isoformat(),
            superseded_by_key="team:blue",
        )
    )
    compile_query, publish = falkordb_atomic._compile, falkordb_journal._publish_query

    def broken_compile(*args, **kwargs):
        query, params, result = compile_query(*args, **kwargs)
        return (
            query.replace(
                "RETURN v.version,receipt.fingerprint",
                "WITH v,receipt RETURN keys(v.version),receipt.fingerprint",
            ),
            params,
            result,
        )

    def broken_publish(*args, **kwargs):
        query, params = publish(*args, **kwargs)
        return (
            query.replace("RETURN v.version", "WITH v RETURN keys(v.version)"),
            params,
        )

    monkeypatch.setattr(falkordb_atomic, "_compile", broken_compile)
    monkeypatch.setattr(falkordb_journal, "_publish_query", broken_publish)
    before, fingerprint = snapshot(graph), batch_fingerprint(plan)
    with pytest.raises(ResponseError, match="Type mismatch"):
        apply(store, plan, "fail-derived")
    assert snapshot(graph) == before
    assert batch_fingerprint(plan) == fingerprint
    assert (
        store.mutation.lookup_execution(
            plan, expected_pot_id="p", mutation_id="fail-derived"
        ).state
        == "absent"
    )


def test_later_incarnation_after_preparation_conflicts_without_reselection(
    store, graph
):
    apply(store, batch(edge()), "old")
    old = history(store, include_invalidated=True)[0]
    plan = MutationBatch(invalidations=[retract(old, FEB)])
    context = prepared(store, plan, "preview")
    close(store, old)
    apply(store, batch(edge(MAR)), "new-incarnation")
    before = snapshot(graph)
    with pytest.raises(GraphMutationVersionConflict):
        execute(store, plan, context)
    assert snapshot(graph) == before


def test_frozen_before_interval_drift_rejects_without_receipt(store, graph):
    apply(store, batch(edge()), "old")
    old = history(store, include_invalidated=True)[0]
    plan = MutationBatch(invalidations=[retract(old, FEB)])
    context = prepared(store, plan, "before-drift")
    graph.query(
        "MATCH ()-[r:RELATES_TO {group_id:'p'}]->() SET r.invalid_at=$end",
        params={"end": MAR.isoformat()},
    )
    before = snapshot(graph)
    with pytest.raises(
        ClaimLifecycleConflict, match="frozen correction interval changed"
    ):
        execute(store, plan, context)
    assert snapshot(graph) == before


def test_raw_frozen_empty_injection_is_rejected(store, graph):
    seed_endpoints(store)
    plan = MutationBatch(
        edge_deletes=[EdgeDelete("OWNED_BY", "service:api", "team:red", closures=())]
    )
    before = snapshot(graph)
    with pytest.raises(MutationExecutionReuseError, match="current prepared"):
        apply(store, plan, "raw-injection")
    assert snapshot(graph) == before


@pytest.mark.parametrize("axis", ("claim-key", "edge", "entity"))
@pytest.mark.parametrize("frozen", (False, True), ids=("raw", "prepared"))
def test_correction_target_axes_intersect_physical_selection(
    store, graph, axis, frozen
):
    apply(store, batch(edge()), "old")
    old = history(store, include_invalidated=True)[0]
    plan = MutationBatch(invalidations=[retract(old, FEB)])
    context = (
        prepared(store, plan, "outside-scope")
        if frozen
        else ProvenanceContext(mutation_id="outside-scope")
    )
    op = plan.invalidations[0]
    if axis == "claim-key":
        op.target_claim_keys = ("different-key",)
    elif axis == "edge":
        op.target_edge = ("OWNED_BY", "service:api", "team:blue")
    else:
        op.target_entity_key = "team:blue"
    if frozen:
        context = replace(
            context,
            prepared=replace(
                context.prepared, batch_fingerprint=batch_fingerprint(plan)
            ),
        )
    before = snapshot(graph)
    with pytest.raises(ClaimLifecycleConflict, match="scope|selector"):
        (
            execute(store, plan, context)
            if frozen
            else store.mutation.apply(
                plan, expected_pot_id="p", provenance_context=context
            )
        )
    assert snapshot(graph) == before


def test_key_only_selection_between_incarnations_is_ambiguous(store, graph):
    apply(store, batch(edge()), "old")
    old = history(store, include_invalidated=True)[0]
    close(store, old)
    apply(store, batch(edge(MAR)), "new")
    plan = MutationBatch(
        invalidations=[
            InvalidationOp(
                None,
                None,
                "ambiguous",
                valid_to=APR.isoformat(),
                target_claim_keys=(old.claim_key,),
            )
        ]
    )
    before = snapshot(graph)
    with pytest.raises(ClaimLifecycleConflict, match="multiple incarnations"):
        apply(store, plan, "ambiguous")
    assert snapshot(graph) == before


@pytest.mark.parametrize("frozen", (False, True), ids=("raw", "prepared"))
def test_correction_keeps_absent_prod_stage_and_competing_sources_separate(
    store, graph, frozen
):
    observations = [
        edge(),
        edge(environment="prod"),
        edge(environment="staging"),
        edge(environment="prod", source_system="ticketing"),
        edge(environment="prod", source_ref="file:other-owners"),
    ]
    apply(store, batch(*observations), "scoped-observations")
    rows = history(store, include_invalidated=True)
    target = next(
        row
        for row in rows
        if row.environment == "prod"
        and row.source_system == "git"
        and row.source_ref == "file:owners"
    )
    preserved = {
        identity: props
        for identity, props in raw(graph).items()
        if identity != target.record_id
    }
    plan = MutationBatch(invalidations=[retract(target, FEB)])
    context = (
        prepared(store, plan, "scoped-correction")
        if frozen
        else ProvenanceContext(mutation_id="scoped-correction")
    )
    context = replace(
        context, source_system="correcting-actor", source_ref="ticket:correction"
    )
    result = (
        execute(store, plan, context)
        if frozen
        else store.mutation.apply(plan, expected_pot_id="p", provenance_context=context)
    )
    assert result.execution.affected_claim_record_ids == (target.record_id,)
    after = raw(graph)
    assert all(after[identity] == props for identity, props in preserved.items())
    assert len(history(store, as_of=MAR)) == 4


def test_touched_history_rejects_closed_endpoint_extension_into_new_incarnation(
    store, graph
):
    apply(store, batch(edge()), "old")
    old = history(store, include_invalidated=True)[0]
    close(store, old)
    apply(store, batch(edge(MAR)), "new")
    plan = MutationBatch(invalidations=[retract(old, APR)])
    before = snapshot(graph)
    with pytest.raises(ClaimLifecycleConflict, match="overlap"):
        apply(store, plan, "overlapping-correction")
    assert snapshot(graph) == before


@pytest.mark.parametrize("frozen", (False, True), ids=("raw", "prepared"))
def test_structural_entity_projection_is_an_explicit_shared_planner_write(
    store, graph, frozen
):
    apply(store, batch(edge()), "old")
    original = history(store, include_invalidated=True)[0]
    entity_id = store.claim_query.entity_properties(pot_id="p", entity_key="team:red")[
        "record_id"
    ]
    plan = MutationBatch(
        invalidations=[
            InvalidationOp(
                "team:red", None, "structural retirement", valid_to=FEB.isoformat()
            )
        ]
    )
    context = (
        prepared(store, plan, "project")
        if frozen
        else ProvenanceContext(mutation_id="project")
    )
    if frozen:
        assert len(plan.entity_upserts) == 1
        assert plan.entity_upserts[0].properties["invalidated"] is True
    fingerprint = batch_fingerprint(plan)
    result = (
        execute(store, plan, context)
        if frozen
        else store.mutation.apply(plan, expected_pot_id="p", provenance_context=context)
    )
    props = store.claim_query.entity_properties(pot_id="p", entity_key="team:red")
    assert (props["record_id"], props["invalidated"], props["valid_to"]) == (
        entity_id,
        True,
        FEB.isoformat(),
    )
    assert props["invalidation_reason"] == "structural retirement"
    assert props["invalid_at"] == FEB.isoformat()
    assert result.mutation_summary.entity_upserts_applied == 1
    assert result.execution.affected_claim_record_ids == (original.record_id,)
    assert result.execution.batch_fingerprint == fingerprint == batch_fingerprint(plan)
    assert history(store, include_invalidated=True)[0].invalid_at == FEB


@pytest.mark.parametrize("frozen", (False, True), ids=("raw", "prepared"))
def test_entity_projection_preserves_existing_labels_and_unrelated_properties(
    store, graph, frozen
):
    graph.query(
        "CREATE (a:Entity:Service {group_id:'p',entity_key:'service:api',uuid:'api',record_id:'api'}),"
        "(b:Entity:Team:LegacyMarker {group_id:'p',entity_key:'legacy-owner:red',uuid:'owner',record_id:'owner',"
        "summary:'authored summary',description:'authored description',custom:'retained'}),"
        "(a)-[:RELATES_TO {group_id:'p',name:'OWNED_BY',subject_key:'service:api',object_key:'legacy-owner:red',"
        "uuid:'legacy-claim',record_id:'legacy-claim',claim_key:'legacy-owner',subgraph:'working',source_system:'git',"
        "source_ref:'file:owners',truth:'source_observation',valid_at:$start,valid_from:$start,fact:'API owned by legacy team'}]->(b)",
        params={"start": JAN.isoformat()},
    )
    before_labels = graph.ro_query(
        "MATCH (e:Entity {group_id:'p',entity_key:'legacy-owner:red'}) RETURN labels(e)"
    ).result_set
    plan = MutationBatch(
        invalidations=[
            InvalidationOp(
                "legacy-owner:red",
                None,
                "retire legacy owner",
                valid_to=FEB.isoformat(),
            )
        ]
    )
    context = (
        prepared(store, plan, "legacy-projection")
        if frozen
        else ProvenanceContext(mutation_id="legacy-projection")
    )
    if frozen:
        assert plan.entity_upserts[0].preserve_labels
    (
        execute(store, plan, context)
        if frozen
        else store.mutation.apply(plan, expected_pot_id="p", provenance_context=context)
    )
    assert (
        graph.ro_query(
            "MATCH (e:Entity {group_id:'p',entity_key:'legacy-owner:red'}) RETURN labels(e)"
        ).result_set
        == before_labels
    )
    props = store.claim_query.entity_properties(
        pot_id="p", entity_key="legacy-owner:red"
    )
    assert (
        props["summary"],
        props["description"],
        props["custom"],
        props["record_id"],
    ) == ("authored summary", "authored description", "retained", "owner")
    assert props["invalidated"] is True


def test_label_preserving_entity_patch_rejects_absent_target(store, graph):
    seed_endpoints(store)
    plan = MutationBatch(
        entity_upserts=[
            EntityUpsert(
                "missing:target", (), {"invalidated": True}, preserve_labels=True
            )
        ]
    )
    before = snapshot(graph)
    with pytest.raises(ClaimLifecycleConflict, match="projection target is missing"):
        apply(store, plan, "missing-projection")
    assert snapshot(graph) == before
