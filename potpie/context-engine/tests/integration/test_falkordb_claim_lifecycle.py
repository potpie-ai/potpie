# ruff: noqa: F811 - imported pytest fixtures are injected as test parameters.
"""Real server/Lite lifecycle policy, ordinary and journal publication parity."""

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json

import pytest

from potpie_context_engine.core.claim_lifecycle import ClaimLifecycleConflict
from potpie_context_engine.core.claim_preparation import prepare_claim_assertions
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
from potpie_context_engine.adapters.outbound.graph.falkordb_writer import (
    FalkorDBGraphWriter,
)
from potpie_context_engine.core.graph_mutations import ProvenanceRef
from potpie_context_engine.adapters.outbound.graph._mutation_execution import (
    MutationExecutionReuseError,
)

from tests.integration.test_falkordb_reliability import (
    backend,
    graph as graph,
    snapshot,
)

pytestmark = pytest.mark.integration
JAN = datetime(2020, 1, 1, tzinfo=timezone.utc)
FEB = datetime(2020, 2, 1, tzinfo=timezone.utc)
MAR = datetime(2020, 3, 1, tzinfo=timezone.utc)
APR = datetime(2020, 4, 1, tzinfo=timezone.utc)


@pytest.fixture(params=(False, True), ids=("ordinary", "journal"))
def store(graph, request):
    store = backend(graph)
    if request.param:
        store.journal.activate(pot_id="p", rollback_enabled=True)
    return store


def edge(point=JAN, target="team:red", **properties):
    return EdgeUpsert(
        "OWNED_BY",
        "service:api",
        target,
        {
            "claim_key": "owner:" + target,
            "subgraph": "working",
            "source_system": "git",
            "source_ref": "file:owners",
            "truth": "source_observation",
            "fact": "API is owned by " + target,
            "valid_from": point.isoformat(),
            **properties,
        },
    )


def batch(*edges):
    return MutationBatch(
        entity_upserts=[
            EntityUpsert("service:api", ("Service",)),
            *[
                EntityUpsert(key, ("Team",))
                for key in sorted({item.to_entity_key for item in edges})
            ],
        ],
        edge_upserts=list(edges),
    )


def apply(store, plan, mid):
    return store.mutation.apply(
        plan, expected_pot_id="p", provenance_context=ProvenanceContext(mutation_id=mid)
    )


def history(store, **filters):
    return store.claim_query.find_claims(
        ClaimQueryFilter(pot_id="p", predicate_in=("OWNED_BY",), **filters)
    )


def close(store, row, point=FEB, mid="close"):
    return apply(
        store,
        MutationBatch(
            invalidations=[
                InvalidationOp(
                    None,
                    None,
                    "source retracted",
                    valid_to=point.isoformat(),
                    target_claim_keys=(row.claim_key,),
                    target_claim_record_ids=(row.record_id,),
                )
            ]
        ),
        mid,
    )


def raw(graph):
    return {
        props["record_id"]: props
        for (props,) in graph.ro_query(
            "MATCH ()-[r:RELATES_TO {group_id:'p'}]->() RETURN properties(r)"
        ).result_set
    }


def test_reassertion_preserves_closed_incarnation_and_february_gap(store, graph):
    first_batch = batch(edge())
    first = apply(store, first_batch, "january")
    original = history(store, include_invalidated=True)[0]
    close(store, original)
    closed = raw(graph)[original.record_id]
    result = apply(store, batch(edge(MAR)), "march")
    rows = sorted(
        history(store, include_invalidated=True), key=lambda row: row.valid_at
    )
    assert len(rows) == 2
    old, current = rows
    assert old.claim_key == current.claim_key and old.record_id != current.record_id
    assert (old.valid_at, old.invalid_at, current.valid_at, current.invalid_at) == (
        JAN,
        FEB,
        MAR,
        None,
    )
    assert raw(graph)[old.record_id] == closed
    assert result.execution.affected_claim_record_ids == (current.record_id,)
    for point, expected in (
        (JAN, [old.record_id]),
        (FEB, []),
        (FEB + timedelta(days=14), []),
        (MAR, [current.record_id]),
    ):
        assert [row.record_id for row in history(store, as_of=point)] == expected
    revision = store.mutation.current_version("p")
    assert apply(store, first_batch, "january") == first
    assert store.mutation.current_version("p") == revision
    payload = store.snapshot.export_data(pot_id="p")
    assert len(payload["claims"]) == 2
    assert {row["record_id"] for row in payload["claims"]} == {
        old.record_id,
        current.record_id,
    }
    store.snapshot.import_data(pot_id="copy", payload=payload)
    version = store.mutation.current_version("copy")
    store.snapshot.import_data(pot_id="copy", payload=payload)
    assert store.mutation.current_version("copy") == version
    copied = store.snapshot.export_data(pot_id="copy")["claims"]
    assert {row["record_id"] for row in copied} == {old.record_id, current.record_id}
    assert len(copied) == 2


def test_active_assertion_retains_start_and_scheduled_endpoint(store):
    ending = datetime.now(timezone.utc) + timedelta(days=30)
    apply(store, batch(edge(valid_until=ending.isoformat())), "scheduled")
    original = history(store, include_invalidated=True)[0]
    apply(store, batch(edge(APR)), "active-refresh")
    rows = history(store, include_invalidated=True)
    assert len(rows) == 1
    assert (rows[0].record_id, rows[0].valid_at, rows[0].valid_until) == (
        original.record_id,
        JAN,
        ending,
    )
    apply(store, batch(edge(ending)), "after-scheduled-end")
    rows = history(store, include_invalidated=True)
    assert len(rows) == 2
    assert (
        next(row for row in rows if row.record_id == original.record_id).valid_until
        == ending
    )
    assert len(history(store, as_of=ending)) == 1


@pytest.mark.parametrize(
    "other_scope",
    [
        {"environment": None},
        {"environment": "staging"},
        {"environment": "prod", "subgraph": "episodic"},
        {"environment": "prod", "source_system": "ticketing"},
        {"environment": "prod", "source_ref": "file:other"},
    ],
    ids=("missing-env", "staging", "subgraph", "source-system", "source-ref"),
)
def test_singleton_closes_only_exact_stable_source_scope(store, graph, other_scope):
    apply(store, batch(edge(environment="prod")), "original")
    original = history(store, include_invalidated=True)[0]
    apply(store, batch(edge(**other_scope)), "other-scope")
    competing = next(
        row
        for row in history(store, include_invalidated=True)
        if row.record_id != original.record_id
    )
    preserved = raw(graph)[competing.record_id]
    result = apply(store, batch(edge(MAR, "team:blue", environment="prod")), "replace")
    rows = {row.record_id: row for row in history(store, include_invalidated=True)}
    winner = next(row for row in rows.values() if row.object_key == "team:blue")
    assert rows[original.record_id].invalid_at == MAR
    assert rows[competing.record_id].invalid_at is None
    assert raw(graph)[competing.record_id] == preserved
    assert set(result.execution.affected_claim_record_ids) == {
        original.record_id,
        winner.record_id,
    }


@pytest.mark.parametrize(
    "source_mode", ("generated", "explicit-false", "legacy-missing")
)
def test_unstable_source_identity_never_automatically_supersedes(
    store, graph, source_mode
):
    original, replacement = edge(), edge(MAR, "team:blue")
    if source_mode == "generated":
        for item in (original, replacement):
            item.properties.pop("source_system")
            item.properties.pop("source_ref")
    elif source_mode == "explicit-false":
        for item in (original, replacement):
            item.properties["source_identity_stable"] = False
    apply(store, batch(original), "old")
    if source_mode == "legacy-missing":
        graph.query("MATCH ()-[r:RELATES_TO]->() REMOVE r.source_identity_stable")
    apply(store, batch(replacement), "new")
    assert len(history(store, as_of=MAR)) == 2
    assert all(
        row.invalid_at is None for row in history(store, include_invalidated=True)
    )


def test_metadata_review_selects_exact_closed_record(store, graph):
    apply(store, batch(edge()), "old")
    original = history(store, include_invalidated=True)[0]
    close(store, original)
    apply(store, batch(edge(MAR)), "new")
    before = raw(graph)
    review = edge(
        APR, evidence_review_required=True, evidence_review_reason="source changed"
    )
    review.metadata_review_record_id = original.record_id
    result = apply(store, batch(review), "review")
    rows = {row.record_id: row for row in history(store, include_invalidated=True)}
    assert (rows[original.record_id].valid_at, rows[original.record_id].invalid_at) == (
        JAN,
        FEB,
    )
    assert rows[original.record_id].properties["evidence_review_required"] is True
    current = next(row for row in rows.values() if row.record_id != original.record_id)
    assert raw(graph)[current.record_id] == before[current.record_id]
    assert result.execution.affected_claim_record_ids == (original.record_id,)
    for row in rows.values():
        assert [
            item.record_id
            for item in history(
                store, include_invalidated=True, record_id_in=(row.record_id,)
            )
        ] == [row.record_id]


def test_delete_preserves_closed_history_and_only_closes_open_incarnation(store, graph):
    apply(store, batch(edge()), "old")
    original = history(store, include_invalidated=True)[0]
    close(store, original)
    closed = raw(graph)[original.record_id]
    apply(store, batch(edge(MAR)), "new")
    result = apply(
        store,
        MutationBatch(edge_deletes=[EdgeDelete("OWNED_BY", "service:api", "team:red")]),
        "delete",
    )
    assert result.mutation_summary.edge_deletes_applied == 1
    assert raw(graph)[original.record_id] == closed
    assert not history(store)
    repeat = apply(
        store,
        MutationBatch(edge_deletes=[EdgeDelete("OWNED_BY", "service:api", "team:red")]),
        "delete-again",
    )
    assert repeat.mutation_summary.edge_deletes_applied == 0


def test_frozen_resolution_preserves_physical_selection_and_replay(store, graph):
    apply(store, batch(edge()), "old")
    original = history(store, include_invalidated=True)[0]
    close(store, original)
    plan = batch(edge(MAR))
    prepare_claim_assertions(
        plan,
        pot_id="p",
        mutation_id="prepared",
        anchor=APR,
        query=store.claim_query,
        definition=store.definition,
    )
    resolution = plan.edge_upserts[0].resolution
    prepared = PreparedMutation(
        "p",
        "prepared",
        "request",
        batch_fingerprint(plan),
        store.mutation.current_version("p"),
        APR,
    )
    context = ProvenanceContext(mutation_id="prepared", prepared=prepared)
    first = store.mutation.compare_and_apply(
        plan,
        expected_pot_id="p",
        expected_version=prepared.expected_version,
        provenance_context=context,
    )
    assert first.execution.affected_claim_record_ids == (resolution.record_id,)
    apply(store, batch(edge(APR)), "refresh")
    before = snapshot(graph)
    assert (
        store.mutation.compare_and_apply(
            plan,
            expected_pot_id="p",
            expected_version=prepared.expected_version,
            provenance_context=context,
        )
        == first
    )
    assert snapshot(graph) == before


@pytest.mark.parametrize(
    "property_name,value",
    (
        ("fact", "different fact"),
        ("description", "different description"),
        ("truth", "inferred"),
    ),
)
def test_existing_truth_content_changes_require_explicit_correction(
    store, graph, property_name, value
):
    apply(store, batch(edge(description="original description")), "old")
    before = snapshot(graph)
    changed = edge(APR, description="original description")
    changed.properties[property_name] = value
    with pytest.raises(ClaimLifecycleConflict, match="explicit correction"):
        apply(store, batch(changed), "reject")
    assert snapshot(graph) == before


def test_frozen_incarnation_never_falls_back_to_another_matching_record(store, graph):
    apply(store, batch(edge()), "old")
    plan = batch(edge(APR))
    prepare_claim_assertions(
        plan,
        pot_id="p",
        mutation_id="frozen",
        anchor=APR,
        query=store.claim_query,
        definition=store.definition,
    )
    original = plan.edge_upserts[0].resolution.record_id
    prepared = PreparedMutation(
        "p",
        "frozen",
        "request",
        batch_fingerprint(plan),
        store.mutation.current_version("p"),
        APR,
    )
    graph.query("MATCH ()-[r:RELATES_TO]->() SET r.record_id='replacement-identity'")
    before = snapshot(graph)
    with pytest.raises(ClaimLifecycleConflict, match="frozen incarnation"):
        store.mutation.compare_and_apply(
            deepcopy(plan),
            expected_pot_id="p",
            expected_version=prepared.expected_version,
            provenance_context=ProvenanceContext(
                mutation_id="frozen", prepared=prepared
            ),
        )
    assert snapshot(graph) == before
    assert original != history(store, include_invalidated=True)[0].record_id


class MutableEmbedder:
    name = "lifecycle-test"
    dimensions = 3
    value = (1.0, 0.0, 0.0)

    def embed(self, text):
        return self.value


def test_native_vector_reassertion_does_not_change_closed_incarnation(store, graph):
    embedder = MutableEmbedder()
    store.writer._embedder = embedder
    apply(store, batch(edge()), "old-vector")
    original = history(store, include_invalidated=True)[0]
    close(store, original)
    preserved = raw(graph)[original.record_id]
    embedder.value = (0.0, 1.0, 0.0)
    apply(store, batch(edge(MAR)), "new-vector")
    rows = raw(graph)
    assert rows[original.record_id] == preserved
    assert list(rows[original.record_id]["fact_embedding"]) == [1.0, 0.0, 0.0]
    current = next(
        props for identity, props in rows.items() if identity != original.record_id
    )
    assert list(current["fact_embedding"]) == [0.0, 1.0, 0.0]


@pytest.mark.asyncio
async def test_internal_writer_vector_binding_updates_only_exact_incarnation(graph):
    store = backend(graph)
    await store.mutation.apply_async(
        batch(edge()),
        expected_pot_id="p",
        provenance_context=ProvenanceContext(mutation_id="old"),
    )
    original = history(store, include_invalidated=True)[0]
    await store.mutation.apply_async(
        MutationBatch(
            invalidations=[
                InvalidationOp(
                    None,
                    None,
                    "source retracted",
                    valid_to=FEB.isoformat(),
                    target_claim_record_ids=(original.record_id,),
                    target_claim_keys=(original.claim_key,),
                )
            ]
        ),
        expected_pot_id="p",
        provenance_context=ProvenanceContext(mutation_id="close"),
    )
    preserved = raw(graph)[original.record_id]
    writer = FalkorDBGraphWriter(
        store.settings, graph=graph, embedder=MutableEmbedder()
    )
    await writer.upsert_edges(
        "p",
        [edge(MAR)],
        ProvenanceRef(
            pot_id="p",
            source_event_id="writer-event",
            mutation_id="writer-primitive",
        ),
    )
    rows = raw(graph)
    assert rows[original.record_id] == preserved
    current = next(
        props for identity, props in rows.items() if identity != original.record_id
    )
    assert list(current["fact_embedding"]) == [1.0, 0.0, 0.0]


def test_unexecuted_legacy_preparation_fails_before_revision_bootstrap(graph):
    store, plan = backend(graph), batch(edge())
    graph.query("CREATE (:LifecycleSentinel {value:'outside-pot'})")
    context = ProvenanceContext(
        mutation_id="legacy",
        prepared=PreparedMutation(
            "p",
            "legacy",
            "request",
            batch_fingerprint(plan),
            0,
            APR,
            schema_version=1,
        ),
    )
    before = snapshot(graph)
    assert (
        store.mutation.lookup_execution(
            plan, expected_pot_id="p", mutation_id="legacy"
        ).state
        == "absent"
    )
    with pytest.raises(MutationExecutionReuseError, match="migration_required"):
        store.mutation.compare_and_apply(
            plan, expected_pot_id="p", expected_version=0, provenance_context=context
        )
    assert snapshot(graph) == before


def test_completed_legacy_preparation_replays_before_revision_rejection(graph):
    store, plan = backend(graph), batch(edge())
    graph.query(
        "CREATE (:PotpieMutationReceipt {pot_id:'p',mutation_id:'legacy',fingerprint:$fp,"
        "payload:$payload,entities:0,edges:0,deletes:0,invalidations:0})",
        params={
            "fp": batch_fingerprint(plan),
            "payload": json.dumps({"mutation_summary": {}}),
        },
    )
    context = ProvenanceContext(
        mutation_id="legacy",
        prepared=PreparedMutation(
            "p",
            "legacy",
            "unknown-old-request",
            batch_fingerprint(plan),
            42,
            APR,
            schema_version=1,
        ),
    )
    before = snapshot(graph)
    result = store.mutation.compare_and_apply(
        plan, expected_pot_id="p", expected_version=42, provenance_context=context
    )
    assert result.ok and result.execution is None
    assert snapshot(graph) == before


def test_missing_frozen_resolution_fails_before_revision_bootstrap(graph):
    store, plan = backend(graph), batch(edge())
    graph.query("CREATE (:LifecycleSentinel {value:'outside-pot'})")
    context = ProvenanceContext(
        mutation_id="unfrozen",
        prepared=PreparedMutation(
            "p",
            "unfrozen",
            "request",
            batch_fingerprint(plan),
            0,
            APR,
        ),
    )
    before = snapshot(graph)
    with pytest.raises(MutationExecutionReuseError, match="missing its frozen"):
        store.mutation.compare_and_apply(
            plan, expected_pot_id="p", expected_version=0, provenance_context=context
        )
    assert snapshot(graph) == before


@pytest.mark.parametrize("collision", ("claim-outside-scope", "entity"))
def test_discovered_record_id_is_unique_beyond_candidate_scope(store, graph, collision):
    apply(store, batch(edge()), "old")
    original = history(store, include_invalidated=True)[0]
    if collision == "entity":
        graph.query(
            "MATCH (e:Entity {group_id:'p',entity_key:'team:red'}) SET e.record_id=$identity",
            params={"identity": original.record_id},
        )
    else:
        graph.query(
            "MATCH (a:Entity {group_id:'p',entity_key:'service:api'}),"
            "(b:Entity {group_id:'p',entity_key:'team:red'}) "
            "CREATE (a)-[:RELATES_TO {group_id:'p',name:'DEPENDS_ON',subject_key:'service:api',"
            "object_key:'team:red',claim_key:'outside-scope',uuid:'outside-uuid',record_id:$identity}]->(b)",
            params={"identity": original.record_id},
        )
    before = snapshot(graph)
    with pytest.raises(ValueError, match="identity"):
        apply(store, batch(edge(APR)), "reject-duplicate")
    assert snapshot(graph) == before


def test_delete_uses_source_occurrence_and_preserves_past_endpoints(store, graph):
    apply(store, batch(edge()), "old")
    original = history(store, include_invalidated=True)[0]
    plan = MutationBatch(
        edge_deletes=[EdgeDelete("OWNED_BY", "service:api", "team:red")]
    )
    result = store.mutation.apply(
        plan,
        expected_pot_id="p",
        provenance_context=ProvenanceContext(
            mutation_id="occurred-in-february", event_occurred_at=FEB
        ),
    )
    assert result.mutation_summary.edge_deletes_applied == 1
    closed = history(store, include_invalidated=True)[0]
    assert (closed.record_id, closed.valid_at, closed.invalid_at) == (
        original.record_id,
        JAN,
        FEB,
    )
    assert not history(store, as_of=FEB)
    preserved = raw(graph)[original.record_id]
    repeat = store.mutation.apply(
        plan,
        expected_pot_id="p",
        provenance_context=ProvenanceContext(
            mutation_id="later-deletion", event_occurred_at=APR
        ),
    )
    assert repeat.mutation_summary.edge_deletes_applied == 0
    assert raw(graph)[original.record_id] == preserved


@pytest.mark.parametrize("missing", ("subject", "object"))
def test_raw_assertion_missing_endpoint_has_no_success_receipt_or_partial_write(
    store, graph, missing
):
    available = "team:red" if missing == "subject" else "service:api"
    label = "Team" if missing == "subject" else "Service"
    plan = MutationBatch(
        entity_upserts=[EntityUpsert(available, (label,))], edge_upserts=[edge()]
    )
    with pytest.raises(ClaimLifecycleConflict, match="endpoint is missing"):
        apply(store, plan, "missing-endpoint")
    assert store.mutation.current_version("p") == 0
    assert not store.claim_query.entity_properties(pot_id="p", entity_key=available)
    assert (
        store.mutation.lookup_execution(
            plan, expected_pot_id="p", mutation_id="missing-endpoint"
        ).state
        == "absent"
    )
    assert not history(store, include_invalidated=True)


def test_authored_annotations_survive_canonical_hydration_without_creation_id_override(
    store, graph
):
    apply(
        store,
        batch(
            edge(
                created_by="source-author",
                identity_key="authored-identity",
                idempotency_key="authored-retry",
                uuid="must-not-replace-native-uuid",
                created_at="must-not-replace-native-creation-time",
            )
        ),
        "annotations",
    )
    props = next(iter(raw(graph).values()))
    assert {
        key: props[key] for key in ("created_by", "identity_key", "idempotency_key")
    } == {
        "created_by": "source-author",
        "identity_key": "authored-identity",
        "idempotency_key": "authored-retry",
    }
    assert props["uuid"] == props["record_id"]
    assert props["uuid"] != "must-not-replace-native-uuid"
    assert props["created_at"] != "must-not-replace-native-creation-time"


def seed_endpoints(store):
    apply(
        store,
        MutationBatch(
            entity_upserts=[
                EntityUpsert("service:api", ("Service",)),
                EntityUpsert("team:red", ("Team",)),
            ]
        ),
        "endpoint-seed",
    )


@pytest.mark.parametrize(
    "endpoint,label", (("service:api", "Service"), ("team:red", "Team"))
)
@pytest.mark.parametrize(
    "refresh", (False, True), ids=("existing-retired", "same-batch-refresh")
)
def test_retired_endpoint_rejects_assertion_without_receipt_or_revision(
    store, graph, endpoint, label, refresh
):
    seed_endpoints(store)
    apply(
        store,
        MutationBatch(
            entity_upserts=[EntityUpsert(endpoint, (label,), {"retired": True})]
        ),
        "retire-endpoint",
    )
    assert not store.claim_query.entity_properties(pot_id="p", entity_key=endpoint)
    plan = MutationBatch(edge_upserts=[edge()])
    if refresh:
        plan.entity_upserts.append(
            EntityUpsert(endpoint, (label,), {"description": "must roll back"})
        )
    before = snapshot(graph)
    version = store.mutation.current_version("p")
    with pytest.raises(ClaimLifecycleConflict, match="endpoint is missing or retired"):
        apply(store, plan, "reject-retired")
    assert snapshot(graph) == before
    assert store.mutation.current_version("p") == version
    assert (
        store.mutation.lookup_execution(
            plan, expected_pot_id="p", mutation_id="reject-retired"
        ).state
        == "absent"
    )


@pytest.mark.parametrize(
    "endpoint,label", (("service:api", "Service"), ("team:red", "Team"))
)
@pytest.mark.parametrize("retired", (False, None), ids=("false", "remove-marker"))
def test_explicit_same_batch_reactivation_allows_assertion(
    store, endpoint, label, retired
):
    seed_endpoints(store)
    apply(
        store,
        MutationBatch(
            entity_upserts=[EntityUpsert(endpoint, (label,), {"retired": True})]
        ),
        "retire-endpoint",
    )
    plan = MutationBatch(
        entity_upserts=[EntityUpsert(endpoint, (label,), {"retired": retired})],
        edge_upserts=[edge()],
    )
    version = store.mutation.current_version("p")
    result = apply(store, plan, "reactivate-and-assert")
    rows = history(store)
    assert len(rows) == 1
    assert store.claim_query.entity_properties(pot_id="p", entity_key=endpoint)
    assert result.execution.affected_claim_record_ids == (rows[0].record_id,)
    assert store.mutation.current_version("p") == version + 1


@pytest.mark.parametrize(
    "endpoint,label", (("service:api", "Service"), ("team:red", "Team"))
)
def test_same_batch_retirement_rejects_assertion_without_partial_entity_write(
    store, graph, endpoint, label
):
    seed_endpoints(store)
    plan = MutationBatch(
        entity_upserts=[EntityUpsert(endpoint, (label,), {"retired": True})],
        edge_upserts=[edge()],
    )
    before = snapshot(graph)
    with pytest.raises(ClaimLifecycleConflict, match="endpoint is missing or retired"):
        apply(store, plan, "retire-and-assert")
    assert snapshot(graph) == before
    assert store.claim_query.entity_properties(pot_id="p", entity_key=endpoint)


def test_entity_only_update_preserves_retirement_until_explicit_reactivation(
    store, graph
):
    seed_endpoints(store)
    target = "team:red"
    apply(
        store,
        MutationBatch(
            entity_upserts=[EntityUpsert(target, ("Team",), {"retired": True})]
        ),
        "retire-endpoint",
    )
    apply(
        store,
        MutationBatch(
            entity_upserts=[
                EntityUpsert(target, ("Team",), {"description": "still retired"})
            ]
        ),
        "refresh-retired",
    )
    assert not store.claim_query.entity_properties(pot_id="p", entity_key=target)
    assert graph.ro_query(
        "MATCH (e:Entity {group_id:'p',entity_key:'team:red'}) RETURN e.retired"
    ).result_set == [[True]]
    apply(
        store,
        MutationBatch(
            entity_upserts=[EntityUpsert(target, ("Team",), {"retired": False})]
        ),
        "reactivate-endpoint",
    )
    assert (
        store.claim_query.entity_properties(pot_id="p", entity_key=target)[
            "description"
        ]
        == "still retired"
    )


def test_journal_reactivation_restore_preserves_entity_activity_and_identity(graph):
    store = backend(graph)
    store.journal.activate(pot_id="p", rollback_enabled=True)
    seed_endpoints(store)
    target = "team:red"
    identity = store.claim_query.entity_properties(pot_id="p", entity_key=target)[
        "record_id"
    ]
    apply(
        store,
        MutationBatch(
            entity_upserts=[EntityUpsert(target, ("Team",), {"retired": True})]
        ),
        "retire-endpoint",
    )
    apply(
        store,
        MutationBatch(
            entity_upserts=[EntityUpsert(target, ("Team",), {"retired": False})]
        ),
        "reactivate-endpoint",
    )
    inverse = store.journal.plan_restore(
        pot_id="p",
        target_commit_id="reactivate-endpoint",
        expected_head="reactivate-endpoint",
    )
    store.journal.apply_restore(
        inverse, mutation_id="undo-reactivation", actor="system"
    )
    assert not store.claim_query.entity_properties(pot_id="p", entity_key=target)
    redo = store.journal.plan_restore(
        pot_id="p",
        target_commit_id="undo-reactivation",
        expected_head="undo-reactivation",
    )
    store.journal.apply_restore(redo, mutation_id="redo-reactivation", actor="system")
    assert (
        store.claim_query.entity_properties(pot_id="p", entity_key=target)["record_id"]
        == identity
    )


@pytest.mark.parametrize(
    "entity_state",
    ("new-endpoint", "new-unrelated", "existing-endpoint", "existing-unrelated"),
)
@pytest.mark.parametrize("identity_field", ("record_id", "uuid-fallback"))
def test_prepared_claim_rejects_same_batch_entity_identity_collision(
    store, graph, entity_state, identity_field
):
    seed_endpoints(store)
    if entity_state == "existing-unrelated":
        apply(
            store,
            MutationBatch(entity_upserts=[EntityUpsert("service:other", ("Service",))]),
            "extra-entity",
        )
    assertion = edge()
    if entity_state == "new-endpoint":
        assertion.from_entity_key = "service:new"
        key, label = "service:new", "Service"
    elif entity_state == "new-unrelated":
        key, label = "service:new", "Service"
    elif entity_state == "existing-endpoint":
        key, label = "team:red", "Team"
    else:
        key, label = "service:other", "Service"
    item = EntityUpsert(key, (label,))
    plan = MutationBatch(entity_upserts=[item], edge_upserts=[assertion])
    prepare_claim_assertions(
        plan,
        pot_id="p",
        mutation_id="staged-collision",
        anchor=APR,
        query=store.claim_query,
        definition=store.definition,
    )
    claim_id = assertion.resolution.record_id
    if identity_field == "record_id":
        item.properties["record_id"] = claim_id
    else:
        item.properties["uuid"] = claim_id
    version = store.mutation.current_version("p")
    prepared = PreparedMutation(
        "p", "staged-collision", "request", batch_fingerprint(plan), version, APR
    )
    before = snapshot(graph)
    rejection = (
        "immutable entity identity"
        if entity_state.startswith("existing")
        else "collides with an entity identity"
    )
    with pytest.raises(ValueError, match=rejection):
        store.mutation.compare_and_apply(
            plan,
            expected_pot_id="p",
            expected_version=version,
            provenance_context=ProvenanceContext(
                mutation_id="staged-collision", prepared=prepared
            ),
        )
    assert snapshot(graph) == before
    assert store.mutation.current_version("p") == version
    assert (
        store.mutation.lookup_execution(
            plan, expected_pot_id="p", mutation_id="staged-collision"
        ).state
        == "absent"
    )


@pytest.mark.parametrize("journal", (False, True), ids=("ordinary", "journal"))
def test_legacy_entity_uuid_is_not_promoted_by_ordinary_write_or_activation(
    graph, journal
):
    store = backend(graph)
    apply(store, MutationBatch(), "bootstrap")
    graph.query(
        "CREATE (:Entity:Service {group_id:'p',entity_key:'service:api',uuid:'legacy-api'}),"
        "(:Entity:Team {group_id:'p',entity_key:'team:red',uuid:'legacy-team'})"
    )
    plan = MutationBatch(
        entity_upserts=[
            EntityUpsert("service:api", ("Service",), {"uuid": "safe-replacement-uuid"})
        ],
        edge_upserts=[edge()],
    )
    prepare_claim_assertions(
        plan,
        pot_id="p",
        mutation_id="legacy-planned-collision",
        anchor=APR,
        query=store.claim_query,
        definition=store.definition,
    )
    claim_id = plan.edge_upserts[0].resolution.record_id
    graph.query(
        "MATCH (e:Entity {group_id:'p',entity_key:'service:api'}) SET e.uuid=$identity",
        params={"identity": claim_id},
    )
    before = snapshot(graph)
    if journal:
        from potpie_context_engine.core.graph_journal import JournalError

        with pytest.raises(JournalError, match="migration_required"):
            store.journal.activate(pot_id="p", rollback_enabled=True)
        assert snapshot(graph) == before
    version = store.mutation.current_version("p")
    prepared = PreparedMutation(
        "p",
        "legacy-planned-collision",
        "request",
        batch_fingerprint(plan),
        version,
        APR,
    )
    with pytest.raises(ValueError, match="migration_required"):
        store.mutation.compare_and_apply(
            plan,
            expected_pot_id="p",
            expected_version=version,
            provenance_context=ProvenanceContext(
                mutation_id="legacy-planned-collision", prepared=prepared
            ),
        )
    assert snapshot(graph) == before
    assert (
        store.mutation.lookup_execution(
            plan, expected_pot_id="p", mutation_id="legacy-planned-collision"
        ).state
        == "absent"
    )
    assert graph.ro_query(
        "MATCH (e:Entity {group_id:'p',entity_key:'service:api'}) RETURN e.record_id,e.uuid"
    ).result_set == [[None, claim_id]]
