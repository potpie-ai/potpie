"""Incarnation history and scoped singleton policy on a disposable Neo4j server."""

from datetime import datetime, timedelta, timezone
import os
from types import SimpleNamespace
import uuid

import pytest

from potpie_context_engine.adapters.outbound.graph.backends.neo4j_backend import (
    Neo4jGraphBackend,
)
from potpie_context_engine.adapters.outbound.graph.neo4j_writer import Neo4jGraphWriter
from potpie_context_engine.adapters.outbound.graph import neo4j_reader
from potpie_context_engine.core.errors import ReconciliationApplyError
from potpie_context_engine.core.claim_preparation import prepare_claim_assertions
from potpie_context_engine.core.graph_mutations import (
    EdgeUpsert,
    EntityUpsert,
    InvalidationOp,
    ProvenanceContext,
)
from potpie_context_engine.core.graph_snapshot_mutation import normalize_snapshot_import
from potpie_context_engine.core.mutation_execution import (
    PreparedMutation,
    batch_fingerprint,
    request_fingerprint,
)
from potpie_context_engine.core.ports.claim_query import ClaimQueryFilter
from potpie_context_engine.core.reconciliation import MutationBatch

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.environ.get("PROTOCOL_TEST_NEO4J_URI"),
        reason="PROTOCOL_TEST_NEO4J_URI must name a disposable test server",
    ),
]

JAN = datetime(2020, 1, 1, tzinfo=timezone.utc)
FEB = datetime(2020, 2, 1, tzinfo=timezone.utc)
MAR = datetime(2020, 3, 1, tzinfo=timezone.utc)
APR = datetime(2020, 4, 1, tzinfo=timezone.utc)


@pytest.fixture
def graph():
    from neo4j import GraphDatabase

    settings = SimpleNamespace(
        is_enabled=lambda: True,
        neo4j_uri=lambda: os.environ["PROTOCOL_TEST_NEO4J_URI"],
        neo4j_user=lambda: os.environ.get("PROTOCOL_TEST_NEO4J_USERNAME", "neo4j"),
        neo4j_password=lambda: os.environ.get("PROTOCOL_TEST_NEO4J_PASSWORD", ""),
    )
    backend = Neo4jGraphBackend(settings)
    pot = "lifecycle-" + uuid.uuid4().hex
    with GraphDatabase.driver(
        settings.neo4j_uri(), auth=(settings.neo4j_user(), settings.neo4j_password())
    ) as driver:
        try:
            yield pot, backend, driver
        finally:
            backend.claim_query.close()
            _run(
                driver,
                "MATCH (n) WHERE n.group_id=$pot OR n.pot_id=$pot DETACH DELETE n",
                pot=pot,
            )


def _run(driver, query, **params):
    with driver.session() as session:
        return [dict(row) for row in session.run(query, **params)]


def _edge(point, *, target="team:red", environment=None, **properties):
    props = {
        "claim_key": "claim:api:" + target,
        "subgraph": "working",
        "source_system": "git",
        "source_ref": "file:owners",
        "truth": "source_observation",
        "fact": "API is owned by " + target,
        "valid_from": point.isoformat(),
        **properties,
    }
    if environment is not None:
        props["environment"] = environment
    return EdgeUpsert("OWNED_BY", "service:api", target, props)


def _batch(*edges):
    return MutationBatch(
        entity_upserts=[
            EntityUpsert("service:api", ("Service",)),
            *[
                EntityUpsert(key, ("Team",))
                for key in sorted({edge.to_entity_key for edge in edges})
            ],
        ],
        edge_upserts=list(edges),
    )


def _apply(graph, batch, mutation_id):
    pot, backend, _ = graph
    return backend.mutation.apply(
        batch,
        expected_pot_id=pot,
        provenance_context=ProvenanceContext(mutation_id=mutation_id),
    )


def _history(graph, **filters):
    pot, backend, _ = graph
    return backend.claim_query.find_claims(
        ClaimQueryFilter(pot_id=pot, predicate_in=("OWNED_BY",), **filters)
    )


def _raw_claims(graph):
    pot, _, driver = graph
    return {
        item["props"].get("record_id", item["props"].get("uuid")): item["props"]
        for item in _run(
            driver,
            "MATCH ()-[r:RELATES_TO {group_id:$pot}]->() RETURN properties(r) AS props",
            pot=pot,
        )
    }


def _close(graph, row, point=FEB, mutation_id="retract"):
    return _apply(
        graph,
        MutationBatch(
            invalidations=[
                InvalidationOp(
                    None,
                    None,
                    "assertion no longer applies",
                    valid_to=point.isoformat(),
                    target_claim_keys=(row.claim_key,),
                    target_claim_record_ids=(row.record_id,),
                )
            ]
        ),
        mutation_id,
    )


def test_reassertion_creates_new_incarnation_preserves_gap_and_replays(graph):
    pot, backend, _ = graph
    initial_batch = _batch(_edge(JAN))
    first = _apply(graph, initial_batch, "january")
    original = _history(graph, include_invalidated=True)[0]
    assert original.record_id
    _close(graph, original)
    closed = _raw_claims(graph)[original.record_id]

    march = _apply(graph, _batch(_edge(MAR)), "march")
    rows = sorted(
        _history(graph, include_invalidated=True), key=lambda row: row.valid_at
    )
    assert len(rows) == 2
    old, new = rows
    assert old.claim_key == new.claim_key
    assert old.record_id != new.record_id
    assert (old.valid_at, old.invalid_at) == (JAN, FEB)
    assert (new.valid_at, new.invalid_at) == (MAR, None)
    assert _raw_claims(graph)[old.record_id] == closed
    assert march.execution.affected_claim_record_ids == (new.record_id,)
    for point, expected in (
        (JAN, [old.record_id]),
        (JAN + timedelta(days=14), [old.record_id]),
        (FEB, []),
        (FEB + timedelta(days=14), []),
        (MAR, [new.record_id]),
    ):
        assert [row.record_id for row in _history(graph, as_of=point)] == expected

    revision = backend.mutation.current_version(pot)
    assert _apply(graph, initial_batch, "january") == first
    assert backend.mutation.current_version(pot) == revision
    assert _raw_claims(graph)[old.record_id] == closed
    assert len(_history(graph, include_invalidated=True)) == 2


def test_refresh_keeps_start_and_scheduled_end_then_reasserts_at_end(graph):
    ending = datetime.now(timezone.utc) + timedelta(days=30)
    _apply(graph, _batch(_edge(JAN, valid_until=ending.isoformat())), "scheduled")
    before = _history(graph, include_invalidated=True)[0]
    _apply(graph, _batch(_edge(APR)), "refresh")
    refreshed = _history(graph, include_invalidated=True)
    assert len(refreshed) == 1
    assert (
        refreshed[0].record_id,
        refreshed[0].valid_at,
        refreshed[0].valid_until,
    ) == (before.record_id, JAN, ending)
    _apply(graph, _batch(_edge(ending)), "at-scheduled-end")
    rows = _history(graph, include_invalidated=True)
    assert len(rows) == 2
    previous = next(row for row in rows if row.record_id == before.record_id)
    current = next(row for row in rows if row.record_id != before.record_id)
    assert (previous.valid_at, previous.valid_until) == (JAN, ending)
    assert (current.valid_at, current.invalid_at, current.valid_until) == (
        ending,
        None,
        None,
    )
    assert [row.record_id for row in _history(graph, as_of=ending)] == [
        current.record_id
    ]


def test_native_execution_honors_frozen_incarnation_and_lifecycle_on_replay(graph):
    pot, backend, _ = graph
    _apply(graph, _batch(_edge(JAN)), "old")
    _close(graph, _history(graph, include_invalidated=True)[0])
    batch = _batch(_edge(MAR))
    prepare_claim_assertions(
        batch,
        pot_id=pot,
        mutation_id="prepared-march",
        anchor=APR,
        query=backend.claim_query,
        definition=backend.definition,
    )
    resolution = batch.edge_upserts[0].resolution
    context = ProvenanceContext(
        mutation_id="prepared-march",
        prepared=PreparedMutation(
            pot_id=pot,
            mutation_id="prepared-march",
            request_fingerprint=request_fingerprint({"assert": "march"}),
            batch_fingerprint=batch_fingerprint(batch),
            expected_version=backend.mutation.current_version(pot),
            prepared_at=APR,
        ),
    )
    first = backend.mutation.compare_and_apply(
        batch,
        expected_pot_id=pot,
        expected_version=context.prepared.expected_version,
        provenance_context=context,
    )
    row = next(iter(_history(graph, as_of=MAR)))
    assert (row.record_id, row.valid_at) == (resolution.record_id, resolution.valid_at)
    assert row.valid_at == MAR
    assert first.execution.affected_claim_record_ids == (resolution.record_id,)
    _apply(graph, _batch(_edge(APR)), "later-refresh")
    before = _raw_claims(graph)
    version = backend.mutation.current_version(pot)
    assert (
        backend.mutation.compare_and_apply(
            batch,
            expected_pot_id=pot,
            expected_version=context.prepared.expected_version,
            provenance_context=context,
        )
        == first
    )
    assert backend.mutation.current_version(pot) == version
    assert _raw_claims(graph) == before


@pytest.mark.parametrize(
    "other_scope",
    [
        {"environment": None},
        {"environment": "staging"},
        {"environment": "prod", "subgraph": "episodic"},
        {"environment": "prod", "source_system": "ticketing"},
        {"environment": "prod", "source_ref": "file:other-owners"},
    ],
    ids=["missing-environment", "staging", "subgraph", "source-system", "source-ref"],
)
def test_singleton_replaces_only_same_stable_source_and_exact_scope(graph, other_scope):
    _apply(graph, _batch(_edge(JAN, environment="prod")), "same-source-old")
    original = _history(graph, include_invalidated=True)[0]
    _apply(graph, _batch(_edge(JAN, **other_scope)), "competing-scope")
    competing = next(
        row
        for row in _history(graph, include_invalidated=True)
        if row.record_id != original.record_id
    )
    preserved = _raw_claims(graph)[competing.record_id]
    result = _apply(
        graph, _batch(_edge(MAR, target="team:blue", environment="prod")), "replace"
    )
    rows = {row.record_id: row for row in _history(graph, include_invalidated=True)}
    assert rows[original.record_id].invalid_at == MAR
    assert rows[competing.record_id].invalid_at is None
    assert _raw_claims(graph)[competing.record_id] == preserved
    winner = next(row for row in rows.values() if row.object_key == "team:blue")
    assert set(result.execution.affected_claim_record_ids) == {
        original.record_id,
        winner.record_id,
    }
    assert {row.record_id for row in _history(graph, as_of=MAR)} == {
        competing.record_id,
        winner.record_id,
    }


def test_generated_source_identity_does_not_supersede_other_objects(graph):
    edges = [_edge(JAN), _edge(MAR, target="team:blue")]
    for edge in edges:
        edge.properties.pop("source_system")
        edge.properties.pop("source_ref")
    _apply(graph, _batch(edges[0]), "source-less-old")
    _apply(graph, _batch(edges[1]), "source-less-new")
    assert len(_history(graph, as_of=MAR)) == 2
    assert all(
        row.invalid_at is None for row in _history(graph, include_invalidated=True)
    )


def test_metadata_review_patches_exact_closed_record_without_reopening(graph):
    _apply(graph, _batch(_edge(JAN)), "old")
    original = _history(graph, include_invalidated=True)[0]
    _close(graph, original)
    _apply(graph, _batch(_edge(MAR)), "new")
    before = _raw_claims(graph)
    edge = _edge(
        APR,
        evidence_review_required=True,
        evidence_review_reason="source version changed",
    )
    edge.metadata_review_record_id = original.record_id
    result = _apply(graph, _batch(edge), "review-closed")
    rows = {row.record_id: row for row in _history(graph, include_invalidated=True)}
    reviewed = rows[original.record_id]
    assert (reviewed.valid_at, reviewed.invalid_at) == (JAN, FEB)
    assert reviewed.fact == original.fact
    assert reviewed.properties["evidence_review_required"] is True
    assert reviewed.properties["evidence_review_reason"] == "source version changed"
    assert result.execution.affected_claim_record_ids == (original.record_id,)
    current = next(row for row in rows.values() if row.record_id != original.record_id)
    assert _raw_claims(graph)[current.record_id] == before[current.record_id]
    assert [row.record_id for row in _history(graph, as_of=APR)] == [current.record_id]


def test_physical_reader_selector_distinguishes_same_key_incarnations(graph):
    _apply(graph, _batch(_edge(JAN)), "old")
    original = _history(graph, include_invalidated=True)[0]
    _close(graph, original)
    _apply(graph, _batch(_edge(MAR)), "new")
    rows = _history(graph, include_invalidated=True)
    current = next(row for row in rows if row.record_id != original.record_id)
    for row in (original, current):
        selected = _history(
            graph, include_invalidated=True, record_id_in=(row.record_id,)
        )
        assert len(selected) == 1
        assert selected[0].record_id == row.record_id
    assert _history(graph, include_invalidated=True, record_id_in=("absent",)) == []


def test_physical_reader_selector_preserves_legacy_uuid_alias(graph):
    pot, _, driver = graph
    _run(
        driver,
        "CREATE (a:Entity:Service {group_id:$pot,entity_key:'service:api'}),"
        "(b:Entity:Team {group_id:$pot,entity_key:'team:red'}),"
        "(a)-[:RELATES_TO {group_id:$pot,name:'OWNED_BY',subject_key:'service:api',"
        "object_key:'team:red',claim_key:'claim:legacy',uuid:'legacy-record'}]->(b)",
        pot=pot,
    )
    rows = _history(graph, record_id_in=("legacy-record",))
    assert len(rows) == 1
    assert rows[0].record_id == "legacy-record"
    assert "record_id" not in _raw_claims(graph)["legacy-record"]


def test_vector_reader_filters_physical_identity_and_environment_before_limit(
    graph, monkeypatch
):
    pot, backend, driver = graph
    index_name = "lifecycle_vector_" + uuid.uuid4().hex
    rows = [
        {"record_id": "prod-best", "environment": "prod", "embedding": [1.0, 0.0, 0.0]},
        {"record_id": "stage", "environment": "staging", "embedding": [1.0, 0.0, 0.0]},
        {"record_id": "missing", "environment": None, "embedding": [1.0, 0.0, 0.0]},
        {
            "record_id": "prod-selected",
            "environment": "prod",
            "embedding": [0.8, 0.2, 0.0],
        },
    ]
    _run(
        driver,
        "CREATE (a:Entity:Service {group_id:$pot,entity_key:'service:api'}),"
        "(b:Entity:Team {group_id:$pot,entity_key:'team:red'}) "
        "WITH a,b UNWIND $rows AS row CREATE (a)-[r:RELATES_TO]->(b) "
        "SET r.group_id=$pot,r.name='OWNED_BY',r.subject_key='service:api',"
        "r.object_key='team:red',r.record_id=row.record_id,r.environment=row.environment,"
        "r.lifecycle_embedding=row.embedding",
        pot=pot,
        rows=rows,
    )
    _run(
        driver,
        f"CREATE VECTOR INDEX {index_name} FOR ()-[r:RELATES_TO]-() "
        "ON (r.lifecycle_embedding) OPTIONS {indexConfig: {"
        "`vector.dimensions`:3,`vector.similarity_function`:'cosine'}}",
    )
    try:
        _run(driver, "CALL db.awaitIndexes(30)")
        monkeypatch.setattr(neo4j_reader, "_VECTOR_INDEX_NAME", index_name)
        reader = neo4j_reader.Neo4jClaimQueryStore(
            backend.settings,
            driver=driver,
            embedder=SimpleNamespace(embed=lambda _: (1.0, 0.0, 0.0)),
        )

        def no_lexical_fallback(*args):
            raise AssertionError("native vector query must supply the selected row")

        monkeypatch.setattr(reader, "_find_claims_lexical", no_lexical_fallback)

        def find(**filters):
            return reader.find_claims(
                ClaimQueryFilter(
                    pot_id=pot,
                    predicate_in=("OWNED_BY",),
                    fact_query="ownership",
                    **filters,
                )
            )

        assert {row.record_id for row in find(environment_in=("PrOd",))} == {
            "prod-best",
            "prod-selected",
        }
        assert [row.record_id for row in find(environment_in=("",))] == ["missing"]
        assert [
            row.record_id
            for row in find(
                environment_in=("prod",), record_id_in=("prod-selected",), limit=1
            )
        ] == ["prod-selected"]
    finally:
        _run(driver, f"DROP INDEX {index_name} IF EXISTS")


@pytest.mark.parametrize(
    "field,value",
    [
        ("fact", "Changed fact"),
        ("description", "Changed description"),
        ("truth", "agent_claim"),
    ],
)
def test_ordinary_refresh_rejects_changed_content_atomically(graph, field, value):
    pot, backend, _ = graph
    _apply(graph, _batch(_edge(JAN, description="Original description")), "original")
    before = _raw_claims(graph)
    version = backend.mutation.current_version(pot)
    batch = _batch(_edge(MAR, **{field: value}))
    with pytest.raises(ReconciliationApplyError, match="explicit correction"):
        _apply(graph, batch, "changed-content")
    assert _raw_claims(graph) == before
    assert backend.mutation.current_version(pot) == version
    assert (
        backend.mutation.lookup_execution(
            batch, expected_pot_id=pot, mutation_id="changed-content"
        ).state
        == "absent"
    )


def test_lifecycle_effects_roll_back_with_revision_and_receipt(graph, monkeypatch):
    pot, backend, _ = graph
    _apply(graph, _batch(_edge(JAN)), "original")
    before = _raw_claims(graph)
    version = backend.mutation.current_version(pot)

    async def fail_after_assertions(*args):
        raise RuntimeError("injected after lifecycle writes")

    monkeypatch.setattr(Neo4jGraphWriter, "delete_edges", fail_after_assertions)
    batch = _batch(_edge(MAR, target="team:blue"))
    with pytest.raises(ReconciliationApplyError, match="injected after lifecycle"):
        _apply(graph, batch, "failed-replacement")
    assert _raw_claims(graph) == before
    assert backend.mutation.current_version(pot) == version
    assert (
        backend.mutation.lookup_execution(
            batch, expected_pot_id=pot, mutation_id="failed-replacement"
        ).state
        == "absent"
    )


def test_snapshot_roundtrip_preserves_same_key_incarnations_and_import_replay(graph):
    pot, backend, _ = graph
    _apply(graph, _batch(_edge(JAN)), "old")
    _close(graph, _history(graph, include_invalidated=True)[0])
    _apply(graph, _batch(_edge(MAR)), "new")
    payload = backend.snapshot.export_data(pot_id=pot)
    assert len(payload["claims"]) == 2
    assert len({row["claim_key"] for row in payload["claims"]}) == 1
    before = sorted(
        (row.record_id, row.valid_at, row.invalid_at)
        for row in _history(graph, include_invalidated=True)
    )
    backend.mutation.reset_pot(pot, mutation_id="clear-for-import")
    first = backend.snapshot.import_data(pot_id=pot, payload=payload)
    revision = backend.mutation.current_version(pot)
    second = backend.snapshot.import_data(pot_id=pot, payload=payload)
    assert second.metadata == first.metadata
    assert backend.mutation.current_version(pot) == revision
    assert (
        sorted(
            (row.record_id, row.valid_at, row.invalid_at)
            for row in _history(graph, include_invalidated=True)
        )
        == before
    )
    assert backend.snapshot.export_data(pot_id=pot) == normalize_snapshot_import(
        payload, pot_id=pot
    )


@pytest.mark.parametrize("collision", ["entity", "unrelated_claim"])
def test_refresh_rejects_global_incarnation_identity_collision(graph, collision):
    pot, backend, driver = graph
    _apply(graph, _batch(_edge(JAN)), "seed")
    original = _history(graph)[0]
    if collision == "entity":
        _run(
            driver,
            "MATCH (n:Entity {group_id:$pot,entity_key:'service:api'}) "
            "SET n.record_id=$identity",
            pot=pot,
            identity=original.record_id,
        )
    else:
        _run(
            driver,
            "MATCH (n:Entity {group_id:$pot,entity_key:'team:red'}) "
            "CREATE (n)-[:RELATES_TO {group_id:$pot, record_id:$identity, "
            "name:'RELATED_TO',subject_key:'team:red',object_key:'team:red'}]->(n)",
            pot=pot,
            identity=original.record_id,
        )
    before = _raw_claims(graph)
    version = backend.mutation.current_version(pot)
    batch = _batch(_edge(MAR))
    with pytest.raises(ReconciliationApplyError, match="incarnation"):
        _apply(graph, batch, "colliding-refresh")
    assert _raw_claims(graph) == before
    assert backend.mutation.current_version(pot) == version
    assert (
        backend.mutation.lookup_execution(
            batch, expected_pot_id=pot, mutation_id="colliding-refresh"
        ).state
        == "absent"
    )


def test_retired_endpoint_requires_explicit_reactivation(graph):
    pot, backend, driver = graph
    _apply(graph, _batch(_edge(JAN)), "seed")
    _run(
        driver,
        "MATCH (n:Entity {group_id:$pot,entity_key:'service:api'}) "
        "SET n.retired=true",
        pot=pot,
    )
    before = _raw_claims(graph)
    version = backend.mutation.current_version(pot)
    batch = _batch(_edge(MAR))
    with pytest.raises(ReconciliationApplyError, match="endpoints"):
        _apply(graph, batch, "inactive-endpoint")
    assert _raw_claims(graph) == before
    assert backend.mutation.current_version(pot) == version
    assert (
        backend.mutation.lookup_execution(
            batch, expected_pot_id=pot, mutation_id="inactive-endpoint"
        ).state
        == "absent"
    )
    batch.entity_upserts[0].properties["retired"] = False
    assert _apply(graph, batch, "reactivated").ok
    assert len(_history(graph)) == 1
