"""UTC instant semantics across native reads, writes and snapshot imports."""

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import os
from types import SimpleNamespace
import uuid

import pytest

from potpie_context_engine.adapters.outbound.graph import neo4j_reader
from potpie_context_engine.adapters.outbound.graph import neo4j_snapshot_write
from potpie_context_engine.adapters.outbound.graph._mutation_execution import (
    CompletedMutationExecution,
    execution_receipts_to_json,
)
from potpie_context_engine.adapters.outbound.graph.backends.neo4j_backend import (
    Neo4jGraphBackend,
)
from potpie_context_engine.core.graph_mutations import (
    EdgeUpsert,
    EntityUpsert,
    ProvenanceContext,
)
from potpie_context_engine.core.graph_snapshot_mutation import prepare_snapshot_import
from potpie_context_engine.core.mutation_execution import (
    ExecutionMetadata,
    batch_fingerprint,
)
from potpie_context_engine.core.ports.claim_query import ClaimQueryFilter
from potpie_context_engine.core.reconciliation import (
    MutationBatch,
    MutationResult,
    MutationSummary,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.environ.get("PROTOCOL_TEST_NEO4J_URI"),
        reason="PROTOCOL_TEST_NEO4J_URI must name a disposable test server",
    ),
]
UTC = timezone.utc
OFFSET = timezone(timedelta(hours=5, minutes=30))
START = datetime(2026, 1, 1, tzinfo=UTC)
END = START + timedelta(microseconds=2)


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
    pot = "temporal-" + uuid.uuid4().hex
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


def _legacy_claims(graph, claims):
    pot, _, driver = graph
    _run(
        driver,
        "CREATE (a:Entity:Service {group_id:$pot,entity_key:'service:api'}),"
        "(b:Entity:Service:Wanted {group_id:$pot,entity_key:'service:db'}) "
        "WITH a,b UNWIND $claims AS claim CREATE (a)-[r:RELATES_TO]->(b) "
        "SET r += claim,r.group_id=$pot,r.name='DEPENDS_ON',"
        "r.subject_key='service:api',r.object_key='service:db'",
        pot=pot,
        claims=claims,
    )


def _find(graph, **filters):
    pot, backend, _ = graph
    return backend.claim_query.find_claims(ClaimQueryFilter(pot_id=pot, **filters))


def _payload(*, alias=None):
    return {
        "format_version": "2",
        "pot_id": "source",
        "entities": [
            {
                "key": "service:api",
                "labels": ["Entity", "Service"],
                "properties": {"record_id": "entity:api"},
            },
            {
                "key": "service:db",
                "labels": ["Entity", "Service"],
                "properties": {"record_id": "entity:db"},
            },
        ],
        "claims": [
            {
                "record_id": "claim:record",
                "claim_key": "claim:source:offset",
                "predicate": "DEPENDS_ON",
                "subject_key": "service:api",
                "object_key": "service:db",
                "valid_at": START.isoformat().replace("+00:00", "Z"),
                "valid_until": END.astimezone(OFFSET).isoformat(),
                "observed_at": "2026-01-01T05:30:00+05:30",
                "properties": {
                    "valid_from": (
                        START.astimezone(OFFSET).isoformat() if alias is None else alias
                    )
                },
            }
        ],
    }


def _physical_claims(graph):
    pot, _, driver = graph
    return _run(
        driver,
        "MATCH ()-[r:RELATES_TO {group_id:$pot}]->() RETURN properties(r) AS props",
        pot=pot,
    )


def _receipts(graph):
    pot, _, driver = graph
    return _run(
        driver,
        "MATCH (r:PotpieMutationReceipt {pot_id:$pot}) RETURN r.mutation_id AS id,r.payload AS payload ORDER BY id",
        pot=pot,
    )


@pytest.mark.parametrize("representation", ["offset", "z", "native"])
@pytest.mark.parametrize("end_field", ["valid_until", "invalid_at"])
def test_legacy_instants_obey_microsecond_half_open_boundaries_and_range_alias(
    graph, representation, end_field
):
    def encode(value):
        if representation == "native":
            return value.astimezone(OFFSET)
        if representation == "z":
            return value.isoformat().replace("+00:00", "Z")
        return value.astimezone(OFFSET).isoformat()

    _legacy_claims(
        graph,
        [
            {
                "record_id": "legacy",
                "valid_from": encode(START),
                end_field: encode(END),
            }
        ],
    )
    for point, expected in (
        (START - timedelta(microseconds=1), []),
        (START, ["legacy"]),
        (END - timedelta(microseconds=1), ["legacy"]),
        (END, []),
        (START.astimezone(OFFSET), ["legacy"]),
    ):
        assert [row.record_id for row in _find(graph, as_of=point)] == expected
    rows = _find(graph, as_of=START, valid_at_after=START, valid_at_before=START)
    assert [row.record_id for row in rows] == ["legacy"]


def test_snapshot_import_canonicalizes_physical_times_but_replays_original_identity(
    graph,
):
    pot, backend, _ = graph
    payload = _payload()
    original = deepcopy(payload)
    batch, _ = prepare_snapshot_import(
        pot_id=pot, payload=payload, expected_version=0, mutation_id="offset-import"
    )
    fingerprint = batch_fingerprint(batch)
    first = backend.snapshot.import_data(
        pot_id=pot, payload=payload, mutation_id="offset-import"
    )
    assert payload == original
    assert first.metadata["execution"]["batch_fingerprint"] == fingerprint
    props = _physical_claims(graph)[0]["props"]
    assert props["valid_at"] == START.isoformat()
    assert "valid_from" not in props
    assert props["valid_until"] == END.isoformat()
    assert props["observed_at"] == START.isoformat()
    assert [
        row.record_id for row in _find(graph, as_of=END - timedelta(microseconds=1))
    ] == ["claim:record"]
    assert _find(graph, as_of=END) == []
    receipt_bytes = _receipts(graph)
    version = backend.mutation.current_version(pot)
    replay = backend.snapshot.import_data(
        pot_id=pot, payload=payload, mutation_id="offset-import"
    )
    assert replay.metadata == first.metadata
    assert _receipts(graph) == receipt_bytes
    assert backend.mutation.current_version(pot) == version
    assert (
        backend.mutation.lookup_execution(
            batch, expected_pot_id=pot, mutation_id="offset-import"
        ).result.execution.batch_fingerprint
        == fingerprint
    )


def test_import_compares_existing_offset_identity_without_rewriting_legacy_row(graph):
    pot, backend, driver = graph
    payload = _payload()
    backend.snapshot.import_data(
        pot_id=pot, payload=payload, mutation_id="first-import"
    )
    _run(
        driver,
        "MATCH ()-[r:RELATES_TO {group_id:$pot}]->() "
        "SET r.valid_at=$start,r.valid_from=$start,r.valid_until=$end,r.observed_at=$start",
        pot=pot,
        start=START.astimezone(OFFSET).isoformat(),
        end=END.astimezone(OFFSET).isoformat(),
    )
    before = _physical_claims(graph)
    result = backend.snapshot.import_data(
        pot_id=pot, payload=payload, mutation_id="same-record-import"
    )
    assert result.metadata["administrative_summary"] == {
        "imported_entities": 0,
        "imported_claims": 0,
    }
    assert _physical_claims(graph) == before
    assert [row.record_id for row in _find(graph, as_of=START)] == ["claim:record"]


def test_typed_native_times_export_and_same_identity_import_preserve_native_values(
    graph,
):
    pot, backend, driver = graph
    _legacy_claims(
        graph,
        [
            {
                "record_id": "typed:claim",
                "claim_key": f"claim:{pot}:typed",
                "valid_at": START.astimezone(OFFSET),
                "valid_from": START.astimezone(OFFSET),
                "valid_until": END.astimezone(OFFSET),
                "observed_at": START.astimezone(OFFSET),
            }
        ],
    )
    _run(
        driver,
        "MATCH (e:Entity {group_id:$pot}) "
        "SET e.record_id='typed:entity:' + e.entity_key,e.deployed_at=$deployed",
        pot=pot,
        deployed=START.astimezone(OFFSET),
    )
    before_claims = _physical_claims(graph)
    before_entities = _run(
        driver,
        "MATCH (e:Entity {group_id:$pot}) RETURN properties(e) AS props "
        "ORDER BY e.entity_key",
        pot=pot,
    )
    assert callable(before_claims[0]["props"]["valid_at"].to_native)
    assert callable(before_entities[0]["props"]["deployed_at"].to_native)

    payload = backend.snapshot.export_data(pot_id=pot)
    claim = payload["claims"][0]
    assert datetime.fromisoformat(claim["valid_at"]) == START
    assert datetime.fromisoformat(claim["valid_until"]) == END
    assert datetime.fromisoformat(claim["properties"]["valid_from"]) == START
    assert all(
        datetime.fromisoformat(row["properties"]["deployed_at"]) == START
        for row in payload["entities"]
    )

    result = backend.snapshot.import_data(
        pot_id=pot, payload=payload, mutation_id="typed-same-identity"
    )
    assert result.metadata["administrative_summary"] == {
        "imported_entities": 0,
        "imported_claims": 0,
    }
    assert _physical_claims(graph) == before_claims
    assert (
        _run(
            driver,
            "MATCH (e:Entity {group_id:$pot}) RETURN properties(e) AS props "
            "ORDER BY e.entity_key",
            pot=pot,
        )
        == before_entities
    )
    assert backend.snapshot.export_data(pot_id=pot) == payload
    assert [row.record_id for row in _find(graph, as_of=START)] == ["typed:claim"]


@pytest.mark.parametrize("alias", ["2030-01-01T00:00:00+00:00", "invalid-time"])
def test_new_import_rejects_conflicting_or_invalid_start_alias_atomically(graph, alias):
    pot, backend, _ = graph
    with pytest.raises(ValueError, match="(?i)(alias|timestamp|valid_from)"):
        backend.snapshot.import_data(
            pot_id=pot, payload=_payload(alias=alias), mutation_id="bad-alias"
        )
    assert backend.mutation.current_version(pot) == 0
    assert _physical_claims(graph) == []
    assert _receipts(graph) == []


@pytest.mark.parametrize(
    "explicit", [False, True], ids=["source-defaults", "explicit-overrides"]
)
def test_raw_occurrence_and_observation_use_separate_frozen_source_times(
    graph, explicit
):
    pot, backend, _ = graph
    occurred = datetime(2020, 1, 1, tzinfo=UTC)
    received = datetime(2020, 2, 1, tzinfo=UTC)
    props = {"fact": "Activity touched API", "truth": "source_observation"}
    if explicit:
        props.update(
            valid_at="2019-01-01T05:30:00+05:30", observed_at="2020-03-01T00:00:00Z"
        )
    backend.mutation.apply(
        MutationBatch(
            entity_upserts=[
                EntityUpsert("activity:test", ("Activity",)),
                EntityUpsert("service:api", ("Service",)),
            ],
            edge_upserts=[EdgeUpsert("TOUCHED", "activity:test", "service:api", props)],
        ),
        expected_pot_id=pot,
        provenance_context=ProvenanceContext(
            mutation_id="source-times",
            event_occurred_at=occurred,
            event_received_at=received,
        ),
    )
    row = _find(graph, include_invalidated=True)[0]
    assert row.valid_at == (datetime(2019, 1, 1, tzinfo=UTC) if explicit else occurred)
    assert row.observed_at == (
        datetime(2020, 3, 1, tzinfo=UTC) if explicit else received
    )


def test_vector_filters_temporal_candidates_and_endpoint_state_before_limit(
    graph, monkeypatch
):
    pot, backend, driver = graph
    _legacy_claims(
        graph,
        [
            {
                "record_id": "expired",
                "valid_from": "2025-01-01T00:00:00Z",
                "valid_until": START.astimezone(OFFSET).isoformat(),
                "temporal_embedding": [1.0, 0.0, 0.0],
            },
            {
                "record_id": "eligible",
                "valid_from": START.astimezone(OFFSET).isoformat(),
                "temporal_embedding": [0.8, 0.2, 0.0],
            },
        ],
    )
    _run(
        driver,
        "MATCH (a:Entity {group_id:$pot,entity_key:'service:api'}) "
        "CREATE (b:Entity:Wanted {group_id:$pot,entity_key:'service:retired',retired:true}),"
        "(a)-[:RELATES_TO {group_id:$pot,name:'DEPENDS_ON',record_id:'retired-endpoint',temporal_embedding:[1.0,0.0,0.0]}]->(b)",
        pot=pot,
    )
    index = "temporal_vector_" + uuid.uuid4().hex
    _run(
        driver,
        f"CREATE VECTOR INDEX {index} FOR ()-[r:RELATES_TO]-() ON (r.temporal_embedding) "
        "OPTIONS {indexConfig:{`vector.dimensions`:3,`vector.similarity_function`:'cosine'}}",
    )
    try:
        _run(driver, "CALL db.awaitIndexes(30)")
        monkeypatch.setattr(neo4j_reader, "_VECTOR_INDEX_NAME", index)
        reader = neo4j_reader.Neo4jClaimQueryStore(
            backend.settings,
            driver=driver,
            embedder=SimpleNamespace(embed=lambda _: (1.0, 0.0, 0.0)),
        )

        def no_fallback(*args, **kwargs):
            raise AssertionError(
                "native vector candidates must contain the eligible claim"
            )

        monkeypatch.setattr(reader, "_find_claims_lexical", no_fallback)
        rows = reader.find_claims(
            ClaimQueryFilter(
                pot_id=pot,
                as_of=START,
                fact_query="dependency",
                endpoint_label="Wanted",
                limit=1,
            )
        )
        assert [row.record_id for row in rows] == ["eligible"]
    finally:
        _run(driver, f"DROP INDEX {index} IF EXISTS")


def test_legacy_snapshot_receipt_replays_before_strict_alias_canonicalization(
    graph, monkeypatch
):
    pot, backend, driver = graph
    payload = _payload(alias="2030-01-01T00:00:00+00:00")
    batch, context = prepare_snapshot_import(
        pot_id=pot, payload=payload, expected_version=0, mutation_id="legacy-import"
    )
    fingerprint = batch_fingerprint(batch)
    completed = MutationResult(
        ok=True,
        mutation_id="legacy-import",
        mutation_summary=MutationSummary(
            entity_upserts_applied=2, edge_upserts_applied=1
        ),
        execution=ExecutionMetadata(
            pot_id=pot,
            applied_at=START,
            version_before=0,
            version_after=1,
            batch_fingerprint=fingerprint,
            request_fingerprint=context.prepared.request_fingerprint,
            affected_claim_record_ids=("claim:record",),
        ),
    )
    receipt = CompletedMutationExecution(pot, "legacy-import", fingerprint, completed)
    _run(
        driver,
        "CREATE (:PotpieRevision {pot_id:$pot,version:1}),"
        "(:PotpieMutationReceipt {pot_id:$pot,mutation_id:'legacy-import',payload:$payload})",
        pot=pot,
        payload=json.dumps(execution_receipts_to_json([receipt]), sort_keys=True),
    )
    original_receipts = _receipts(graph)

    def unexpected(*args):
        pytest.fail("completed import must replay before enforcing new alias policy")

    monkeypatch.setattr(neo4j_snapshot_write, "canonicalize_snapshot_times", unexpected)
    recovered = backend.snapshot.import_data(
        pot_id=pot, payload=payload, mutation_id="legacy-import"
    )
    assert recovered.metadata["execution"] == completed.execution.to_dict()
    assert backend.mutation.current_version(pot) == 1
    assert _receipts(graph) == original_receipts
    assert _physical_claims(graph) == []
