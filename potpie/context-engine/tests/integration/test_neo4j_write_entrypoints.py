"""Supported administrative writes share native revision and receipt boundaries."""

from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
import asyncio
import json
import os
import threading
from types import SimpleNamespace
import uuid

import pytest

from potpie_context_engine.adapters.outbound.graph.backends.neo4j_backend import (
    Neo4jGraphBackend,
)
from potpie_context_engine.adapters.outbound.graph.neo4j_admin import Neo4jAnalytics
from potpie_context_engine.adapters.outbound.graph.neo4j_writer import Neo4jGraphWriter
from potpie_context_engine.adapters.outbound.graph import (
    neo4j_atomic,
    neo4j_snapshot_write,
)
from potpie_context_engine.core.errors import (
    GraphMutationVersionConflict,
    MutationBatchValidationError,
    ReconciliationApplyError,
)
from potpie_context_engine.adapters.outbound.graph._mutation_execution import (
    MutationExecutionReuseError,
)
from potpie_context_engine.core.graph_snapshot_mutation import prepare_snapshot_import
from potpie_context_engine.core.graph_mutations import (
    EdgeUpsert,
    EntityUpsert,
    ProvenanceContext,
)
from potpie_context_engine.core.mutation_execution import (
    PreparedMutation,
    batch_fingerprint,
)
from potpie_context_engine.core.reconciliation import MutationBatch

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.environ.get("PROTOCOL_TEST_NEO4J_URI"),
        reason="PROTOCOL_TEST_NEO4J_URI must name a disposable test server",
    ),
]


@pytest.fixture
def graph():
    from neo4j import GraphDatabase

    settings = SimpleNamespace(
        is_enabled=lambda: True,
        neo4j_uri=lambda: os.environ["PROTOCOL_TEST_NEO4J_URI"],
        neo4j_user=lambda: os.environ.get("PROTOCOL_TEST_NEO4J_USERNAME", "neo4j"),
        neo4j_password=lambda: os.environ.get("PROTOCOL_TEST_NEO4J_PASSWORD", ""),
    )
    with GraphDatabase.driver(
        settings.neo4j_uri(), auth=(settings.neo4j_user(), settings.neo4j_password())
    ) as driver:
        yield "entrypoints-" + uuid.uuid4().hex, Neo4jGraphBackend(settings), driver


def _run(driver, query, **params):
    with driver.session() as session:
        return [dict(row) for row in session.run(query, **params)]


def _receipts(driver, pot):
    rows = _run(
        driver,
        "MATCH (r:PotpieMutationReceipt {pot_id:$pot}) RETURN r.payload AS payload",
        pot=pot,
    )
    return [json.loads(row["payload"])[0] for row in rows]


def test_reset_then_old_execution_replay_keeps_graph_empty_and_revision_fixed(graph):
    pot, backend, driver = graph
    batch = MutationBatch(entity_upserts=[EntityUpsert("service:api", ("Service",))])
    context = ProvenanceContext(mutation_id="before-reset")
    first = backend.mutation.apply(
        batch, expected_pot_id=pot, provenance_context=context
    )
    assert backend.mutation.reset_pot(pot)["ok"]
    assert backend.mutation.current_version(pot) == 2
    assert (
        backend.mutation.apply(batch, expected_pot_id=pot, provenance_context=context)
        == first
    )
    assert backend.mutation.current_version(pot) == 2
    assert _run(
        driver, "MATCH (e:Entity {group_id:$pot}) RETURN count(e) AS count", pot=pot
    ) == [{"count": 0}]


def test_supported_invalidation_publishes_revision_and_affected_record_receipt(graph):
    pot, backend, driver = graph
    _run(
        driver,
        "CREATE (a:Entity:Service {group_id:$pot,entity_key:'service:api'}),"
        "(b:Entity:Service {group_id:$pot,entity_key:'service:db'}),"
        "(a)-[:RELATES_TO {group_id:$pot,name:'DEPENDS_ON',subject_key:'service:api',"
        "object_key:'service:db',claim_key:'claim:entrypoint',uuid:'record:entrypoint',record_id:'record:entrypoint'}]->(b)",
        pot=pot,
    )
    assert backend.mutation.invalidate(pot_id=pot, claim_keys=["claim:entrypoint"]) == 1
    assert backend.mutation.current_version(pot) == 1
    receipt = _receipts(driver, pot)[0]["result"]["execution"]
    assert receipt["affected_claim_record_ids"] == ["record:entrypoint"]
    assert (receipt["version_before"], receipt["version_after"]) == (0, 1)


def _malformed_entities(driver, pot):
    _run(
        driver,
        "CREATE (:Entity:Repository {group_id:$pot,entity_key:'service:api',record_id:'entity:api'}),"
        "(:Entity:Repository {group_id:$pot,entity_key:'service:db',record_id:'entity:db'})",
        pot=pot,
    )


def test_combined_entity_repair_advances_once_with_receipt_and_noop_does_not(graph):
    pot, backend, driver = graph
    _malformed_entities(driver, pot)
    report = backend.analytics.repair(
        pot, targets=("entity_summaries", "entity_labels")
    )
    assert report.repaired == {"entity_summaries": 2, "entity_labels": 2}
    assert backend.mutation.current_version(pot) == 1
    receipts = _receipts(driver, pot)
    assert len(receipts) == 1
    result = receipts[0]["result"]
    assert result["mutation_summary"]["entity_upserts_applied"] == 2
    assert (
        result["execution"]["version_before"],
        result["execution"]["version_after"],
    ) == (0, 1)
    assert result["execution"]["request_fingerprint"] is not None
    rows = _run(
        driver,
        "MATCH (e:Entity {group_id:$pot}) RETURN labels(e) AS labels,e.summary AS summary",
        pot=pot,
    )
    assert len(rows) == 2
    assert all(
        "Service" in row["labels"]
        and "Repository" not in row["labels"]
        and row["summary"]
        for row in rows
    )
    noop = backend.analytics.repair(pot, targets=("entity_summaries", "entity_labels"))
    assert noop.repaired == {"entity_summaries": 0, "entity_labels": 0}
    assert backend.mutation.current_version(pot) == 1
    assert _receipts(driver, pot) == receipts


def test_entity_repair_failure_rolls_back_all_entities_revision_and_receipt(
    graph, monkeypatch
):
    pot, backend, driver = graph
    _malformed_entities(driver, pot)

    async def fail(*args):
        raise RuntimeError("injected repair failure after entities")

    monkeypatch.setattr(Neo4jGraphWriter, "upsert_edges", fail)
    with pytest.raises(ReconciliationApplyError, match="injected repair"):
        backend.analytics.repair(pot, targets=("entity_summaries", "entity_labels"))
    assert backend.mutation.current_version(pot) == 0
    assert _receipts(driver, pot) == []
    rows = _run(
        driver,
        "MATCH (e:Entity {group_id:$pot}) RETURN labels(e) AS labels,e.summary AS summary",
        pot=pot,
    )
    assert len(rows) == 2
    assert all(
        "Repository" in row["labels"]
        and "Service" not in row["labels"]
        and row["summary"] is None
        for row in rows
    )


def test_entity_repair_rejects_stale_selection_before_repreparing(graph, monkeypatch):
    pot, backend, driver = graph
    _malformed_entities(driver, pot)
    original = Neo4jAnalytics._entity_rows
    concurrent = Neo4jGraphBackend(backend.settings)
    interrupted = False

    async def read_then_write(self, selected_pot):
        nonlocal interrupted
        rows = await original(self, selected_pot)
        if not interrupted:
            interrupted = True
            await concurrent.mutation.apply_async(
                MutationBatch(
                    entity_upserts=[
                        EntityUpsert(
                            "service:api",
                            ("Service",),
                            {"summary": "new authored summary"},
                        )
                    ]
                ),
                expected_pot_id=selected_pot,
            )
        return rows

    monkeypatch.setattr(Neo4jAnalytics, "_entity_rows", read_then_write)
    report = backend.analytics.repair(
        pot, targets=("entity_summaries", "entity_labels")
    )
    assert report.repaired == {"entity_summaries": 1, "entity_labels": 1}
    assert backend.mutation.current_version(pot) == 2
    assert (
        backend.claim_query.entity_properties(pot_id=pot, entity_key="service:api")[
            "summary"
        ]
        == "new authored summary"
    )
    receipts = _receipts(driver, pot)
    assert len(receipts) == 2
    assert sorted(
        item["result"]["execution"]["version_after"] for item in receipts
    ) == [1, 2]


def _snapshot_payload():
    return {
        "format_version": "2",
        "pot_id": "source",
        "entities": [
            {
                "key": "service:api",
                "labels": ["Entity", "Service"],
                "properties": {
                    "name": "API",
                    "uuid": "uuid:api",
                    "record_id": "record:api",
                },
            },
            {
                "key": "service:db",
                "labels": ["Entity", "Service"],
                "properties": {"name": "DB"},
            },
        ],
        "claims": [
            {
                "claim_key": "claim:source:snapshot",
                "record_id": "record:claim",
                "predicate": "DEPENDS_ON",
                "subject_key": "service:api",
                "object_key": "service:db",
                "truth": "source_observation",
                "fact": "API depends on DB",
                "source_ref": "fixture:import",
                "valid_at": "2026-01-01T00:00:00+00:00",
                "properties": {
                    "uuid": "uuid:claim",
                    "nested": {"preserve": [1, None, "two"]},
                },
            }
        ],
    }


def test_snapshot_import_preserves_incarnation_ids_and_replays_stable_manifest(graph):
    pot, backend, driver = graph
    payload = _snapshot_payload()
    first = backend.snapshot.import_data(pot_id=pot, payload=payload)
    assert first.metadata["mutation_id"].startswith("snapshot:")
    execution = first.metadata["execution"]
    assert (execution["version_before"], execution["version_after"]) == (0, 1)
    assert execution["affected_claim_record_ids"] == ["record:claim"]
    assert first.metadata["administrative_summary"] == {
        "imported_entities": 2,
        "imported_claims": 1,
    }
    restarted = Neo4jGraphBackend(backend.settings)
    assert restarted.snapshot.import_data(pot_id=pot, payload=payload) == first
    assert backend.mutation.current_version(pot) == 1
    assert len(_receipts(driver, pot)) == 1
    exported = backend.snapshot.export_data(pot_id=pot)
    api = next(row for row in exported["entities"] if row["key"] == "service:api")
    assert api["properties"]["uuid"] == "uuid:api"
    assert api["properties"]["record_id"] == "record:api"
    assert exported["claims"][0]["record_id"] == "record:claim"
    assert exported["claims"][0]["properties"]["uuid"] == "uuid:claim"
    assert exported["claims"][0]["properties"]["nested"] == {
        "preserve": [1, None, "two"]
    }


def test_empty_snapshot_import_has_stable_receipt_without_revision_advance(graph):
    pot, backend, driver = graph
    payload = {"format_version": "2", "pot_id": "source", "entities": [], "claims": []}
    first = backend.snapshot.import_data(pot_id=pot, payload=payload)
    assert backend.snapshot.import_data(pot_id=pot, payload=payload) == first
    execution = first.metadata["execution"]
    assert (execution["version_before"], execution["version_after"]) == (0, 0)
    assert backend.mutation.current_version(pot) == 0
    assert len(_receipts(driver, pot)) == 1


def test_snapshot_reset_replay_requires_explicit_new_identity_to_restore(graph):
    pot, backend, driver = graph
    payload = _snapshot_payload()
    first = backend.snapshot.import_data(pot_id=pot, payload=payload)
    reset = backend.mutation.reset_pot(pot)
    assert reset["ok"]
    assert reset["group_id_nodes_before"] == 2
    assert reset["group_id_nodes_remaining"] == 0
    assert reset["mutation_id"]
    assert reset["execution"]["version_after"] == 2
    assert backend.mutation.current_version(pot) == 2
    assert backend.snapshot.import_data(pot_id=pot, payload=payload) == first
    assert backend.snapshot.export_data(pot_id=pot)["entities"] == []
    restored = backend.snapshot.import_data(
        pot_id=pot, payload=payload, mutation_id="intentional-restore"
    )
    assert restored.metadata["execution"]["affected_claim_record_ids"] == [
        "record:claim"
    ]
    assert (
        restored.metadata["execution"]["version_before"],
        restored.metadata["execution"]["version_after"],
    ) == (2, 3)
    receipts = _receipts(driver, pot)
    assert len(receipts) == 3
    reset_receipt = next(
        item
        for item in receipts
        if item["result"]["administrative_summary"].get("group_id_nodes_before") == 2
    )
    assert reset_receipt["result"]["execution"]["affected_claim_record_ids"] == [
        "record:claim"
    ]


def test_snapshot_failure_rolls_back_entities_revision_and_receipt(graph, monkeypatch):
    pot, backend, driver = graph

    def fail(*args):
        raise RuntimeError("injected failure after imported entities")

    monkeypatch.setattr(neo4j_snapshot_write, "stored_claim", fail)
    with pytest.raises(RuntimeError, match="after imported entities"):
        backend.snapshot.import_data(pot_id=pot, payload=_snapshot_payload())
    assert backend.mutation.current_version(pot) == 0
    assert backend.snapshot.export_data(pot_id=pot)["entities"] == []
    assert _receipts(driver, pot) == []


def test_snapshot_conflict_rejects_every_effect_and_receipt(graph):
    pot, backend, driver = graph
    payload = _snapshot_payload()
    backend.snapshot.import_data(pot_id=pot, payload=payload)
    changed = _snapshot_payload()
    changed["entities"][0]["properties"]["name"] = "conflicting API"
    with pytest.raises(ValueError, match="conflicts with target"):
        backend.snapshot.import_data(pot_id=pot, payload=changed)
    assert backend.mutation.current_version(pot) == 1
    assert len(_receipts(driver, pot)) == 1


def test_concurrent_identical_snapshot_import_returns_one_receipt(graph):
    pot, backend, driver = graph
    barrier = threading.Barrier(2)

    def import_one(_):
        barrier.wait(timeout=10)
        return Neo4jGraphBackend(backend.settings).snapshot.import_data(
            pot_id=pot, payload=_snapshot_payload()
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        first, second = list(pool.map(import_one, range(2)))
    assert first == second
    assert backend.mutation.current_version(pot) == 1
    assert len(_receipts(driver, pot)) == 1


def test_reset_receipt_failure_rolls_back_deleted_graph_and_revision(
    graph, monkeypatch
):
    pot, backend, driver = graph
    backend.snapshot.import_data(pot_id=pot, payload=_snapshot_payload())

    def fail(*args):
        raise RuntimeError("injected reset receipt failure")

    monkeypatch.setattr(neo4j_atomic, "execution_receipts_to_json", fail)
    with pytest.raises(RuntimeError, match="reset receipt failure"):
        backend.mutation.reset_pot(pot)
    assert backend.mutation.current_version(pot) == 1
    assert len(backend.snapshot.export_data(pot_id=pot)["entities"]) == 2
    assert len(_receipts(driver, pot)) == 1


@pytest.mark.parametrize("kind", ["reset", "snapshot"])
def test_administrative_batches_enforce_frozen_revision(graph, kind):
    pot, backend, driver = graph
    if kind == "snapshot":
        batch, context = prepare_snapshot_import(
            pot_id=pot, payload=_snapshot_payload(), expected_version=0
        )
    else:
        batch = MutationBatch(reset_pot=True)
        fingerprint = batch_fingerprint(batch)
        context = ProvenanceContext(
            mutation_id="frozen-reset",
            prepared=PreparedMutation(
                pot_id=pot,
                mutation_id="frozen-reset",
                request_fingerprint="reset-request",
                batch_fingerprint=fingerprint,
                expected_version=0,
                prepared_at=datetime.now(timezone.utc),
            ),
        )
    backend.mutation.apply(
        MutationBatch(entity_upserts=[EntityUpsert("service:existing", ("Service",))]),
        expected_pot_id=pot,
    )
    with pytest.raises(GraphMutationVersionConflict):
        backend.mutation.compare_and_apply(
            batch, expected_pot_id=pot, expected_version=1, provenance_context=context
        )
    assert backend.mutation.current_version(pot) == 1
    assert len(_receipts(driver, pot)) == 1
    assert _run(
        driver, "MATCH (e:Entity {group_id:$pot}) RETURN e.entity_key AS key", pot=pot
    ) == [{"key": "service:existing"}]


async def test_direct_writer_reset_uses_canonical_receipt_and_revision(graph):
    pot, backend, driver = graph
    await backend.mutation.apply_async(
        MutationBatch(entity_upserts=[EntityUpsert("service:api", ("Service",))]),
        expected_pot_id=pot,
    )
    result = await backend.graph_writer.reset_pot(pot)
    assert result["ok"]
    assert result["group_id_nodes_before"] == 1
    assert result["group_id_nodes_remaining"] == 0
    assert result["mutation_id"]
    assert result["execution"]["version_after"] == 2
    assert await backend.mutation.current_version_async(pot) == 2
    receipts = _receipts(driver, pot)
    assert len(receipts) == 2
    assert any(
        item["result"].get("administrative_summary", {}).get("group_id_nodes_before")
        == 1
        for item in receipts
    )


def test_entity_repair_over_batch_cap_fails_without_partial_writes(graph):
    pot, backend, driver = graph
    _run(
        driver,
        "UNWIND range(0,5000) AS i CREATE (:Entity:Service {group_id:$pot,entity_key:'service:cap-'+toString(i)})",
        pot=pot,
    )
    with pytest.raises(
        MutationBatchValidationError, match="entity upsert cap exceeded"
    ):
        backend.analytics.repair(pot, targets=("entity_summaries",))
    assert backend.mutation.current_version(pot) == 0
    assert _receipts(driver, pot) == []
    assert _run(
        driver,
        "MATCH (e:Entity {group_id:$pot}) RETURN count(e) AS count,count(e.summary) AS summarized",
        pot=pot,
    ) == [{"count": 5001, "summarized": 0}]


@pytest.mark.parametrize("kind", ["mixed", "empty_snapshot"])
def test_mutated_or_empty_administrative_intent_is_rejected_without_effects(
    graph, kind
):
    pot, backend, driver = graph
    if kind == "mixed":
        batch = MutationBatch(reset_pot=True)
        batch.entity_upserts.append(EntityUpsert("service:forbidden", ("Service",)))
    else:
        batch = MutationBatch(snapshot_import={})
    with pytest.raises(MutationBatchValidationError):
        backend.mutation.apply(batch, expected_pot_id=pot)
    assert backend.mutation.current_version(pot) == 0
    assert _receipts(driver, pot) == []
    assert backend.snapshot.export_data(pot_id=pot)["entities"] == []


def test_snapshot_explicit_identity_rejects_changed_content(graph):
    pot, backend, driver = graph
    payload = _snapshot_payload()
    backend.snapshot.import_data(
        pot_id=pot, payload=payload, mutation_id="reserved-import"
    )
    changed = _snapshot_payload()
    changed["claims"][0]["fact"] = "changed import content"
    with pytest.raises(MutationExecutionReuseError):
        backend.snapshot.import_data(
            pot_id=pot, payload=changed, mutation_id="reserved-import"
        )
    assert backend.mutation.current_version(pot) == 1
    assert len(_receipts(driver, pot)) == 1


def test_snapshot_cannot_reassign_existing_incarnation_to_another_record(graph):
    pot, backend, driver = graph
    backend.snapshot.import_data(pot_id=pot, payload=_snapshot_payload())
    collision = {
        "format_version": "2",
        "pot_id": "source",
        "entities": [
            {
                "key": "service:new",
                "labels": ["Entity", "Service"],
                "properties": {"record_id": "record:claim"},
            }
        ],
        "claims": [],
    }
    with pytest.raises(ValueError, match="incarnation conflicts with target"):
        backend.snapshot.import_data(pot_id=pot, payload=collision)
    assert backend.mutation.current_version(pot) == 1
    assert len(_receipts(driver, pot)) == 1


def test_snapshot_export_reimport_keeps_existing_record_ids_and_has_no_effects(graph):
    pot, backend, driver = graph
    backend.snapshot.import_data(pot_id=pot, payload=_snapshot_payload())
    exported = backend.snapshot.export_data(pot_id=pot)
    merged = backend.snapshot.import_data(
        pot_id=pot, payload=exported, mutation_id="merge-existing"
    )
    assert merged.metadata["administrative_summary"] == {
        "imported_entities": 0,
        "imported_claims": 0,
    }
    assert merged.metadata["execution"]["affected_claim_record_ids"] == []
    assert backend.snapshot.export_data(pot_id=pot) == exported
    assert backend.mutation.current_version(pot) == 2
    assert len(_receipts(driver, pot)) == 2


def test_snapshot_response_loss_recovers_original_manifest_from_receipt(
    graph, monkeypatch
):
    pot, backend, driver = graph
    original = neo4j_atomic.apply_atomic

    async def lose_response(*args, **kwargs):
        await original(*args, **kwargs)
        raise ConnectionError("injected lost snapshot response")

    with monkeypatch.context() as patch:
        patch.setattr(neo4j_atomic, "apply_atomic", lose_response)
        with pytest.raises(ConnectionError, match="lost snapshot response"):
            backend.snapshot.import_data(pot_id=pot, payload=_snapshot_payload())
    recovered = Neo4jGraphBackend(backend.settings).snapshot.import_data(
        pot_id=pot, payload=_snapshot_payload()
    )
    assert recovered.metadata["administrative_summary"] == {
        "imported_entities": 2,
        "imported_claims": 1,
    }
    assert recovered.metadata["execution"]["version_after"] == 1
    assert backend.mutation.current_version(pot) == 1
    assert len(_receipts(driver, pot)) == 1


def test_import_existing_ordinary_uuid_records_preserves_their_identity(graph):
    pot, backend, driver = graph
    backend.mutation.apply(
        MutationBatch(
            entity_upserts=[
                EntityUpsert("service:api", ("Service",)),
                EntityUpsert("service:db", ("Service",)),
            ],
            edge_upserts=[
                EdgeUpsert(
                    "DEPENDS_ON",
                    "service:api",
                    "service:db",
                    {"claim_key": "claim:ordinary", "fact": "API depends on DB"},
                )
            ],
        ),
        expected_pot_id=pot,
    )
    before = backend.snapshot.export_data(pot_id=pot)
    result = backend.snapshot.import_data(pot_id=pot, payload=before)
    assert result.metadata["administrative_summary"] == {
        "imported_entities": 0,
        "imported_claims": 0,
    }
    assert backend.snapshot.export_data(pot_id=pot) == before
    assert backend.mutation.current_version(pot) == 2


@pytest.mark.parametrize("door", ["sync", "async", "writer"])
async def test_reset_wrapper_identity_replays_without_clearing_newer_graph(graph, door):
    pot, backend, driver = graph

    async def reset(identity):
        if door == "sync":
            return await asyncio.to_thread(
                backend.mutation.reset_pot, pot, mutation_id=identity
            )
        if door == "async":
            return await backend.mutation.reset_pot_async(pot, mutation_id=identity)
        return await backend.graph_writer.reset_pot(pot, mutation_id=identity)

    await backend.mutation.apply_async(
        MutationBatch(entity_upserts=[EntityUpsert("service:old", ("Service",))]),
        expected_pot_id=pot,
    )
    first = await reset("reset:recovery")
    assert first["ok"] and first["mutation_id"] == "reset:recovery"
    assert first["group_id_nodes_before"] == 1
    assert (
        first["execution"]["version_before"],
        first["execution"]["version_after"],
    ) == (1, 2)
    await backend.mutation.apply_async(
        MutationBatch(entity_upserts=[EntityUpsert("service:new", ("Service",))]),
        expected_pot_id=pot,
    )
    assert await reset("reset:recovery") == first
    assert await reset("reset:recovery") == first
    assert await backend.mutation.current_version_async(pot) == 3
    assert _run(
        driver, "MATCH (e:Entity {group_id:$pot}) RETURN e.entity_key AS key", pot=pot
    ) == [{"key": "service:new"}]
    with pytest.raises(MutationExecutionReuseError):
        await backend.mutation.apply_async(
            MutationBatch(
                entity_upserts=[EntityUpsert("service:forbidden", ("Service",))]
            ),
            expected_pot_id=pot,
            provenance_context=ProvenanceContext(mutation_id="reset:recovery"),
        )
    assert await backend.mutation.current_version_async(pot) == 3
    assert len(_receipts(driver, pot)) == 3


def test_reset_generated_identity_is_returned_and_can_be_replayed(graph):
    pot, backend, driver = graph
    first = backend.mutation.reset_pot(pot)
    assert (
        first["ok"] and isinstance(first["mutation_id"], str) and first["mutation_id"]
    )
    assert first["execution"]["version_after"] == 1
    assert backend.mutation.reset_pot(pot, mutation_id=first["mutation_id"]) == first
    assert backend.mutation.current_version(pot) == 1
    assert len(_receipts(driver, pot)) == 1


def test_reset_wrapper_recovers_authoritative_receipt_after_lost_response(
    graph, monkeypatch
):
    pot, backend, driver = graph
    backend.mutation.apply(
        MutationBatch(entity_upserts=[EntityUpsert("service:api", ("Service",))]),
        expected_pot_id=pot,
    )
    original = neo4j_atomic.apply_atomic

    async def lose_response(*args, **kwargs):
        await original(*args, **kwargs)
        raise ConnectionError("injected lost reset response")

    with monkeypatch.context() as patch:
        patch.setattr(neo4j_atomic, "apply_atomic", lose_response)
        with pytest.raises(ConnectionError, match="lost reset response"):
            backend.mutation.reset_pot(pot, mutation_id="reset:lost-response")
    restarted = Neo4jGraphBackend(backend.settings)
    recovered = restarted.mutation.reset_pot(pot, mutation_id="reset:lost-response")
    assert recovered["ok"] and recovered["mutation_id"] == "reset:lost-response"
    assert recovered["group_id_nodes_before"] == 1
    assert (
        recovered["execution"]["version_before"],
        recovered["execution"]["version_after"],
    ) == (1, 2)
    assert (
        restarted.mutation.reset_pot(pot, mutation_id="reset:lost-response")
        == recovered
    )
    assert restarted.mutation.current_version(pot) == 2
    assert len(_receipts(driver, pot)) == 2


async def test_writer_reset_failure_exposes_generated_identity_for_recovery(
    graph, monkeypatch
):
    pot, backend, driver = graph
    original = neo4j_atomic.apply_atomic

    async def lose_response(*args, **kwargs):
        await original(*args, **kwargs)
        raise ConnectionError("injected lost writer reset response")

    with monkeypatch.context() as patch:
        patch.setattr(neo4j_atomic, "apply_atomic", lose_response)
        failed = await backend.graph_writer.reset_pot(pot)
    assert not failed["ok"] and failed["execution"] is None
    assert failed["mutation_id"]
    recovered = await backend.graph_writer.reset_pot(
        pot, mutation_id=failed["mutation_id"]
    )
    assert recovered["ok"]
    assert recovered["execution"]["version_after"] == 1
    assert (
        await backend.graph_writer.reset_pot(pot, mutation_id=failed["mutation_id"])
        == recovered
    )
    assert await backend.mutation.current_version_async(pot) == 1
    assert len(_receipts(driver, pot)) == 1
