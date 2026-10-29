"""Exact, receipt-bearing legacy repair against a disposable Neo4j server."""

import asyncio
from datetime import datetime, timezone
import json
import os
from types import SimpleNamespace
import uuid

import pytest

from potpie_context_engine.adapters.outbound.graph import neo4j_legacy
from potpie_context_engine.adapters.outbound.graph._mutation_execution import (
    MutationExecutionReuseError,
)
from potpie_context_engine.adapters.outbound.graph.backends.neo4j_backend import (
    Neo4jGraphBackend,
)
from potpie_context_engine.adapters.outbound.graph.plan_stores.local_json import (
    LocalJsonGraphPlanStore,
)
from potpie_context_engine.core.graph_mutations import EntityUpsert, ProvenanceContext
from potpie_context_engine.core.errors import (
    MutationBatchValidationError,
    ReconciliationApplyError,
)
from potpie_context_engine.core.legacy_repair_policy import (
    plan_legacy_repairs,
    record_patch,
)
from potpie_context_engine.core.legacy_records import RecordRepair
from potpie_context_engine.core.legacy_repair_service import prepare_legacy_repair
from potpie_context_engine.core.mutation_execution import (
    PreparedMutation,
    batch_fingerprint,
)
from potpie_context_engine.core.reconciliation import MutationBatch
from potpie_context_engine.core.runtime import build_graph_runtime

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.environ.get("PROTOCOL_TEST_NEO4J_URI"),
        reason="PROTOCOL_TEST_NEO4J_URI must name a disposable test server",
    ),
]
UTC = timezone.utc
OFFSET_START = "2020-01-01T05:30:00+05:30"
UTC_START = "2020-01-01T00:00:00+00:00"


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
    pot = "legacy-" + uuid.uuid4().hex
    with GraphDatabase.driver(
        settings.neo4j_uri(), auth=(settings.neo4j_user(), settings.neo4j_password())
    ) as driver:
        try:
            yield pot, backend, driver, settings
        finally:
            backend.claim_query.close()
            _run(
                driver,
                "MATCH (n) WHERE n.group_id=$pot OR n.pot_id=$pot OR n.fixture_pot=$pot DETACH DELETE n",
                pot=pot,
            )


def _run(driver, query, **params):
    with driver.session() as session:
        return [dict(row) for row in session.run(query, **params)]


def _seed(graph, *, no_ids=False):
    pot, _, driver, _ = graph
    a = {"group_id": pot, "entity_key": "service:a", "observed_at": OFFSET_START}
    b = {
        "group_id": pot,
        "entity_key": "service:b",
        "record_id": "entity:b",
        "uuid": "alias:b",
    }
    claim = {
        "group_id": pot,
        "claim_key": "claim:dependency",
        "name": "DEPENDS_ON",
        "subject_key": "service:a",
        "object_key": "service:b",
        "valid_at": OFFSET_START,
        "valid_from": OFFSET_START,
    }
    if not no_ids:
        a["uuid"], claim["uuid"] = "entity:a", "claim:a"
    _run(
        driver,
        "CREATE (a:Entity:Service),(b:Entity:Service),(a)-[r:RELATES_TO]->(b) SET a=$a,b=$b,r=$claim,r.observed_at=datetime($observed)",
        a=a,
        b=b,
        claim=claim,
        observed=OFFSET_START,
    )


def _physical(graph):
    pot, _, driver, _ = graph
    return (
        _run(
            driver,
            "MATCH (n {group_id:$pot}) RETURN properties(n) AS props,labels(n) AS labels ORDER BY n.entity_key",
            pot=pot,
        ),
        _run(
            driver,
            "MATCH ()-[r {group_id:$pot}]->() RETURN properties(r) AS props ORDER BY r.claim_key",
            pot=pot,
        ),
    )


def _receipts(graph):
    pot, _, driver, _ = graph
    return _run(
        driver,
        "MATCH (r:PotpieMutationReceipt {pot_id:$pot}) RETURN r.mutation_id AS id,r.payload AS payload ORDER BY id",
        pot=pot,
    )


def _plan(graph):
    pot, backend, _, _ = graph
    inventory = backend.mutation.legacy_inventory(pot)
    effects, findings = plan_legacy_repairs(inventory)
    assert not [finding for finding in findings if finding.blocking], findings
    return inventory, effects


def _prepared(graph, effects, mutation_id):
    pot, backend, _, _ = graph
    batch = MutationBatch(record_repairs=tuple(effects))
    fingerprint = batch_fingerprint(batch)
    context = ProvenanceContext(
        mutation_id=mutation_id,
        prepared=PreparedMutation(
            pot,
            mutation_id,
            fingerprint,
            fingerprint,
            backend.mutation.current_version(pot),
            datetime.now(UTC),
        ),
    )
    return batch, context


def _apply(graph, batch, context):
    pot, backend, _, _ = graph
    return backend.mutation.compare_and_apply(
        batch,
        expected_pot_id=pot,
        expected_version=context.prepared.expected_version,
        provenance_context=context,
    )


def test_inventory_retains_retired_missing_labels_malformed_endpoints_and_native_precision(
    graph,
):
    pot, backend, driver, _ = graph
    _run(
        driver,
        "CREATE (a:Service {group_id:$pot,entity_key:'service:a',retired:true}),(b {entity_key:'service:foreign',fixture_pot:$pot}),(a)-[r:UNEXPECTED]->(b) SET r.group_id=$pot,r.claim_key='claim:broken',r.name='DEPENDS_ON',r.valid_at=datetime('2020-01-01T00:00:00.000000001Z'),r.retired=true",
        pot=pot,
    )
    before = _physical(graph)
    inventory = backend.mutation.legacy_inventory(pot)
    assert len(inventory.records) == 2
    claim = next(record for record in inventory.records if record.kind == "claim")
    entity = next(record for record in inventory.records if record.kind == "entity")
    assert claim.property_types["valid_at"] == "DateTime"
    assert claim.properties["valid_at"] == "2020-01-01T00:00:00.000000001+00:00"
    assert claim.subject_pot_id == pot and claim.object_pot_id is None
    assert claim.relationship_type == "UNEXPECTED"
    assert claim.properties["retired"] is True
    assert "Entity" not in entity.labels
    encoded = json.dumps([record.to_dict() for record in inventory.records])
    assert "locator" not in encoded and "elementId" not in encoded
    assert not inventory.journal_active
    assert inventory.max_repair_records == 5000
    assert inventory.max_repair_bytes == 8_000_000
    _, findings = plan_legacy_repairs(inventory)
    assert any(
        finding.blocking and "precision" in finding.detail for finding in findings
    )
    assert _physical(graph) == before
    assert (
        _run(driver, "MATCH (v:PotpieRevision {pot_id:$pot}) RETURN v", pot=pot) == []
    )


def test_inventory_preserves_nonstring_keys_and_reports_them_without_sort_failure(
    graph,
):
    pot, backend, driver, _ = graph
    _run(
        driver,
        "CREATE (a:Entity {group_id:$pot,entity_key:7}),"
        "(b:Entity {group_id:$pot,entity_key:'service:valid'}),"
        "(a)-[:RELATES_TO {group_id:$pot,claim_key:8,name:'DEPENDS_ON',"
        "subject_key:7,object_key:'service:valid'}]->(b)",
        pot=pot,
    )
    before = _physical(graph)
    inventory = backend.mutation.legacy_inventory(pot)
    assert len(inventory.records) == 3
    claim = next(record for record in inventory.records if record.kind == "claim")
    assert claim.properties["claim_key"] == 8
    assert claim.subject_key == 7
    _, findings = plan_legacy_repairs(inventory)
    assert any(
        finding.blocking and "logical key" in finding.detail for finding in findings
    )
    assert any(
        finding.blocking and "endpoints" in finding.detail for finding in findings
    )
    assert _physical(graph) == before
    assert _receipts(graph) == []
    assert backend.mutation.current_version(pot) == 0


@pytest.mark.parametrize("retired", [False, True])
def test_repair_normalizes_aware_native_times_backfills_ids_and_preserves_aliases(
    graph,
    retired,
):
    pot, backend, driver, _ = graph
    _seed(graph)
    if retired:
        _run(
            driver,
            "MATCH (a {group_id:$pot})-[r]->() SET a.retired=true,r.retired=true",
            pot=pot,
        )
    backend.mutation.apply(
        MutationBatch(),
        expected_pot_id=pot,
        provenance_context=ProvenanceContext(mutation_id="older-empty-receipt"),
    )
    old_receipts = _receipts(graph)
    inventory, effects = _plan(graph)
    native_claim = next(
        record for record in inventory.records if record.kind == "claim"
    )
    assert native_claim.property_types["observed_at"] == "DateTime"
    assert len(effects) == 2
    batch, context = _prepared(graph, effects, "normalize-and-backfill")
    original_fingerprint = batch_fingerprint(batch)
    result = _apply(graph, batch, context)
    assert result.administrative_summary == {
        "repaired_entities": 1,
        "repaired_claims": 1,
    }
    assert result.execution.affected_claim_record_ids == ("claim:a",)
    assert result.execution.batch_fingerprint == original_fingerprint
    after, again = _plan(graph)
    assert again == ()
    by_key = {record.key: record for record in after.records}
    assert by_key["service:a"].properties["record_id"] == "entity:a"
    assert by_key["service:a"].properties["uuid"] == "entity:a"
    assert by_key["service:b"].properties["record_id"] == "entity:b"
    assert by_key["service:b"].properties["uuid"] == "alias:b"
    claim = by_key["claim:dependency"]
    assert claim.properties["record_id"] == claim.properties["uuid"] == "claim:a"
    assert (
        claim.properties["valid_at"]
        == claim.properties["valid_from"]
        == claim.properties["observed_at"]
        == UTC_START
    )
    assert claim.property_types == {}
    assert claim.properties.get("retired", False) is retired
    assert all(receipt in _receipts(graph) for receipt in old_receipts)
    assert backend.mutation.current_version(pot) == 1


def test_native_submicrosecond_precision_blocks_planning_and_direct_repair(graph):
    pot, backend, driver, _ = graph
    _seed(graph)
    _run(
        driver,
        "MATCH ()-[r {group_id:$pot}]->() SET r.observed_at=datetime('2020-01-01T00:00:00.000000001Z')",
        pot=pot,
    )
    inventory = backend.mutation.legacy_inventory(pot)
    claim = next(record for record in inventory.records if record.kind == "claim")
    assert claim.properties["observed_at"].endswith("000000001+00:00")
    _, findings = plan_legacy_repairs(inventory)
    assert any(
        finding.blocking and "precision" in finding.detail for finding in findings
    )
    effect = RecordRepair(
        claim, property_updates={"record_id": "claim:a", "observed_at": UTC_START}
    )
    batch, context = _prepared(graph, (effect,), "forged-lossy-repair")
    before = _physical(graph)
    with pytest.raises(MutationBatchValidationError, match="precision"):
        _apply(graph, batch, context)
    assert _physical(graph) == before
    assert _receipts(graph) == []
    assert backend.mutation.current_version(pot) == 0


def test_frozen_generated_ids_receipt_replay_and_reset_never_restore_old_repair(graph):
    pot, backend, _, settings = graph
    _seed(graph, no_ids=True)
    _, effects = _plan(graph)
    frozen_ids = tuple(effect.after().value("record_id") for effect in effects)
    batch, context = _prepared(graph, effects, "frozen-backfill")
    first = _apply(graph, batch, context)
    receipts = _receipts(graph)
    assert _apply(graph, batch, context) == first
    assert _receipts(graph) == receipts
    assert frozen_ids == tuple(
        effect.after().value("record_id") for effect in batch.record_repairs
    )
    restarted = Neo4jGraphBackend(settings)
    lookup = restarted.mutation.lookup_execution(
        batch, expected_pot_id=pot, mutation_id=context.mutation_id
    )
    assert lookup.result == first
    backend.mutation.reset_pot(pot, mutation_id="reset-after-repair")
    assert _apply(graph, batch, context) == first
    assert backend.mutation.current_version(pot) == 2
    assert _physical(graph) == ([], [])
    assert all(receipt in _receipts(graph) for receipt in receipts)
    with pytest.raises(MutationExecutionReuseError):
        backend.mutation.apply(
            MutationBatch(),
            expected_pot_id=pot,
            provenance_context=ProvenanceContext(mutation_id=context.mutation_id),
        )


@pytest.mark.parametrize("changed", ["property", "label", "endpoint"])
def test_complete_before_image_rejects_native_drift_without_partial_effects(
    graph, changed
):
    pot, backend, driver, _ = graph
    _seed(graph)
    _, effects = _plan(graph)
    batch, context = _prepared(graph, effects, "drifted-repair")
    if changed == "property":
        query = "MATCH (e {group_id:$pot,entity_key:'service:a'}) SET e.annotation='changed'"
    elif changed == "label":
        query = "MATCH (e {group_id:$pot,entity_key:'service:a'}) SET e:Extra"
    else:
        query = "MATCH ()-[r {group_id:$pot}]->() SET r.subject_key='service:changed'"
    _run(driver, query, pot=pot)
    before = _physical(graph)
    with pytest.raises(ValueError, match="missing, changed, or ambiguous"):
        _apply(graph, batch, context)
    assert _physical(graph) == before
    assert _receipts(graph) == []
    assert backend.mutation.current_version(pot) == 0


def test_indistinguishable_missing_id_records_and_duplicate_aliases_block_repair(graph):
    pot, backend, driver, _ = graph
    _seed(graph, no_ids=True)
    _run(
        driver,
        "MATCH (a)-[r {group_id:$pot}]->(b) CREATE (a)-[duplicate:RELATES_TO]->(b) SET duplicate=properties(r)",
        pot=pot,
    )
    inventory = backend.mutation.legacy_inventory(pot)
    _, findings = plan_legacy_repairs(inventory)
    assert any(
        finding.blocking and "indistinguishable" in finding.detail
        for finding in findings
    )
    claim = next(record for record in inventory.records if record.kind == "claim")
    effect = record_patch(claim, new_record_id="frozen:new-id")
    batch, context = _prepared(graph, (effect,), "ambiguous-repair")
    before = _physical(graph)
    with pytest.raises(ValueError, match="ambiguous"):
        _apply(graph, batch, context)
    assert _physical(graph) == before
    _run(
        driver,
        "MATCH (e {group_id:$pot,entity_key:'service:a'}) SET e.uuid='alias:b'",
        pot=pot,
    )
    inventory = backend.mutation.legacy_inventory(pot)
    _, findings = plan_legacy_repairs(inventory)
    assert any(finding.code == "duplicate_identity" for finding in findings)
    assert backend.mutation.current_version(pot) == 0
    assert _receipts(graph) == []


def test_duplicate_entity_logical_keys_block_planning_and_native_publication(graph):
    pot, backend, driver, _ = graph
    _run(
        driver,
        "CREATE (:Entity {group_id:$pot,entity_key:'service:duplicate',uuid:'first',note:'one'}),(:Entity {group_id:$pot,entity_key:'service:duplicate',uuid:'second',note:'two'})",
        pot=pot,
    )
    inventory = backend.mutation.legacy_inventory(pot)
    assert len(inventory.records) == 2
    _, findings = plan_legacy_repairs(inventory)
    assert any(
        finding.blocking and "logical entity key" in finding.detail
        for finding in findings
    )
    effect = record_patch(inventory.records[0], new_record_id="unused")
    batch, context = _prepared(graph, (effect,), "duplicate-key-repair")
    before = _physical(graph)
    with pytest.raises(ValueError, match="logical entity key"):
        _apply(graph, batch, context)
    assert _physical(graph) == before
    assert _receipts(graph) == []
    assert backend.mutation.current_version(pot) == 0


def test_internal_failure_rolls_back_entire_repair_chunk_revision_and_receipt(
    graph, monkeypatch
):
    pot, backend, _, _ = graph
    _seed(graph)
    _, effects = _plan(graph)
    batch, context = _prepared(graph, effects, "rolled-back-repair")
    before = _physical(graph)
    patch = neo4j_legacy.patch_records
    calls = []

    async def fail_after_patch(transaction, kind, patches):
        await patch(transaction, kind, patches)
        calls.append(kind)
        if kind == "claim":
            raise RuntimeError("injected after native repair patches")

    monkeypatch.setattr(neo4j_legacy, "patch_records", fail_after_patch)
    with pytest.raises(RuntimeError, match="injected after native repair"):
        _apply(graph, batch, context)
    assert calls == ["entity", "claim"]
    assert _physical(graph) == before
    assert _receipts(graph) == []
    assert backend.mutation.current_version(pot) == 0


@pytest.mark.parametrize("field", ["record_id", "uuid"])
def test_ordinary_entity_writes_cannot_overwrite_existing_identity(graph, field):
    pot, backend, _, _ = graph
    first = backend.mutation.apply(
        MutationBatch(entity_upserts=[EntityUpsert("service:new", ("Service",))]),
        expected_pot_id=pot,
        provenance_context=ProvenanceContext(mutation_id="new-entity"),
    )
    before, receipts = _physical(graph), _receipts(graph)
    with pytest.raises(Exception, match="immutable entity identity"):
        backend.mutation.apply(
            MutationBatch(
                entity_upserts=[
                    EntityUpsert(
                        "service:new", (), {field: "replacement"}, preserve_labels=True
                    )
                ]
            ),
            expected_pot_id=pot,
            provenance_context=ProvenanceContext(mutation_id="identity-overwrite"),
        )
    assert _physical(graph) == before
    assert _receipts(graph) == receipts
    assert backend.mutation.current_version(pot) == first.execution.version_after == 1


def test_ordinary_missing_identity_and_cross_kind_alias_reuse_require_repair(graph):
    pot, backend, _, _ = graph
    _seed(graph)
    before = _physical(graph)
    with pytest.raises(Exception, match="migration_required"):
        backend.mutation.apply(
            MutationBatch(entity_upserts=[EntityUpsert("service:a", ("Service",))]),
            expected_pot_id=pot,
            provenance_context=ProvenanceContext(mutation_id="ordinary-backfill"),
        )
    with pytest.raises(Exception, match="shared by multiple records"):
        backend.mutation.apply(
            MutationBatch(
                entity_upserts=[
                    EntityUpsert("service:new", ("Service",), {"uuid": "claim:a"})
                ]
            ),
            expected_pot_id=pot,
            provenance_context=ProvenanceContext(mutation_id="alias-reuse"),
        )
    assert _physical(graph) == before
    assert _receipts(graph) == []
    assert backend.mutation.current_version(pot) == 0


def test_ordinary_claim_closure_cannot_silently_backfill_uuid_only_identity(graph):
    pot, backend, _, _ = graph
    _seed(graph)
    before = _physical(graph)
    with pytest.raises(
        ReconciliationApplyError, match="migration_required.*physical record_id"
    ):
        backend.mutation.invalidate(pot_id=pot, claim_keys=("claim:dependency",))
    assert _physical(graph) == before
    assert _receipts(graph) == []
    assert backend.mutation.current_version(pot) == 0


def test_ordinary_upsert_cannot_duplicate_existing_node_missing_entity_label(graph):
    pot, backend, driver, _ = graph
    _run(
        driver,
        "CREATE (:Service {group_id:$pot,entity_key:'service:a',record_id:'existing:entity'})",
        pot=pot,
    )
    before = _physical(graph)
    with pytest.raises(
        ReconciliationApplyError, match="migration_required.*Entity label"
    ):
        backend.mutation.apply(
            MutationBatch(entity_upserts=[EntityUpsert("service:a", ("Service",))]),
            expected_pot_id=pot,
            provenance_context=ProvenanceContext(mutation_id="missing-label-upsert"),
        )
    assert _physical(graph) == before
    assert _receipts(graph) == []
    assert backend.mutation.current_version(pot) == 0


@pytest.mark.asyncio
async def test_async_inventory_owns_driver_on_callers_loop_and_sync_door_rejects_active_loop(
    graph,
):
    pot, backend, _, _ = graph
    inventory = await backend.mutation.legacy_inventory_async(pot)
    assert inventory.records == ()
    with pytest.raises(RuntimeError, match="async method"):
        backend.mutation.legacy_inventory(pot)
    await asyncio.sleep(0)


def test_5001_record_backfill_completes_in_bounded_revisioned_chunks(graph, tmp_path):
    pot, backend, driver, _ = graph
    _run(
        driver,
        "UNWIND range(0,5000) AS i CREATE (:Entity:Service {group_id:$pot,entity_key:'service:'+toString(i),uuid:'entity:'+toString(i)})",
        pot=pot,
    )
    _, effects = _plan(graph)
    assert len(effects) == 5001
    oversized, context = _prepared(graph, effects[:5000], "over-byte-limit")
    with pytest.raises(ValueError, match="backend transaction limits"):
        _apply(graph, oversized, context)
    assert backend.mutation.current_version(pot) == 0
    assert _receipts(graph) == []
    store = LocalJsonGraphPlanStore(home=tmp_path)
    preview = prepare_legacy_repair(
        backend, store, pot_id=pot, repair_id="bounded-repair", chunk_size=5000
    )
    assert preview.ok and preview.total_batches > 1
    assert preview.repairable_records == 5001
    assert backend.mutation.current_version(pot) == 0
    assert _receipts(graph) == []
    authorizations = []

    async def allow(selected_pot, access):
        authorizations.append((selected_pot, access))

    restarted = build_graph_runtime(
        backend=backend,
        plan_store=LocalJsonGraphPlanStore(home=tmp_path),
        commit_authorize=allow,
        commit_actor=lambda: "test:admin",
    ).workbench
    result = restarted.repair_legacy(
        pot_id=pot, mode="apply", repair_id=preview.repair_id
    )
    assert result.ok, result
    assert result.status == "completed"
    assert result.completed_batches == result.total_batches == preview.total_batches
    assert result.repaired["entity"] == 5001
    assert _plan(graph)[1] == ()
    assert backend.mutation.current_version(pot) == result.total_batches
    receipts = _receipts(graph)
    assert len(receipts) == result.total_batches
    replay = restarted.repair_legacy(
        pot_id=pot, mode="apply", repair_id=preview.repair_id
    )
    assert replay.ok and replay.executions == result.executions
    assert _receipts(graph) == receipts
    assert authorizations == [(pot, "admin"), (pot, "admin")]
