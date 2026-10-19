"""Lossless native inventory, explicit legacy repair and identity invariants."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import os
import subprocess
import threading
import time

import pytest
from redis.exceptions import ResponseError

from potpie_context_engine.adapters.outbound.graph import falkordb_legacy
from potpie_context_engine.adapters.outbound.graph._mutation_execution import (
    MutationExecutionReuseError,
)
from potpie_context_engine.adapters.outbound.graph.plan_stores.local_json import (
    LocalJsonGraphPlanStore,
)
from potpie_context_engine.core.graph_journal import JournalError, JournalLimits
from potpie_context_engine.core.graph_mutations import (
    EdgeDelete,
    EntityUpsert,
    ProvenanceContext,
)
from potpie_context_engine.core.legacy_records import resolve_entity_identity
from potpie_context_engine.core.legacy_repair_policy import (
    plan_legacy_repairs,
    record_patch,
)
from potpie_context_engine.core.legacy_repair_service import prepare_legacy_repair
from potpie_context_engine.core.reconciliation import MutationBatch

from tests.integration.test_falkordb_claim_lifecycle import store as store
from tests.integration.test_falkordb_reliability import (
    backend,
    graph as graph,
    snapshot,
)

pytestmark = pytest.mark.integration
OFFSET = "2020-01-01T05:30:00+05:30"
UTC = "2020-01-01T00:00:00+00:00"


def seed(graph, *, missing=False):
    a = {
        "group_id": "p",
        "entity_key": "service:a",
        "uuid": "old-a",
        "observed_at": OFFSET,
        "created_at": 1577836800000,
        "retired": True,
    }
    b = {"group_id": "p", "entity_key": "service:b", "uuid": "old-b"}
    r = {
        "group_id": "p",
        "name": "DEPENDS_ON",
        "subject_key": "service:a",
        "object_key": "service:b",
        "claim_key": "legacy-dependency",
        "uuid": "old-claim",
        "valid_at": OFFSET,
        "valid_from": OFFSET,
        "observed_at": "2020-01-02T05:30:00+05:30",
        "custom": "preserved",
        "retired": True,
    }
    if not missing:
        a["record_id"], b["record_id"], r["record_id"] = (
            "record-a",
            "record-b",
            "record-claim",
        )
    graph.query(
        "CREATE (a:Entity:Service),(b:Entity:Service) SET a=$a,b=$b "
        "CREATE (a)-[r:RELATES_TO]->(b) SET r=$r "
        "MERGE (v:PotpieRevision {pot_id:'p'}) ON CREATE SET v.version=0",
        params={"a": a, "b": b, "r": r},
    )


def effects(store):
    inventory = store.mutation.legacy_inventory("p")
    patches, findings = plan_legacy_repairs(inventory)
    assert not any(finding.blocking for finding in findings), findings
    return patches


def apply(store, patches, mid="legacy-repair", *, expected=None):
    plan = MutationBatch(record_repairs=tuple(patches))
    return store.mutation.compare_and_apply(
        plan,
        expected_pot_id="p",
        expected_version=(
            store.mutation.current_version("p") if expected is None else expected
        ),
        provenance_context=ProvenanceContext(mutation_id=mid),
    )


def native_receipt(graph, mid):
    return graph.ro_query(
        "MATCH (r:PotpieMutationReceipt {pot_id:'p',mutation_id:$mid}) RETURN properties(r)",
        params={"mid": mid},
    ).result_set[0][0]


def test_inventory_is_lossless_including_retired_unlabeled_and_cross_pot_records(graph):
    store = backend(graph)
    seed(graph, missing=True)
    graph.query(
        "CREATE (a:Legacy {group_id:'p',entity_key:'service:unlabeled',raw:'unchanged'}),"
        "(b:Entity {group_id:'other',entity_key:'service:foreign'}) "
        "CREATE (a)-[:ODD_EDGE {name:'USES',subject_key:'wrong',valid_until:'bad'}]->(b)"
    )
    before = snapshot(graph)
    inventory = store.mutation.legacy_inventory("p")
    assert len(inventory.records) == 5
    assert inventory.journal_active is False
    assert inventory.max_repair_records == falkordb_legacy.MAX_REPAIR_RECORDS
    assert inventory.max_repair_bytes == falkordb_legacy.MAX_REPAIR_BYTES
    assert any(record.labels == ("Legacy",) for record in inventory.records)
    malformed = next(
        record for record in inventory.records if record.relationship_type == "ODD_EDGE"
    )
    assert malformed.subject_pot_id == "p" and malformed.object_pot_id == "other"
    assert malformed.properties["valid_until"] == "bad"
    assert malformed.issues
    encoded = json.dumps([record.to_dict() for record in inventory.records])
    assert "locator" not in encoded
    assert all("record_id" not in record.properties for record in inventory.records)
    assert asyncio.run(store.mutation.legacy_inventory_async("p")) == inventory
    assert snapshot(graph) == before


def test_explicit_backfill_preserves_uuids_and_enables_validate_only_activation(graph):
    store = backend(graph)
    seed(graph, missing=True)
    before = snapshot(graph)
    with pytest.raises(JournalError, match="migration_required"):
        store.journal.activate(pot_id="p", rollback_enabled=True)
    assert snapshot(graph) == before
    patches = effects(store)
    result = apply(store, patches)
    assert result.administrative_summary == {
        "repaired_entities": 2,
        "repaired_claims": 1,
    }
    inventory = store.mutation.legacy_inventory("p")
    assert all(
        record.properties["record_id"] == record.properties["uuid"]
        for record in inventory.records
    )
    assert {record.properties["uuid"] for record in inventory.records} == {
        "old-a",
        "old-b",
        "old-claim",
    }
    assert result.execution.affected_claim_record_ids == ("old-claim",)
    assert effects(store) == ()
    data = snapshot(graph)[:2]
    store.journal.activate(pot_id="p", rollback_enabled=True)
    assert snapshot(graph)[:2] == data


def test_time_normalization_is_atomic_audited_replayable_and_has_no_more_work(
    store, graph, tmp_path
):
    seed(graph)
    patches = effects(store)
    original = deepcopy(patches)
    old_state = store.journal.journal_state("p")
    result = apply(store, patches)
    assert patches == original
    assert result.administrative_summary == {
        "repaired_entities": 1,
        "repaired_claims": 1,
    }
    assert result.execution.version_after == 1
    assert result.execution.affected_claim_record_ids == ("record-claim",)
    inventory = store.mutation.legacy_inventory("p")
    entity = next(record for record in inventory.records if record.key == "service:a")
    claim = next(record for record in inventory.records if record.kind == "claim")
    assert entity.properties["observed_at"] == claim.properties["valid_at"] == UTC
    assert claim.properties["valid_from"] == UTC
    assert entity.properties["created_at"] == 1577836800000
    assert (
        claim.properties["uuid"] == "old-claim"
        and claim.properties["custom"] == "preserved"
    )
    assert claim.properties["retired"] is True
    assert effects(store) == ()
    receipt = native_receipt(graph, result.mutation_id)
    assert json.loads(receipt["record_repairs"]) == [
        patch.to_dict() for patch in patches
    ]
    assert "repair_id" not in receipt["record_repairs"]
    if old_state is not None:
        state = store.journal.journal_state("p")
        assert (
            state.sequence == old_state.sequence + 1
            and state.head == result.mutation_id
        )
        journal = store.journal.get_receipt(pot_id="p", commit_id=result.mutation_id)
        assert journal.origin == "repair" and journal.rollback_supported is False
        assert journal.semantic_changes == ()
        with pytest.raises(JournalError):
            store.journal.plan_restore(
                pot_id="p",
                target_commit_id=result.mutation_id,
                expected_head=state.head,
            )
    before = snapshot(graph)
    assert apply(store, patches, expected=0) == result
    assert snapshot(graph) == before
    no_work = prepare_legacy_repair(
        store,
        LocalJsonGraphPlanStore(home=tmp_path / "plans"),
        pot_id="p",
        repair_id="nothing-left",
    )
    assert no_work.repairable_records == no_work.total_batches == 0
    assert snapshot(graph) == before


def test_changed_full_beforeimage_and_repeated_target_reject_without_writes(
    store, graph
):
    seed(graph)
    patches = effects(store)
    graph.query("MATCH (e {group_id:'p',entity_key:'service:a'}) SET e.extra='changed'")
    before = snapshot(graph)
    with pytest.raises(ValueError, match="missing, changed, or ambiguous"):
        apply(store, patches)
    assert snapshot(graph) == before
    current = effects(store)
    with pytest.raises(ValueError, match="missing, changed, or ambiguous"):
        apply(store, (current[0], current[0]), "duplicate-target")
    assert snapshot(graph) == before


def test_runtime_failure_after_repairs_receipt_revision_and_head_rolls_back(
    store, graph, monkeypatch
):
    seed(graph)
    patches = effects(store)
    compile_query = falkordb_legacy.compile_repairs

    def broken(*args, **kwargs):
        query, params, result = compile_query(*args, **kwargs)
        return (
            query.replace(
                "RETURN v.version,receipt.fingerprint",
                "WITH v,receipt RETURN keys(v.version),receipt.fingerprint",
            ),
            params,
            result,
        )

    monkeypatch.setattr(falkordb_legacy, "compile_repairs", broken)
    before = snapshot(graph)
    with pytest.raises(ResponseError, match="Type mismatch"):
        apply(store, patches)
    assert snapshot(graph) == before
    assert (
        store.mutation.lookup_execution(
            MutationBatch(record_repairs=patches),
            expected_pot_id="p",
            mutation_id="legacy-repair",
        ).state
        == "absent"
    )


def test_concurrent_identical_repair_has_one_receipt_and_changed_batch_cannot_rebind(
    store, graph
):
    seed(graph)
    patches = effects(store)
    ready = threading.Barrier(2)

    def execute():
        ready.wait(timeout=5)
        return apply(store, patches, expected=0)

    with ThreadPoolExecutor(max_workers=2) as workers:
        first, second = tuple(workers.map(lambda _: execute(), range(2)))
    assert first == second
    assert store.mutation.current_version("p") == 1
    assert graph.ro_query(
        "MATCH (r:PotpieMutationReceipt) RETURN count(r)"
    ).result_set == [[1]]
    before = snapshot(graph)
    with pytest.raises(MutationExecutionReuseError):
        apply(store, patches[:1], expected=0)
    assert snapshot(graph) == before


@pytest.mark.parametrize(
    "corruption",
    ("duplicate-alias", "naive-time", "contradictory-alias", "missing-label"),
)
def test_ambiguous_legacy_data_blocks_repair_preview_and_apply(graph, corruption):
    store = backend(graph)
    seed(graph)
    patches = effects(store)
    if corruption == "duplicate-alias":
        graph.query("MATCH (e {entity_key:'service:a'}) SET e.uuid='old-claim'")
    elif corruption == "naive-time":
        graph.query(
            "MATCH ()-[r:RELATES_TO]->() SET r.valid_until='2020-02-01T00:00:00'"
        )
    elif corruption == "contradictory-alias":
        graph.query("MATCH ()-[r:RELATES_TO]->() SET r.valid_at='2020-02-01T00:00:00Z'")
    else:
        graph.query("MATCH (e {entity_key:'service:a'}) REMOVE e:Entity")
    _, findings = plan_legacy_repairs(store.mutation.legacy_inventory("p"))
    assert any(finding.blocking for finding in findings)
    before = snapshot(graph)
    with pytest.raises(ValueError):
        apply(store, patches)
    assert snapshot(graph) == before


def test_indistinguishable_native_records_do_not_leak_locators_or_pick_a_target(graph):
    store = backend(graph)
    seed(graph)
    graph.query(
        "CREATE (:Entity:Service {group_id:'p',entity_key:'service:identityless',observed_at:$time})",
        params={"time": OFFSET},
    )
    record = next(
        record
        for record in store.mutation.legacy_inventory("p").records
        if record.key == "service:identityless"
    )
    patch = record_patch(record, new_record_id="frozen-new-id")
    graph.query(
        "MATCH (e {entity_key:'service:identityless'}) CREATE (copy:Entity:Service) SET copy=properties(e)"
    )
    before = snapshot(graph)
    with pytest.raises(ValueError, match="ambiguous|logical entity key is shared"):
        apply(store, (patch,))
    assert snapshot(graph) == before


@pytest.mark.parametrize("field", ("record_id", "uuid"))
@pytest.mark.parametrize("value", ("different-id", None))
def test_ordinary_entity_identity_overwrite_is_rejected(store, graph, field, value):
    plan = MutationBatch(entity_upserts=[EntityUpsert("service:new", ("Service",))])
    context = ProvenanceContext(mutation_id="new-entity")
    result = store.mutation.apply(plan, expected_pot_id="p", provenance_context=context)
    props = store.claim_query.entity_properties(pot_id="p", entity_key="service:new")
    expected = resolve_entity_identity(
        None, {}, pot_id="p", mutation_id="new-entity", entity_key="service:new"
    )
    assert props["record_id"] == props["uuid"] == expected["record_id"]
    assert plan.entity_upserts[0].properties == {}
    before = snapshot(graph)
    with pytest.raises(ValueError, match="immutable entity identity"):
        store.mutation.apply(
            MutationBatch(
                entity_upserts=[
                    EntityUpsert("service:new", ("Service",), {field: value})
                ]
            ),
            expected_pot_id="p",
            provenance_context=ProvenanceContext(mutation_id="overwrite"),
        )
    assert snapshot(graph) == before
    assert (
        store.mutation.apply(plan, expected_pot_id="p", provenance_context=context)
        == result
    )


def test_existing_missing_entity_or_claim_id_requires_explicit_repair(store, graph):
    seed(graph, missing=True)
    before = snapshot(graph)
    with pytest.raises(ValueError, match="migration_required"):
        store.mutation.apply(
            MutationBatch(
                entity_upserts=[
                    EntityUpsert("service:a", ("Service",), {"uuid": "old-a"})
                ]
            ),
            expected_pot_id="p",
            provenance_context=ProvenanceContext(mutation_id="ordinary-backfill"),
        )
    assert snapshot(graph) == before
    if store.journal.journal_state("p") is not None:
        _, findings = plan_legacy_repairs(store.mutation.legacy_inventory("p"))
        assert any(
            finding.blocking and "active journal" in finding.detail
            for finding in findings
        )


def test_touched_claim_cannot_promote_uuid_into_record_id_during_ordinary_delete(
    store, graph
):
    seed(graph)
    graph.query("MATCH ()-[r:RELATES_TO]->() REMOVE r.record_id SET r.retired=false")
    before = snapshot(graph)
    with pytest.raises(ValueError, match="migration_required"):
        store.mutation.apply(
            MutationBatch(
                edge_deletes=[EdgeDelete("DEPENDS_ON", "service:a", "service:b")]
            ),
            expected_pot_id="p",
            provenance_context=ProvenanceContext(
                mutation_id="delete-without-record-id"
            ),
        )
    assert snapshot(graph) == before


def test_global_alias_guard_rejects_new_entity_using_a_claim_uuid(store, graph):
    seed(graph)
    before = snapshot(graph)
    with pytest.raises(ValueError, match="physical identity"):
        store.mutation.apply(
            MutationBatch(
                entity_upserts=[
                    EntityUpsert(
                        "service:collision", ("Service",), {"record_id": "old-claim"}
                    )
                ]
            ),
            expected_pot_id="p",
            provenance_context=ProvenanceContext(mutation_id="cross-kind-alias"),
        )
    assert snapshot(graph) == before


def test_advertised_journal_repair_limits_are_enforced_before_publication(graph):
    store = backend(graph)
    store.journal.activate(
        pot_id="p",
        rollback_enabled=True,
        limits=JournalLimits(max_records=1, max_bytes=40000),
    )
    seed(graph)
    inventory = store.mutation.legacy_inventory("p")
    assert inventory.journal_active and inventory.max_repair_records == 1
    assert inventory.max_repair_bytes == 40000
    patches = effects(store)
    before = snapshot(graph)
    with pytest.raises(ValueError, match="limits"):
        apply(store, patches)
    assert snapshot(graph) == before


def test_old_repair_receipt_replays_after_reset_without_restoring_data(graph):
    store = backend(graph)
    seed(graph, missing=True)
    patches = effects(store)
    repaired = apply(store, patches)
    receipt = native_receipt(graph, repaired.mutation_id)
    reset = store.mutation.reset_pot("p", mutation_id="reset-after-repair")
    before = snapshot(graph)
    assert apply(store, patches, expected=0) == repaired
    assert snapshot(graph) == before
    assert snapshot(graph)[:2] == ([], [])
    assert store.mutation.current_version("p") == reset["execution"]["version_after"]
    assert native_receipt(graph, repaired.mutation_id) == receipt


def test_repaired_receipt_survives_server_restart(graph, request):
    container = os.getenv("POTPIE_TEST_FALKORDB_RESTART_CONTAINER")
    if request.node.callspec.params["graph"] != "server" or not container:
        pytest.skip("requires the isolated server restart opt-in")
    store = backend(graph)
    seed(graph, missing=True)
    patches = effects(store)
    result = apply(store, patches)
    before = snapshot(graph)
    subprocess.run(
        ["rtk", "docker", "restart", container], check=True, capture_output=True
    )
    deadline = time.monotonic() + 20
    while True:
        try:
            assert (
                store.mutation.lookup_execution(
                    MutationBatch(record_repairs=patches),
                    expected_pot_id="p",
                    mutation_id=result.mutation_id,
                ).result
                == result
            )
            break
        except Exception:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.1)
    assert snapshot(graph) == before
