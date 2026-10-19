"""Supported administrative FalkorDB writes share the native mutation boundary."""

from concurrent.futures import ThreadPoolExecutor
import json

import pytest
from redis.exceptions import ConnectionError, ResponseError

from potpie_context_engine.adapters.outbound.graph import falkordb_admin
from potpie_context_engine.adapters.outbound.graph import falkordb_atomic
from potpie_context_engine.adapters.outbound.graph._mutation_execution import (
    MutationExecutionReuseError,
)
from potpie_context_engine.core.graph_mutations import EntityUpsert, ProvenanceContext
from potpie_context_engine.core.graph_snapshot_mutation import prepare_snapshot_import
from potpie_context_engine.core.mutation_execution import MutationPersistenceError
from potpie_context_engine.core.reconciliation import MutationBatch

from tests.integration.test_falkordb_reliability import (
    backend,
    batch,
    context,
    graph as graph,
    snapshot,
)

pytestmark = pytest.mark.integration


def payload():
    return {
        "format_version": "2",
        "pot_id": "source",
        "entities": [
            {
                "key": "service:web",
                "labels": ["Entity", "Service"],
                "properties": {"name": "web"},
            },
            {
                "key": "service:db",
                "labels": ["Entity", "Service"],
                "properties": {"name": "db"},
            },
        ],
        "claims": [
            {
                "claim_key": "claim:source:dependency",
                "predicate": "DEPENDS_ON",
                "subject_key": "service:web",
                "object_key": "service:db",
                "source_ref": "config:one",
                "source_system": "github",
                "properties": {},
            }
        ],
    }


def receipt_count(graph):
    return graph.ro_query("MATCH (r:PotpieMutationReceipt) RETURN count(r)").result_set[
        0
    ][0]


def test_snapshot_import_has_native_receipt_and_replay_cannot_undo_reset(graph):
    store, incoming = backend(graph), payload()
    imported = store.snapshot.import_data(pot_id="p", payload=incoming)
    normalized, _ = prepare_snapshot_import(
        pot_id="p", payload=incoming, expected_version=0
    )
    mid = imported.metadata["mutation_id"]
    lookup = store.mutation.lookup_execution(
        normalized, expected_pot_id="p", mutation_id=mid
    )
    assert lookup.state == "completed"
    assert lookup.result.administrative_summary == {
        "imported_entities": 2,
        "imported_claims": 1,
    }
    assert (
        lookup.result.execution.version_before,
        lookup.result.execution.version_after,
    ) == (0, 1)
    assert len(lookup.result.execution.affected_claim_record_ids) == 1
    report = store.mutation.reset_pot("p")
    assert report["ok"] and report["group_id_nodes_before"] == 2
    assert report["group_id_nodes_remaining"] == 0 and report["version"] == 2
    assert report["mutation_id"]
    assert report["execution"]["version_after"] == 2
    assert receipt_count(graph) == 2
    replay = backend(graph).snapshot.import_data(pot_id="p", payload=incoming)
    assert replay.metadata == imported.metadata
    assert store.snapshot.export_data(pot_id="p")["entities"] == []
    assert store.mutation.current_version("p") == 2
    assert receipt_count(graph) == 2
    store.mutation.reset_pot("p")
    assert store.mutation.current_version("p") == 3
    assert receipt_count(graph) == 3


@pytest.mark.asyncio
async def test_reset_wrapper_replays_supplied_identity_without_deleting_newer_data(
    graph,
):
    store = backend(graph)
    first = await store.mutation.reset_pot_async("p", mutation_id="reset-one")
    plan = batch()
    await store.mutation.apply_async(
        plan, expected_pot_id="p", provenance_context=context(plan, version=1)
    )
    replay = await store.mutation.reset_pot_async("p", mutation_id="reset-one")
    assert replay == first
    assert store.mutation.current_version("p") == 2
    assert receipt_count(graph) == 2
    assert len(store.snapshot.export_data(pot_id="p")["entities"]) == 2


@pytest.mark.parametrize("operation", ("reset", "snapshot"))
def test_admin_runtime_failure_rolls_back_effects_revision_and_receipt(
    graph, monkeypatch, operation
):
    store, plan = backend(graph), batch()
    store.mutation.compare_and_apply(
        plan, expected_pot_id="p", expected_version=0, provenance_context=context(plan)
    )
    before = snapshot(graph)
    original = falkordb_admin.compile_administration

    def broken(*args, **kwargs):
        query, params, result = original(*args, **kwargs)
        return (
            query.replace(
                "RETURN v.version,receipt.fingerprint",
                "WITH v,receipt RETURN keys(v.version),receipt.fingerprint",
            ),
            params,
            result,
        )

    monkeypatch.setattr(falkordb_admin, "compile_administration", broken)
    if operation == "reset":
        with pytest.raises(ResponseError, match="Type mismatch"):
            store.mutation.reset_pot("p")
    else:
        incoming = payload()
        incoming["entities"] = [
            {"key": "service:extra", "labels": ["Entity", "Service"], "properties": {}}
        ]
        incoming["claims"] = []
        with pytest.raises(ResponseError, match="Type mismatch"):
            store.snapshot.import_data(pot_id="p", payload=incoming)
    assert snapshot(graph) == before


def test_concurrent_snapshot_imports_publish_one_complete_receipt(graph):
    incoming = payload()
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(
            pool.map(
                lambda _: backend(graph).snapshot.import_data(
                    pot_id="p", payload=incoming
                ),
                range(6),
            )
        )
    assert all(result == results[0] for result in results)
    assert backend(graph).mutation.current_version("p") == 1
    assert receipt_count(graph) == 1
    assert len(backend(graph).snapshot.export_data(pot_id="p")["claims"]) == 1


def test_snapshot_identity_binding_and_empty_import_receipt(graph):
    store, incoming = backend(graph), payload()
    first = store.snapshot.import_data(
        pot_id="p", payload=incoming, mutation_id="import-one"
    )
    changed = payload()
    changed["entities"][0]["properties"]["name"] = "changed"
    with pytest.raises(MutationExecutionReuseError):
        store.snapshot.import_data(
            pot_id="p", payload=changed, mutation_id="import-one"
        )
    assert store.mutation.current_version("p") == 1
    again = store.snapshot.import_data(
        pot_id="p", payload=incoming, mutation_id="import-two"
    )
    assert again.metadata["mutation_id"] != first.metadata["mutation_id"]
    assert store.mutation.current_version("p") == 2
    empty = {"format_version": "2", "pot_id": "source", "entities": [], "claims": []}
    result = store.snapshot.import_data(
        pot_id="p", payload=empty, mutation_id="empty-import"
    )
    assert (
        result.metadata["execution"]["version_before"],
        result.metadata["execution"]["version_after"],
    ) == (2, 2)
    assert receipt_count(graph) == 3


@pytest.mark.parametrize("labels", ([], ["Service"]))
def test_snapshot_import_normalizes_entity_label_for_explicit_repeated_requests(
    graph, labels
):
    incoming = {
        "format_version": "2",
        "pot_id": "source",
        "entities": [{"key": "service:web", "labels": labels, "properties": {}}],
        "claims": [],
    }
    store = backend(graph)
    store.snapshot.import_data(pot_id="p", payload=incoming, mutation_id="first")
    store.snapshot.import_data(pot_id="p", payload=incoming, mutation_id="second")
    exported = store.snapshot.export_data(pot_id="p")
    assert exported["entities"][0]["labels"] == sorted({"Entity", *labels})
    assert store.mutation.current_version("p") == 2
    assert receipt_count(graph) == 2


def test_snapshot_response_loss_replays_authoritative_admin_receipt(graph, monkeypatch):
    original, lost = falkordb_atomic._rows, False

    def lose_response(native, response):
        nonlocal lost
        rows = original(native, response)
        if not lost and rows and len(rows[0]) == 8 and isinstance(rows[0][1], str):
            lost = True
            raise ConnectionError("lost admin publication response")
        return rows

    monkeypatch.setattr(falkordb_atomic, "_rows", lose_response)
    store, incoming = backend(graph), payload()
    with pytest.raises(ConnectionError, match="lost admin"):
        store.snapshot.import_data(
            pot_id="p", payload=incoming, mutation_id="import-lost"
        )
    recovered = backend(graph).snapshot.import_data(
        pot_id="p", payload=incoming, mutation_id="import-lost"
    )
    assert recovered.metadata["mutation_id"] == "import-lost"
    assert recovered.metadata["execution"]["version_after"] == 1
    assert store.mutation.current_version("p") == 1
    assert receipt_count(graph) == 1


def test_repair_combines_all_entity_deltas_and_no_work_has_no_receipt(graph):
    graph.query(
        "CREATE (:Entity:Activity {group_id:'p',entity_key:'service:web',description:'Web service'}),"
        "(:Entity:Activity {group_id:'p',entity_key:'service:db',name:'db'})"
    )
    store = backend(graph)
    report = store.analytics.repair("p", targets=("entity_summaries", "entity_labels"))
    assert report.repaired == {"entity_summaries": 2, "entity_labels": 2}
    assert store.mutation.current_version("p") == 1
    assert receipt_count(graph) == 1
    again = store.analytics.repair("p", targets=("entity_summaries", "entity_labels"))
    assert again.repaired == {"entity_summaries": 0, "entity_labels": 0}
    assert store.mutation.current_version("p") == 1
    assert receipt_count(graph) == 1


def test_repair_rejects_stale_selection_and_reselects_without_partial_write(
    graph, monkeypatch
):
    from potpie_context_engine.adapters.outbound.graph.falkordb_repair import (
        FalkorDBRepair,
    )

    graph.query(
        "CREATE (:Entity:Activity {group_id:'p',entity_key:'service:web',name:'web'})"
    )
    store, selected, original = backend(graph), False, FalkorDBRepair._entity_rows

    def concurrent_change(self, pot_id):
        nonlocal selected
        rows = list(original(self, pot_id))
        if not selected:
            selected = True
            store.mutation.apply(
                MutationBatch(
                    entity_upserts=[
                        EntityUpsert("service:extra", ("Entity", "Service"))
                    ]
                ),
                expected_pot_id="p",
                provenance_context=ProvenanceContext(mutation_id="concurrent"),
            )
        return rows

    monkeypatch.setattr(FalkorDBRepair, "_entity_rows", concurrent_change)
    report = store.analytics.repair("p", targets=("entity_summaries", "entity_labels"))
    assert report.repaired == {"entity_summaries": 1, "entity_labels": 1}
    assert store.mutation.current_version("p") == 2
    assert receipt_count(graph) == 2


def test_exact_invalidation_does_not_expand_entity_or_alias_keys(graph):
    store, plan = backend(graph), batch()
    plan.edge_upserts[0].properties["claim_key"] = "dependency"
    store.mutation.apply(plan, expected_pot_id="p", provenance_context=context(plan))
    assert (
        store.mutation.invalidate(
            pot_id="p", claim_keys=("service:web",), reason="exact"
        )
        == 0
    )
    assert graph.ro_query(
        "MATCH ()-[r:RELATES_TO {claim_key:'dependency'}]->() RETURN r.invalid_at"
    ).result_set == [[None]]
    assert (
        store.mutation.invalidate(
            pot_id="p", claim_keys=("dependency",), reason="exact"
        )
        == 1
    )


def test_nonlegacy_receipt_missing_ok_is_corrupt(graph):
    store, plan = backend(graph), batch()
    store.mutation.apply(plan, expected_pot_id="p", provenance_context=context(plan))
    raw = graph.ro_query("MATCH (r:PotpieMutationReceipt) RETURN r.payload").result_set[
        0
    ][0]
    damaged = json.loads(raw)
    damaged.pop("ok")
    graph.query(
        "MATCH (r:PotpieMutationReceipt) SET r.payload=$payload",
        params={"payload": json.dumps(damaged)},
    )
    with pytest.raises(MutationPersistenceError):
        store.mutation.lookup_execution(plan, expected_pot_id="p", mutation_id="m1")


def test_journal_reset_and_snapshot_import_remain_unsupported(graph):
    from potpie_context_engine.core.graph_journal import JournalError

    store = backend(graph)
    store.journal.activate(pot_id="p", rollback_enabled=True)
    with pytest.raises(JournalError, match="unsupported"):
        store.mutation.reset_pot("p")
    with pytest.raises(JournalError, match="unsupported"):
        store.snapshot.import_data(pot_id="p", payload=payload())
    assert store.mutation.current_version("p") == 0
    assert receipt_count(graph) == 0
