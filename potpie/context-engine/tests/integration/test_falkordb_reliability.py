"""Native reliability evidence shared by isolated FalkorDB server and Lite."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import subprocess
import threading
import time
from urllib.parse import urlparse
import uuid
from types import SimpleNamespace

import pytest
from redis.exceptions import ConnectionError, ResponseError

from potpie_context_engine.adapters.outbound.graph import falkordb_atomic
from potpie_context_engine.adapters.outbound.graph._mutation_execution import (
    MutationExecutionReuseError,
)
from potpie_context_engine.adapters.outbound.graph.backends.falkordb_backend import (
    FalkorDBGraphBackend,
)
from potpie_context_engine.core.errors import GraphMutationVersionConflict
from potpie_context_engine.core.claim_preparation import prepare_claim_assertions
from potpie_context_engine.core.definition import DEFAULT_GRAPH_DEFINITION
from potpie_context_engine.core.graph_mutations import (
    EdgeDelete,
    EdgeUpsert,
    EntityUpsert,
    InvalidationOp,
    ProvenanceContext,
)
from potpie_context_engine.core.mutation_execution import (
    MutationPersistenceError,
    PreparedMutation,
    batch_fingerprint,
)
from potpie_context_engine.core.reconciliation import MutationBatch

from tests.embedded_falkordb import embedded_falkordb

pytestmark = pytest.mark.integration


class Settings:
    def is_enabled(self):
        return True

    def falkordb_graph_name(self):
        return "reliability"


@pytest.fixture(params=("lite", "server"))
def graph(request, tmp_path):
    if request.param == "lite":
        with embedded_falkordb(tmp_path / "reliability.db") as database:
            yield database.select_graph("reliability")
        return
    endpoint = os.getenv("PROTOCOL_TEST_FALKORDB_URL")
    if not endpoint:
        pytest.skip("set PROTOCOL_TEST_FALKORDB_URL for native server evidence")
    from falkordb import FalkorDB

    parsed = urlparse(endpoint)
    database = FalkorDB(host=parsed.hostname, port=parsed.port or 6379)
    native = database.select_graph(f"reliability_{uuid.uuid4().hex}")
    try:
        yield native
    finally:
        try:
            native.delete()
        except ResponseError as exc:
            if "empty key" not in str(exc).lower():
                raise
        database.connection.close()


def batch(value="before"):
    return MutationBatch(
        entity_upserts=[
            EntityUpsert("service:web", ("Entity", "Service"), {"description": value}),
            EntityUpsert("service:db", ("Entity", "Service")),
        ],
        edge_upserts=[
            EdgeUpsert(
                "DEPENDS_ON",
                "service:web",
                "service:db",
                {
                    "claim_key": "dependency",
                    "source_ref": "config",
                    "truth": "source_observation",
                },
            )
        ],
    )


def context(plan, mid="m1", *, version=0, request="request-one", query=None):
    if any(edge.resolution is None for edge in plan.edge_upserts) or any(
        op.closures is None for op in (*plan.edge_deletes, *plan.invalidations)
    ):
        prepare_claim_assertions(
            plan,
            pot_id="p",
            mutation_id=mid,
            anchor=datetime(2026, 1, 1, tzinfo=timezone.utc)
            + timedelta(seconds=version),
            query=query or SimpleNamespace(find_claims=lambda _: []),
            definition=DEFAULT_GRAPH_DEFINITION,
        )
    return ProvenanceContext(
        mutation_id=mid,
        source_event_id=f"event:{mid}",
        prepared=PreparedMutation(
            "p",
            mid,
            request,
            batch_fingerprint(plan),
            version,
            datetime.now(timezone.utc),
        ),
    )


def backend(graph):
    return FalkorDBGraphBackend(Settings(), graph_provider=lambda: graph)


def snapshot(graph):
    return (
        graph.ro_query(
            "MATCH (e:Entity) RETURN properties(e) ORDER BY e.entity_key"
        ).result_set,
        graph.ro_query(
            "MATCH ()-[r:RELATES_TO]->() RETURN properties(r) ORDER BY r.uuid"
        ).result_set,
        graph.ro_query("MATCH (v:PotpieRevision) RETURN properties(v)").result_set,
        graph.ro_query(
            "MATCH (r:PotpieMutationReceipt) RETURN properties(r) ORDER BY r.mutation_id"
        ).result_set,
    )


def test_metadata_request_binding_and_empty_revision(graph):
    port, plan = backend(graph).mutation, batch()
    prepared = context(plan)
    result = port.compare_and_apply(
        plan, expected_pot_id="p", expected_version=0, provenance_context=prepared
    )
    execution = result.execution
    assert execution is not None
    assert (execution.version_before, execution.version_after) == (0, 1)
    assert execution.request_fingerprint == "request-one"
    assert execution.batch_fingerprint == batch_fingerprint(plan)
    ids = graph.ro_query(
        "MATCH ()-[r:RELATES_TO]->() RETURN coalesce(r.record_id,r.uuid)"
    ).result_set
    assert execution.affected_claim_record_ids == tuple(row[0] for row in ids)
    assert (
        port.lookup_execution(plan, expected_pot_id="p", mutation_id="m1").result
        == result
    )
    assert (
        port.compare_and_apply(
            plan, expected_pot_id="p", expected_version=999, provenance_context=prepared
        )
        == result
    )
    with pytest.raises(MutationExecutionReuseError):
        port.compare_and_apply(
            plan,
            expected_pot_id="p",
            expected_version=0,
            provenance_context=context(plan, request="other"),
        )
    empty = MutationBatch()
    receipt = port.compare_and_apply(
        empty,
        expected_pot_id="p",
        expected_version=1,
        provenance_context=context(empty, "empty", version=1),
    )
    assert (receipt.execution.version_before, receipt.execution.version_after) == (1, 1)
    assert receipt.execution.affected_claim_record_ids == ()
    assert port.current_version("p") == 1


def test_frozen_revision_cannot_be_replaced_by_caller(graph):
    port, plan = backend(graph).mutation, batch()
    port.compare_and_apply(
        plan, expected_pot_id="p", expected_version=0, provenance_context=context(plan)
    )
    stale = batch("stale")
    with pytest.raises(GraphMutationVersionConflict) as raised:
        port.compare_and_apply(
            stale,
            expected_pot_id="p",
            expected_version=1,
            provenance_context=context(stale, "stale", version=0),
        )
    assert raised.value.expected == 0
    assert port.current_version("p") == 1
    assert (
        port.lookup_execution(stale, expected_pot_id="p", mutation_id="stale").state
        == "absent"
    )


@pytest.mark.parametrize("journal", (False, True))
def test_concurrent_identical_and_changed_batches_have_one_receipt(graph, journal):
    if journal:
        backend(graph).journal.activate(pot_id="p", rollback_enabled=True)
    plan, barrier = batch(), threading.Barrier(8)

    def apply():
        barrier.wait()
        return backend(graph).mutation.compare_and_apply(
            plan,
            expected_pot_id="p",
            expected_version=0,
            provenance_context=context(plan),
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: apply(), range(8)))
    assert all(result == results[0] for result in results)
    assert backend(graph).mutation.current_version("p") == 1
    assert graph.ro_query(
        "MATCH (r:PotpieMutationReceipt) RETURN count(r)"
    ).result_set == [[1]]
    assert graph.ro_query("MATCH (v:PotpieRevision) RETURN count(v)").result_set == [
        [1]
    ]
    changed = batch("changed")
    with pytest.raises(MutationExecutionReuseError):
        backend(graph).mutation.compare_and_apply(
            changed,
            expected_pot_id="p",
            expected_version=1,
            provenance_context=context(changed),
        )


def test_runtime_failure_after_graph_and_receipt_writes_rolls_back(graph, monkeypatch):
    port, seed = backend(graph).mutation, batch()
    seed.entity_upserts.append(EntityUpsert("service:cache", ("Service",)))
    seed.edge_upserts.append(
        EdgeUpsert(
            "DEPENDS_ON",
            "service:web",
            "service:cache",
            {
                "claim_key": "cache",
                "source_ref": "other-config",
                "truth": "source_observation",
            },
        )
    )
    port.compare_and_apply(
        seed, expected_pot_id="p", expected_version=0, provenance_context=context(seed)
    )
    before = snapshot(graph)
    failing = batch("must roll back")
    failing.entity_upserts.append(EntityUpsert("service:worker", ("Entity", "Service")))
    failing.edge_deletes.append(EdgeDelete("DEPENDS_ON", "service:web", "service:db"))
    failing.invalidations.append(
        InvalidationOp(
            target_entity_key="service:cache", target_edge=None, reason="rollback"
        )
    )
    compile_query = falkordb_atomic._compile

    def broken(*args, **kwargs):
        query, params, result = compile_query(*args, **kwargs)
        # The type error depends on a stored property and occurs after entity,
        # relationship, revision and receipt writes have all executed.
        query = query.replace(
            "RETURN v.version,receipt.fingerprint",
            "WITH v,receipt RETURN keys(v.version),receipt.fingerprint",
        )
        return query, params, result

    monkeypatch.setattr(falkordb_atomic, "_compile", broken)
    with pytest.raises(ResponseError, match="Type mismatch"):
        port.compare_and_apply(
            failing,
            expected_pot_id="p",
            expected_version=1,
            provenance_context=context(
                failing, "failed", version=1, query=backend(graph).claim_query
            ),
        )
    assert snapshot(graph) == before
    assert (
        port.lookup_execution(failing, expected_pot_id="p", mutation_id="failed").state
        == "absent"
    )


def test_response_loss_recovers_committed_receipt(graph, monkeypatch):
    plan, original = batch(), falkordb_atomic._rows
    lost = False

    def lose_response(native, response):
        nonlocal lost
        rows = original(native, response)
        if (
            not lost
            and rows
            and len(rows[0]) == 8
            and rows[0][1] == batch_fingerprint(plan)
        ):
            lost = True
            raise ConnectionError("response lost after native commit")
        return rows

    monkeypatch.setattr(falkordb_atomic, "_rows", lose_response)
    with pytest.raises(ConnectionError, match="response lost"):
        backend(graph).mutation.compare_and_apply(
            plan,
            expected_pot_id="p",
            expected_version=0,
            provenance_context=context(plan),
        )
    fresh = backend(graph).mutation
    recovered = fresh.lookup_execution(plan, expected_pot_id="p", mutation_id="m1")
    assert recovered.state == "completed"
    assert (
        fresh.compare_and_apply(
            plan,
            expected_pot_id="p",
            expected_version=0,
            provenance_context=context(plan),
        )
        == recovered.result
    )
    assert fresh.current_version("p") == 1


def test_duplicate_receipts_fail_closed(graph):
    plan, port = batch(), backend(graph).mutation
    port.compare_and_apply(
        plan, expected_pot_id="p", expected_version=0, provenance_context=context(plan)
    )
    graph.query(
        "MATCH (r:PotpieMutationReceipt) CREATE (duplicate:PotpieMutationReceipt) SET duplicate=properties(r)"
    )
    with pytest.raises(MutationPersistenceError, match="duplicate"):
        port.lookup_execution(plan, expected_pot_id="p", mutation_id="m1")
    with pytest.raises(MutationPersistenceError, match="duplicate"):
        port.compare_and_apply(
            plan,
            expected_pot_id="p",
            expected_version=0,
            provenance_context=context(plan),
        )


def test_existing_claim_record_identity_is_preserved(graph):
    store, plan = backend(graph), batch()
    store.mutation.compare_and_apply(
        plan, expected_pot_id="p", expected_version=0, provenance_context=context(plan)
    )
    graph.query("MATCH ()-[r:RELATES_TO]->() SET r.record_id='existing-record'")
    before = graph.ro_query("MATCH ()-[r:RELATES_TO]->() RETURN r.uuid").result_set
    plan = batch()
    result = store.mutation.compare_and_apply(
        plan,
        expected_pot_id="p",
        expected_version=1,
        provenance_context=context(plan, "again", version=1, query=store.claim_query),
    )
    assert result.execution.affected_claim_record_ids == ("existing-record",)
    assert (
        graph.ro_query("MATCH ()-[r:RELATES_TO]->() RETURN r.uuid").result_set == before
    )


def test_corrupt_native_affected_id_metadata_fails_closed(graph):
    store, plan = backend(graph), batch()
    store.mutation.compare_and_apply(
        plan, expected_pot_id="p", expected_version=0, provenance_context=context(plan)
    )
    graph.query("MATCH (r:PotpieMutationReceipt) SET r.affected_claim_record_ids=[1]")
    with pytest.raises(MutationPersistenceError):
        store.mutation.lookup_execution(plan, expected_pot_id="p", mutation_id="m1")


@pytest.mark.parametrize("operation", ("upsert", "delete", "invalidate"))
def test_touched_legacy_claim_without_identity_requires_migration(graph, operation):
    store, plan = backend(graph), batch()
    store.mutation.compare_and_apply(
        plan, expected_pot_id="p", expected_version=0, provenance_context=context(plan)
    )
    graph.query("MATCH ()-[r:RELATES_TO]->() REMOVE r.uuid,r.record_id")
    before = snapshot(graph)
    if operation == "upsert":
        mutation = batch("must remain unchanged")
    elif operation == "delete":
        mutation = MutationBatch(
            edge_deletes=[EdgeDelete("DEPENDS_ON", "service:web", "service:db")]
        )
    else:
        mutation = MutationBatch(
            invalidations=[
                InvalidationOp(
                    target_entity_key=None,
                    target_edge=None,
                    target_claim_keys=("dependency",),
                    reason="legacy",
                )
            ]
        )
    with pytest.raises(MutationPersistenceError, match="migration required"):
        store.mutation.compare_and_apply(
            mutation,
            expected_pot_id="p",
            expected_version=1,
            provenance_context=ProvenanceContext(mutation_id="legacy"),
        )
    assert snapshot(graph) == before
    assert (
        store.mutation.lookup_execution(
            mutation, expected_pot_id="p", mutation_id="legacy"
        ).state
        == "absent"
    )


def test_legacy_receipt_preserves_unknown_execution_metadata(graph):
    import json

    store, plan = backend(graph), batch()
    graph.query(
        "CREATE (:PotpieMutationReceipt {pot_id:'p',mutation_id:'legacy',fingerprint:$fp,"
        "payload:$payload,entities:0,edges:0,deletes:0,invalidations:0})",
        params={
            "fp": batch_fingerprint(plan),
            "payload": json.dumps({"mutation_summary": {}}),
        },
    )
    receipt = store.mutation.lookup_execution(
        plan, expected_pot_id="p", mutation_id="legacy"
    )
    assert receipt.state == "completed" and receipt.result.execution is None
    replay = store.mutation.compare_and_apply(
        plan,
        expected_pot_id="p",
        expected_version=999,
        provenance_context=ProvenanceContext(mutation_id="legacy"),
    )
    assert replay == receipt.result
    assert store.mutation.current_version("p") == 0


@pytest.mark.asyncio
async def test_native_async_contract(graph):
    port, plan = backend(graph).mutation, batch()
    assert port.capabilities.atomic_writes
    assert port.capabilities.revision_check
    assert port.capabilities.execution_lookup
    assert port.capabilities.durable_receipts
    assert await port.current_version_async("p") == 0
    result = await port.compare_and_apply_async(
        plan, expected_pot_id="p", expected_version=0, provenance_context=context(plan)
    )
    assert (
        await port.lookup_execution_async(plan, expected_pot_id="p", mutation_id="m1")
    ).result == result
    assert await port.current_version_async("p") == 1


def test_journal_metadata_empty_revision_and_native_rollback(graph, monkeypatch):
    from potpie_context_engine.adapters.outbound.graph import falkordb_journal

    store, plan = backend(graph), batch()
    store.journal.activate(pot_id="p", rollback_enabled=True)
    result = store.mutation.compare_and_apply(
        plan, expected_pot_id="p", expected_version=0, provenance_context=context(plan)
    )
    assert result.execution is not None
    ids = graph.ro_query("MATCH ()-[r:RELATES_TO]->() RETURN r.record_id").result_set
    assert result.execution.affected_claim_record_ids == tuple(row[0] for row in ids)
    assert (
        store.mutation.lookup_execution(
            plan, expected_pot_id="p", mutation_id="m1"
        ).result
        == result
    )
    empty = MutationBatch()
    empty_result = store.mutation.compare_and_apply(
        empty,
        expected_pot_id="p",
        expected_version=1,
        provenance_context=context(empty, "empty", version=1),
    )
    assert (
        empty_result.execution.version_before,
        empty_result.execution.version_after,
    ) == (1, 1)
    before = snapshot(graph)
    publish = falkordb_journal._publish_query

    def broken(*args, **kwargs):
        query, params = publish(*args, **kwargs)
        return query.replace(
            "RETURN v.version", "WITH v RETURN keys(v.version)"
        ), params

    monkeypatch.setattr(falkordb_journal, "_publish_query", broken)
    failing = batch("rolled back")
    with pytest.raises(ResponseError, match="Type mismatch"):
        store.mutation.compare_and_apply(
            failing,
            expected_pot_id="p",
            expected_version=1,
            provenance_context=context(
                failing, "failed", version=1, query=store.claim_query
            ),
        )
    assert snapshot(graph) == before
    assert (
        store.mutation.lookup_execution(
            failing, expected_pot_id="p", mutation_id="failed"
        ).state
        == "absent"
    )


def test_lite_receipt_survives_process_restart(tmp_path: Path):
    path, plan = tmp_path / "restart.db", batch()
    with embedded_falkordb(path) as database:
        result = backend(database.select_graph("restart")).mutation.compare_and_apply(
            plan,
            expected_pot_id="p",
            expected_version=0,
            provenance_context=context(plan),
        )
        database.connection.save()
    with embedded_falkordb(path) as database:
        port = backend(database.select_graph("restart")).mutation
        assert (
            port.lookup_execution(plan, expected_pot_id="p", mutation_id="m1").result
            == result
        )
        assert (
            port.compare_and_apply(
                plan,
                expected_pot_id="p",
                expected_version=0,
                provenance_context=context(plan),
            )
            == result
        )
        assert port.current_version("p") == 1


def test_server_aof_receipt_survives_restart():
    container = os.getenv("POTPIE_TEST_FALKORDB_RESTART_CONTAINER")
    endpoint = os.getenv("PROTOCOL_TEST_FALKORDB_URL")
    if not container or not endpoint:
        pytest.skip("requires an explicitly dedicated restartable FalkorDB container")
    from falkordb import FalkorDB

    parsed = urlparse(endpoint)
    database = FalkorDB(host=parsed.hostname, port=parsed.port or 6379)
    graph = database.select_graph(f"restart_{uuid.uuid4().hex}")
    plan = batch()
    try:
        result = backend(graph).mutation.compare_and_apply(
            plan,
            expected_pot_id="p",
            expected_version=0,
            provenance_context=context(plan),
        )
        subprocess.run(
            ["rtk", "docker", "restart", container], check=True, capture_output=True
        )
        deadline = time.monotonic() + 15
        while True:
            try:
                # Redis PING becomes ready before FalkorDB finishes replaying
                # its module state from the append-only log.
                graph.ro_query("RETURN 1")
                break
            except (ConnectionError, ResponseError) as exc:
                if isinstance(exc, ResponseError) and not any(
                    marker in str(exc).lower()
                    for marker in ("pending queries", "loading")
                ):
                    raise
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.1)
        fresh = backend(database.select_graph(graph.name)).mutation
        assert (
            fresh.lookup_execution(plan, expected_pot_id="p", mutation_id="m1").result
            == result
        )
        assert (
            fresh.compare_and_apply(
                plan,
                expected_pot_id="p",
                expected_version=0,
                provenance_context=context(plan),
            )
            == result
        )
        assert fresh.current_version("p") == 1
    finally:
        graph.delete()
        database.connection.close()
