"""Native execution guarantees; endpoint must name a disposable Neo4j server."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
import asyncio
import json
import os
import subprocess
import threading
import time
import uuid

import pytest

from potpie_context_engine.adapters.outbound.graph import neo4j_atomic
from potpie_context_engine.adapters.outbound.graph._mutation_execution import (
    MutationExecutionReuseError,
)
from potpie_context_engine.adapters.outbound.graph.backends.neo4j_backend import (
    Neo4jGraphBackend,
)
from potpie_context_engine.adapters.outbound.graph.neo4j_writer import Neo4jGraphWriter
from potpie_context_engine.adapters.outbound.graph.neo4j_reader import (
    Neo4jClaimQueryStore,
)
from potpie_context_engine.core.claim_preparation import prepare_claim_assertions
from potpie_context_engine.core.definition import DEFAULT_GRAPH_DEFINITION
from potpie_context_engine.core.errors import (
    GraphMutationVersionConflict,
    ReconciliationApplyError,
)
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
from potpie_context_engine.core.ports.graph.mutation import (
    AsyncAtomicGraphMutationPort,
    AsyncRecoverableGraphMutationPort,
)
from potpie_context_engine.core.reconciliation import MutationBatch

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.environ.get("PROTOCOL_TEST_NEO4J_URI"),
        reason="PROTOCOL_TEST_NEO4J_URI must name a disposable test server",
    ),
]


class Settings:
    def is_enabled(self):
        return True

    def neo4j_uri(self):
        return os.environ["PROTOCOL_TEST_NEO4J_URI"]

    def neo4j_user(self):
        return os.environ.get("PROTOCOL_TEST_NEO4J_USERNAME", "neo4j")

    def neo4j_password(self):
        return os.environ.get("PROTOCOL_TEST_NEO4J_PASSWORD", "")


@pytest.fixture
def graph():
    return "execution-" + uuid.uuid4().hex, Neo4jGraphBackend(Settings())


def _batch(description="API depends on DB"):
    return MutationBatch(
        entity_upserts=[
            EntityUpsert("service:api", ("Service",)),
            EntityUpsert("service:db", ("Service",)),
        ],
        edge_upserts=[
            EdgeUpsert(
                "DEPENDS_ON",
                "service:api",
                "service:db",
                {
                    "claim_key": "claim:execution",
                    "fact": description,
                    "source_ref": "fixture:execution",
                    "truth": "source_observation",
                },
            )
        ],
    )


def _context(pot, batch, *, mutation="once", request="request", version=0):
    anchor = datetime.now(timezone.utc)
    for edge in batch.edge_upserts:
        edge.properties.setdefault("source_system", "fixture")
        edge.properties.setdefault("source_ref", f"fixture:{mutation}")
        edge.properties.setdefault("valid_at", anchor.isoformat())
    query = Neo4jClaimQueryStore(Settings())
    try:
        prepare_claim_assertions(
            batch,
            pot_id=pot,
            mutation_id=mutation,
            anchor=anchor,
            query=query,
            definition=DEFAULT_GRAPH_DEFINITION,
        )
    finally:
        query.close()
    return ProvenanceContext(
        mutation_id=mutation,
        prepared=PreparedMutation(
            pot_id=pot,
            mutation_id=mutation,
            request_fingerprint=request,
            batch_fingerprint=batch_fingerprint(batch),
            expected_version=version,
            prepared_at=anchor,
        ),
    )


def _query(query, **parameters):
    from neo4j import GraphDatabase

    settings = Settings()
    with GraphDatabase.driver(
        settings.neo4j_uri(), auth=(settings.neo4j_user(), settings.neo4j_password())
    ) as driver:
        with driver.session() as session:
            return [dict(row) for row in session.run(query, **parameters)]


def _record_ids(pot):
    return tuple(
        row["id"]
        for row in _query(
            "MATCH ()-[r:RELATES_TO {group_id:$pot}]->() "
            "RETURN coalesce(r.record_id,r.uuid) AS id ORDER BY id",
            pot=pot,
        )
    )


def test_prepared_execution_metadata_and_stale_replay(graph):
    pot, backend = graph
    batch = _batch()
    context = _context(pot, batch)
    before = datetime.now(timezone.utc)
    result = backend.mutation.compare_and_apply(
        batch, expected_pot_id=pot, expected_version=0, provenance_context=context
    )
    execution = result.execution
    assert execution is not None
    assert execution.pot_id == pot
    assert before <= execution.applied_at <= datetime.now(timezone.utc)
    assert (execution.version_before, execution.version_after) == (0, 1)
    assert execution.batch_fingerprint == batch_fingerprint(batch)
    assert execution.request_fingerprint == "request"
    assert execution.affected_claim_record_ids == _record_ids(pot)
    assert len(execution.affected_claim_record_ids) == 1
    restarted = Neo4jGraphBackend(Settings())
    assert (
        restarted.mutation.compare_and_apply(
            batch, expected_pot_id=pot, expected_version=999, provenance_context=context
        )
        == result
    )
    assert restarted.mutation.current_version(pot) == 1


def test_empty_execution_has_receipt_and_does_not_advance_revision(graph):
    pot, backend = graph
    batch = MutationBatch()
    context = _context(pot, batch)
    result = backend.mutation.compare_and_apply(
        batch, expected_pot_id=pot, expected_version=0, provenance_context=context
    )
    assert result.execution is not None
    assert (result.execution.version_before, result.execution.version_after) == (0, 0)
    assert result.execution.affected_claim_record_ids == ()
    assert backend.mutation.current_version(pot) == 0
    assert (
        backend.mutation.lookup_execution(
            batch, expected_pot_id=pot, mutation_id="once"
        ).result
        == result
    )
    assert (
        backend.mutation.compare_and_apply(
            batch, expected_pot_id=pot, expected_version=10, provenance_context=context
        )
        == result
    )


def test_concurrent_identical_execution_returns_one_receipt(graph):
    pot, backend = graph
    batch = _batch()
    context = _context(pot, batch)
    barrier = threading.Barrier(2)

    def apply(_):
        barrier.wait(timeout=10)
        return Neo4jGraphBackend(Settings()).mutation.compare_and_apply(
            batch, expected_pot_id=pot, expected_version=0, provenance_context=context
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        first, second = list(pool.map(apply, range(2)))
    assert first == second
    assert backend.mutation.current_version(pot) == 1
    assert len(_record_ids(pot)) == 1
    assert _query(
        "MATCH (r:PotpieMutationReceipt {pot_id:$pot}) RETURN count(r) AS count",
        pot=pot,
    ) == [{"count": 1}]


@pytest.mark.parametrize("compare", [False, True], ids=["apply", "compare"])
def test_frozen_revision_cannot_be_bypassed_by_current_caller_revision(graph, compare):
    pot, backend = graph
    batch = _batch()
    context = _context(pot, batch)
    backend.mutation.apply(
        MutationBatch(entity_upserts=[EntityUpsert("service:other", ("Service",))]),
        expected_pot_id=pot,
    )
    operation = (
        backend.mutation.compare_and_apply if compare else backend.mutation.apply
    )
    kwargs = {"expected_version": 1} if compare else {}
    with pytest.raises(GraphMutationVersionConflict) as caught:
        operation(batch, expected_pot_id=pot, provenance_context=context, **kwargs)
    assert caught.value.expected == 0 and caught.value.current == 1
    assert _record_ids(pot) == ()
    assert backend.mutation.current_version(pot) == 1
    assert (
        backend.mutation.lookup_execution(
            batch, expected_pot_id=pot, mutation_id="once"
        ).state
        == "absent"
    )


def test_replay_rejects_changed_request_and_changed_batch(graph):
    pot, backend = graph
    batch = _batch()
    context = _context(pot, batch)
    backend.mutation.apply(batch, expected_pot_id=pot, provenance_context=context)
    changed = replace(
        context, prepared=replace(context.prepared, request_fingerprint="other-request")
    )
    with pytest.raises(MutationExecutionReuseError, match="different request"):
        backend.mutation.compare_and_apply(
            batch, expected_pot_id=pot, expected_version=999, provenance_context=changed
        )
    with pytest.raises(MutationExecutionReuseError, match="different batch"):
        backend.mutation.apply(
            _batch("changed"),
            expected_pot_id=pot,
            provenance_context=ProvenanceContext(mutation_id="once"),
        )
    assert backend.mutation.current_version(pot) == 1


def test_receipt_serialization_failure_rolls_back_graph_revision_and_receipt(
    graph, monkeypatch
):
    pot, backend = graph
    batch = _batch()

    def fail(_):
        raise RuntimeError("injected receipt serialization failure")

    monkeypatch.setattr(neo4j_atomic, "execution_receipts_to_json", fail)
    with pytest.raises(RuntimeError, match="injected receipt"):
        backend.mutation.apply(
            batch, expected_pot_id=pot, provenance_context=_context(pot, batch)
        )
    assert backend.mutation.current_version(pot) == 0
    assert _record_ids(pot) == ()
    assert (
        backend.claim_query.entity_properties(pot_id=pot, entity_key="service:api")
        == {}
    )
    assert (
        backend.mutation.lookup_execution(
            batch, expected_pot_id=pot, mutation_id="once"
        ).state
        == "absent"
    )


def test_uncommitted_graph_and_execution_are_invisible_to_other_connections(
    graph, monkeypatch
):
    pot, backend = graph
    batch = _batch()
    entered, release = threading.Event(), threading.Event()
    original = Neo4jGraphWriter.upsert_edges

    async def pause(self, *args):
        entered.set()
        assert await asyncio.to_thread(release.wait, 15)
        return await original(self, *args)

    monkeypatch.setattr(Neo4jGraphWriter, "upsert_edges", pause)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(
            backend.mutation.apply,
            batch,
            expected_pot_id=pot,
            provenance_context=_context(pot, batch),
        )
        try:
            assert entered.wait(15)
            observer = Neo4jGraphBackend(Settings())
            assert observer.mutation.current_version(pot) == 0
            assert (
                observer.claim_query.entity_properties(
                    pot_id=pot, entity_key="service:api"
                )
                == {}
            )
            assert (
                observer.mutation.lookup_execution(
                    batch, expected_pot_id=pot, mutation_id="once"
                ).state
                == "absent"
            )
        finally:
            release.set()
        assert pending.result(timeout=15).ok
    assert backend.mutation.current_version(pot) == 1
    assert len(_record_ids(pot)) == 1


def test_response_loss_recovers_from_authoritative_receipt(graph, monkeypatch):
    pot, backend = graph
    batch = _batch()
    context = _context(pot, batch)
    original = neo4j_atomic.apply_atomic

    async def lose_response(*args, **kwargs):
        await original(*args, **kwargs)
        raise ConnectionError("injected loss after native commit")

    with monkeypatch.context() as patch:
        patch.setattr(neo4j_atomic, "apply_atomic", lose_response)
        with pytest.raises(ConnectionError, match="after native commit"):
            backend.mutation.apply(
                batch, expected_pot_id=pot, provenance_context=context
            )
    recovered = Neo4jGraphBackend(Settings())
    lookup = recovered.mutation.lookup_execution(
        batch, expected_pot_id=pot, mutation_id="once"
    )
    assert lookup.state == "completed"
    assert (
        recovered.mutation.apply(batch, expected_pot_id=pot, provenance_context=context)
        == lookup.result
    )
    assert recovered.mutation.current_version(pot) == 1


@pytest.mark.parametrize("operation", ["delete", "invalidate"])
def test_lifecycle_receipt_includes_affected_existing_claim_record(graph, operation):
    pot, backend = graph
    backend.mutation.apply(_batch(), expected_pot_id=pot)
    record_ids = _record_ids(pot)
    batch = (
        MutationBatch(
            edge_deletes=[EdgeDelete("DEPENDS_ON", "service:api", "service:db")]
        )
        if operation == "delete"
        else MutationBatch(
            invalidations=[
                InvalidationOp(
                    None, None, "retired", target_claim_keys=("claim:execution",)
                )
            ]
        )
    )
    result = backend.mutation.apply(batch, expected_pot_id=pot)
    assert result.execution is not None
    assert result.execution.affected_claim_record_ids == record_ids
    assert (result.execution.version_before, result.execution.version_after) == (1, 2)


def test_receipt_identity_is_unique_in_native_store(graph):
    from neo4j.exceptions import ConstraintError

    pot, backend = graph
    batch = _batch()
    backend.mutation.apply(
        batch, expected_pot_id=pot, provenance_context=_context(pot, batch)
    )
    with pytest.raises(ConstraintError):
        _query(
            "CREATE (:PotpieMutationReceipt {pot_id:$pot,mutation_id:'once',payload:'[]'})",
            pot=pot,
        )


def test_corrupt_receipt_lookup_fails_closed(graph):
    pot, backend = graph
    _query(
        "CREATE (:PotpieMutationReceipt {pot_id:$pot,mutation_id:'corrupt',payload:'not-json'})",
        pot=pot,
    )
    with pytest.raises(MutationPersistenceError, match="invalid durable"):
        backend.mutation.lookup_execution(
            _batch(), expected_pot_id=pot, mutation_id="corrupt"
        )


def test_corrupt_revision_fails_closed_for_reads_writes_and_reset(graph):
    pot, backend = graph
    _query(
        "CREATE (:PotpieRevision {pot_id:$pot,version:'invalid',lock_sequence:0})",
        pot=pot,
    )
    calls = (
        lambda: backend.mutation.current_version(pot),
        lambda: backend.mutation.apply(_batch(), expected_pot_id=pot),
        lambda: backend.mutation.reset_pot(pot),
    )
    for call in calls:
        with pytest.raises(MutationPersistenceError, match="pot revision"):
            call()
    assert _record_ids(pot) == ()


def test_missing_affected_record_identity_rolls_back_lifecycle_update(graph):
    pot, backend = graph
    backend.mutation.apply(_batch(), expected_pot_id=pot)
    _query(
        "MATCH ()-[r:RELATES_TO {group_id:$pot}]->() REMOVE r.uuid,r.record_id", pot=pot
    )
    batch = MutationBatch(
        invalidations=[
            InvalidationOp(
                None, None, "retired", target_claim_keys=("claim:execution",)
            )
        ]
    )
    with pytest.raises(ReconciliationApplyError, match="incarnation identity"):
        backend.mutation.apply(
            batch,
            expected_pot_id=pot,
            provenance_context=ProvenanceContext(mutation_id="retire"),
        )
    assert backend.mutation.current_version(pot) == 1
    assert _query(
        "MATCH ()-[r:RELATES_TO {group_id:$pot}]->() RETURN r.invalid_at AS invalid_at",
        pot=pot,
    ) == [{"invalid_at": None}]
    assert (
        backend.mutation.lookup_execution(
            batch, expected_pot_id=pot, mutation_id="retire"
        ).state
        == "absent"
    )


def test_legacy_receipt_keeps_unknown_metadata_and_original_fingerprint(graph):
    pot, backend = graph
    batch = _batch()
    context = _context(pot, batch)
    backend.mutation.apply(batch, expected_pot_id=pot, provenance_context=context)
    payload = _query(
        "MATCH (r:PotpieMutationReceipt {pot_id:$pot,mutation_id:'once'}) "
        "RETURN r.payload AS payload",
        pot=pot,
    )[0]["payload"]
    legacy = json.loads(payload)
    legacy[0]["result"].pop("execution")
    original_fingerprint = legacy[0]["batch_fingerprint"]
    _query(
        "MATCH (r:PotpieMutationReceipt {pot_id:$pot,mutation_id:'once'}) SET r.payload=$payload",
        pot=pot,
        payload=json.dumps(legacy),
    )
    lookup = backend.mutation.lookup_execution(
        batch, expected_pot_id=pot, mutation_id="once"
    )
    assert lookup.result.execution is None
    assert lookup.batch_fingerprint == original_fingerprint == batch_fingerprint(batch)
    replay = backend.mutation.compare_and_apply(
        batch, expected_pot_id=pot, expected_version=0, provenance_context=context
    )
    assert replay == lookup.result
    assert replay.execution is None
    assert backend.mutation.current_version(pot) == 1


@pytest.mark.asyncio
async def test_native_async_atomic_and_recovery_contracts(graph):
    pot, backend = graph
    mutation = backend.mutation
    assert isinstance(mutation, AsyncAtomicGraphMutationPort)
    assert isinstance(mutation, AsyncRecoverableGraphMutationPort)
    assert mutation.capabilities.atomic_writes
    assert mutation.capabilities.revision_check
    assert mutation.capabilities.execution_lookup
    assert mutation.capabilities.durable_receipts
    assert await mutation.current_version_async(pot) == 0
    batch = _batch()
    context = _context(pot, batch)
    result = await mutation.compare_and_apply_async(
        batch, expected_pot_id=pot, expected_version=0, provenance_context=context
    )
    assert (
        await mutation.lookup_execution_async(
            batch, expected_pot_id=pot, mutation_id="once"
        )
    ).result == result
    assert (
        await mutation.apply_async(
            batch, expected_pot_id=pot, provenance_context=context
        )
        == result
    )
    assert await mutation.current_version_async(pot) == 1
    assert (await mutation.reset_pot_async(pot))["ok"]
    assert await mutation.current_version_async(pot) == 2


def test_empty_reset_is_revision_barrier_and_retains_receipts(graph):
    pot, backend = graph
    batch = MutationBatch()
    result = backend.mutation.apply(
        batch, expected_pot_id=pot, provenance_context=_context(pot, batch)
    )
    assert backend.mutation.reset_pot(pot)["ok"]
    assert backend.mutation.current_version(pot) == 1
    assert backend.mutation.reset_pot(pot)["ok"]
    assert backend.mutation.current_version(pot) == 2
    assert (
        backend.mutation.lookup_execution(
            batch, expected_pot_id=pot, mutation_id="once"
        ).result
        == result
    )


@pytest.mark.skipif(
    not os.environ.get("PROTOCOL_TEST_NEO4J_CONTAINER"),
    reason="restart requires an explicitly named disposable Docker container",
)
def test_native_server_restart_retains_execution_receipt(graph):
    from neo4j import GraphDatabase
    from neo4j.exceptions import DriverError, Neo4jError

    pot, backend = graph
    batch = _batch()
    context = _context(pot, batch)
    first = backend.mutation.apply(
        batch, expected_pot_id=pot, provenance_context=context
    )
    subprocess.run(
        ["docker", "restart", os.environ["PROTOCOL_TEST_NEO4J_CONTAINER"]],
        check=True,
        capture_output=True,
        timeout=60,
    )
    settings = Settings()
    deadline = time.monotonic() + 60
    with GraphDatabase.driver(
        settings.neo4j_uri(), auth=(settings.neo4j_user(), settings.neo4j_password())
    ) as driver:
        while True:
            try:
                driver.verify_connectivity()
                break
            except (DriverError, Neo4jError, OSError):
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.5)
    restarted = Neo4jGraphBackend(Settings())
    assert (
        restarted.mutation.lookup_execution(
            batch, expected_pot_id=pot, mutation_id="once"
        ).result
        == first
    )
    assert (
        restarted.mutation.apply(batch, expected_pot_id=pot, provenance_context=context)
        == first
    )
    assert restarted.mutation.current_version(pot) == 1
    assert _record_ids(pot) == first.execution.affected_claim_record_ids
