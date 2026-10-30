"""Authoritative receipts fence missing/regressed Neo4j revision metadata."""

from concurrent.futures import ThreadPoolExecutor
import asyncio
import json
import os
import threading
from types import SimpleNamespace
import uuid

import pytest

from potpie_context_engine.adapters.outbound.graph import neo4j_execution_state
from potpie_context_engine.adapters.outbound.graph.backends.neo4j_backend import (
    Neo4jGraphBackend,
)
from potpie_context_engine.core.graph_mutations import EntityUpsert, ProvenanceContext
from potpie_context_engine.core.mutation_execution import MutationPersistenceError
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
        yield "revision-" + uuid.uuid4().hex, Neo4jGraphBackend(settings), driver


def _run(driver, query, **parameters):
    with driver.session() as session:
        return [dict(row) for row in session.run(query, **parameters)]


def _batch(name):
    return MutationBatch(entity_upserts=[EntityUpsert("service:" + name, ("Service",))])


def _apply(backend, pot, name):
    return backend.mutation.apply(
        _batch(name),
        expected_pot_id=pot,
        provenance_context=ProvenanceContext(mutation_id=name),
    )


@pytest.mark.parametrize("fault", ["missing", "regressed"])
def test_receipt_floor_fences_reads_lookup_replay_and_new_execution(graph, fault):
    pot, backend, driver = graph
    first = _apply(backend, pot, "a")
    _apply(backend, pot, "b")
    if fault == "missing":
        _run(driver, "MATCH (r:PotpieRevision {pot_id:$pot}) DELETE r", pot=pot)
        expected = 0
    else:
        _run(driver, "MATCH (r:PotpieRevision {pot_id:$pot}) SET r.version=1", pot=pot)
        expected = 1
    restarted = Neo4jGraphBackend(backend.settings)
    attempts = (
        lambda: restarted.mutation.current_version(pot),
        lambda: restarted.mutation.lookup_execution(
            _batch("a"), expected_pot_id=pot, mutation_id="a"
        ),
        lambda: restarted.mutation.lookup_execution(
            _batch("c"), expected_pot_id=pot, mutation_id="unknown"
        ),
        lambda: _apply(restarted, pot, "a"),
        lambda: restarted.mutation.compare_and_apply(
            _batch("c"),
            expected_pot_id=pot,
            expected_version=expected,
            provenance_context=ProvenanceContext(mutation_id="c"),
        ),
        lambda: restarted.mutation.reset_pot(pot, mutation_id="reset:corrupt"),
    )
    for attempt in attempts:
        with pytest.raises(MutationPersistenceError, match="receipt floor"):
            attempt()
    assert first.execution.version_after == 1
    assert _run(
        driver,
        "MATCH (e:Entity {group_id:$pot}) RETURN e.entity_key AS key ORDER BY key",
        pot=pot,
    ) == [
        {"key": "service:a"},
        {"key": "service:b"},
    ]
    assert _run(
        driver,
        "MATCH (r:PotpieMutationReceipt {pot_id:$pot}) RETURN count(r) AS count",
        pot=pot,
    ) == [{"count": 2}]
    assert _run(
        driver,
        "MATCH (r:PotpieRevision {pot_id:$pot}) RETURN r.version AS version",
        pot=pot,
    ) == ([] if fault == "missing" else [{"version": 1}])


def test_modern_empty_receipt_requires_revision_even_at_zero(graph):
    pot, backend, driver = graph
    batch = MutationBatch()
    context = ProvenanceContext(mutation_id="empty")
    result = backend.mutation.apply(
        batch, expected_pot_id=pot, provenance_context=context
    )
    assert result.execution.version_after == 0
    _run(driver, "MATCH (r:PotpieRevision {pot_id:$pot}) DELETE r", pot=pot)
    attempts = (
        lambda: backend.mutation.current_version(pot),
        lambda: backend.mutation.lookup_execution(
            batch, expected_pot_id=pot, mutation_id="empty"
        ),
        lambda: backend.mutation.apply(
            batch, expected_pot_id=pot, provenance_context=context
        ),
        lambda: _apply(backend, pot, "new"),
    )
    for attempt in attempts:
        with pytest.raises(MutationPersistenceError, match="receipt floor"):
            attempt()
    assert _run(
        driver,
        "MATCH (r:PotpieRevision {pot_id:$pot}) RETURN count(r) AS count",
        pot=pot,
    ) == [{"count": 0}]


def test_null_execution_metadata_cannot_erase_revision_floor(graph):
    pot, backend, driver = graph
    _apply(backend, pot, "modern")
    payload = _run(
        driver,
        "MATCH (r:PotpieMutationReceipt {pot_id:$pot,mutation_id:'modern'}) RETURN r.payload AS payload",
        pot=pot,
    )[0]["payload"]
    corrupt = json.loads(payload)
    corrupt[0]["result"]["execution"] = None
    _run(
        driver,
        "MATCH (r:PotpieMutationReceipt {pot_id:$pot,mutation_id:'modern'}) SET r.payload=$payload",
        pot=pot,
        payload=json.dumps(corrupt),
    )
    _run(driver, "MATCH (r:PotpieRevision {pot_id:$pot}) DELETE r", pot=pot)
    attempts = (
        lambda: backend.mutation.current_version(pot),
        lambda: backend.mutation.lookup_execution(
            _batch("modern"), expected_pot_id=pot, mutation_id="modern"
        ),
        lambda: _apply(backend, pot, "new"),
    )
    for attempt in attempts:
        with pytest.raises(MutationPersistenceError, match="invalid durable"):
            attempt()
    assert _run(
        driver,
        "MATCH (r:PotpieRevision {pot_id:$pot}) RETURN count(r) AS count",
        pot=pot,
    ) == [{"count": 0}]


def test_duplicate_receipt_key_cannot_lower_revision_floor(graph):
    pot, backend, driver = graph
    _apply(backend, pot, "modern")
    payload = _run(
        driver,
        "MATCH (r:PotpieMutationReceipt {pot_id:$pot,mutation_id:'modern'}) RETURN r.payload AS payload",
        pot=pot,
    )[0]["payload"]
    assert payload.count('"version_after": 1') == 1
    corrupt = payload.replace(
        '"version_after": 1', '"version_after": 1, "version_after": 0', 1
    )
    _run(
        driver,
        "MATCH (r:PotpieMutationReceipt {pot_id:$pot,mutation_id:'modern'}) SET r.payload=$payload",
        pot=pot,
        payload=corrupt,
    )
    _run(driver, "MATCH (r:PotpieRevision {pot_id:$pot}) SET r.version=0", pot=pot)
    attempts = (
        lambda: backend.mutation.current_version(pot),
        lambda: backend.mutation.lookup_execution(
            _batch("modern"), expected_pot_id=pot, mutation_id="modern"
        ),
        lambda: backend.mutation.compare_and_apply(
            _batch("new"),
            expected_pot_id=pot,
            expected_version=0,
            provenance_context=ProvenanceContext(mutation_id="new"),
        ),
    )
    for attempt in attempts:
        with pytest.raises(MutationPersistenceError, match="invalid durable") as caught:
            attempt()
        assert "duplicate storage key: version_after" in str(caught.value.__cause__)
    assert _run(
        driver,
        "MATCH (r:PotpieRevision {pot_id:$pot}) RETURN r.version AS version",
        pot=pot,
    ) == [{"version": 0}]
    assert _run(
        driver,
        "MATCH (r:PotpieMutationReceipt {pot_id:$pot}) RETURN count(r) AS count",
        pot=pot,
    ) == [{"count": 1}]
    assert _run(
        driver, "MATCH (e:Entity {group_id:$pot}) RETURN e.entity_key AS key", pot=pot
    ) == [{"key": "service:modern"}]


def test_fresh_bootstrap_serializes_concurrent_identical_callers(graph):
    pot, backend, driver = graph
    ready = threading.Barrier(4)

    def apply(_):
        ready.wait(timeout=10)
        return _apply(Neo4jGraphBackend(backend.settings), pot, "fresh")

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(apply, range(4)))
    assert all(result == results[0] for result in results)
    assert backend.mutation.current_version(pot) == 1
    assert _run(
        driver,
        "MATCH (r:PotpieMutationReceipt {pot_id:$pot}) RETURN count(r) AS count",
        pot=pot,
    ) == [{"count": 1}]


def test_revision_read_allows_commit_between_receipt_scan_and_revision_read(
    graph, monkeypatch
):
    pot, backend, _ = graph
    _apply(backend, pot, "before")
    entered, release = threading.Event(), threading.Event()
    original = neo4j_execution_state._one

    async def pause_revision_read(transaction, query, **parameters):
        if query == neo4j_execution_state._REVISION and parameters.get("pot") == pot:
            entered.set()
            assert await asyncio.to_thread(release.wait, 15)
        return await original(transaction, query, **parameters)

    monkeypatch.setattr(neo4j_execution_state, "_one", pause_revision_read)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(backend.mutation.current_version, pot)
        try:
            assert entered.wait(15)
            assert (
                _apply(
                    Neo4jGraphBackend(backend.settings), pot, "after"
                ).execution.version_after
                == 2
            )
        finally:
            release.set()
        assert pending.result(timeout=15) == 2


def test_revision_above_modern_receipt_floor_remains_valid(graph):
    pot, backend, driver = graph
    _apply(backend, pot, "before")
    _run(driver, "MATCH (r:PotpieRevision {pot_id:$pot}) SET r.version=7", pot=pot)
    assert backend.mutation.current_version(pot) == 7
    assert (
        backend.mutation.lookup_execution(
            _batch("before"), expected_pot_id=pot, mutation_id="before"
        ).state
        == "completed"
    )
    result = backend.mutation.compare_and_apply(
        _batch("after"),
        expected_pot_id=pot,
        expected_version=7,
        provenance_context=ProvenanceContext(mutation_id="after"),
    )
    assert (result.execution.version_before, result.execution.version_after) == (7, 8)
