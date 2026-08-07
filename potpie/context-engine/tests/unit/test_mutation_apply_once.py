from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import multiprocessing
from pathlib import Path
import threading

import pytest

from potpie_context_core.context_events import EventRef
from potpie_context_core.graph_mutations import (
    EdgeUpsert,
    EntityUpsert,
    ProvenanceContext,
)
from potpie_context_core.lifecycle import SetupPlan
from potpie_context_core.ports.claim_query import ClaimQueryFilter
from potpie_context_core.ports.graph.mutation import MutationExecutionState
from potpie_context_core.reconciliation import MutationBatch
from potpie_context_engine.adapters.outbound.graph._mutation_execution import (
    MutationExecutionReuseError,
)
from potpie_context_engine.adapters.outbound.graph.backends.embedded_backend import (
    EmbeddedGraphBackend,
)
from potpie_context_engine.adapters.outbound.graph.backends.in_memory_backend import (
    InMemoryGraphBackend,
)

pytestmark = pytest.mark.unit

POT = "apply-once-pot"
MUTATION_ID = "mutation:reserved-apply-once"


def _plan(summary: str = "API uses structured logging") -> MutationBatch:
    return MutationBatch(
        event_ref=EventRef(
            event_id="event:apply-once",
            source_system="agent",
            pot_id=POT,
        ),
        summary=summary,
        entity_upserts=[
            EntityUpsert(entity_key="pref:logging", labels=("Preference",))
        ],
        edge_upserts=[
            EdgeUpsert(
                edge_type="RELATES_TO",
                from_entity_key="pref:logging",
                to_entity_key="service:api",
                properties={"fact": summary},
            )
        ],
    )


def _context() -> ProvenanceContext:
    return ProvenanceContext(mutation_id=MUTATION_ID)


def _embedded_apply_worker(
    home: str,
    ready: multiprocessing.synchronize.Event,
    results: multiprocessing.queues.Queue,
) -> None:
    ready.wait(timeout=10)
    try:
        result = EmbeddedGraphBackend(home=Path(home)).mutation.apply(
            _plan(),
            expected_pot_id=POT,
            provenance_context=_context(),
        )
    except Exception as exc:
        results.put(("error", type(exc).__name__, str(exc)))
    else:
        results.put(("ok", result.mutation_id))


def test_in_memory_completed_retry_returns_cached_receipt_without_notifying() -> None:
    changes = 0

    def on_change() -> None:
        nonlocal changes
        changes += 1

    backend = InMemoryGraphBackend(on_change=on_change)
    plan = _plan()

    first = backend.mutation.apply(
        plan,
        expected_pot_id=POT,
        provenance_context=_context(),
    )
    retried = backend.mutation.apply(
        plan,
        expected_pot_id=POT,
        provenance_context=_context(),
    )

    assert retried == first
    assert retried.mutation_id == MUTATION_ID
    assert changes == 1
    assert len(backend.claim_query.find_claims(ClaimQueryFilter(pot_id=POT))) == 1


def test_in_memory_overlapping_retry_waits_for_the_owner_and_applies_once() -> None:
    entered_callback = threading.Event()
    release_callback = threading.Event()
    changes = 0

    def on_change() -> None:
        nonlocal changes
        changes += 1
        entered_callback.set()
        assert release_callback.wait(timeout=10)

    backend = InMemoryGraphBackend(on_change=on_change)
    plan = _plan()

    with ThreadPoolExecutor(max_workers=2) as pool:
        owner = pool.submit(
            backend.mutation.apply,
            plan,
            expected_pot_id=POT,
            provenance_context=_context(),
        )
        assert entered_callback.wait(timeout=10)
        retry = pool.submit(
            backend.mutation.apply,
            plan,
            expected_pot_id=POT,
            provenance_context=_context(),
        )
        assert not retry.done()
        release_callback.set()
        owner_result = owner.result(timeout=10)
        retry_result = retry.result(timeout=10)

    assert owner_result == retry_result
    assert changes == 1
    assert len(backend.claim_query.find_claims(ClaimQueryFilter(pot_id=POT))) == 1


def test_same_mutation_id_with_different_batch_is_rejected() -> None:
    backend = InMemoryGraphBackend()
    backend.mutation.apply(
        _plan(),
        expected_pot_id=POT,
        provenance_context=_context(),
    )

    with pytest.raises(MutationExecutionReuseError, match="different mutation batch"):
        backend.mutation.apply(
            _plan("API uses JSON logging"),
            expected_pot_id=POT,
            provenance_context=_context(),
        )


def test_embedded_receipt_survives_new_backend_without_rewriting(tmp_path) -> None:
    plan = _plan()
    first = EmbeddedGraphBackend(home=tmp_path).mutation.apply(
        plan,
        expected_pot_id=POT,
        provenance_context=_context(),
    )
    graph_path = tmp_path / "graph.json"
    before_bytes = graph_path.read_bytes()
    before_mtime = graph_path.stat().st_mtime_ns

    fresh = EmbeddedGraphBackend(home=tmp_path)
    lookup = fresh.mutation.lookup_execution(
        plan,
        expected_pot_id=POT,
        mutation_id=MUTATION_ID,
    )
    retried = fresh.mutation.apply(
        plan,
        expected_pot_id=POT,
        provenance_context=_context(),
    )

    assert lookup.state == MutationExecutionState.completed.value
    assert lookup.result == first
    assert retried == first
    assert graph_path.read_bytes() == before_bytes
    assert graph_path.stat().st_mtime_ns == before_mtime
    assert len(fresh.claim_query.find_claims(ClaimQueryFilter(pot_id=POT))) == 1


def test_embedded_rejects_reused_id_with_different_batch_across_instances(
    tmp_path,
) -> None:
    EmbeddedGraphBackend(home=tmp_path).mutation.apply(
        _plan(),
        expected_pot_id=POT,
        provenance_context=_context(),
    )

    with pytest.raises(MutationExecutionReuseError, match="different mutation batch"):
        EmbeddedGraphBackend(home=tmp_path).mutation.apply(
            _plan("API uses JSON logging"),
            expected_pot_id=POT,
            provenance_context=_context(),
        )


def test_embedded_stale_lifecycle_writes_preserve_newer_graph_and_receipt(
    tmp_path,
) -> None:
    stale = EmbeddedGraphBackend(home=tmp_path)
    writer = EmbeddedGraphBackend(home=tmp_path)
    applied = writer.mutation.apply(
        _plan(),
        expected_pot_id=POT,
        provenance_context=_context(),
    )

    assert (
        stale.mutation.invalidate(
            pot_id=POT,
            claim_keys=("claim:does-not-exist",),
        )
        == 0
    )
    assert stale.mutation.reset_pot("different-pot") == {"removed_claims": 0}
    assert stale.provision(SetupPlan()).ok

    fresh = EmbeddedGraphBackend(home=tmp_path)
    rows = fresh.claim_query.find_claims(ClaimQueryFilter(pot_id=POT))
    lookup = fresh.mutation.lookup_execution(
        _plan(),
        expected_pot_id=POT,
        mutation_id=MUTATION_ID,
    )
    assert len(rows) == 1
    assert lookup.state == MutationExecutionState.completed.value
    assert lookup.result == applied


def test_embedded_concurrent_processes_share_one_durable_receipt(tmp_path) -> None:
    process_context = multiprocessing.get_context("spawn")
    ready = process_context.Event()
    results = process_context.Queue()
    processes = [
        process_context.Process(
            target=_embedded_apply_worker,
            args=(str(tmp_path), ready, results),
        )
        for _ in range(2)
    ]
    for process in processes:
        process.start()
    ready.set()
    outcomes = [results.get(timeout=20) for _ in processes]
    for process in processes:
        process.join(timeout=20)

    assert all(process.exitcode == 0 for process in processes)
    assert outcomes == [("ok", MUTATION_ID), ("ok", MUTATION_ID)]
    persisted = json.loads((tmp_path / "graph.json").read_text(encoding="utf-8"))
    assert len(persisted["mutation_receipts"]) == 1
    backend = EmbeddedGraphBackend(home=tmp_path)
    assert len(backend.claim_query.find_claims(ClaimQueryFilter(pot_id=POT))) == 1
