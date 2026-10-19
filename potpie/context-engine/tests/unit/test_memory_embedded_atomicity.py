"""Atomic graph/revision/receipt publication for process-local and JSON stores."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
import json
import threading

import pytest

from potpie_context_engine.adapters.outbound.graph._mutation_execution import (
    MutationExecutionReuseError,
)
from potpie_context_engine.adapters.outbound.graph.backends.embedded_backend import (
    EmbeddedGraphBackend,
)
from potpie_context_engine.adapters.outbound.graph.backends.in_memory_backend import (
    InMemoryGraphBackend,
    _Mutation,
)
from potpie_context_engine.core.errors import GraphMutationVersionConflict
from potpie_context_engine.core.graph_mutations import (
    EdgeUpsert,
    EntityUpsert,
    ProvenanceContext,
)
from potpie_context_engine.core.mutation_execution import (
    PreparedMutation,
    batch_fingerprint,
)
from potpie_context_engine.core.ports.claim_query import ClaimQueryFilter
from potpie_context_engine.core.ports.graph.mutation import (
    AsyncAtomicGraphMutationPort,
    AsyncRecoverableGraphMutationPort,
    RecoverableGraphMutationPort,
)
from potpie_context_engine.core.reconciliation import MutationBatch

pytestmark = pytest.mark.unit
POT = "atomic-pot"


@pytest.fixture(params=["memory", "embedded"])
def backend(request, tmp_path):
    return (
        InMemoryGraphBackend()
        if request.param == "memory"
        else EmbeddedGraphBackend(home=tmp_path)
    )


def _plan() -> MutationBatch:
    return MutationBatch(
        summary="Two related services",
        entity_upserts=[
            EntityUpsert("service:a", ("Service",)),
            EntityUpsert("service:b", ("Service",)),
        ],
        edge_upserts=[
            EdgeUpsert("RELATES_TO", "service:a", "service:b", {"fact": "first"}),
            EdgeUpsert("RELATES_TO", "service:b", "service:a", {"fact": "second"}),
        ],
    )


def _context(plan, *, mutation_id="write", request="request", version=0):
    return ProvenanceContext(
        mutation_id=mutation_id,
        prepared=PreparedMutation(
            POT,
            mutation_id,
            request,
            batch_fingerprint(plan),
            version,
            datetime(2026, 10, 19, tzinfo=timezone.utc),
        ),
    )


def _lookup(backend, plan, mutation_id="write"):
    return backend.mutation.lookup_execution(
        plan, expected_pot_id=POT, mutation_id=mutation_id
    )


@pytest.mark.parametrize("journal", [False, True])
def test_internal_failure_leaves_no_data_revision_or_receipt(
    backend, monkeypatch, journal
):
    if journal:
        backend.journal.activate(pot_id=POT)
    plan = _plan()
    original = _Mutation._upsert_claim_row

    def fail_after_first_write(self, row):
        original(self, row)
        raise RuntimeError("injected internal apply failure")

    with monkeypatch.context() as patch:
        patch.setattr(_Mutation, "_upsert_claim_row", fail_after_first_write)
        with pytest.raises(RuntimeError, match="internal apply failure"):
            backend.mutation.apply(
                plan, expected_pot_id=POT, provenance_context=_context(plan)
            )
    assert backend.claim_query.find_claims(ClaimQueryFilter(pot_id=POT)) == []
    assert (
        backend.claim_query.entity_labels(pot_id=POT, entity_keys=["service:a"]) == {}
    )
    assert backend.mutation.current_version(POT) == 0
    assert _lookup(backend, plan).state == "absent"
    # A proven rolled-back execution can safely use the same frozen identity.
    assert backend.mutation.apply(
        plan, expected_pot_id=POT, provenance_context=_context(plan)
    ).ok


@pytest.mark.parametrize("journal", [False, True])
def test_persistence_failure_does_not_publish_graph_revision_or_receipt(
    tmp_path, monkeypatch, journal
):
    backend = EmbeddedGraphBackend(home=tmp_path)
    if journal:
        backend.journal.activate(pot_id=POT)
    before = (tmp_path / "graph.json").read_bytes() if journal else None
    plan = _plan()

    def fail_persist(self, *args, **kwargs):
        raise OSError("injected disk persistence failure")

    with monkeypatch.context() as patch:
        patch.setattr(EmbeddedGraphBackend, "_write_state", fail_persist)
        with pytest.raises(OSError, match="disk persistence failure"):
            backend.mutation.apply(
                plan, expected_pot_id=POT, provenance_context=_context(plan)
            )
    assert backend._inner.store.rows == []
    assert backend._inner.revisions.get(POT, 0) == 0
    assert backend._inner.execution_registry.completed() == ()
    if before is not None:
        assert (tmp_path / "graph.json").read_bytes() == before
    else:
        assert not (tmp_path / "graph.json").exists()
    restarted = EmbeddedGraphBackend(home=tmp_path)
    assert restarted.claim_query.find_claims(ClaimQueryFilter(pot_id=POT)) == []
    assert restarted.mutation.current_version(POT) == 0
    assert _lookup(restarted, plan).state == "absent"


def test_partial_json_write_leaves_original_state_and_receipts(tmp_path, monkeypatch):
    backend = EmbeddedGraphBackend(home=tmp_path)
    plan = _plan()
    original = backend.mutation.apply(
        plan, expected_pot_id=POT, provenance_context=_context(plan)
    )
    before = (tmp_path / "graph.json").read_bytes()

    def partial_write(payload, stream, **kwargs):
        stream.write('{"incomplete":')
        raise OSError("injected partial JSON write")

    with monkeypatch.context() as patch:
        patch.setattr(json, "dump", partial_write)
        with pytest.raises(OSError, match="partial JSON write"):
            backend.mutation.apply(
                plan,
                expected_pot_id=POT,
                provenance_context=_context(plan, mutation_id="second", version=1),
            )
    assert (tmp_path / "graph.json").read_bytes() == before
    restarted = EmbeddedGraphBackend(home=tmp_path)
    assert restarted.mutation.current_version(POT) == 1
    assert _lookup(restarted, plan).result == original
    assert _lookup(restarted, plan, mutation_id="second").state == "absent"


@pytest.mark.parametrize("journal", [False, True])
def test_reader_waits_for_complete_batch(backend, monkeypatch, journal):
    if journal:
        backend.journal.activate(pot_id=POT)
    first_write = threading.Event()
    release = threading.Event()
    reader_started = threading.Event()
    original = _Mutation._upsert_claim_row

    def pause_after_write(self, row):
        original(self, row)
        if not first_write.is_set():
            first_write.set()
            assert release.wait(timeout=5)

    monkeypatch.setattr(_Mutation, "_upsert_claim_row", pause_after_write)

    def read():
        reader_started.set()
        return backend.claim_query.find_claims(
            ClaimQueryFilter(pot_id=POT, subject_label="Service")
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        writer = pool.submit(backend.mutation.apply, _plan(), expected_pot_id=POT)
        try:
            assert first_write.wait(timeout=5)
            reader = pool.submit(read)
            assert reader_started.wait(timeout=5)
            assert not reader.done()
        finally:
            release.set()
        assert writer.result(timeout=5).ok
        assert len(reader.result(timeout=5)) == 2


@pytest.mark.parametrize("journal", [False, True])
def test_observer_failure_retains_committed_revision_and_receipt(journal):
    fail = False

    def observer():
        if fail:
            raise RuntimeError("injected observer failure")

    backend = InMemoryGraphBackend(on_change=observer)
    if journal:
        backend.journal.activate(pot_id=POT)
    fail = True
    plan = _plan()
    context = _context(plan)
    with pytest.raises(RuntimeError, match="observer failure"):
        backend.mutation.apply(plan, expected_pot_id=POT, provenance_context=context)
    assert len(backend.claim_query.find_claims(ClaimQueryFilter(pot_id=POT))) == 2
    assert backend.mutation.current_version(POT) == 1
    lookup = _lookup(backend, plan)
    assert lookup.state == "completed"
    assert lookup.result.execution.version_after == 1
    assert (
        backend.mutation.compare_and_apply(
            plan, expected_pot_id=POT, expected_version=0, provenance_context=context
        )
        == lookup.result
    )


def test_observer_does_not_hold_graph_or_receipt_readers():
    entered = threading.Event()
    release = threading.Event()

    def observer():
        entered.set()
        assert release.wait(timeout=5)

    backend = InMemoryGraphBackend(on_change=observer)
    plan = _plan()
    with ThreadPoolExecutor(max_workers=2) as pool:
        writer = pool.submit(
            backend.mutation.apply,
            plan,
            expected_pot_id=POT,
            provenance_context=_context(plan),
        )
        try:
            assert entered.wait(timeout=5)
            lookup = pool.submit(_lookup, backend, plan).result(timeout=2)
            assert lookup.state == "completed"
            assert (
                pool.submit(backend.mutation.current_version, POT).result(timeout=2)
                == 1
            )
            assert (
                len(backend.claim_query.find_claims(ClaimQueryFilter(pot_id=POT))) == 2
            )
        finally:
            release.set()
        assert writer.result(timeout=5) == lookup.result


@pytest.mark.parametrize("journal", [False, True])
def test_empty_and_nonempty_receipts_have_exact_revisions_and_replay_metadata(
    backend, journal
):
    if journal:
        backend.journal.activate(pot_id=POT)
    empty = MutationBatch()
    empty_result = backend.mutation.compare_and_apply(
        empty,
        expected_pot_id=POT,
        expected_version=0,
        provenance_context=_context(empty, mutation_id="empty"),
    )
    assert (
        empty_result.execution.version_before
        == empty_result.execution.version_after
        == 0
    )
    assert backend.mutation.current_version(POT) == 0
    plan = _plan()
    context = _context(plan)
    result = backend.mutation.compare_and_apply(
        plan, expected_pot_id=POT, expected_version=0, provenance_context=context
    )
    assert result.execution.version_before == 0
    assert result.execution.version_after == 1
    assert result.execution.request_fingerprint == "request"
    assert len(result.execution.affected_claim_record_ids) == 2
    if isinstance(backend, EmbeddedGraphBackend):
        backend = EmbeddedGraphBackend(home=backend.home)
    assert (
        backend.mutation.compare_and_apply(
            plan, expected_pot_id=POT, expected_version=0, provenance_context=context
        )
        == result
    )
    assert backend.mutation.current_version(POT) == 1
    assert _lookup(backend, plan).result == result


def test_changed_request_or_batch_cannot_reuse_completed_identity(backend):
    plan = _plan()
    context = _context(plan)
    backend.mutation.apply(plan, expected_pot_id=POT, provenance_context=context)
    with pytest.raises(MutationExecutionReuseError, match="different request"):
        backend.mutation.compare_and_apply(
            plan,
            expected_pot_id=POT,
            expected_version=0,
            provenance_context=replace(
                context,
                prepared=replace(context.prepared, request_fingerprint="changed"),
            ),
        )
    changed = replace(plan, summary="changed batch")
    with pytest.raises(MutationExecutionReuseError, match="different mutation batch"):
        backend.mutation.apply(
            changed, expected_pot_id=POT, provenance_context=_context(changed)
        )
    assert backend.mutation.current_version(POT) == 1


@pytest.mark.parametrize("use_compare", [False, True])
def test_frozen_expected_revision_is_enforced_even_with_current_port_version(
    backend, use_compare
):
    plan = _plan()
    backend.mutation.apply(plan, expected_pot_id=POT)
    apply = (
        backend.mutation.compare_and_apply if use_compare else backend.mutation.apply
    )
    version_args = {"expected_version": 1} if use_compare else {}
    with pytest.raises(GraphMutationVersionConflict):
        apply(
            plan, expected_pot_id=POT, provenance_context=_context(plan), **version_args
        )
    assert backend.mutation.current_version(POT) == 1
    assert _lookup(backend, plan).state == "absent"


@pytest.mark.parametrize(
    "surface",
    ["claim", "properties", "semantic", "inspection", "analytics", "snapshot"],
)
def test_existing_embedded_reader_sees_another_instances_commit(tmp_path, surface):
    stale = EmbeddedGraphBackend(home=tmp_path)
    readers = {
        "claim": lambda: len(
            stale.claim_query.find_claims(ClaimQueryFilter(pot_id=POT))
        ),
        "properties": lambda: len(
            stale.claim_query.entity_labels(
                pot_id=POT, entity_keys=["service:a", "service:b"]
            )
        ),
        "semantic": lambda: len(stale.semantic.search(pot_id=POT, query="first")),
        "inspection": lambda: len(
            stale.inspection.neighborhood(pot_id=POT, entity_key="service:a").nodes
        ),
        "analytics": lambda: stale.analytics.counts(POT)["claims"],
        "snapshot": lambda: len(stale.snapshot.export_data(pot_id=POT)["claims"]),
    }
    reader = readers[surface]
    assert reader() == 0
    EmbeddedGraphBackend(home=tmp_path).mutation.apply(_plan(), expected_pot_id=POT)
    assert reader() == 2


def test_legacy_receipt_preserves_unknown_metadata_on_restart(tmp_path):
    plan = _plan()
    backend = EmbeddedGraphBackend(home=tmp_path)
    backend.mutation.apply(
        plan,
        expected_pot_id=POT,
        provenance_context=ProvenanceContext(mutation_id="legacy"),
    )
    path = tmp_path / "graph.json"
    state = json.loads(path.read_text())
    state["mutation_receipts"][0]["result"].pop("execution")
    path.write_text(json.dumps(state))
    restarted = EmbeddedGraphBackend(home=tmp_path)
    replay = restarted.mutation.apply(
        plan,
        expected_pot_id=POT,
        provenance_context=ProvenanceContext(mutation_id="legacy"),
    )
    assert replay.execution is None
    assert restarted.mutation.current_version(POT) == 1


def test_sync_async_recovery_contracts_share_atomic_results(backend):
    assert isinstance(backend.mutation, RecoverableGraphMutationPort)
    assert isinstance(backend.mutation, AsyncRecoverableGraphMutationPort)
    assert isinstance(backend.mutation, AsyncAtomicGraphMutationPort)
    assert backend.mutation.capabilities.atomic_writes
    assert backend.mutation.capabilities.durable_receipts == isinstance(
        backend, EmbeddedGraphBackend
    )
    plan = _plan()

    async def execute():
        result = await backend.mutation.compare_and_apply_async(
            plan,
            expected_pot_id=POT,
            expected_version=0,
            provenance_context=_context(plan),
        )
        assert await backend.mutation.current_version_async(POT) == 1
        assert (
            await backend.mutation.lookup_execution_async(
                plan, expected_pot_id=POT, mutation_id="write"
            )
        ).result == result
        return result

    result = asyncio.run(execute())
    assert (
        backend.mutation.compare_and_apply(
            plan,
            expected_pot_id=POT,
            expected_version=0,
            provenance_context=_context(plan),
        )
        == result
    )
