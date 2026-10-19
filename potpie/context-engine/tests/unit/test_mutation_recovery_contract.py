from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import datetime, timezone, timedelta
import threading

import pytest

from potpie_context_engine.adapters.outbound.graph.backends.in_memory_backend import (
    InMemoryGraphBackend,
)
from potpie_context_engine.core.graph_mutations import ProvenanceContext
from potpie_context_engine.core.mutation_plan_store import InMemoryGraphPlanStore
from potpie_context_engine.core.runtime import build_graph_runtime
from potpie_context_engine.core.workbench_service import GraphWorkbenchService


def _prepared(backend=None):
    backend = backend or InMemoryGraphBackend()
    store = InMemoryGraphPlanStore()
    wb = GraphWorkbenchService(backend=backend, plan_store=store)
    proposal = wb.propose(
        {
            "operations": [
                {
                    "op": "upsert_entity",
                    "subject": {"key": "service:a", "type": "Service"},
                }
            ]
        },
        pot_id="p",
    )
    record = store.get(pot_id="p", plan_id=proposal.plan_id)
    return backend, store, wb, record


def _execute(backend, record):
    return backend.mutation.compare_and_apply(
        record.lowered_batch,
        expected_pot_id="p",
        expected_version=record.prepared.expected_version,
        provenance_context=replace(
            record.provenance or ProvenanceContext(),
            mutation_id=record.mutation_id,
            prepared=record.prepared,
        ),
    )


@pytest.mark.parametrize("recover_stale", [True, False])
@pytest.mark.parametrize("status", ["committing", "error", "expired", "conflict"])
def test_receipt_precedes_workflow_status_lease_and_current_revision(
    status, recover_stale
):
    backend, store, wb, record = _prepared()
    receipt = _execute(backend, record)
    backend.mutation.reset_pot("p")
    store.save(
        replace(
            record,
            status=status,
            commit_attempt_id="old-worker",
            commit_attempt_started_at=datetime.now(timezone.utc),
        )
    )
    recovered = wb.commit(record.plan_id, pot_id="p", recover_stale=recover_stale)
    assert recovered.ok
    assert recovered.execution == receipt.execution
    assert recovered.applied_at == receipt.execution.applied_at
    assert recovered.current_subgraph_versions == {"_global": 0}
    assert recovered.new_subgraph_versions == {"_global": 1}
    assert backend.mutation.current_version("p") == 2
    assert store.get(pot_id="p", plan_id=record.plan_id).execution == receipt.execution


@pytest.mark.parametrize("stale", [False, True])
def test_false_suppresses_takeover_of_unexecuted_attempt(stale):
    backend, store, wb, record = _prepared()
    store.save(
        replace(
            record,
            status="committing",
            commit_attempt_id="owner",
            commit_attempt_started_at=datetime.now(timezone.utc)
            - timedelta(hours=1 if stale else 0),
        )
    )
    result = wb.commit(record.plan_id, pot_id="p", recover_stale=False)
    assert not result.ok and result.status == "committing"
    assert backend.mutation.current_version("p") == 0


def test_default_recovers_stale_unexecuted_attempt():
    backend, store, wb, record = _prepared()
    store.save(
        replace(
            record,
            status="committing",
            commit_attempt_id="dead",
            commit_attempt_started_at=datetime.now(timezone.utc) - timedelta(hours=1),
        )
    )
    assert wb.commit(record.plan_id, pot_id="p").ok
    assert backend.mutation.current_version("p") == 1


def test_unexecuted_legacy_plan_requires_new_preparation():
    backend, store, wb, record = _prepared()
    store.save(replace(record, prepared=None))
    result = wb.commit(record.plan_id, pot_id="p")
    assert not result.ok and "migration_required" in result.detail
    assert backend.mutation.current_version("p") == 0


def test_receipt_metadata_mismatch_is_ambiguous_not_success():
    backend, store, wb, record = _prepared()
    receipt = _execute(backend, record)
    inner = backend.mutation

    class WrongReceipt:
        def __getattr__(self, name):
            return getattr(inner, name)

        def lookup_execution(self, *args, **kwargs):
            lookup = inner.lookup_execution(*args, **kwargs)
            return replace(
                lookup,
                result=replace(
                    receipt, execution=replace(receipt.execution, pot_id="other")
                ),
            )

    backend._mutation = WrongReceipt()
    result = wb.commit(record.plan_id, pot_id="p")
    assert not result.ok and "ambiguous" in result.detail
    assert backend.mutation.current_version("p") == 1


def test_cancelled_public_commit_recovers_while_original_observer_is_pending():
    published, release = threading.Event(), threading.Event()

    def observer():
        published.set()
        assert release.wait(5)

    backend = InMemoryGraphBackend(on_change=observer)
    store = InMemoryGraphPlanStore()
    runtime = build_graph_runtime(backend, store)

    async def journey():
        proposal = await runtime.propose_async(
            {
                "operations": [
                    {
                        "op": "upsert_entity",
                        "subject": {"key": "service:a", "type": "Service"},
                    }
                ]
            },
            pot_id="p",
        )
        task = asyncio.create_task(runtime.commit_async(proposal.plan_id, pot_id="p"))
        try:
            assert await asyncio.wait_for(asyncio.to_thread(published.wait, 3), 4)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            receipt = await runtime.commit_async(
                proposal.plan_id, pot_id="p", recover_stale=False
            )
            assert receipt.ok and receipt.execution.version_after == 1
        finally:
            release.set()

    asyncio.run(journey())
    assert backend.mutation.current_version("p") == 1


def test_legacy_completed_receipt_keeps_unknown_metadata():
    backend, store, wb, record = _prepared()
    receipt = _execute(backend, record)
    store.save(replace(record, prepared=None))
    inner = backend.mutation

    class LegacyReceipt:
        def __getattr__(self, name):
            return getattr(inner, name)

        def lookup_execution(self, *args, **kwargs):
            return replace(
                inner.lookup_execution(*args, **kwargs),
                result=replace(receipt, execution=None),
            )

    backend._mutation = LegacyReceipt()
    recovered = wb.commit(record.plan_id, pot_id="p")
    assert recovered.ok and recovered.mutation_id == record.mutation_id
    assert recovered.execution is None and recovered.applied_at is None
    assert recovered.current_subgraph_versions == recovered.new_subgraph_versions == {}
    assert backend.mutation.current_version("p") == 1


def test_status_recovers_backend_receipt_without_applying_again():
    backend, store, wb, record = _prepared()
    original = _execute(backend, record)
    recovered = wb.commit_status(record.plan_id, pot_id="p")
    assert recovered.ok and recovered.execution == original.execution
    assert backend.mutation.current_version("p") == 1


def test_process_dies_after_graph_commit_before_plan_finalization(tmp_path):
    import subprocess
    import sys
    from potpie_context_engine.adapters.outbound.graph.backends.embedded_backend import (
        EmbeddedGraphBackend,
    )
    from potpie_context_engine.adapters.outbound.graph.plan_stores.local_json import (
        LocalJsonGraphPlanStore,
    )

    store = LocalJsonGraphPlanStore(tmp_path)
    wb = GraphWorkbenchService(
        backend=EmbeddedGraphBackend(home=tmp_path), plan_store=store
    )
    proposal = wb.propose(
        {
            "idempotency_key": "crashed",
            "operations": [
                {
                    "op": "upsert_entity",
                    "subject": {"key": "service:a", "type": "Service"},
                }
            ],
        },
        pot_id="p",
    )
    record = store.get(pot_id="p", plan_id=proposal.plan_id)
    store.save(
        replace(
            record,
            status="committing",
            commit_attempt_id="dead-worker",
            commit_attempt_started_at=datetime.now(timezone.utc),
        )
    )
    script = """
import os, sys
from pathlib import Path
from dataclasses import replace
from potpie_context_engine.adapters.outbound.graph.backends.embedded_backend import EmbeddedGraphBackend
from potpie_context_engine.adapters.outbound.graph.plan_stores.local_json import LocalJsonGraphPlanStore
from potpie_context_engine.core.graph_mutations import ProvenanceContext
home=Path(sys.argv[1]); store=LocalJsonGraphPlanStore(home)
r=store.get(pot_id='p',plan_id=sys.argv[2]); backend=EmbeddedGraphBackend(home=home)
backend.mutation.compare_and_apply(r.lowered_batch,expected_pot_id='p',expected_version=r.prepared.expected_version,
    provenance_context=replace(r.provenance or ProvenanceContext(),mutation_id=r.mutation_id,prepared=r.prepared))
os._exit(17)
"""
    child = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path), record.plan_id], timeout=20
    )
    assert child.returncode == 17
    backend = EmbeddedGraphBackend(home=tmp_path)
    prior = backend.mutation.lookup_execution(
        record.lowered_batch, expected_pot_id="p", mutation_id=record.mutation_id
    )
    recovered = GraphWorkbenchService(
        backend=backend, plan_store=LocalJsonGraphPlanStore(tmp_path)
    ).commit(record.plan_id, pot_id="p", recover_stale=False)
    assert recovered.ok and recovered.execution == prior.result.execution
    assert backend.mutation.current_version("p") == 1


@pytest.mark.parametrize("state", ["absent", "ambiguous", "unsupported"])
def test_status_never_substitutes_workflow_bookkeeping_for_authority(state):
    backend, store, wb, record = _prepared()
    assert wb.commit(record.plan_id, pot_id="p").ok
    inner = backend.mutation

    class MissingReceipt:
        def __getattr__(self, name):
            return getattr(inner, name)

        def lookup_execution(self, *args, **kwargs):
            return replace(
                inner.lookup_execution(*args, **kwargs), state=state, result=None
            )

    backend._mutation = MissingReceipt()
    assert not wb.commit_status(record.plan_id, pot_id="p").ok
    assert not wb.commit(record.plan_id, pot_id="p").ok
    assert backend.mutation.current_version("p") == 1
