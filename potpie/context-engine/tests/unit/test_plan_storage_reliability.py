from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from potpie_context_engine.adapters.outbound.graph import _local_json_atomic as atomic
from potpie_context_engine.adapters.outbound.graph.backends.in_memory_backend import (
    InMemoryGraphBackend,
)
from potpie_context_engine.adapters.outbound.graph.plan_stores.local_json import (
    LocalJsonGraphPlanStore,
)
from potpie_context_engine.core.mutation_execution import MutationPersistenceError
from potpie_context_engine.core.mutation_plan_store import InMemoryGraphPlanStore
from potpie_context_engine.core.workbench_service import GraphWorkbenchService


def _proposal(store):
    workbench = GraphWorkbenchService(backend=InMemoryGraphBackend(), plan_store=store)
    proposal = workbench.propose(
        {
            "idempotency_key": "reserved",
            "operations": [
                {
                    "op": "upsert_entity",
                    "subject": {"key": "service:a", "type": "Service"},
                }
            ],
        },
        pot_id="p",
    )
    return store.get(pot_id="p", plan_id=proposal.plan_id)


@pytest.mark.parametrize(
    "payload",
    [
        "{",
        "[]",
        "null",
        '{"plans":null}',
        '{"plans":{"p":[]}}',
        '{"plans":{"p":{"bad":null}}}',
        '{"plans":{},"plans":{}}',
        '{"schema_version":99,"plans":{}}',
    ],
)
def test_corrupt_plan_store_fails_closed_without_overwriting(tmp_path, payload):
    store = LocalJsonGraphPlanStore(tmp_path)
    record = _proposal(store)
    store._path.write_text(payload)
    original = store._path.read_bytes()
    operations = [
        lambda: store.get(pot_id="p", plan_id=record.plan_id),
        lambda: store.list(pot_id="p"),
        lambda: store.save(record),
        lambda: store.reserve_idempotency(
            record=record,
            idempotency_key="reserved",
            request_fingerprint=record.prepared.request_fingerprint,
        ),
    ]
    for operation in operations:
        with pytest.raises(MutationPersistenceError):
            operation()
        assert store._path.read_bytes() == original


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("created_at", "bad"),
        ("expires_at", None),
        ("plan_id", "other"),
        ("pot_id", "other"),
        ("status", "unknown"),
        ("expected_subgraph_versions", {"_global": True}),
        ("expected_subgraph_versions", {"_global": -1}),
        ("current_subgraph_versions", []),
        ("original_payload", []),
        ("mutation_id", 7),
    ],
)
def test_invalid_record_is_never_silently_dropped(tmp_path, field, value):
    store = LocalJsonGraphPlanStore(tmp_path)
    record = _proposal(store)
    data = json.loads(store._path.read_text())
    data["plans"]["p"][record.plan_id][field] = value
    store._path.write_text(json.dumps(data))
    before = store._path.read_bytes()
    with pytest.raises(MutationPersistenceError):
        store.get(pot_id="p", plan_id=record.plan_id)
    assert store._path.read_bytes() == before


def test_unreadable_store_is_not_absence(tmp_path, monkeypatch):
    store = LocalJsonGraphPlanStore(tmp_path)
    record = _proposal(store)
    original = Path.read_text

    def denied(path, *args, **kwargs):
        if path == store._path:
            raise PermissionError("denied")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", denied)
    with pytest.raises(MutationPersistenceError, match="unreadable"):
        store.save(record)


def test_memory_store_detaches_inputs_outputs_and_cas_state():
    store = InMemoryGraphPlanStore()
    record = _proposal(store)
    original = deepcopy(record)
    record.original_payload["idempotency_key"] = "changed"
    record.lowered_batch.entity_upserts.clear()
    assert store.get(pot_id="p", plan_id=record.plan_id) == original
    listed = store.list(pot_id="p")[0]
    listed.original_payload["operations"].clear()
    assert store.get(pot_id="p", plan_id=record.plan_id) == original
    assert not store.compare_and_set(expected=record, replacement=original)
    assert store.compare_and_set(expected=original, replacement=original)
    original.original_payload["operations"].clear()
    assert store.get(pot_id="p", plan_id=record.plan_id).original_payload["operations"]


@pytest.mark.parametrize("failure", ["fsync", "replace"])
def test_failed_durable_replace_preserves_old_bytes(tmp_path, monkeypatch, failure):
    path = tmp_path / "state.json"
    atomic.durable_json_replace(path, {"generation": 1})
    before = path.read_bytes()

    def fail(*args, **kwargs):
        raise OSError("injected publication failure")

    monkeypatch.setattr(atomic.os, failure, fail)
    with pytest.raises(OSError):
        atomic.durable_json_replace(path, {"generation": 2})
    assert path.read_bytes() == before
    assert list(tmp_path.glob("*.tmp")) == []


def test_process_lock_unavailable_refuses_storage(tmp_path, monkeypatch):
    monkeypatch.setattr(atomic, "_fcntl", None)
    monkeypatch.setattr(atomic, "_msvcrt", None)
    with pytest.raises(RuntimeError, match="process locking"):
        with atomic.locked_json_store(tmp_path / "state.json"):
            pytest.fail("must not enter an unprotected critical section")


def test_abrupt_process_exit_after_reservation_preserves_request_identity(tmp_path):
    script = """
import os, sys
from pathlib import Path
from potpie_context_engine.adapters.outbound.graph.backends.in_memory_backend import InMemoryGraphBackend
from potpie_context_engine.adapters.outbound.graph.plan_stores.local_json import LocalJsonGraphPlanStore
from potpie_context_engine.core.workbench_service import GraphWorkbenchService
store=LocalJsonGraphPlanStore(Path(sys.argv[1]))
wb=GraphWorkbenchService(backend=InMemoryGraphBackend(),plan_store=store)
wb.propose({'idempotency_key':'reserved','operations':[{'op':'upsert_entity','subject':{'key':'service:a','type':'Service'}}]},pot_id='p')
os._exit(23)
"""
    child = subprocess.run([sys.executable, "-c", script, str(tmp_path)], timeout=20)
    assert child.returncode == 23
    store = LocalJsonGraphPlanStore(tmp_path)
    prior = store.list(pot_id="p")[0]
    retry = _proposal(store)
    assert retry == prior
