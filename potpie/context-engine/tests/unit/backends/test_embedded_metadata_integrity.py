"""Embedded state decoding and acknowledgement failures preserve durable history."""

import json
import os
import stat

import pytest

from potpie_context_engine.adapters.outbound.graph import _local_json_atomic
from potpie_context_engine.adapters.outbound.graph.backends.embedded_backend import (
    EmbeddedGraphBackend,
)
from potpie_context_engine.core.graph_mutations import EntityUpsert, ProvenanceContext
from potpie_context_engine.core.mutation_execution import MutationPersistenceError
from potpie_context_engine.core.reconciliation import MutationBatch

pytestmark = pytest.mark.unit
POT = "metadata-pot"


def _seed(tmp_path):
    backend = EmbeddedGraphBackend(home=tmp_path)
    plan = MutationBatch(entity_upserts=[EntityUpsert("service:a", ("Service",))])
    result = backend.mutation.apply(
        plan,
        expected_pot_id=POT,
        provenance_context=ProvenanceContext(mutation_id="original"),
    )
    path = tmp_path / "graph.json"
    return backend, plan, result, path, json.loads(path.read_text())


def _lookup(backend, plan):
    return backend.mutation.lookup_execution(
        plan, expected_pot_id=POT, mutation_id="original"
    )


@pytest.mark.parametrize(
    "revisions",
    [
        None,
        [],
        True,
        False,
        0,
        "",
        {POT: True},
        {POT: False},
        {POT: -1},
        {POT: 1.0},
        {POT: "1"},
        {"": 1},
    ],
)
def test_invalid_revision_metadata_fails_closed_without_rewriting(tmp_path, revisions):
    backend, plan, _, path, state = _seed(tmp_path)
    state["graph_revisions"] = revisions
    path.write_text(json.dumps(state))
    before = path.read_bytes()
    with pytest.raises(MutationPersistenceError):
        backend.mutation.apply(
            plan,
            expected_pot_id=POT,
            provenance_context=ProvenanceContext(mutation_id="new"),
        )
    assert path.read_bytes() == before
    with pytest.raises(MutationPersistenceError):
        EmbeddedGraphBackend(home=tmp_path)


@pytest.mark.parametrize(
    "field",
    [
        "rows",
        "labels",
        "entity_properties",
        "graph_revisions",
        "mutation_receipts",
        "graph_journal",
    ],
)
def test_versioned_state_requires_complete_collection_metadata(tmp_path, field):
    _, _, _, path, state = _seed(tmp_path)
    assert state["schema_version"] == 1
    state.pop(field)
    path.write_text(json.dumps(state))
    with pytest.raises(MutationPersistenceError):
        EmbeddedGraphBackend(home=tmp_path)


@pytest.mark.parametrize("version", [None, True, False, "1", 0, -1, 2])
def test_unknown_or_malformed_embedded_schema_fails_closed(tmp_path, version):
    _, _, _, path, state = _seed(tmp_path)
    state["schema_version"] = version
    path.write_text(json.dumps(state))
    with pytest.raises(MutationPersistenceError, match="schema"):
        EmbeddedGraphBackend(home=tmp_path)


@pytest.mark.parametrize("revisions", [None, {}, {POT: 0}])
def test_modern_receipt_rejects_missing_or_older_unversioned_revision(
    tmp_path, revisions
):
    _, _, _, path, state = _seed(tmp_path)
    state.pop("schema_version")
    if revisions is None:
        state.pop("graph_revisions")
    else:
        state["graph_revisions"] = revisions
    path.write_text(json.dumps(state))
    with pytest.raises(MutationPersistenceError, match="revision"):
        EmbeddedGraphBackend(home=tmp_path)


@pytest.mark.parametrize(
    "journal",
    [None, [], True, {"type": "list", "value": []}, {"type": "map", "value": []}],
)
def test_corrupt_journal_metadata_fails_closed(tmp_path, journal):
    _, _, _, path, state = _seed(tmp_path)
    state["graph_journal"] = journal
    path.write_text(json.dumps(state))
    with pytest.raises(MutationPersistenceError):
        EmbeddedGraphBackend(home=tmp_path)


@pytest.mark.skipif(
    os.name == "nt", reason="POSIX directory synchronization acknowledgement"
)
def test_error_after_replace_leaves_complete_execution_authoritative(
    tmp_path, monkeypatch
):
    backend, plan, original, _, _ = _seed(tmp_path)
    original_fsync = _local_json_atomic.os.fsync

    def fail_directory_flush(descriptor):
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise OSError("directory synchronization failed after replacement")
        return original_fsync(descriptor)

    next_plan = MutationBatch(entity_upserts=[EntityUpsert("service:b", ("Service",))])
    context = ProvenanceContext(mutation_id="after-replace")
    with monkeypatch.context() as patch:
        patch.setattr(_local_json_atomic.os, "fsync", fail_directory_flush)
        with pytest.raises(OSError, match="after replacement"):
            backend.mutation.apply(
                next_plan, expected_pot_id=POT, provenance_context=context
            )
    restarted = EmbeddedGraphBackend(home=tmp_path)
    receipt = restarted.mutation.lookup_execution(
        next_plan, expected_pot_id=POT, mutation_id="after-replace"
    )
    assert receipt.state == "completed"
    assert (
        backend.mutation.lookup_execution(
            next_plan, expected_pot_id=POT, mutation_id="after-replace"
        )
        == receipt
    )
    assert backend.mutation.current_version(POT) == 2
    assert receipt.result.execution.version_before == 1
    assert receipt.result.execution.version_after == 2
    assert (
        restarted.mutation.apply(
            next_plan, expected_pot_id=POT, provenance_context=context
        )
        == receipt.result
    )
    assert restarted.mutation.current_version(POT) == 2
    assert _lookup(restarted, plan).result == original
    assert restarted.claim_query.entity_properties(pot_id=POT, entity_key="service:b")


@pytest.mark.parametrize("raw", ["{", "[]", "null"])
def test_corrupt_json_or_root_never_becomes_absent_execution(tmp_path, raw):
    backend, plan, _, path, _ = _seed(tmp_path)
    path.write_text(raw)
    with pytest.raises(MutationPersistenceError):
        _lookup(backend, plan)
    assert path.read_text() == raw


def test_unreadable_store_does_not_become_absent_execution(tmp_path, monkeypatch):
    import builtins

    backend, plan, original, path, _ = _seed(tmp_path)
    original_open = builtins.open

    def unreadable(file, *args, **kwargs):
        if file == path:
            raise PermissionError("injected graph read failure")
        return original_open(file, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(builtins, "open", unreadable)
        with pytest.raises(PermissionError, match="graph read failure"):
            _lookup(backend, plan)
    assert _lookup(backend, plan).result == original


@pytest.mark.parametrize("operation", ["version", "lookup", "apply"])
def test_disappearing_previously_observed_store_fails_closed(tmp_path, operation):
    backend, plan, _, path, _ = _seed(tmp_path)
    path.unlink()
    with pytest.raises(MutationPersistenceError, match="previously observed"):
        if operation == "version":
            backend.mutation.current_version(POT)
        elif operation == "lookup":
            _lookup(backend, plan)
        else:
            backend.mutation.apply(plan, expected_pot_id=POT)
    assert not path.exists()


def test_initially_missing_store_is_fresh_until_first_publication(tmp_path):
    backend = EmbeddedGraphBackend(home=tmp_path)
    assert backend.mutation.current_version(POT) == 0
    assert backend.mutation.current_version(POT) == 0
    backend.provision()
    path = tmp_path / "graph.json"
    assert path.exists()
    path.unlink()
    with pytest.raises(MutationPersistenceError, match="previously observed"):
        backend.mutation.current_version(POT)


@pytest.mark.skipif(
    os.name == "nt", reason="POSIX directory synchronization acknowledgement"
)
def test_postreplace_failure_records_publication_before_another_read(
    tmp_path, monkeypatch
):
    backend = EmbeddedGraphBackend(home=tmp_path)
    original_fsync = _local_json_atomic.os.fsync

    def fail_directory_flush(descriptor):
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise OSError("postreplace acknowledgement failed")
        return original_fsync(descriptor)

    plan = MutationBatch(entity_upserts=[EntityUpsert("service:a", ("Service",))])
    with monkeypatch.context() as patch:
        patch.setattr(_local_json_atomic.os, "fsync", fail_directory_flush)
        with pytest.raises(OSError, match="acknowledgement failed"):
            backend.mutation.apply(plan, expected_pot_id=POT)
    path = tmp_path / "graph.json"
    assert path.exists()
    path.unlink()
    with pytest.raises(MutationPersistenceError, match="previously observed"):
        backend.mutation.current_version(POT)


@pytest.mark.parametrize(
    "location",
    [
        "root_receipts",
        "root_schema",
        "revisions",
        "result",
        "execution",
        "journal",
        "entity_properties",
    ],
)
@pytest.mark.parametrize("operation", ["lookup", "replay", "new_write", "restart"])
def test_duplicate_json_keys_never_erase_execution_authority(
    tmp_path, location, operation
):
    backend, plan, original, path, state = _seed(tmp_path)
    receipt = state["mutation_receipts"][0]
    targets = {
        "root_receipts": (state, "mutation_receipts", []),
        "root_schema": (state, "schema_version", state["schema_version"]),
        "revisions": (state["graph_revisions"], POT, state["graph_revisions"][POT]),
        "result": (receipt["result"], "ok", True),
        "execution": (receipt["result"]["execution"], "version_after", 1),
        "journal": (state["graph_journal"], "type", state["graph_journal"]["type"]),
        "entity_properties": (state["entity_properties"][0][2], "name", "changed name"),
    }
    target, key, value = targets[location]
    assert key in target
    encoded = json.dumps(target)
    duplicated = encoded[:-1] + ", " + json.dumps(key) + ": " + json.dumps(value) + "}"
    raw = json.dumps(state).replace(encoded, duplicated, 1)
    assert raw != json.dumps(state)
    path.write_text(raw)
    before = path.read_bytes()

    with pytest.raises(MutationPersistenceError, match="malformed JSON"):
        if operation == "lookup":
            _lookup(backend, plan)
        elif operation == "replay":
            backend.mutation.apply(
                plan,
                expected_pot_id=POT,
                provenance_context=ProvenanceContext(mutation_id="original"),
            )
        elif operation == "new_write":
            backend.mutation.apply(
                plan,
                expected_pot_id=POT,
                provenance_context=ProvenanceContext(mutation_id="new"),
            )
        else:
            EmbeddedGraphBackend(home=tmp_path)

    assert path.read_bytes() == before
    cached = backend._inner.execution_registry.lookup(
        plan, expected_pot_id=POT, mutation_id="original"
    )
    assert cached.state == "completed"
    assert cached.result == original
    assert backend._inner.revisions[POT] == 1
