"""Durable reviewed repair preparation and receipt-first resumability."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import subprocess
import sys

import pytest

from potpie_context_engine.adapters.outbound.graph.backends.embedded_backend import (
    EmbeddedGraphBackend,
)
from potpie_context_engine.adapters.outbound.graph.plan_stores.local_json import (
    LocalJsonGraphPlanStore,
)
from potpie_context_engine.core.graph_mutations import EntityUpsert
from potpie_context_engine.core.legacy_repair_policy import plan_legacy_repairs
from potpie_context_engine.core.legacy_repair_service import prepare_legacy_repair
from potpie_context_engine.core.mutation_execution import (
    MutationPersistenceError,
    batch_fingerprint,
)
from potpie_context_engine.core.mutation_plan_store import InMemoryGraphPlanStore
from potpie_context_engine.core.reconciliation import MutationBatch
from potpie_context_engine.core.runtime import build_graph_runtime
from tests.unit.test_memory_legacy_repair import _state, _backend, POT

pytestmark = pytest.mark.unit


async def _allow(pot_id, access):
    pass


def _workbench(backend, store):
    return build_graph_runtime(
        backend=backend,
        plan_store=store,
        commit_authorize=_allow,
        commit_actor=lambda: "admin",
    ).workbench


@pytest.fixture(params=["memory", "json"])
def repair_context(request, tmp_path):
    backend = _backend(request.param, tmp_path, _state(claim_id=False, entity_id=False))
    store = (
        LocalJsonGraphPlanStore(tmp_path)
        if request.param == "json"
        else InMemoryGraphPlanStore()
    )
    workbench = _workbench(backend, store)
    return backend, store, workbench


def _prepare(context, **kwargs):
    backend, store, _ = context
    return prepare_legacy_repair(backend, store, pot_id=POT, **kwargs)


def _apply(context, repair_id):
    return context[2].repair_legacy(pot_id=POT, mode="apply", repair_id=repair_id)


def test_review_is_frozen_and_apply_replay_and_second_scan_are_idempotent(
    repair_context,
):
    backend, store, _ = repair_context
    before = backend.mutation.legacy_inventory(POT)
    preview = _prepare(repair_context, chunk_size=1)
    assert preview.ok and preview.status == "ready" and preview.total_batches == 2
    assert preview.repairable_records == 2 and len(preview.changes) == 2
    assert backend.mutation.legacy_inventory(POT) == before
    assert backend.mutation.current_version(POT) == 0
    run = store.get_repair_run(pot_id=POT, repair_id=preview.repair_id)
    batches = tuple(
        store.get(pot_id=POT, plan_id=child.plan_id).lowered_batch
        for child in run.children
    )
    result = _apply(repair_context, preview.repair_id)
    assert result.ok and result.completed_batches == 2
    assert result.repaired == {"entity": 1, "claim": 1}
    assert [execution["version_after"] for execution in result.executions] == [1, 2]
    replay = _apply(repair_context, preview.repair_id)
    assert replay.executions == result.executions
    assert backend.mutation.current_version(POT) == 2
    assert [batch_fingerprint(batch) for batch in batches] == [
        child.batch_fingerprint for child in run.children
    ]
    second = _prepare(repair_context)
    assert second.ok and second.repairable_records == second.total_batches == 0
    assert _apply(repair_context, second.repair_id).status == "completed"
    assert backend.mutation.current_version(POT) == 2


def test_concurrent_previews_converge_and_changed_policy_rejects(repair_context):
    with ThreadPoolExecutor(4) as pool:
        previews = list(
            pool.map(lambda _: _prepare(repair_context, repair_id="shared"), range(8))
        )
    assert all(preview.changes == previews[0].changes for preview in previews)
    assert repair_context[0].mutation.current_version(POT) == 0
    with pytest.raises(MutationPersistenceError, match="different request"):
        _prepare(repair_context, repair_id="shared", chunk_size=1)


def test_generic_commit_cannot_apply_repair_child(repair_context):
    _, store, workbench = repair_context
    preview = _prepare(repair_context)
    run = store.get_repair_run(pot_id=POT, repair_id=preview.repair_id)
    child = run.children[0]
    with pytest.raises(PermissionError, match="administrative repair"):
        workbench.commit(child.plan_id, pot_id=POT)
    assert repair_context[0].mutation.current_version(POT) == 0
    applied = _apply(repair_context, preview.repair_id)
    # A completed receipt is always recoverable, even through the generic door.
    recovered = workbench.commit(child.plan_id, pot_id=POT)
    assert recovered.ok and recovered.execution.to_dict() == {
        key: value
        for key, value in applied.executions[0].items()
        if key != "mutation_id"
    }


def test_intervening_write_conflicts_without_reselection(repair_context):
    backend, store, _ = repair_context
    preview = _prepare(repair_context)
    frozen = preview.changes
    backend.mutation.apply(
        MutationBatch(
            entity_upserts=[EntityUpsert("service:new", ("Entity", "Service"), {})]
        ),
        expected_pot_id=POT,
    )
    result = _apply(repair_context, preview.repair_id)
    assert not result.ok and result.status == "conflict"
    assert result.completed_batches == 0 and result.changes == frozen
    assert backend.mutation.current_version(POT) == 1
    assert _prepare(repair_context, repair_id=preview.repair_id).changes == frozen


def test_reset_cannot_be_undone_by_old_repair(repair_context):
    backend, _, _ = repair_context
    preview = _prepare(repair_context)
    applied = _apply(repair_context, preview.repair_id)
    backend.mutation.reset_pot(POT, mutation_id="reset-after-repair")
    revision = backend.mutation.current_version(POT)
    replay = _apply(repair_context, preview.repair_id)
    assert replay.executions == applied.executions
    assert backend.mutation.current_version(POT) == revision
    assert backend.mutation.legacy_inventory(POT).records == ()


def test_blocking_preview_reserves_no_executable_children(tmp_path):
    state = _state(claim_id=False)
    state["rows"][0]["valid_at"] = "not-a-timestamp"
    backend = _backend("json", tmp_path, state)
    store = LocalJsonGraphPlanStore(tmp_path)
    workbench = _workbench(backend, store)
    context = backend, store, workbench
    preview = _prepare(context)
    assert not preview.ok and preview.status == "blocked" and preview.total_batches == 0
    assert store.list(pot_id=POT) == ()
    assert _apply(context, preview.repair_id).status == "blocked"
    assert backend.mutation.current_version(POT) == 0
    assert (
        json.loads((tmp_path / "graph.json").read_text())["rows"][0]["valid_at"]
        == "not-a-timestamp"
    )


@pytest.mark.parametrize(
    "corruption",
    [
        "missing_manifest",
        "missing_child",
        "changed_binding",
        "duplicate_child",
        "changed_batch",
        "wrong_schema",
        "bool_schema",
        "float_schema",
    ],
)
def test_corrupt_manifest_fails_closed(tmp_path, corruption):
    backend = _backend("json", tmp_path, _state(claim_id=False))
    store = LocalJsonGraphPlanStore(tmp_path)
    preview = prepare_legacy_repair(backend, store, pot_id=POT)
    data = json.loads(store._path.read_text())
    manifest = data["repair_runs"][POT][preview.repair_id]
    child = manifest["children"][0]
    if corruption == "missing_manifest":
        del data["repair_runs"]
    elif corruption == "missing_child":
        del data["plans"][POT][child["plan_id"]]
    elif corruption == "changed_binding":
        child["batch_fingerprint"] = "changed"
    elif corruption == "duplicate_child":
        manifest["children"].append(dict(child))
    elif corruption == "changed_batch":
        data["plans"][POT][child["plan_id"]]["lowered_batch"]["record_repairs"][0][
            "property_updates"
        ]["record_id"] = "changed"
    elif corruption == "wrong_schema":
        data["schema_version"] = 1
    else:
        manifest["schema_version"] = True if corruption == "bool_schema" else 1.0
    store._path.write_text(json.dumps(data))
    before = store._path.read_bytes()
    with pytest.raises(MutationPersistenceError):
        store.get_repair_run(pot_id=POT, repair_id=preview.repair_id)
    assert store._path.read_bytes() == before
    assert backend.mutation.current_version(POT) == 0


def test_process_crash_after_graph_commit_resumes_original_receipt(tmp_path):
    backend = _backend("json", tmp_path, _state(claim_id=False, entity_id=False))
    store = LocalJsonGraphPlanStore(tmp_path)
    preview = prepare_legacy_repair(backend, store, pot_id=POT, chunk_size=1)
    script = """
import os,sys
from pathlib import Path
from potpie_context_engine.adapters.outbound.graph.backends.embedded_backend import EmbeddedGraphBackend
from potpie_context_engine.adapters.outbound.graph.plan_stores.local_json import LocalJsonGraphPlanStore
from potpie_context_engine.core.runtime import build_graph_runtime
class CrashStore(LocalJsonGraphPlanStore):
    def compare_and_set(self, *, expected, replacement):
        if replacement.status == "committed":
            os._exit(29)
        return super().compare_and_set(expected=expected,replacement=replacement)
home=Path(sys.argv[1])
async def allow(pot, access):
    pass
service=build_graph_runtime(backend=EmbeddedGraphBackend(home=home),plan_store=CrashStore(home),commit_authorize=allow,commit_actor=lambda:"admin").workbench
service.repair_legacy(pot_id=sys.argv[2],mode="apply",repair_id=sys.argv[3])
"""
    process = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path), POT, preview.repair_id],
        capture_output=True,
        text=True,
    )
    assert process.returncode == 29, process.stderr
    backend = EmbeddedGraphBackend(home=tmp_path)
    store = LocalJsonGraphPlanStore(tmp_path)
    run = store.get_repair_run(pot_id=POT, repair_id=preview.repair_id)
    child = store.get(pot_id=POT, plan_id=run.children[0].plan_id)
    assert child.status == "committing"
    original = backend.mutation.lookup_execution(
        child.lowered_batch, expected_pot_id=POT, mutation_id=child.mutation_id
    ).result
    workbench = _workbench(backend, store)
    result = _apply((backend, store, workbench), preview.repair_id)
    assert result.ok and result.completed_batches == 2
    assert (
        result.executions[0]["applied_at"] == original.execution.applied_at.isoformat()
    )
    assert result.executions[0]["version_after"] == 1
    assert backend.mutation.current_version(POT) == 2


def test_raw_tag_codec_is_injective_and_precision_is_not_truncated():
    from datetime import datetime, timezone
    from potpie_context_engine.core.legacy_records import (
        LegacyRecord,
        LegacyInventory,
        raw_properties,
    )

    literal = {
        "observed_at": {"type": "datetime", "value": "2025-01-01T00:00:00+00:00"}
    }
    typed = {"observed_at": datetime(2025, 1, 1, tzinfo=timezone.utc)}
    assert raw_properties(literal) != raw_properties(typed)
    row = LegacyRecord(
        "p",
        "entity",
        "service:a",
        {"record_id": "e", "observed_at": "2025-01-01T00:00:00.000000001Z"},
        labels=("Entity",),
    )
    effects, findings = plan_legacy_repairs(LegacyInventory((row,)))
    assert effects == () and any(
        f.blocking and "precision" in f.detail for f in findings
    )


@pytest.mark.parametrize("changed", [True, 1.0])
def test_exact_beforeimage_distinguishes_numeric_types(changed):
    from potpie_context_engine.core.legacy_records import (
        LegacyRecord,
        match_record_repairs,
    )
    from potpie_context_engine.core.legacy_repair_policy import record_patch

    before = LegacyRecord(
        "p", "entity", "service:a", {"counter": 1}, labels=("Entity",)
    )
    effect = record_patch(before, new_record_id="new-id")
    actual = replace(before, properties={"counter": changed})
    with pytest.raises(ValueError, match="missing, changed, or ambiguous"):
        match_record_repairs((actual,), (effect,), pot_id="p")


@pytest.mark.parametrize("namespace", ["fields", "properties"])
@pytest.mark.parametrize("end", ["2024-01-01T00:00:00Z", "2025-01-01T05:30:00+05:30"])
def test_entity_nonpositive_valid_to_blocks_repair(namespace, end):
    from potpie_context_engine.core.legacy_records import LegacyRecord, LegacyInventory

    values = {"valid_from": "2025-01-01T00:00:00Z", "valid_to": end}
    record = LegacyRecord("p", "entity", "service:a", {}, labels=("Entity",))
    record = replace(record, **{namespace: values})
    effects, findings = plan_legacy_repairs(LegacyInventory((record,)))
    assert not effects and any(
        f.blocking and "not positive" in f.detail for f in findings
    )


@pytest.mark.parametrize("namespace", ["fields", "properties"])
@pytest.mark.parametrize("alias", ["pot_id", "group_id"])
def test_contradictory_reserved_pot_alias_blocks_repair(namespace, alias):
    from potpie_context_engine.core.legacy_records import LegacyRecord, LegacyInventory

    record = LegacyRecord("p", "entity", "service:a", {}, labels=("Entity",))
    record = replace(record, **{namespace: {alias: "other"}})
    effects, findings = plan_legacy_repairs(LegacyInventory((record,)))
    assert not effects and any(
        f.blocking and "contradicts" in f.detail for f in findings
    )


def test_keyless_legacy_claim_repairs_identity_without_inventing_logical_key():
    from potpie_context_engine.core.legacy_records import LegacyRecord, LegacyInventory

    record = LegacyRecord(
        "p",
        "claim",
        "",
        {"name": "DEPENDS_ON"},
        subject_key="service:a",
        object_key="service:b",
        relationship_type="RELATES_TO",
        subject_pot_id="p",
        object_pot_id="p",
    )
    effects, findings = plan_legacy_repairs(LegacyInventory((record,)))
    assert len(effects) == 1 and not any(f.blocking for f in findings)
    assert set(effects[0].property_updates) == {"record_id"}


@pytest.mark.parametrize("name", ["stored_source", "temporal_diagnostics"])
def test_unknown_source_keys_matching_internal_metadata_survive_repair(tmp_path, name):
    state = _state(claim_id=False)
    state["rows"][0][name] = {"legacy": "authored value"}
    backend = _backend("json", tmp_path, state)
    store = LocalJsonGraphPlanStore(tmp_path)
    context = backend, store, _workbench(backend, store)
    preview = _prepare(context)
    assert preview.ok
    assert _apply(context, preview.repair_id).ok
    assert json.loads((tmp_path / "graph.json").read_text())["rows"][0][name] == {
        "legacy": "authored value"
    }


def test_repair_receipt_never_claims_current_matching_from_empty_semantic_batch(
    repair_context,
):
    backend, store, workbench = repair_context
    preview = _prepare(repair_context)
    applied = _apply(repair_context, preview.repair_id)
    run = store.get_repair_run(pot_id=POT, repair_id=preview.repair_id)
    backend.mutation.reset_pot(POT)
    replay = workbench.commit(run.children[0].plan_id, pot_id=POT, verify=True)
    assert replay.ok and replay.verification.state == "unsupported"
    assert replay.verification.outcome == "inconclusive"
    assert replay.to_dict()["verification"]["unsupported"] == [{"capability": "verification.record_repairs"}]
    assert replay.execution.to_dict() == {
        key: value
        for key, value in applied.executions[0].items()
        if key != "mutation_id"
    }
