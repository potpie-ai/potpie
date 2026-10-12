"""One semantic/concurrency contract against reference and durable journals."""

import json
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest
from potpie_context_engine.core.graph_journal import (
    FieldValue,
    JournalError,
    JournalLimits,
    JournalRecord,
    JournalState,
    decode_journal,
    journal_json,
)
from potpie_context_engine.core.graph_mutations import (
    EdgeUpsert,
    EntityUpsert,
    InvalidationOp,
    ProvenanceContext,
)
from potpie_context_engine.core.journal_fields import capture_changes
from potpie_context_engine.core.journal_inverse import plan_inverse, validate_state
from potpie_context_engine.core.ports.claim_query import ClaimQueryFilter
from potpie_context_engine.core.reconciliation import MutationBatch

from potpie_context_engine.adapters.outbound.graph.backends.embedded_backend import (
    EmbeddedGraphBackend,
)
from potpie_context_engine.adapters.outbound.graph.backends.in_memory_backend import (
    InMemoryGraphBackend,
)


@pytest.fixture(params=("in_memory", "embedded", "falkordb"))
def backend(request, tmp_path):
    if request.param == "in_memory":
        backend = InMemoryGraphBackend()
    elif request.param == "embedded":
        backend = EmbeddedGraphBackend(home=tmp_path)
    else:
        port = os.environ.get("POTPIE_JOURNAL_FALKOR_PORT")
        if port is None:
            pytest.skip("set POTPIE_JOURNAL_FALKOR_PORT for a private FalkorDB server")
        from falkordb import FalkorDB

        from potpie_context_engine.adapters.outbound.graph.backends.falkordb_backend import (
            FalkorDBGraphBackend,
        )

        class Settings:
            def is_enabled(self):
                return True

            def falkordb_graph_name(self):
                return "journal_tests"

        graph = FalkorDB(host="127.0.0.1", port=int(port)).select_graph(
            uuid.uuid4().hex
        )
        backend = FalkorDBGraphBackend(Settings(), graph_provider=lambda: graph)
        request.addfinalizer(graph.delete)
    backend.journal.activate(pot_id="p", rollback_enabled=True)
    return backend


def test_disabling_rollback_preserves_capture_and_receipts(backend):
    write(backend, "disable-before", summary="before")
    original = backend.journal.journal_state("p")
    backend.journal.set_rollback_enabled(pot_id="p", enabled=False)
    with pytest.raises(JournalError, match="disabled"):
        backend.journal.plan_restore(
            pot_id="p", target_commit_id="disable-before", expected_head=original.head
        )
    write(backend, "disable-after", summary="after")
    state = backend.journal.journal_state("p")
    assert not state.rollback_enabled and state.sequence == 2
    assert state.generation == original.generation
    assert (
        backend.journal.get_receipt(pot_id="p", commit_id="disable-before") is not None
    )
    assert (
        backend.journal.get_receipt(pot_id="p", commit_id="disable-after") is not None
    )


def write(backend, mid, *, summary=None, props=None, entities=(), edges=()):
    operations = list(entities)
    if summary is not None or props is not None:
        operations.append(
            EntityUpsert(
                "service:a", ("Entity", "Service"), props or {"summary": summary}
            )
        )
    return backend.mutation.compare_and_apply(
        MutationBatch(entity_upserts=operations, edge_upserts=list(edges)),
        expected_pot_id="p",
        expected_version=backend.mutation.current_version("p"),
        provenance_context=ProvenanceContext(mutation_id=mid),
    )


def revert(backend, target, head, mid, mode="revert"):
    plan = backend.journal.plan_restore(
        pot_id="p", target_commit_id=target, expected_head=head, mode=mode
    )
    return backend.journal.apply_restore(plan, mutation_id=mid, actor="actor")


def test_overwrite_disjoint_edit_and_undo(backend):
    write(backend, "one", summary="original")
    write(backend, "two", summary="changed")
    write(backend, "three", props={"owner": "other"})
    receipt = revert(backend, "two", "three", "four")
    props = backend.claim_query.entity_properties(pot_id="p", entity_key="service:a")
    assert props["summary"] == "original" and props["owner"] == "other"
    assert receipt.affected_record_count == 1
    assert receipt.reverts_commit_id == "two"
    revert(backend, "four", "four", "five")
    assert (
        backend.claim_query.entity_properties(pot_id="p", entity_key="service:a")[
            "summary"
        ]
        == "changed"
    )


def test_same_field_conflict_and_stale_head(backend):
    for mid in ("one", "two", "three"):
        write(backend, mid, summary=mid)
    with pytest.raises(JournalError, match="field conflict"):
        revert(backend, "two", "three", "four")
    with pytest.raises(JournalError, match="stale HEAD"):
        revert(backend, "three", "two", "four")
    assert backend.journal.journal_state("p").head == "three"


def test_creation_retirement_keeps_history_and_reactivation(backend):
    write(backend, "one", summary="a")
    record_id = backend.claim_query.entity_properties(
        pot_id="p", entity_key="service:a"
    )["record_id"]
    revert(backend, "one", "one", "two")
    assert (
        backend.claim_query.entity_properties(pot_id="p", entity_key="service:a") == {}
    )
    assert not backend.claim_query.entity_labels(pot_id="p", entity_keys=("service:a",))
    assert not backend.inspection.neighborhood(pot_id="p", entity_key="service:a").nodes
    revert(backend, "two", "two", "three")
    assert (
        backend.claim_query.entity_properties(pot_id="p", entity_key="service:a")[
            "record_id"
        ]
        == record_id
    )


def test_new_dependency_blocks_retiring_endpoint(backend):
    write(backend, "one", summary="a")
    write(
        backend,
        "two",
        entities=[EntityUpsert("service:b", ("Entity", "Service"))],
        edges=[
            EdgeUpsert(
                "DEPENDS_ON",
                "service:b",
                "service:a",
                {"source_ref": "test", "truth": "authoritative_fact"},
            )
        ],
    )
    with pytest.raises(JournalError, match="endpoint"):
        revert(backend, "one", "two", "three")
    # Both inverses can be composed atomically if the range also removes the dependency.
    revert(backend, "one", "two", "three", mode="rollback")
    assert not backend.claim_query.find_claims(ClaimQueryFilter(pot_id="p"))


def test_range_is_one_commit_and_can_be_undone(backend):
    for mid in ("one", "two", "three"):
        write(backend, mid, summary=mid)
    receipt = revert(backend, "one", "three", "four", mode="rollback")
    assert receipt.sequence == 4 and receipt.rollback_target_commit_id == "one"
    assert receipt.affected_record_count == 1
    assert (
        backend.claim_query.entity_properties(pot_id="p", entity_key="service:a")[
            "summary"
        ]
        == "one"
    )
    revert(backend, "four", "four", "five")
    assert (
        backend.claim_query.entity_properties(pot_id="p", entity_key="service:a")[
            "summary"
        ]
        == "three"
    )


def test_restore_retry_returns_receipt_before_stale_head_and_binds_actor(backend):
    write(backend, "one", summary="a")
    plan = backend.journal.plan_restore(
        pot_id="p", target_commit_id="one", expected_head="one"
    )
    first = backend.journal.apply_restore(plan, mutation_id="two", actor="actor")
    assert (
        backend.journal.apply_restore(plan, mutation_id="two", actor="actor") == first
    )
    with pytest.raises(ValueError, match="reused"):
        backend.journal.apply_restore(plan, mutation_id="two", actor="other")
    assert backend.journal.journal_state("p").sequence == 2


def test_resource_generation_fences_plan_and_incomplete_operation(backend):
    write(backend, "one", summary="a")
    plan = backend.journal.plan_restore(
        pot_id="p", target_commit_id="one", expected_head="one"
    )
    backend.journal.begin_resource(pot_id="p", operation_id="import", owner="worker")
    with pytest.raises(JournalError, match="resource"):
        backend.journal.apply_restore(plan, mutation_id="two", actor="actor")
    with pytest.raises(JournalError, match="guard"):
        write(backend, "bad", props={"owner": "x"})
    barrier = backend.journal.complete_resource(
        pot_id="p", operation_id="import", owner="worker"
    )
    assert not barrier.rollback_supported and barrier.origin == "resource"
    with pytest.raises(JournalError, match="barrier"):
        revert(backend, "one", barrier.commit_id, "two", mode="rollback")
    # Completed resources allow an unrelated selective revert under current generation.
    revert(backend, "one", barrier.commit_id, "two")


def test_admin_writers_rejected_and_other_pot_isolated(backend):
    write(backend, "one", summary="a")
    with pytest.raises(JournalError, match="reset"):
        backend.mutation.reset_pot("p")
    payload = backend.snapshot.export_data(pot_id="p")
    with pytest.raises(JournalError, match="snapshot import"):
        backend.snapshot.import_data(pot_id="p", payload=payload)
    assert backend.journal.get_receipt(pot_id="other", commit_id="one") is None
    assert backend.journal.journal_state("p").head == "one"


def test_embedded_restart_and_failure_before_publication(tmp_path, monkeypatch):
    backend = EmbeddedGraphBackend(home=tmp_path)
    backend.journal.activate(pot_id="p", rollback_enabled=True)
    write(backend, "one", summary="before")
    original = EmbeddedGraphBackend._write_state

    def fail(*args, **kwargs):
        raise OSError("lost disk")

    monkeypatch.setattr(EmbeddedGraphBackend, "_write_state", fail)
    with pytest.raises(OSError):
        write(backend, "two", summary="unpublished")
    monkeypatch.setattr(EmbeddedGraphBackend, "_write_state", original)
    restarted = EmbeddedGraphBackend(home=tmp_path)
    assert (
        restarted.claim_query.entity_properties(pot_id="p", entity_key="service:a")[
            "summary"
        ]
        == "before"
    )
    assert restarted.journal.get_receipt(pot_id="p", commit_id="two") is None
    assert restarted.journal.get_receipt(pot_id="p", commit_id="one") is not None


def test_embedded_replicas_do_not_lose_head_or_guards(tmp_path):
    first, second = [EmbeddedGraphBackend(home=tmp_path) for _ in range(2)]
    first.journal.activate(pot_id="p", rollback_enabled=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(
                b.mutation.apply,
                MutationBatch(
                    entity_upserts=[EntityUpsert(f"service:{i}", ("Entity", "Service"))]
                ),
                expected_pot_id="p",
            )
            for i, b in enumerate((first, second))
        ]
        assert all(f.result().ok for f in futures)
    assert first.journal.journal_state("p").sequence == 2
    second.journal.begin_resource(pot_id="p", operation_id="resource", owner="worker")
    assert (
        EmbeddedGraphBackend(home=tmp_path)
        .journal.journal_state("p")
        .resource_guard.operation_id
        == "resource"
    )


def test_caps_reject_without_canonical_change(tmp_path):
    backend = EmbeddedGraphBackend(home=tmp_path)
    backend.journal.activate(pot_id="p", limits=JournalLimits(max_records=1))
    with pytest.raises(JournalError, match="record limit"):
        write(
            backend,
            "one",
            summary="a",
            entities=[EntityUpsert("service:b", ("Entity", "Service"))],
        )
    assert backend.journal.journal_state("p").sequence == 0
    assert not EmbeddedGraphBackend(home=tmp_path).claim_query.entity_properties(
        pot_id="p", entity_key="service:a"
    )


def test_codec_null_unset_and_nested_disjoint_fields():
    old = JournalRecord(
        "record",
        "p",
        "key",
        "entity",
        {"properties": {"nested": {"a": None}, "empty": {}}, "active": True},
    )
    new = replace(
        old, fields={"properties": {"nested": {"a": "value", "b": 2}}, "active": True}
    )
    semantic, audit = capture_changes({"record": old}, {"record": new})
    assert not audit
    assert decode_journal(json.loads(journal_json(semantic))) == semantic
    assert next(
        c for c in semantic[0].fields if c.path == ("properties", "nested", "a")
    ).before == FieldValue(True, None)
    assert next(
        c for c in semantic[0].fields if c.path == ("properties", "nested", "b")
    ).before == FieldValue(False)
    from potpie_context_engine.core.graph_journal import CommitReceipt

    receipt = CommitReceipt(
        "commit",
        "p",
        "generation",
        1,
        None,
        "fp",
        datetime.now(timezone.utc),
        "actor",
        "message",
        "graph",
        "write",
        True,
        semantic,
    )
    state = JournalState("p", "generation", 1, "commit", rollback_enabled=True)
    plan = plan_inverse(
        state=state,
        current={"record": new},
        receipts=[receipt],
        target_commit_id="commit",
        expected_head="commit",
        validator=lambda _: None,
    )
    assert plan.records[0].after == old


def test_singleton_implicit_effects_are_captured_and_restored(backend):
    write(
        backend,
        "one",
        summary="a",
        entities=[
            EntityUpsert("team:first", ("Entity", "Team")),
            EntityUpsert("team:second", ("Entity", "Team")),
        ],
        edges=[
            EdgeUpsert(
                "OWNED_BY",
                "service:a",
                "team:first",
                {
                    "claim_key": "owner1",
                    "source_ref": "one",
                    "truth": "authoritative_fact",
                },
            )
        ],
    )
    write(
        backend,
        "two",
        edges=[
            EdgeUpsert(
                "OWNED_BY",
                "service:a",
                "team:second",
                {
                    "claim_key": "owner2",
                    "source_ref": "two",
                    "truth": "authoritative_fact",
                },
            )
        ],
    )
    receipt = backend.journal.get_receipt(pot_id="p", commit_id="two")
    assert {c.logical_key for c in receipt.semantic_changes} == {"owner1", "owner2"}
    assert receipt.affected_record_count == 2
    revert(backend, "two", "two", "three")
    rows = backend.claim_query.find_claims(ClaimQueryFilter(pot_id="p"))
    assert len(rows) == 1 and rows[0].object_key == "team:first"
    revert(backend, "three", "three", "four")
    rows = backend.claim_query.find_claims(ClaimQueryFilter(pot_id="p"))
    assert len(rows) == 1 and rows[0].object_key == "team:second"


def test_parallel_claims_share_logical_key_but_keep_incarnations(backend):
    write(
        backend,
        "one",
        summary="a",
        entities=[EntityUpsert("service:b", ("Entity", "Service"))],
        edges=[
            EdgeUpsert(
                "DEPENDS_ON",
                "service:a",
                "service:b",
                {
                    "claim_key": "shared",
                    "source_ref": source,
                    "truth": "authoritative_fact",
                },
            )
            for source in ("first", "second")
        ],
    )
    rows = backend.claim_query.find_claims(ClaimQueryFilter(pot_id="p"))
    assert len(rows) == 2 and len({r.record_id for r in rows}) == 2
    assert (
        len(backend.inspection.neighborhood(pot_id="p", entity_key="service:a").edges)
        == 2
    )
    revert(backend, "one", "one", "two")
    assert not backend.claim_query.find_claims(ClaimQueryFilter(pot_id="p"))
    assert (
        len(
            backend.claim_query.find_claims(
                ClaimQueryFilter(
                    pot_id="p", include_invalidated=True, include_retired=True
                )
            )
        )
        == 2
    )


def test_native_writer_bypass_is_rejected(backend):
    if backend.profile != "falkordb":
        return
    import asyncio

    from potpie_context_engine.core.graph_mutations import ProvenanceRef

    with pytest.raises(JournalError, match="legacy direct writer"):
        asyncio.run(
            backend.graph_writer.upsert_entities(
                "p",
                [EntityUpsert("service:a", ("Entity", "Service"))],
                ProvenanceRef("p", "event"),
            )
        )
    assert backend.journal.journal_state("p").sequence == 0


def test_recovery_fences_old_owner_and_requires_worker_proof(backend):
    from potpie_context_engine.core.journal_context import (
        JournalWriteContext,
        journal_write_context,
    )

    backend.journal.begin_resource(pot_id="p", operation_id="op", owner="old")
    with pytest.raises(JournalError, match="stopped"):
        backend.journal.recover_resource_guard(
            pot_id="p",
            operation_id="op",
            expected_owner="old",
            new_owner="new",
            verify_worker_stopped=lambda _: False,
        )
    backend.journal.recover_resource_guard(
        pot_id="p",
        operation_id="op",
        expected_owner="old",
        new_owner="new",
        verify_worker_stopped=lambda _: True,
    )
    with (
        journal_write_context(
            JournalWriteContext(resource_operation_id="op", resource_owner="old")
        ),
        pytest.raises(JournalError, match="guard"),
    ):
        write(backend, "bad", summary="bad")
    with journal_write_context(
        JournalWriteContext(resource_operation_id="op", resource_owner="new")
    ):
        write(backend, "good", summary="good")
    backend.journal.complete_resource(pot_id="p", operation_id="op", owner="new")
    with (
        journal_write_context(
            JournalWriteContext(resource_operation_id="op", resource_owner="old")
        ),
        pytest.raises(JournalError, match="guard"),
    ):
        write(backend, "late", summary="late")


def test_resource_pause_before_import_and_after_final_delete(
    backend, tmp_path, monkeypatch
):
    from potpie_context_engine.core.runtime import build_graph_runtime

    from potpie_context_engine.adapters.outbound.resources.local_resource_store import (
        LocalResourceStore,
    )
    from potpie_context_engine.application.services.resource_facade import (
        ResourceFacade,
    )
    from potpie_context_engine.testing import InMemoryGraphPlanStore

    store = LocalResourceStore(home=tmp_path / "resources")
    runtime = build_graph_runtime(
        backend=backend, plan_store=InMemoryGraphPlanStore(), resource_store=store
    )
    journal = runtime.backend.journal
    facade = ResourceFacade(
        store=store,
        graph=runtime.graph,
        claims=runtime.backend.claim_query,
        journal=journal,
    )
    write(backend, "one", summary="a")
    plan = journal.plan_restore(pot_id="p", target_commit_id="one", expected_head="one")
    content = {
        "meta.json": json.dumps(
            {
                "source_ref": "synthetic:test",
                "sections": [
                    {
                        "slug": "body",
                        "title": "Body",
                        "summary": "Source text",
                        "ordinal": 0,
                        "chunks": [{"seq": 0, "label": "Body"}],
                    }
                ],
            }
        ),
        "body/0000.txt": "Evidence",
    }
    original_import = LocalResourceStore.import_dir

    def paused_import(self, **kwargs):
        assert journal.journal_state("p").resource_guard is not None
        with pytest.raises(JournalError, match="resource"):
            journal.apply_restore(plan, mutation_id="unsafe", actor="actor")
        return original_import(self, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(LocalResourceStore, "import_dir", paused_import)
        result = facade.import_dir(pot_id="p", slug="test", files=content)
    assert result.graph.ok, result.graph
    assert journal.journal_state("p").resource_guard is None
    receipts = journal.read_receipts(
        pot_id="p", generation=journal.journal_state("p").generation, after_sequence=1
    )
    assert receipts and all(
        r.origin == "resource" and not r.rollback_supported and r.resource_operation_id
        for r in receipts
    )
    plan = journal.plan_restore(
        pot_id="p",
        target_commit_id="one",
        expected_head=journal.journal_state("p").head,
    )
    original_delete = LocalResourceStore.delete

    def paused_delete(self, **kwargs):
        assert journal.journal_state("p").resource_guard is not None
        with pytest.raises(JournalError, match="resource"):
            journal.apply_restore(plan, mutation_id="unsafe-delete", actor="actor")
        return original_delete(self, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(LocalResourceStore, "delete", paused_delete)
        result = facade.delete(pot_id="p", slug="test")
    assert result.removed
    assert journal.journal_state("p").resource_guard is None


def test_falkor_lost_response_preserves_receipt(backend, monkeypatch):
    if backend.profile != "falkordb":
        return
    from potpie_context_engine.adapters.outbound.graph.falkordb_journal import (
        FalkorJournal,
    )

    original = FalkorJournal._execute

    def lose_response(self, *args, **kwargs):
        original(self, *args, **kwargs)
        raise OSError("response lost after publication")

    with monkeypatch.context() as patch:
        patch.setattr(FalkorJournal, "_execute", lose_response)
        with pytest.raises(OSError):
            write(backend, "one", summary="a")
    assert backend.journal.get_receipt(pot_id="p", commit_id="one") is not None
    write(backend, "one", summary="a")
    assert backend.journal.journal_state("p").sequence == 1


def test_resource_completion_retry_is_idempotent_and_owner_bound(backend):
    backend.journal.begin_resource(pot_id="p", operation_id="op", owner="worker")
    first = backend.journal.complete_resource(
        pot_id="p", operation_id="op", owner="worker"
    )
    assert (
        backend.journal.complete_resource(pot_id="p", operation_id="op", owner="worker")
        == first
    )
    with pytest.raises(JournalError, match="reused"):
        backend.journal.complete_resource(pot_id="p", operation_id="op", owner="other")
    assert backend.journal.journal_state("p").sequence == 1


def test_restore_commit_id_cannot_be_reused_as_ordinary_mutation(backend):
    write(backend, "one", summary="a")
    write(backend, "two", summary="b")
    revert(backend, "two", "two", "undo")
    with pytest.raises(ValueError, match="reused|already used"):
        write(backend, "undo", summary="c")
    assert backend.journal.journal_state("p").sequence == 3
    assert (
        backend.claim_query.entity_properties(pot_id="p", entity_key="service:a")[
            "summary"
        ]
        == "a"
    )


def test_review_marker_effect_cannot_be_labeled_as_reversible(backend):
    write(
        backend,
        "one",
        summary="a",
        entities=[EntityUpsert("service:b", ("Entity", "Service"))],
        edges=[
            EdgeUpsert(
                "DEPENDS_ON",
                "service:a",
                "service:b",
                {
                    "claim_key": "ordinary-looking-key",
                    "source_ref": "source",
                    "truth": "authoritative_fact",
                    "evidence": [
                        {
                            "source_ref": "source",
                            "authority": "external_system",
                            "metadata": {"evidence_review_required": True},
                        }
                    ],
                },
            )
        ],
    )
    receipt = backend.journal.get_receipt(pot_id="p", commit_id="one")
    assert not receipt.rollback_supported and receipt.origin == "resource"
    assert {c["logical_key"] for c in receipt.display_context} >= {
        "service:a",
        "service:b",
    }
    with pytest.raises(JournalError, match="barrier"):
        revert(backend, "one", "one", "two")


def test_activation_preserves_legacy_native_identity_and_marks_boundary(backend):
    result = backend.mutation.compare_and_apply(
        MutationBatch(
            entity_upserts=[
                EntityUpsert(
                    "service:legacy", ("Entity", "Service"), {"summary": "before"}
                )
            ]
        ),
        expected_pot_id="legacy",
        expected_version=backend.mutation.current_version("legacy"),
        provenance_context=ProvenanceContext(mutation_id="legacy-write"),
    )
    assert result.ok
    before = backend.claim_query.entity_properties(
        pot_id="legacy", entity_key="service:legacy"
    )
    state = backend.journal.activate(pot_id="legacy")
    assert state.sequence == 0 and not state.rollback_enabled
    assert not backend.journal.get_receipt(pot_id="legacy", commit_id="legacy-write")
    after = backend.claim_query.entity_properties(
        pot_id="legacy", entity_key="service:legacy"
    )
    assert after["record_id"]
    if before.get("record_id") or before.get("uuid"):
        assert after["record_id"] == (before.get("record_id") or before["uuid"])
    assert state == backend.journal.activate(pot_id="legacy")


def test_normal_upsert_reactivates_retired_entity_without_changing_identity(backend):
    write(backend, "one", summary="a")
    original = backend.claim_query.entity_properties(
        pot_id="p", entity_key="service:a"
    )["record_id"]
    revert(backend, "one", "one", "undo")
    assert write(backend, "new", summary="new").ok
    after = backend.claim_query.entity_properties(pot_id="p", entity_key="service:a")
    assert after["record_id"] == original and after["summary"] == "new"


def test_reasserting_retired_claim_creates_distinct_incarnation_and_keeps_history(
    backend,
):
    entities = [
        EntityUpsert(key, ("Entity", "Service")) for key in ("service:a", "service:b")
    ]
    edge = EdgeUpsert(
        "DEPENDS_ON",
        "service:a",
        "service:b",
        {"claim_key": "edge", "source_ref": "source", "truth": "authoritative_fact"},
    )
    write(backend, "entities", entities=entities)
    write(backend, "one", edges=[edge])
    old = backend.claim_query.find_claims(ClaimQueryFilter(pot_id="p"))[0].record_id
    revert(backend, "one", "one", "undo")
    assert not backend.snapshot.export_data(pot_id="p")["claims"]
    write(backend, "new", edges=[edge])
    current = backend.claim_query.find_claims(ClaimQueryFilter(pot_id="p"))
    assert len(current) == 1 and current[0].record_id != old
    history = backend.claim_query.find_claims(
        ClaimQueryFilter(pot_id="p", include_invalidated=True, include_retired=True)
    )
    assert len(history) == 2 and len({row.record_id for row in history}) == 2
    revert(backend, "undo", "new", "again")
    current = backend.claim_query.find_claims(ClaimQueryFilter(pot_id="p"))
    assert len(current) == 2 and old in {row.record_id for row in current}
    with pytest.raises(JournalError, match="snapshot export.*parallel"):
        backend.snapshot.export_data(pot_id="p")


@pytest.mark.parametrize(
    "limits", [JournalLimits(max_records=1), JournalLimits(max_bytes=200)]
)
def test_native_caps_reject_before_any_canonical_change(backend, limits):
    backend.journal.activate(pot_id="cap", limits=limits)
    with pytest.raises(JournalError, match="limit"):
        backend.mutation.compare_and_apply(
            MutationBatch(
                entity_upserts=[
                    EntityUpsert(key, ("Entity", "Service"))
                    for key in ("service:a", "service:b")
                ]
            ),
            expected_pot_id="cap",
            expected_version=backend.mutation.current_version("cap"),
            provenance_context=ProvenanceContext(mutation_id="large"),
        )
    assert backend.journal.journal_state("cap").sequence == 0
    assert not backend.claim_query.entity_properties(
        pot_id="cap", entity_key="service:a"
    )


def test_neo4j_capability_is_explicitly_unavailable():
    from potpie_context_engine.adapters.outbound.graph.backends.neo4j_backend import (
        Neo4jGraphBackend,
    )

    backend = Neo4jGraphBackend(object())
    assert not backend.journal.journal_capability("p").journal_supported
    with pytest.raises(JournalError, match="deferred"):
        backend.journal.activate(pot_id="p")


def test_identity_migration_rejects_duplicate_ids_without_activation(backend):
    result = backend.mutation.compare_and_apply(
        MutationBatch(
            entity_upserts=[
                EntityUpsert(key, ("Entity", "Service"), {"uuid": "duplicate"})
                for key in ("service:a", "service:b")
            ]
        ),
        expected_pot_id="legacy",
        expected_version=backend.mutation.current_version("legacy"),
    )
    assert result.ok
    with pytest.raises(JournalError, match="ambiguous"):
        backend.journal.activate(pot_id="legacy")
    assert backend.journal.journal_state("legacy") is None
    assert (
        backend.claim_query.entity_properties(pot_id="legacy", entity_key="service:a")[
            "uuid"
        ]
        == "duplicate"
    )


def test_codec_preserves_supported_types_and_reserved_map_keys():
    value = {
        "type": "datetime",
        "value": {
            "none": None,
            "bool": False,
            "int": 2,
            "float": 2.5,
            "string": "☃",
            "tuple": (1, "2", False),
            "list": ["1", 2],
            "empty": {},
            "time": datetime.now(timezone.utc),
        },
    }
    decoded = decode_journal(json.loads(journal_json(value)))
    assert decoded == value and type(decoded["value"]["tuple"]) is tuple
    assert type(decoded["value"]["list"]) is list
    naive = datetime(2026, 1, 1)  # noqa: DTZ001 - exercise rejection of naive datetimes.
    for unsupported in (float("nan"), float("inf"), {1}, naive):
        with pytest.raises(JournalError):
            journal_json(unsupported)


def test_restore_invalidates_current_vectors_and_keeps_semantic_source_fields(backend):
    write(
        backend,
        "entities",
        entities=[
            EntityUpsert(key, ("Entity", "Service"))
            for key in ("service:a", "service:b")
        ],
    )
    for mid, text, vector in (
        ("one", "before", [1.0, 0.0]),
        ("two", "after", [0.0, 1.0]),
    ):
        write(
            backend,
            mid,
            edges=[
                EdgeUpsert(
                    "DEPENDS_ON",
                    "service:a",
                    "service:b",
                    {
                        "claim_key": "edge",
                        "source_ref": "source",
                        "truth": "authoritative_fact",
                        "description": text,
                        "fact_embedding": vector,
                        "embedding_dim": 2,
                    },
                )
            ],
        )
    current = backend.claim_query.find_claims(ClaimQueryFilter(pot_id="p"))[0]
    if backend.profile == "falkordb":
        assert backend.journal.graph.query(
            "MATCH ()-[r:RELATES_TO]->() RETURN r.fact_embedding"
        ).result_set == [[[0.0, 1.0]]]
    else:
        assert current.fact_embedding == (0.0, 1.0)
    revert(backend, "two", "two", "undo")
    restored = backend.claim_query.find_claims(ClaimQueryFilter(pot_id="p"))[0]
    assert restored.description == "before" and restored.fact_embedding is None
    assert restored.source_ref == "source"
    if backend.profile == "falkordb":
        assert backend.journal.graph.query(
            "MATCH ()-[r:RELATES_TO]->() RETURN r.fact_embedding"
        ).result_set == [[None]]


def test_offset_timestamp_reassertion_keeps_invalidated_incarnation(backend):
    entities = [
        EntityUpsert(key, ("Entity", "Service")) for key in ("service:a", "service:b")
    ]
    edge = EdgeUpsert(
        "DEPENDS_ON",
        "service:a",
        "service:b",
        {"claim_key": "edge", "source_ref": "source", "truth": "authoritative_fact"},
    )
    write(backend, "one", entities=entities, edges=[edge])
    original = backend.claim_query.find_claims(ClaimQueryFilter(pot_id="p"))[
        0
    ].record_id
    expired = (
        (datetime.now(timezone.utc) - timedelta(minutes=1))
        .astimezone(timezone(timedelta(hours=5, minutes=30)))
        .isoformat()
    )
    result = backend.mutation.compare_and_apply(
        MutationBatch(
            invalidations=[
                InvalidationOp(
                    None, None, "expired", valid_to=expired, target_claim_keys=("edge",)
                )
            ]
        ),
        expected_pot_id="p",
        expected_version=backend.mutation.current_version("p"),
        provenance_context=ProvenanceContext(mutation_id="invalidate"),
    )
    assert result.ok and not backend.claim_query.find_claims(
        ClaimQueryFilter(pot_id="p")
    )
    write(backend, "new", edges=[edge])
    live = backend.claim_query.find_claims(ClaimQueryFilter(pot_id="p"))
    assert len(live) == 1 and live[0].record_id != original
    history = backend.claim_query.find_claims(
        ClaimQueryFilter(pot_id="p", include_invalidated=True, include_retired=True)
    )
    assert (
        len(history) == 2
        and next(row for row in history if row.record_id == original).invalid_at
        is not None
    )


def test_restore_after_supersession_accepts_native_bookkeeping_edges(backend):
    write(
        backend,
        "initial",
        summary="original",
        entities=[
            EntityUpsert(key, ("Entity", "Service"), {})
            for key in ("service:b", "service:c")
        ],
        edges=[EdgeUpsert("DEPENDS_ON", "service:a", "service:b", {})],
    )
    result = backend.mutation.compare_and_apply(
        MutationBatch(
            edge_upserts=[EdgeUpsert("DEPENDS_ON", "service:a", "service:c", {})],
            invalidations=[
                InvalidationOp(
                    None,
                    ("DEPENDS_ON", "service:a", "service:b"),
                    "replacement dependency",
                    superseded_by_key="service:c",
                )
            ],
        ),
        expected_pot_id="p",
        expected_version=backend.mutation.current_version("p"),
        provenance_context=ProvenanceContext(mutation_id="supersede"),
    )
    assert result.ok
    write(backend, "edit", summary="temporary")
    revert(backend, "edit", "edit", "revert")
    assert (
        backend.claim_query.entity_properties(pot_id="p", entity_key="service:a")[
            "summary"
        ]
        == "original"
    )
    live = backend.claim_query.find_claims(ClaimQueryFilter(pot_id="p"))
    assert any(
        row.predicate == "DEPENDS_ON" and row.object_key == "service:c" for row in live
    )
    if backend.profile == "falkordb":
        assert any(row.predicate == "SUPERSEDES" for row in live)


def test_restore_validation_keeps_system_edges_scoped_and_unknown_edges_closed():
    records = {
        key: JournalRecord(
            key, "p", key, "entity", {"labels": ("Team",), "properties": {}}
        )
        for key in ("team:a", "team:b")
    }
    claim = JournalRecord(
        "system",
        "p",
        "claim:system",
        "claim",
        {"subject_key": "team:a", "object_key": "team:b", "predicate": "SUPERSEDES"},
    )
    validate_state({**records, "system": claim}, pot_id="p")
    with pytest.raises(JournalError, match="absent/retired endpoint"):
        validate_state({"team:a": records["team:a"], "system": claim}, pot_id="p")
    with pytest.raises(JournalError, match="scope mismatch"):
        validate_state(
            {**records, "system": replace(claim, pot_id="other")}, pot_id="p"
        )
    unknown = replace(claim, fields={**claim.fields, "predicate": "UNKNOWN_INTERNAL"})
    with pytest.raises(JournalError, match="unknown edge type"):
        validate_state({**records, "system": unknown}, pot_id="p")
