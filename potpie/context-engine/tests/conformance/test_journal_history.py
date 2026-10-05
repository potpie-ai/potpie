import asyncio
from dataclasses import replace

import pytest
from potpie_context_core.graph_journal import JournalError
from potpie_context_core.graph_mutations import EntityUpsert, ProvenanceContext
from potpie_context_core.journal_history import (
    CommitFilters,
    commit_header,
    list_history,
    rebuild_history,
    receipt_detail,
    reconcile_history,
)
from potpie_context_core.reconciliation import MutationBatch

from potpie_context_engine.adapters.outbound.graph.backends.embedded_backend import (
    EmbeddedGraphBackend,
)
from potpie_context_engine.adapters.outbound.graph.local_commit_mirror import (
    LocalCommitMirror,
)


def seed(tmp_path, count=5):
    backend = EmbeddedGraphBackend(home=tmp_path)
    backend.journal.activate(pot_id="p", rollback_enabled=True)
    for i in range(1, count + 1):
        backend.mutation.apply(
            MutationBatch(
                entity_upserts=[
                    EntityUpsert(
                        "service:a", ("Entity", "Service"), {"summary": str(i)}
                    )
                ]
            ),
            expected_pot_id="p",
            provenance_context=ProvenanceContext(
                mutation_id=f"c{i}", actor_user_id="a" if i % 2 else "b"
            ),
        )
    return backend


async def test_recover_native_receipts_and_rebuild_deleted_index(tmp_path):
    backend = seed(tmp_path)
    for _ in range(2):
        path = tmp_path / "listing.sqlite"
        mirror = LocalCommitMirror(path)
        result = await list_history(backend.journal, mirror, pot_id="p", limit=2)
        assert result["coverage"]["complete"]
        assert [r["sequence"] for r in result["headers"]] == [5, 4]
        page = await list_history(
            backend.journal, mirror, pot_id="p", limit=2, cursor=result["next_cursor"]
        )
        assert [r["sequence"] for r in page["headers"]] == [3, 2]
        path.unlink()
    assert backend.journal.journal_state("p").sequence == 5


async def test_listing_detects_rebuild_between_coverage_and_page(tmp_path, monkeypatch):
    backend = seed(tmp_path)
    mirror = LocalCommitMirror(tmp_path / "listing.sqlite")
    original = mirror.read_page_async

    async def reset_then_read(**kwargs):
        await mirror.reset_for_rebuild_async(
            pot_id=kwargs["pot_id"], generation=kwargs["generation"]
        )
        return await original(**kwargs)

    monkeypatch.setattr(mirror, "read_page_async", reset_then_read)
    with pytest.raises(JournalError) as rejected:
        await list_history(backend.journal, mirror, pot_id="p")
    assert rejected.value.code == "history_rebuilding"


async def test_cursor_rejects_rebuild_behind_newer_reported_coverage(
    tmp_path, monkeypatch
):
    backend = seed(tmp_path)
    mirror = LocalCommitMirror(tmp_path / "listing.sqlite")
    page = await list_history(backend.journal, mirror, pot_id="p", limit=1)
    backend.mutation.apply(
        MutationBatch(
            entity_upserts=[
                EntityUpsert("service:a", ("Entity", "Service"), {"summary": "6"})
            ]
        ),
        expected_pot_id="p",
        provenance_context=ProvenanceContext(mutation_id="c6"),
    )
    original = mirror.read_page_async

    async def older_rebuild_snapshot(**kwargs):
        _, rows = await original(**kwargs)
        # Old cursor rows are available, but the newer reconciled checkpoint
        # disappeared while another worker rebuilt the index.
        return kwargs["ceiling"], rows

    monkeypatch.setattr(mirror, "read_page_async", older_rebuild_snapshot)
    with pytest.raises(JournalError) as rejected:
        await list_history(
            backend.journal, mirror, pot_id="p", cursor=page["next_cursor"]
        )
    assert rejected.value.code == "history_rebuilding"


async def test_cursor_binds_filters_pot_and_generation(tmp_path):
    backend = seed(tmp_path)
    mirror = LocalCommitMirror(tmp_path / "listing.sqlite")
    filters = CommitFilters(actor="a")
    result = await list_history(
        backend.journal, mirror, pot_id="p", limit=1, filters=filters
    )
    assert result["headers"][0]["sequence"] == 5
    with pytest.raises(JournalError, match="cursor"):
        await list_history(
            backend.journal, mirror, pot_id="p", cursor=result["next_cursor"]
        )
    page = await list_history(
        backend.journal,
        mirror,
        pot_id="p",
        limit=1,
        cursor=result["next_cursor"],
        filters=filters,
    )
    assert page["headers"][0]["sequence"] == 3
    from potpie_context_core.journal_history import cursor_decode

    with pytest.raises(JournalError, match="cursor"):
        cursor_decode(
            result["next_cursor"],
            pot_id="other",
            generation=backend.journal.journal_state("p").generation,
            filters=filters,
        )


async def test_mirror_failure_keeps_receipt_and_retry_recovery(tmp_path, monkeypatch):
    backend = seed(tmp_path)
    mirror = LocalCommitMirror(tmp_path / "listing.sqlite")

    async def fail(**kwargs):
        raise OSError("mirror unavailable")

    with monkeypatch.context() as patch:
        patch.setattr(mirror, "append_batch_async", fail)
        with pytest.raises(OSError):
            await reconcile_history(backend.journal, mirror, pot_id="p")
    assert (
        await mirror.progress_async(
            pot_id="p", generation=backend.journal.journal_state("p").generation
        )
        == 0
    )
    assert backend.journal.get_receipt(pot_id="p", commit_id="c5") is not None
    assert (await reconcile_history(backend.journal, mirror, pot_id="p"))["complete"]


async def test_replicas_duplicate_imports_and_mismatch(tmp_path):
    backend = seed(tmp_path)
    mirrors = [LocalCommitMirror(tmp_path / "listing.sqlite") for _ in range(3)]
    await asyncio.gather(
        *(reconcile_history(backend.journal, mirror, pot_id="p") for mirror in mirrors)
    )
    receipt = backend.journal.get_receipt(pot_id="p", commit_id="c1")
    with pytest.raises(JournalError, match="mismatch"):
        await mirrors[0].append_batch_async(
            headers=(commit_header(replace(receipt, message="corrupt")),),
            expected_progress=0,
        )


async def test_bounded_recovery_reports_lag_and_gap_fails_closed(tmp_path, monkeypatch):
    backend = seed(tmp_path, count=105)
    mirror = LocalCommitMirror(tmp_path / "listing.sqlite")
    coverage = await reconcile_history(
        backend.journal, mirror, pot_id="p", max_batches=1
    )
    assert (
        coverage["indexed_through"] == 100
        and coverage["indexing_lag"] == 5
        and not coverage["complete"]
    )
    original = backend.journal.read_receipts
    # EmbeddedJournal instances are lightweight facades, so bind the test's instance.
    journal = backend.journal
    monkeypatch.setattr(journal, "read_receipts", lambda **kw: original(**kw)[1:])
    with pytest.raises(JournalError, match="gap"):
        await reconcile_history(journal, mirror, pot_id="p")


def test_detail_is_bounded_and_explains_partial_history(tmp_path):
    backend = seed(tmp_path)
    result = receipt_detail(backend.journal, pot_id="p", commit_id="c2", limit=1)
    assert len(result["changes"]) == 1
    assert "partial" in result["historical_view"]
    with pytest.raises(JournalError, match="pagination"):
        receipt_detail(backend.journal, pot_id="p", commit_id="c2", limit=100000)


async def test_checkpoint_corruption_fails_closed_and_explicit_rebuild_repairs(
    tmp_path,
):
    import json
    import sqlite3

    backend = seed(tmp_path)
    path = tmp_path / "listing.sqlite"
    mirror = LocalCommitMirror(path)
    await reconcile_history(backend.journal, mirror, pot_id="p")
    with sqlite3.connect(path) as conn:
        raw = conn.execute("SELECT header FROM commits WHERE sequence=5").fetchone()[0]
        header = json.loads(raw)
        header["message"] = "corrupt"
        conn.execute(
            "UPDATE commits SET header=? WHERE sequence=5", (json.dumps(header),)
        )
    with pytest.raises(JournalError, match="hash mismatch"):
        await reconcile_history(backend.journal, mirror, pot_id="p")
    assert (await rebuild_history(backend.journal, mirror, pot_id="p"))["complete"]
    result = await list_history(backend.journal, mirror, pot_id="p")
    assert result["headers"][0]["message"] != "corrupt"
    assert backend.journal.journal_state("p").sequence == 5


async def test_corrupt_older_header_is_detected_without_per_row_native_reads(tmp_path):
    import sqlite3

    backend = seed(tmp_path)
    mirror = LocalCommitMirror(tmp_path / "listing.sqlite")
    await reconcile_history(backend.journal, mirror, pot_id="p")
    with sqlite3.connect(mirror.path) as conn:
        conn.execute(
            "UPDATE commits SET header=json_set(header,'$.receipt_hash','corrupt') WHERE sequence=2"
        )
    with pytest.raises(JournalError, match="hash mismatch"):
        await list_history(backend.journal, mirror, pot_id="p")


async def test_listing_page_stops_at_byte_budget_with_continuation(tmp_path):
    from potpie_context_core.graph_journal import journal_json

    backend = EmbeddedGraphBackend(home=tmp_path)
    backend.journal.activate(pot_id="p")
    for i in range(3):
        backend.mutation.apply(
            MutationBatch(
                entity_upserts=[
                    EntityUpsert(
                        "service:a", ("Entity", "Service"), {"summary": str(i)}
                    )
                ]
            ),
            expected_pot_id="p",
            provenance_context=ProvenanceContext(
                mutation_id=f"c{i}", actor_user_id="a" * 600_000
            ),
        )
    mirror = LocalCommitMirror(tmp_path / "listing.sqlite")
    page = await list_history(backend.journal, mirror, pot_id="p", limit=3)
    assert len(page["headers"]) == 1 and page["next_cursor"]
    assert len(journal_json(page).encode()) <= 1_000_000
    next_page = await list_history(
        backend.journal, mirror, pot_id="p", limit=3, cursor=page["next_cursor"]
    )
    assert next_page["headers"][0]["sequence"] == 2
