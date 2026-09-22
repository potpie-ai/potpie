"""Evidence bytes remain immutable across storage lifecycle operations."""

from concurrent.futures import ThreadPoolExecutor
import json
from threading import Barrier, Event, get_ident

import pytest

from potpie_context_core.ports.resource_store import (
    ResourceStoreError,
    format_resource_id,
)
from potpie_context_engine.adapters.outbound.resources import LocalResourceStore
from potpie_context_engine.testing import InMemoryResourceStore

POT = "revision-acceptance"
DOC = "retry-runbook"


@pytest.fixture(params=("local", "in_memory"))
def store(request, tmp_path):
    return (
        LocalResourceStore(home=tmp_path)
        if request.param == "local"
        else InMemoryResourceStore()
    )


def files(text, source_ref="synthetic:first"):
    return {
        "meta.json": json.dumps(
            {
                "source_ref": source_ref,
                "source_kind": "markdown",
                "sections": [
                    {
                        "slug": "body",
                        "title": "Recovery",
                        "summary": "Recovery instructions.",
                        "ordinal": 0,
                        "content_hash": "caller-reused-this-hash",
                        "chunks": [{"seq": 0, "label": "Recovery"}],
                    }
                ],
            }
        ),
        "body/0000.txt": text,
    }


def uri(revision):
    return format_resource_id(DOC, "body", 0, revision=revision)


def test_changed_bytes_with_reused_hash_preserve_original_text_and_metadata(store):
    original = store.import_dir(pot_id=POT, slug=DOC, files=files("Wait 47 minutes."))
    replacement = store.import_dir(pot_id=POT, slug=DOC, files=files("Wait 5 minutes."))

    assert replacement.revision > original.revision
    assert "body" not in replacement.sections_kept
    assert (
        store.get(pot_id=POT, resource_id=uri(original.revision)).text
        == "Wait 47 minutes."
    )
    assert (
        store.get(pot_id=POT, resource_id=uri(replacement.revision)).text
        == "Wait 5 minutes."
    )
    metadata_change = store.import_dir(
        pot_id=POT, slug=DOC, files=files("Wait 5 minutes.", "synthetic:replacement")
    )
    assert metadata_change.revision > replacement.revision
    assert (
        store.get(pot_id=POT, resource_id=uri(replacement.revision)).source_ref
        == "synthetic:first"
    )
    assert (
        store.get(pot_id=POT, resource_id=uri(metadata_change.revision)).source_ref
        == "synthetic:replacement"
    )


@pytest.mark.parametrize("delete_kind", ("document", "pot"))
def test_deletion_never_reuses_a_citation_revision(store, delete_kind):
    original = store.import_dir(pot_id=POT, slug=DOC, files=files("Wait 47 minutes."))
    store.import_dir(pot_id="other-pot", slug=DOC, files=files("Other pot."))
    if delete_kind == "document":
        assert store.delete(pot_id=POT, slug=DOC)
    else:
        assert store.purge_pot(POT)

    replacement = store.import_dir(pot_id=POT, slug=DOC, files=files("Wait 5 minutes."))

    assert replacement.revision > original.revision
    with pytest.raises(ResourceStoreError) as missing:
        store.get(pot_id=POT, resource_id=uri(original.revision))
    assert missing.value.code == "resource_not_found"
    with pytest.raises(ResourceStoreError) as ambiguous:
        store.get(pot_id=POT, resource_id=format_resource_id(DOC, "body", 0))
    assert ambiguous.value.code == "resource_revision_ambiguous"
    assert store.get(pot_id="other-pot", resource_id=uri(1)).text == "Other pot."


def test_concurrent_imports_allocate_distinct_immutable_versions(store):
    barrier = Barrier(4)
    stores = (
        [LocalResourceStore(home=store.home) for _ in range(4)]
        if isinstance(store, LocalResourceStore)
        else [store] * 4
    )

    def import_one(index):
        text = f"Wait {index} minutes."
        barrier.wait(timeout=10)
        result = stores[index].import_dir(pot_id=POT, slug=DOC, files=files(text))
        return result.revision, text

    with ThreadPoolExecutor(max_workers=4) as pool:
        versions = list(pool.map(import_one, range(4)))

    assert len({revision for revision, _ in versions}) == 4
    for revision, text in versions:
        assert store.get(pot_id=POT, resource_id=uri(revision)).text == text


def test_reader_cannot_return_new_bytes_under_old_revision_during_swap(
    tmp_path, monkeypatch
):
    from potpie_context_engine.adapters.outbound.resources import (
        local_resource_store as storage,
    )

    store = LocalResourceStore(home=tmp_path)
    writer = LocalResourceStore(home=tmp_path)
    store.import_dir(pot_id=POT, slug=DOC, files=files("Wait 47 minutes."))
    reading = Event()
    writing = Event()
    written = Event()
    reader_thread = get_ident()
    read_text = storage._read_text

    def read_during_refresh(path):
        if (
            get_ident() == reader_thread
            and path.name == "0000.txt"
            and not reading.is_set()
        ):
            reading.set()
            assert writing.wait(5)
            # A correctly locked writer waits for this read to finish. An
            # unlocked writer can replace the path after version resolution.
            written.wait(0.5)
        return read_text(path)

    def refresh():
        assert reading.wait(5)
        writing.set()
        writer.import_dir(pot_id=POT, slug=DOC, files=files("Wait 5 minutes."))
        written.set()

    monkeypatch.setattr(storage, "_read_text", read_during_refresh)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(refresh)
        original = store.get(pot_id=POT, resource_id=uri(1))
        future.result(timeout=10)

    assert original.revision == 1
    assert original.text == "Wait 47 minutes."
