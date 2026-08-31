"""Conformance suite for ResourceStorePort and ResourceIndexPort.

Every store/index implementation — local disk today, blob storage and hosted
indexes later — must pass this suite unchanged. New backends add a fixture
param here, nothing else.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pytest

pytestmark = pytest.mark.unit


# --- staging helper ---------------------------------------------------------


def make_staging(
    tmp_path: Path,
    name: str = "stage",
    sections: dict[str, list[str]] | None = None,
) -> Path:
    sections = sections or {"intro": ["Intro text about rollback runbooks."]}
    stage = tmp_path / name
    stage.mkdir(parents=True, exist_ok=True)
    meta_sections = []
    for ordinal, (slug, chunks) in enumerate(sections.items()):
        sdir = stage / slug
        sdir.mkdir(exist_ok=True)
        chunk_refs = []
        for seq, text in enumerate(chunks):
            (sdir / f"{seq:04d}.txt").write_text(text, encoding="utf-8")
            chunk_refs.append({"seq": seq, "label": text[:40] or slug})
        meta_sections.append(
            {
                "slug": slug,
                "title": slug.title(),
                "summary": f"Summary of {slug}.",
                "ordinal": ordinal,
                "content_hash": "",
                "chunks": chunk_refs,
            }
        )
    (stage / "meta.json").write_text(
        json.dumps(
            {
                "source_ref": "file:///doc.md",
                "source_kind": "markdown",
                "sections": meta_sections,
            }
        ),
        encoding="utf-8",
    )
    return stage


# --- store fixtures ---------------------------------------------------------


@pytest.fixture(params=["local", "memory"])
def store_factory(request: pytest.FixtureRequest, tmp_path: Path) -> Callable[[], Any]:
    if request.param == "local":
        from potpie_context_engine.adapters.outbound.resources.local_chunk_store import (
            LocalResourceStore,
        )

        return lambda: LocalResourceStore(home=tmp_path / "home")
    from potpie_context_engine.testing_resources import InMemoryResourceStore

    return lambda: InMemoryResourceStore()


POT = "pot_conf"


def _import(store: Any, tmp_path: Path, doc: str = "guide", **kwargs: Any) -> Any:
    stage = make_staging(tmp_path, name=f"stage-{doc}", **kwargs)
    report = store.import_dir(pot_id=POT, doc_slug=doc, source_dir=stage)
    return report


def test_store_import_then_get_roundtrip(store_factory, tmp_path: Path) -> None:
    store = store_factory()
    report = _import(store, tmp_path)
    assert not report.errors
    chunk = store.get(pot_id=POT, doc_slug="guide", section_slug="intro", seq=0)
    assert chunk.content == "Intro text about rollback runbooks."
    assert chunk.uri == "potpie://res/guide/intro/0000"
    assert chunk.doc_slug == "guide" and chunk.section_slug == "intro" and chunk.seq == 0


def test_store_get_missing_raises_stable_code(store_factory, tmp_path: Path) -> None:
    from potpie_context_core.ports.resource_store import ResourceStoreError

    store = store_factory()
    _import(store, tmp_path)
    with pytest.raises(ResourceStoreError) as err:
        store.get(pot_id=POT, doc_slug="guide", section_slug="intro", seq=9)
    assert err.value.code == "resource_not_found"


def test_store_oversized_chunk_reports_stable_code(store_factory, tmp_path: Path) -> None:
    store = store_factory()
    report = _import(store, tmp_path, doc="big", sections={"body": ["x" * 9000]})
    codes = [e["code"] for e in report.errors]
    assert "resource_chunk_too_large" in codes


def test_store_reimport_reports_removed_sections(store_factory, tmp_path: Path) -> None:
    store = store_factory()
    _import(
        store,
        tmp_path,
        sections={"intro": ["Intro."], "extra": ["Extra text."]},
    )
    report = store.import_dir(
        pot_id=POT,
        doc_slug="guide",
        source_dir=make_staging(tmp_path, name="stage-v2", sections={"intro": ["Intro."]}),
        force=True,
    )
    assert "extra" in report.sections_removed


def test_store_iter_chunks_yields_every_chunk(store_factory, tmp_path: Path) -> None:
    store = store_factory()
    _import(store, tmp_path, sections={"a": ["one", "two"], "b": ["three"]})
    got = {(c.section_slug, c.seq) for c in store.iter_chunks(pot_id=POT, doc_slug="guide")}
    assert got == {("a", 0), ("a", 1), ("b", 0)}


def test_store_delete_removes_document(store_factory, tmp_path: Path) -> None:
    from potpie_context_core.ports.resource_store import ResourceStoreError

    store = store_factory()
    _import(store, tmp_path)
    result = store.delete(pot_id=POT, doc_slug="guide")
    assert result["removed"] is True
    assert all(d["doc_slug"] != "guide" for d in store.list_documents(pot_id=POT))
    with pytest.raises(ResourceStoreError):
        store.get(pot_id=POT, doc_slug="guide", section_slug="intro", seq=0)


def test_store_purge_pot_empties_everything(store_factory, tmp_path: Path) -> None:
    store = store_factory()
    _import(store, tmp_path)
    assert store.purge_pot(POT) is True
    assert store.list_documents(pot_id=POT) == []


def test_store_status_reports_ready_backend(store_factory, tmp_path: Path) -> None:
    store = store_factory()
    status = store.status()
    assert status.ready is True
    assert status.backend


# --- index fixtures ---------------------------------------------------------


@pytest.fixture(params=["sqlite", "memory"])
def index_and_store(request: pytest.FixtureRequest, tmp_path: Path) -> tuple[Any, Any]:
    if request.param == "sqlite":
        from potpie_context_engine.adapters.outbound.resources.fts_index import (
            SqliteFtsResourceIndex,
        )
        from potpie_context_engine.adapters.outbound.resources.local_chunk_store import (
            LocalResourceStore,
        )

        store = LocalResourceStore(home=tmp_path / "home")
        return SqliteFtsResourceIndex(store=store), store
    from potpie_context_engine.testing_resources import (
        InMemoryResourceIndex,
        InMemoryResourceStore,
    )

    store = InMemoryResourceStore()
    return InMemoryResourceIndex(), store


def _indexed_doc(index: Any, store: Any, tmp_path: Path) -> None:
    stage = make_staging(
        tmp_path, name="stage-idx", sections={"ops": ["Rollback steps for payments deploys."]}
    )
    store.import_dir(pot_id=POT, doc_slug="runbook", source_dir=stage)
    index.index_document(
        pot_id=POT,
        doc_slug="runbook",
        chunks=store.iter_chunks(pot_id=POT, doc_slug="runbook"),
    )


def test_index_document_then_search_hits(index_and_store, tmp_path: Path) -> None:
    index, store = index_and_store
    _indexed_doc(index, store, tmp_path)
    hits = index.search(pot_id=POT, query="rollback payments", limit=5)
    assert hits
    assert (hits[0].doc_slug, hits[0].section_slug, hits[0].seq) == ("runbook", "ops", 0)


def test_index_remove_document_clears_hits(index_and_store, tmp_path: Path) -> None:
    index, store = index_and_store
    _indexed_doc(index, store, tmp_path)
    index.remove_document(pot_id=POT, doc_slug="runbook")
    assert index.search(pot_id=POT, query="rollback", limit=5) == []


def test_index_capabilities_and_status(index_and_store, tmp_path: Path) -> None:
    index, _ = index_and_store
    caps = index.capabilities()
    assert caps.lexical is True
    status = index.status()
    assert status.ready is True
    assert status.profile
