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
    assert all(d.doc_slug != "guide" for d in store.list_documents(pot_id=POT))
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
        return SqliteFtsResourceIndex(home=tmp_path / "home"), store
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


# --- contract additions: agent-driven writes, typed catalog, wiring hooks ----


def test_store_put_document_writes_without_a_staging_tree(store_factory, tmp_path: Path) -> None:
    """Agents and RPC callers hold content in memory; the port must accept it
    directly — fabricating an on-disk staging tree is not a contract."""
    from potpie_context_core.ports.resource_store import ChunkWrite
    from potpie_context_core.resource_models import ResourceManifest

    store = store_factory()
    manifest = ResourceManifest.model_validate(
        {
            "source_ref": "agent://session/1",
            "source_kind": "markdown",
            "sections": [
                {
                    "slug": "notes",
                    "title": "Notes",
                    "summary": "Agent notes about zebra migration.",
                    "ordinal": 0,
                    "content_hash": "",
                    "chunks": [{"seq": 0, "label": "notes"}],
                }
            ],
        }
    )
    report = store.put_document(
        pot_id=POT,
        doc_slug="agent-doc",
        manifest=manifest,
        chunks=[ChunkWrite(section_slug="notes", seq=0, content="Zebra migration notes.")],
    )
    assert not report.errors
    assert "notes" in report.sections_added
    chunk = store.get(pot_id=POT, doc_slug="agent-doc", section_slug="notes", seq=0)
    assert chunk.content == "Zebra migration notes."


def test_store_list_documents_returns_typed_document_info(store_factory, tmp_path: Path) -> None:
    from potpie_context_core.ports.resource_store import DocumentInfo

    store = store_factory()
    _import(store, tmp_path)
    docs = store.list_documents(pot_id=POT)
    assert docs and isinstance(docs[0], DocumentInfo)
    info = docs[0]
    assert info.doc_slug == "guide"
    assert info.section_count == 1
    payload = info.to_payload()
    assert set(payload) == {
        "doc_slug",
        "source_ref",
        "source_kind",
        "revision",
        "updated_at",
        "section_count",
    }


def test_store_list_sections_returns_slugs(store_factory, tmp_path: Path) -> None:
    store = store_factory()
    _import(store, tmp_path, sections={"intro": ["Intro."], "ops": ["Ops text."]})
    sections = store.list_sections(pot_id=POT, doc_slug="guide")
    assert {s.slug for s in sections} == {"intro", "ops"}


def test_store_staging_root_is_a_writable_path(store_factory, tmp_path: Path) -> None:
    store = store_factory()
    root = store.staging_root(pot_id=POT)
    root.mkdir(parents=True, exist_ok=True)
    (root / "probe.txt").write_text("ok", encoding="utf-8")
    assert (root / "probe.txt").read_text(encoding="utf-8") == "ok"


def test_store_default_index_satisfies_the_index_port(store_factory, tmp_path: Path) -> None:
    store = store_factory()
    index = store.default_index()
    assert index is not None
    assert index.capabilities().lexical is True


def test_sqlite_index_works_over_a_non_local_store(tmp_path: Path) -> None:
    """The S3-bytes + local-FTS hybrid: the index must not require the local store."""
    from potpie_context_engine.adapters.outbound.resources.fts_index import (
        SqliteFtsResourceIndex,
    )
    from potpie_context_engine.testing_resources import InMemoryResourceStore

    store = InMemoryResourceStore()
    stage = make_staging(tmp_path, name="stage-hybrid", sections={"ops": ["Rollback for payments."]})
    store.import_dir(pot_id=POT, doc_slug="runbook", source_dir=stage)

    index = SqliteFtsResourceIndex(home=tmp_path / "index-home")
    index.index_document(
        pot_id=POT,
        doc_slug="runbook",
        chunks=store.iter_chunks(pot_id=POT, doc_slug="runbook"),
    )
    hits = index.search(pot_id=POT, query="payments rollback", limit=5)
    assert hits and hits[0].doc_slug == "runbook"
