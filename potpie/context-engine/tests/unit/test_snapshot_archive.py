from __future__ import annotations

import json
from copy import deepcopy

import pytest

from potpie_context_core.ports.resource_index import IndexReport
from potpie_context_core.ports.resource_store import format_resource_id
from potpie_context_engine.adapters.outbound.graph.backends.in_memory_backend import (
    InMemoryGraphBackend,
)
from potpie_context_engine.adapters.outbound.resources.local_resource_store import (
    LocalResourceStore,
)
from potpie_context_engine.adapters.outbound.resources import snapshot as resource_snapshot
from potpie_context_engine.application.services.resource_facade import ResourceFacade


def _files(text: str, *, source_ref: str = "test:source") -> dict[str, str]:
    return {
        "meta.json": json.dumps(
            {
                "source_ref": source_ref,
                "source_kind": "markdown",
                "sections": [
                    {
                        "slug": "body",
                        "title": "Body",
                        "summary": "The document body.",
                        "ordinal": 0,
                        "content_hash": text,
                        "chunks": [{"seq": 0, "label": "Body"}],
                    }
                ],
            }
        ),
        "body/0000.txt": text,
    }


def _facade(home, *, graph=None, index=None) -> ResourceFacade:
    backend = graph or InMemoryGraphBackend()
    return ResourceFacade(
        store=LocalResourceStore(home=home), snapshot=backend.snapshot, index=index
    )


def _seed_two_revisions(facade: ResourceFacade, pot_id: str, slug: str = "guide"):
    first = facade.store.import_dir(
        pot_id=pot_id, slug=slug, files=_files("revision one")
    )
    second = facade.store.import_dir(
        pot_id=pot_id, slug=slug, files=_files("revision two")
    )
    return first, second


def _text(facade: ResourceFacade, pot_id: str, slug: str, revision: int) -> str:
    resource_id = format_resource_id(slug, "body", 0, revision=revision)
    return facade.store.get(pot_id=pot_id, resource_id=resource_id).text


def test_archive_round_trip_preserves_revision_bytes_and_retry_is_idempotent(tmp_path):
    source = _facade(tmp_path / "source")
    first, second = _seed_two_revisions(source, "source")
    source.snapshot.import_data(
        pot_id="source",
        payload={
            "format_version": "2",
            "pot_id": "source",
            "entities": [
                {"key": "document:guide", "labels": ["Document"], "properties": {"name": "Guide"}}
            ],
            "claims": [],
        },
    )
    archive = source.export_snapshot(pot_id="source")

    target = _facade(tmp_path / "target")
    manifest = target.import_snapshot(pot_id="target", payload=archive)
    target.import_snapshot(pot_id="target", payload=archive)

    assert manifest.metadata["documents"] == 1
    assert _text(target, "target", "guide", first.revision) == "revision one"
    assert _text(target, "target", "guide", second.revision) == "revision two"
    assert target.export_snapshot(pot_id="target")["resources"] == archive["resources"]
    assert target.snapshot.export_data(pot_id="target")["entities"] == [
        {"key": "document:guide", "labels": ["Document"], "properties": {"name": "Guide"}}
    ]


def test_archive_preserves_unrelated_target_documents(tmp_path):
    source = _facade(tmp_path / "source")
    _seed_two_revisions(source, "source")
    archive = source.export_snapshot(pot_id="source")
    target = _facade(tmp_path / "target")
    target.store.import_dir(
        pot_id="target", slug="unrelated", files=_files("keep this")
    )

    target.import_snapshot(pot_id="target", payload=archive)

    assert target.store.documents(pot_id="target") == ("guide", "unrelated")
    assert _text(target, "target", "unrelated", 1) == "keep this"


def test_export_of_missing_pot_has_empty_resources_and_does_not_fail(tmp_path):
    facade = _facade(tmp_path / "home")

    archive = facade.export_snapshot(pot_id="does-not-exist")

    assert archive["resources"] == {"format_version": "1", "files": {}}
    assert archive["entities"] == []
    assert archive["claims"] == []


def test_archive_preserves_chunk_newlines_exactly(tmp_path):
    source = _facade(tmp_path / "source")
    source.store.import_dir(
        pot_id="source", slug="lines", files=_files("first\nsecond\n\n")
    )
    archive = source.export_snapshot(pot_id="source")
    chunk_path = "lines/body/0000.txt"
    assert archive["resources"]["files"][chunk_path] == "first\nsecond\n\n"

    target = _facade(tmp_path / "target")
    target.import_snapshot(pot_id="target", payload=archive)

    assert target.store.export_snapshot(pot_id="target")["files"][chunk_path] == "first\nsecond\n\n"


def test_conflicting_document_refuses_before_graph_or_bytes_change(tmp_path):
    source = _facade(tmp_path / "source")
    _seed_two_revisions(source, "source")
    archive = source.export_snapshot(pot_id="source")
    graph = InMemoryGraphBackend()
    target = _facade(tmp_path / "target", graph=graph)
    target.store.import_dir(
        pot_id="target", slug="guide", files=_files("target version")
    )
    graph_before = graph.snapshot.export_data(pot_id="target")
    bytes_before = target.store.export_snapshot(pot_id="target")

    with pytest.raises(ValueError, match="document conflicts"):
        target.import_snapshot(pot_id="target", payload=archive)

    assert graph.snapshot.export_data(pot_id="target") == graph_before
    assert target.store.export_snapshot(pot_id="target") == bytes_before


def test_graph_conflict_refuses_before_resource_publication(tmp_path):
    source = _facade(tmp_path / "source")
    _seed_two_revisions(source, "source")
    source.snapshot.import_data(
        pot_id="source",
        payload={
            "format_version": "2",
            "pot_id": "source",
            "entities": [{"key": "shared", "labels": ["Entity"], "properties": {"side": "source"}}],
            "claims": [],
        },
    )
    graph = InMemoryGraphBackend()
    graph.snapshot.import_data(
        pot_id="target",
        payload={
            "format_version": "2",
            "pot_id": "target",
            "entities": [{"key": "shared", "labels": ["Entity"], "properties": {"side": "target"}}],
            "claims": [],
        },
    )
    target = _facade(tmp_path / "target", graph=graph)
    target.store.import_dir(pot_id="target", slug="unrelated", files=_files("keep this"))
    before = target.store.export_snapshot(pot_id="target")

    with pytest.raises(ValueError, match="entity conflicts"):
        target.import_snapshot(pot_id="target", payload=source.export_snapshot(pot_id="source"))

    assert target.store.export_snapshot(pot_id="target") == before


@pytest.mark.parametrize(
    "mutation",
    [
        lambda files: files.pop("guide/body/0000.txt"),
        lambda files: files.__setitem__("../escape.txt", "bad"),
    ],
)
def test_malformed_resources_refuse_before_graph_mutation(tmp_path, mutation):
    source = _facade(tmp_path / "source")
    _seed_two_revisions(source, "source")
    archive = deepcopy(source.export_snapshot(pot_id="source"))
    mutation(archive["resources"]["files"])
    target_graph = InMemoryGraphBackend()
    target = _facade(tmp_path / "target", graph=target_graph)
    before = target_graph.snapshot.export_data(pot_id="target")

    with pytest.raises(ValueError):
        target.import_snapshot(pot_id="target", payload=archive)

    assert target_graph.snapshot.export_data(pot_id="target") == before
    assert target.store.documents(pot_id="target") == ()


class _FailingSnapshot:
    def __init__(self) -> None:
        self.inner = InMemoryGraphBackend().snapshot

    def export_data(self, *, pot_id: str):
        return self.inner.export_data(pot_id=pot_id)

    def import_data(self, *, pot_id: str, payload):
        raise RuntimeError("graph import failed")


def test_graph_failure_rolls_resource_tree_back(tmp_path):
    source = _facade(tmp_path / "source")
    _seed_two_revisions(source, "source")
    archive = source.export_snapshot(pot_id="source")
    target = ResourceFacade(
        store=LocalResourceStore(home=tmp_path / "target"), snapshot=_FailingSnapshot()
    )
    target.store.import_dir(
        pot_id="target", slug="unrelated", files=_files("keep this")
    )
    before = target.store.export_snapshot(pot_id="target")

    with pytest.raises(RuntimeError, match="graph import failed"):
        target.import_snapshot(pot_id="target", payload=archive)

    assert target.store.export_snapshot(pot_id="target") == before


def test_second_document_publish_failure_rolls_back_first_and_keeps_old_data(
    tmp_path, monkeypatch
):
    source = _facade(tmp_path / "source")
    source.store.import_dir(pot_id="source", slug="alpha", files=_files("alpha"))
    source.store.import_dir(pot_id="source", slug="beta", files=_files("beta"))
    archive = source.export_snapshot(pot_id="source")
    target = _facade(tmp_path / "target")
    target.store.import_dir(
        pot_id="target", slug="existing", files=_files("must survive")
    )
    before = target.store.export_snapshot(pot_id="target")
    real_replace = resource_snapshot.os.replace
    publications = 0

    def fail_second_publication(source_path, destination_path):
        nonlocal publications
        if destination_path.parent == target.store._pot_root("target"):
            publications += 1
            if publications == 2:
                raise OSError("synthetic second publish failure")
        return real_replace(source_path, destination_path)

    monkeypatch.setattr(resource_snapshot.os, "replace", fail_second_publication)

    with pytest.raises(OSError, match="second publish failure"):
        target.import_snapshot(pot_id="target", payload=archive)

    assert publications == 2
    assert target.store.documents(pot_id="target") == ("existing",)
    assert target.store.export_snapshot(pot_id="target") == before
    assert _text(target, "target", "existing", 1) == "must survive"


class _RecordingIndex:
    def __init__(self) -> None:
        self.dropped: list[str] = []
        self.indexed: list[str] = []

    def drop_document(self, *, pot_id: str, slug: str) -> int:
        self.dropped.append(slug)
        return 0

    def index_document(self, *, pot_id: str, manifest, chunks) -> IndexReport:
        self.indexed.append(manifest.doc)
        return IndexReport(doc=manifest.doc, profile="test", chunks=len(chunks))


def test_successful_restore_rebuilds_index_for_all_target_documents(tmp_path):
    source = _facade(tmp_path / "source")
    _seed_two_revisions(source, "source")
    index = _RecordingIndex()
    target = _facade(tmp_path / "target", index=index)
    target.store.import_dir(
        pot_id="target", slug="unrelated", files=_files("keep this")
    )

    target.import_snapshot(pot_id="target", payload=source.export_snapshot(pot_id="source"))

    assert index.dropped == ["guide", "unrelated"]
    assert index.indexed == ["guide", "unrelated"]
