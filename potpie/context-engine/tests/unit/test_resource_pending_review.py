from pathlib import Path

import pytest

from potpie_context_engine.adapters.outbound.resources.local_resource_store import (
    LocalResourceStore,
)
from potpie_context_engine.testing import InMemoryResourceStore, write_import_directory


def _source(root: Path, slug: str) -> Path:
    return write_import_directory(
        root,
        [
            {
                "slug": slug,
                "title": slug,
                "summary": slug,
                "ordinal": 0,
                "content_hash": slug,
                "chunks": [{"seq": 0, "label": slug, "text": slug}],
            }
        ],
    )


@pytest.mark.parametrize("kind", ["memory", "local"])
def test_changed_revision_publishes_pending_review_and_identical_retry_keeps_it(
    tmp_path: Path, kind: str
) -> None:
    store = (
        InMemoryResourceStore()
        if kind == "memory"
        else LocalResourceStore(home=tmp_path)
    )
    first = _source(tmp_path / "first", "body")
    store.import_dir(pot_id="pot", slug="guide", source_dir=first)
    second = _source(tmp_path / "second", "other")

    changed = store.import_dir(pot_id="pot", slug="guide", source_dir=second)
    assert changed.pending_review_refs
    assert changed.pending_review_sections == ("body",)

    retried = store.import_dir(pot_id="pot", slug="guide", source_dir=second)
    assert retried.revision == changed.revision
    assert retried.pending_review_refs == changed.pending_review_refs


@pytest.mark.parametrize("kind", ["memory", "local"])
def test_stale_clear_cannot_erase_newer_revision_pending_review(
    tmp_path: Path, kind: str
) -> None:
    store = (
        InMemoryResourceStore()
        if kind == "memory"
        else LocalResourceStore(home=tmp_path)
    )
    first = _source(tmp_path / "first", "one")
    second = _source(tmp_path / "second", "two")
    third = _source(tmp_path / "third", "three")
    store.import_dir(pot_id="pot", slug="guide", source_dir=first)
    rev2 = store.import_dir(pot_id="pot", slug="guide", source_dir=second)
    rev3 = store.import_dir(pot_id="pot", slug="guide", source_dir=third)

    observed = store.clear_pending_review(
        pot_id="pot",
        slug="guide",
        expected_revision=rev2.revision,
        expected_refs=rev2.pending_review_refs,
    )

    assert observed.revision == rev3.revision
    assert observed.pending_review_refs == rev3.pending_review_refs
    assert store.current_manifest(pot_id="pot", slug="guide").pending_review_refs
