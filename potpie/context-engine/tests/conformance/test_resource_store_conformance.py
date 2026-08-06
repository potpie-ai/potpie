"""ResourceStorePort conformance.

Every resource store must answer the same contract, whatever it keeps the bytes
on. ``local`` writes files under the Potpie home; ``in_memory`` is the stub
third parties copy when building their own. Both run every test here, so a
behavior that only the filesystem happens to provide cannot leak into the
contract — that is what R9 ("storage is swappable behind one port") means in
practice.

Tests that are genuinely about the on-disk representation — the pot directory
encoding, the temp-dir-plus-rename swap — are marked local-only and say why.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from potpie_context_core.api import (
    DEFAULT_SECTION_SLUG,
    RECOMMENDED_MAX_SECTION_CHUNKS,
    RESOURCE_CHUNK_MAX_CHARS,
    ResourceStoreError,
    format_resource_id,
)
from potpie_context_core.ports.resource_store import (
    RESOURCE_CHUNK_TOO_LARGE,
    RESOURCE_LABEL_MAX_CHARS,
    RESOURCE_MANIFEST_INVALID,
    RESOURCE_NOT_FOUND,
    RESOURCE_SECTION_MISSING_CHUNK,
    RESOURCE_SLUG_INVALID,
    RESOURCE_SUMMARY_MAX_CHARS,
    RESOURCE_TEXT_TOO_LARGE,
)
from potpie_context_engine.adapters.outbound.resources import (
    LocalResourceStore,
    pot_dir_name,
)
from potpie_context_engine.testing import (
    InMemoryResourceStore,
    run_resource_store_conformance,
    write_import_directory,
)

STORES = ["local", "in_memory"]

POT = "conformance:resources"
OTHER_POT = "conformance:resources-other"
DOC = "q3-review"


def _build(kind, tmp_path):
    if kind == "local":
        return LocalResourceStore(home=tmp_path / "home")
    return InMemoryResourceStore()


def _section(slug, *, chunks, summary="what this section covers", **overrides):
    section = {
        "slug": slug,
        "title": slug.replace("-", " ").title(),
        "summary": summary,
        "ordinal": 0,
        "content_hash": f"{slug}-v1",
        "chunks": chunks,
    }
    section.update(overrides)
    return section


def _simple_dir(root, *, texts=("alpha", "omega"), **overrides):
    return write_import_directory(
        root,
        [
            _section(
                "body",
                chunks=[
                    {"label": f"part {index}", "text": text}
                    for index, text in enumerate(texts)
                ],
                **overrides,
            )
        ],
        source_ref="file:///q3.pdf",
        source_kind="pdf",
    )


@pytest.mark.parametrize("kind", STORES)
def test_store_satisfies_the_shared_conformance_suite(kind, tmp_path):
    run_resource_store_conformance(lambda: _build(kind, tmp_path))


@pytest.mark.parametrize("kind", STORES)
def test_import_then_get_round_trips_the_chunk_and_its_provenance(kind, tmp_path):
    store = _build(kind, tmp_path)
    store.import_dir(
        pot_id=POT, slug=DOC, source_dir=_simple_dir(tmp_path / "src", texts=("alpha",))
    )

    chunk = store.get(pot_id=POT, resource_id=format_resource_id(DOC, "body", 0))

    assert chunk.text == "alpha"
    assert chunk.chars == 5
    assert (chunk.doc, chunk.section, chunk.seq) == (DOC, "body", 0)
    assert chunk.revision == 1
    assert chunk.source_ref == "file:///q3.pdf"


@pytest.mark.parametrize("kind", STORES)
def test_get_many_answers_in_the_order_requested(kind, tmp_path):
    store = _build(kind, tmp_path)
    store.import_dir(
        pot_id=POT,
        slug=DOC,
        source_dir=_simple_dir(tmp_path / "src", texts=("a", "b", "c")),
    )
    ids = tuple(format_resource_id(DOC, "body", seq) for seq in (2, 0, 1))

    chunks = store.get_many(pot_id=POT, resource_ids=ids)

    assert tuple(chunk.text for chunk in chunks) == ("c", "a", "b")
    assert tuple(chunk.resource_id for chunk in chunks) == ids


@pytest.mark.parametrize("kind", STORES)
def test_resource_ids_are_zero_padded(kind, tmp_path):
    store = _build(kind, tmp_path)
    store.import_dir(
        pot_id=POT, slug=DOC, source_dir=_simple_dir(tmp_path / "src", texts=("alpha",))
    )

    chunk = store.get(pot_id=POT, resource_id=f"potpie://res/{DOC}/body/0000")

    assert chunk.resource_id == f"potpie://res/{DOC}/body/0000"


@pytest.mark.parametrize("kind", STORES)
def test_oversized_chunk_is_rejected_and_the_prior_document_survives(kind, tmp_path):
    store = _build(kind, tmp_path)
    store.import_dir(
        pot_id=POT, slug=DOC, source_dir=_simple_dir(tmp_path / "v1", texts=("alpha",))
    )
    oversized = _simple_dir(
        tmp_path / "v2", texts=("x" * (RESOURCE_CHUNK_MAX_CHARS + 1),)
    )

    with pytest.raises(ResourceStoreError) as exc:
        store.import_dir(pot_id=POT, slug=DOC, source_dir=oversized)

    assert exc.value.code == RESOURCE_CHUNK_TOO_LARGE
    # R6: the refused import must not be half-applied.
    surviving = store.get(pot_id=POT, resource_id=format_resource_id(DOC, "body", 0))
    assert surviving.text == "alpha"
    assert surviving.revision == 1


@pytest.mark.parametrize("kind", STORES)
def test_chunk_exactly_at_the_cap_is_accepted(kind, tmp_path):
    store = _build(kind, tmp_path)
    text = "y" * RESOURCE_CHUNK_MAX_CHARS

    store.import_dir(
        pot_id=POT, slug=DOC, source_dir=_simple_dir(tmp_path / "src", texts=(text,))
    )

    chunk = store.get(pot_id=POT, resource_id=format_resource_id(DOC, "body", 0))
    assert chunk.chars == RESOURCE_CHUNK_MAX_CHARS


@pytest.mark.parametrize("kind", STORES)
def test_invalid_document_slug_is_rejected(kind, tmp_path):
    store = _build(kind, tmp_path)
    source = _simple_dir(tmp_path / "src")

    for bad in ("../escape", "Q3 Review", "", "a/b"):
        with pytest.raises(ResourceStoreError) as exc:
            store.import_dir(pot_id=POT, slug=bad, source_dir=source)
        assert exc.value.code == RESOURCE_SLUG_INVALID


@pytest.mark.parametrize("kind", STORES)
def test_invalid_section_slug_is_rejected(kind, tmp_path):
    store = _build(kind, tmp_path)
    source = write_import_directory(
        tmp_path / "src",
        [_section("Not A Slug", chunks=[{"label": "only", "text": "alpha"}])],
    )

    with pytest.raises(ResourceStoreError) as exc:
        store.import_dir(pot_id=POT, slug=DOC, source_dir=source)

    assert exc.value.code == RESOURCE_SLUG_INVALID


@pytest.mark.parametrize("kind", STORES)
def test_chunk_without_a_label_is_rejected(kind, tmp_path):
    store = _build(kind, tmp_path)
    source = write_import_directory(
        tmp_path / "src", [_section("body", chunks=[{"text": "alpha"}])]
    )

    with pytest.raises(ResourceStoreError) as exc:
        store.import_dir(pot_id=POT, slug=DOC, source_dir=source)

    assert exc.value.code == RESOURCE_MANIFEST_INVALID
    assert "label" in str(exc.value)


@pytest.mark.parametrize("kind", STORES)
def test_section_naming_a_missing_chunk_file_is_rejected(kind, tmp_path):
    store = _build(kind, tmp_path)
    source = write_import_directory(
        tmp_path / "src",
        [
            _section(
                "body",
                chunks=[
                    {"label": "present", "text": "alpha"},
                    {"label": "absent"},  # no text -> no file on disk
                ],
            )
        ],
    )

    with pytest.raises(ResourceStoreError) as exc:
        store.import_dir(pot_id=POT, slug=DOC, source_dir=source)

    assert exc.value.code == RESOURCE_SECTION_MISSING_CHUNK


@pytest.mark.parametrize("kind", STORES)
def test_duplicate_section_slug_is_rejected(kind, tmp_path):
    store = _build(kind, tmp_path)
    source = write_import_directory(
        tmp_path / "src",
        [
            _section("body", chunks=[{"label": "one", "text": "alpha"}]),
            _section("body", chunks=[{"label": "two", "text": "omega"}], ordinal=1),
        ],
    )

    with pytest.raises(ResourceStoreError) as exc:
        store.import_dir(pot_id=POT, slug=DOC, source_dir=source)

    assert exc.value.code == RESOURCE_MANIFEST_INVALID


@pytest.mark.parametrize("kind", STORES)
def test_reimport_bumps_the_revision_and_drops_the_old_chunks(kind, tmp_path):
    store = _build(kind, tmp_path)
    store.import_dir(
        pot_id=POT,
        slug=DOC,
        source_dir=_simple_dir(tmp_path / "v1", texts=("alpha", "omega")),
    )

    manifest = store.import_dir(
        pot_id=POT,
        slug=DOC,
        source_dir=_simple_dir(
            tmp_path / "v2", texts=("rewritten",), content_hash="body-v2"
        ),
    )

    assert manifest.revision == 2
    assert store.get(
        pot_id=POT, resource_id=format_resource_id(DOC, "body", 0)
    ).text == ("rewritten")
    with pytest.raises(ResourceStoreError) as exc:
        store.get(pot_id=POT, resource_id=format_resource_id(DOC, "body", 1))
    assert exc.value.code == RESOURCE_NOT_FOUND


@pytest.mark.parametrize("kind", STORES)
def test_reimport_of_an_unchanged_document_does_not_bump_the_revision(kind, tmp_path):
    # P1-3: the counter R7 hangs prior-revision claim invalidation on must mean
    # "the content moved", not "somebody ran the command". Re-running the
    # extraction script — which agents do, e.g. to re-read the report as JSON —
    # used to burn a revision and rewrite every section's revision property.
    store = _build(kind, tmp_path)
    store.import_dir(pot_id=POT, slug=DOC, source_dir=_simple_dir(tmp_path / "v1"))

    second = store.import_dir(
        pot_id=POT, slug=DOC, source_dir=_simple_dir(tmp_path / "v1-again")
    )
    third = store.import_dir(
        pot_id=POT, slug=DOC, source_dir=_simple_dir(tmp_path / "v1-again-again")
    )

    assert second.revision == 1
    assert third.revision == 1
    assert second.sections_kept == ("body",)
    assert not second.sections_added and not second.sections_removed


@pytest.mark.parametrize("kind", STORES)
@pytest.mark.parametrize(
    "second_dir",
    [
        # content changed in place
        pytest.param(
            lambda root: _simple_dir(
                root, texts=("rewritten",), content_hash="body-v2"
            ),
            id="changed",
        ),
        # a section arrived
        pytest.param(
            lambda root: write_import_directory(
                root,
                [
                    _section("body", chunks=[{"label": "part 0", "text": "alpha"}]),
                    _section(
                        "risks", chunks=[{"label": "one", "text": "gamma"}], ordinal=1
                    ),
                ],
            ),
            id="added",
        ),
    ],
)
def test_reimport_bumps_the_revision_when_the_section_set_moves(
    kind, second_dir, tmp_path
):
    store = _build(kind, tmp_path)
    store.import_dir(
        pot_id=POT, slug=DOC, source_dir=_simple_dir(tmp_path / "v1", texts=("alpha",))
    )

    manifest = store.import_dir(
        pot_id=POT, slug=DOC, source_dir=second_dir(tmp_path / "v2")
    )

    assert manifest.revision == 2


@pytest.mark.parametrize("kind", STORES)
def test_reimport_that_only_removes_a_section_bumps_the_revision(kind, tmp_path):
    # A removal is what retracts SECTION_OF claims, so it must be a new revision
    # even though nothing that survived changed.
    store = _build(kind, tmp_path)
    first = write_import_directory(
        tmp_path / "v1",
        [
            _section("body", chunks=[{"label": "one", "text": "alpha"}]),
            _section("capacity", chunks=[{"label": "two", "text": "omega"}], ordinal=1),
        ],
    )
    second = write_import_directory(
        tmp_path / "v2",
        [_section("body", chunks=[{"label": "one", "text": "alpha"}])],
    )
    store.import_dir(pot_id=POT, slug=DOC, source_dir=first)

    manifest = store.import_dir(pot_id=POT, slug=DOC, source_dir=second)

    assert manifest.revision == 2
    assert manifest.sections_removed == ("capacity",)


@pytest.mark.parametrize("kind", STORES)
def test_reimport_reports_added_kept_and_removed_sections(kind, tmp_path):
    store = _build(kind, tmp_path)
    first = write_import_directory(
        tmp_path / "v1",
        [
            _section("body", chunks=[{"label": "one", "text": "alpha"}]),
            _section("capacity", chunks=[{"label": "two", "text": "omega"}], ordinal=1),
        ],
    )
    second = write_import_directory(
        tmp_path / "v2",
        [
            # unchanged content_hash -> kept
            _section("body", chunks=[{"label": "one", "text": "alpha"}]),
            # capacity is gone, risks is new
            _section("risks", chunks=[{"label": "three", "text": "gamma"}], ordinal=1),
        ],
    )
    store.import_dir(pot_id=POT, slug=DOC, source_dir=first)

    manifest = store.import_dir(pot_id=POT, slug=DOC, source_dir=second)

    assert manifest.sections_added == ("risks",)
    assert manifest.sections_kept == ("body",)
    assert manifest.sections_removed == ("capacity",)


@pytest.mark.parametrize("kind", STORES)
def test_changed_section_is_derivable_from_the_manifest(kind, tmp_path):
    # R14: only sections whose content actually moved need re-summarizing, and
    # the import report is what tells a caller which those are.
    store = _build(kind, tmp_path)
    first = write_import_directory(
        tmp_path / "v1",
        [
            _section("body", chunks=[{"label": "one", "text": "alpha"}]),
            _section("capacity", chunks=[{"label": "two", "text": "omega"}], ordinal=1),
        ],
    )
    second = write_import_directory(
        tmp_path / "v2",
        [
            _section("body", chunks=[{"label": "one", "text": "alpha"}]),
            _section(
                "capacity",
                chunks=[{"label": "two", "text": "rewritten"}],
                ordinal=1,
                content_hash="capacity-v2",
            ),
        ],
    )
    store.import_dir(pot_id=POT, slug=DOC, source_dir=first)

    manifest = store.import_dir(pot_id=POT, slug=DOC, source_dir=second)

    present = {section.slug for section in manifest.sections}
    changed = present - set(manifest.sections_added) - set(manifest.sections_kept)
    assert changed == {"capacity"}


@pytest.mark.parametrize("kind", STORES)
def test_section_without_a_summary_imports_as_pending(kind, tmp_path):
    # Two-pass ingest: the script splits, the agent summarizes later.
    store = _build(kind, tmp_path)
    source = write_import_directory(
        tmp_path / "src",
        [_section("body", chunks=[{"label": "one", "text": "alpha"}], summary="")],
    )

    manifest = store.import_dir(pot_id=POT, slug=DOC, source_dir=source)

    assert [section.summary_pending for section in manifest.sections] == [True]
    assert store.list(pot_id=POT, slug=DOC)[0].summary_pending is True


@pytest.mark.parametrize("kind", STORES)
def test_reimport_keeps_the_summary_of_an_unchanged_section(kind, tmp_path):
    # R14: re-running the extraction script emits summaries empty, because a
    # script can split but cannot judge. A section whose content_hash did not
    # move must not lose the index (R12) it already had.
    store = _build(kind, tmp_path)
    store.import_dir(
        pot_id=POT,
        slug=DOC,
        source_dir=_simple_dir(
            tmp_path / "v1", texts=("alpha",), summary="400 rps capacity ceiling"
        ),
    )

    manifest = store.import_dir(
        pot_id=POT,
        slug=DOC,
        source_dir=_simple_dir(tmp_path / "v2", texts=("alpha",), summary=""),
    )

    assert manifest.sections_kept == ("body",)
    assert manifest.sections[0].summary == "400 rps capacity ceiling"
    assert manifest.sections[0].summary_pending is False
    assert store.list(pot_id=POT, slug=DOC)[0].summary == "400 rps capacity ceiling"


@pytest.mark.parametrize("kind", STORES)
def test_reimport_of_changed_content_takes_the_new_summary(kind, tmp_path):
    store = _build(kind, tmp_path)
    store.import_dir(
        pot_id=POT,
        slug=DOC,
        source_dir=_simple_dir(
            tmp_path / "v1", texts=("alpha",), summary="the old one"
        ),
    )

    manifest = store.import_dir(
        pot_id=POT,
        slug=DOC,
        source_dir=_simple_dir(
            tmp_path / "v2", texts=("rewritten",), summary="", content_hash="body-v2"
        ),
    )

    assert manifest.sections_kept == ()
    assert manifest.sections[0].summary == ""
    assert manifest.sections[0].summary_pending is True


@pytest.mark.parametrize("kind", STORES)
def test_manifest_text_destined_for_the_graph_is_bounded(kind, tmp_path):
    # R1: import is the only step that reads the directory, so a section body
    # pasted into summary is stopped here or not at all.
    store = _build(kind, tmp_path)
    source = _simple_dir(
        tmp_path / "src",
        texts=("alpha",),
        summary="x" * (RESOURCE_SUMMARY_MAX_CHARS + 1),
    )

    with pytest.raises(ResourceStoreError) as exc:
        store.import_dir(pot_id=POT, slug=DOC, source_dir=source)

    assert exc.value.code == RESOURCE_TEXT_TOO_LARGE


@pytest.mark.parametrize("kind", STORES)
def test_oversized_chunk_label_is_rejected(kind, tmp_path):
    store = _build(kind, tmp_path)
    source = write_import_directory(
        tmp_path / "src",
        [
            _section(
                "body",
                chunks=[
                    {"label": "l" * (RESOURCE_LABEL_MAX_CHARS + 1), "text": "alpha"}
                ],
            )
        ],
    )

    with pytest.raises(ResourceStoreError) as exc:
        store.import_dir(pot_id=POT, slug=DOC, source_dir=source)

    assert exc.value.code == RESOURCE_TEXT_TOO_LARGE


@pytest.mark.parametrize("kind", STORES)
def test_list_returns_sections_with_their_chunk_labels(kind, tmp_path):
    store = _build(kind, tmp_path)
    source = write_import_directory(
        tmp_path / "src",
        [
            _section(
                "body",
                chunks=[
                    {"label": "opening", "text": "alpha", "page": 1},
                    {"label": "closing", "text": "omega", "offset": 42},
                ],
            ),
            _section(
                "risks", chunks=[{"label": "top risk", "text": "gamma"}], ordinal=1
            ),
        ],
    )
    store.import_dir(pot_id=POT, slug=DOC, source_dir=source)

    sections = store.list(pot_id=POT, slug=DOC)

    assert [section.slug for section in sections] == ["body", "risks"]
    assert [ref.label for ref in sections[0].chunks] == ["opening", "closing"]
    assert sections[0].chunks[0].page == 1
    assert sections[0].chunks[1].offset == 42


@pytest.mark.parametrize("kind", STORES)
def test_list_can_filter_to_one_section(kind, tmp_path):
    store = _build(kind, tmp_path)
    source = write_import_directory(
        tmp_path / "src",
        [
            _section("body", chunks=[{"label": "one", "text": "alpha"}]),
            _section("risks", chunks=[{"label": "two", "text": "gamma"}], ordinal=1),
        ],
    )
    store.import_dir(pot_id=POT, slug=DOC, source_dir=source)

    assert [row.slug for row in store.list(pot_id=POT, slug=DOC, section="risks")] == [
        "risks"
    ]
    with pytest.raises(ResourceStoreError) as exc:
        store.list(pot_id=POT, slug=DOC, section="missing")
    assert exc.value.code == RESOURCE_NOT_FOUND


@pytest.mark.parametrize("kind", STORES)
def test_default_section_slug_carries_a_source_with_no_divisions(kind, tmp_path):
    store = _build(kind, tmp_path)
    source = write_import_directory(
        tmp_path / "src",
        [
            _section(
                DEFAULT_SECTION_SLUG, chunks=[{"label": "whole", "text": "one piece"}]
            )
        ],
    )
    store.import_dir(pot_id=POT, slug=DOC, source_dir=source)

    chunk = store.get(
        pot_id=POT, resource_id=format_resource_id(DOC, DEFAULT_SECTION_SLUG, 0)
    )

    assert chunk.section == DEFAULT_SECTION_SLUG
    assert chunk.text == "one piece"


@pytest.mark.parametrize("kind", STORES)
def test_oversized_section_warns_but_still_imports(kind, tmp_path):
    store = _build(kind, tmp_path)
    count = RECOMMENDED_MAX_SECTION_CHUNKS + 1
    source = _simple_dir(
        tmp_path / "src", texts=tuple(f"part {i}" for i in range(count))
    )

    manifest = store.import_dir(pot_id=POT, slug=DOC, source_dir=source)

    assert len(manifest.sections[0].chunks) == count
    assert any("chunks" in warning for warning in manifest.warnings)


@pytest.mark.parametrize("kind", STORES)
def test_section_with_no_chunks_warns_but_still_imports(kind, tmp_path):
    store = _build(kind, tmp_path)
    source = write_import_directory(tmp_path / "src", [_section("body", chunks=[])])

    manifest = store.import_dir(pot_id=POT, slug=DOC, source_dir=source)

    assert manifest.sections[0].chunks == ()
    assert any("no chunks" in warning for warning in manifest.warnings)


@pytest.mark.parametrize("kind", STORES)
def test_pots_cannot_read_each_others_chunks(kind, tmp_path):
    store = _build(kind, tmp_path)
    store.import_dir(
        pot_id=POT,
        slug=DOC,
        source_dir=_simple_dir(tmp_path / "src", texts=("secret",)),
    )

    with pytest.raises(ResourceStoreError) as exc:
        store.get(pot_id=OTHER_POT, resource_id=format_resource_id(DOC, "body", 0))

    assert exc.value.code == RESOURCE_NOT_FOUND


@pytest.mark.parametrize("kind", STORES)
def test_purge_pot_leaves_every_other_pot_intact(kind, tmp_path):
    store = _build(kind, tmp_path)
    source = _simple_dir(tmp_path / "src", texts=("alpha",))
    store.import_dir(pot_id=POT, slug=DOC, source_dir=source)
    store.import_dir(pot_id=OTHER_POT, slug=DOC, source_dir=source)

    assert store.purge_pot(POT) is True

    assert store.get(pot_id=OTHER_POT, resource_id=format_resource_id(DOC, "body", 0))
    with pytest.raises(ResourceStoreError):
        store.list(pot_id=POT, slug=DOC)
    assert store.purge_pot(POT) is False


@pytest.mark.parametrize("kind", STORES)
def test_delete_removes_one_document_and_is_a_no_op_when_absent(kind, tmp_path):
    store = _build(kind, tmp_path)
    source = _simple_dir(tmp_path / "src", texts=("alpha",))
    store.import_dir(pot_id=POT, slug=DOC, source_dir=source)
    store.import_dir(pot_id=POT, slug="other-doc", source_dir=source)

    assert store.delete(pot_id=POT, slug=DOC) is True
    assert store.delete(pot_id=POT, slug=DOC) is False

    with pytest.raises(ResourceStoreError):
        store.list(pot_id=POT, slug=DOC)
    assert store.list(pot_id=POT, slug="other-doc")


# --- local-only: the on-disk representation ---------------------------------


def test_pot_directory_names_are_deterministic_and_never_collide():
    assert pot_dir_name("conformance:pot-a") == pot_dir_name("conformance:pot-a")
    # Sanitizing alone would fold both of these onto the same name.
    assert pot_dir_name("a:b") != pot_dir_name("a/b")
    for pot_id in ("../escape", "a/b", "pot_1", "", ":"):
        name = pot_dir_name(pot_id)
        assert "/" not in name and ".." not in name
        assert not name.startswith(".")


def test_local_store_writes_the_documented_disk_layout(tmp_path):
    store = LocalResourceStore(home=tmp_path / "home")
    store.import_dir(
        pot_id=POT, slug=DOC, source_dir=_simple_dir(tmp_path / "src", texts=("alpha",))
    )

    doc_root = tmp_path / "home" / "resources" / pot_dir_name(POT) / DOC
    assert (doc_root / "body" / "0000.txt").read_text(encoding="utf-8") == "alpha"
    assert (doc_root / "meta.json").is_file()


def test_local_store_default_home_follows_the_context_engine_home(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("CONTEXT_ENGINE_HOME", str(tmp_path))

    store = LocalResourceStore()
    store.import_dir(
        pot_id=POT, slug=DOC, source_dir=_simple_dir(tmp_path / "src", texts=("alpha",))
    )

    assert (tmp_path / "resources" / pot_dir_name(POT) / DOC / "meta.json").is_file()


def test_local_store_leaves_no_scratch_directories_behind(tmp_path):
    store = LocalResourceStore(home=tmp_path / "home")
    store.import_dir(
        pot_id=POT, slug=DOC, source_dir=_simple_dir(tmp_path / "v1", texts=("alpha",))
    )
    with pytest.raises(ResourceStoreError):
        store.import_dir(
            pot_id=POT,
            slug=DOC,
            source_dir=_simple_dir(
                tmp_path / "v2", texts=("x" * (RESOURCE_CHUNK_MAX_CHARS + 1),)
            ),
        )
    store.import_dir(
        pot_id=POT, slug=DOC, source_dir=_simple_dir(tmp_path / "v3", texts=("omega",))
    )

    pot_root = tmp_path / "home" / "resources" / pot_dir_name(POT)
    assert [path.name for path in pot_root.iterdir()] == [DOC]


def test_local_store_sweeps_stale_leftovers_from_a_crashed_import(tmp_path):
    # A crash leaves scratch dirs; once they are old enough to be nobody's,
    # the next import of the same document clears them rather than accumulate
    # orphan bytes.
    store = LocalResourceStore(home=tmp_path / "home")
    store.import_dir(
        pot_id=POT, slug=DOC, source_dir=_simple_dir(tmp_path / "v1", texts=("alpha",))
    )
    pot_root = tmp_path / "home" / "resources" / pot_dir_name(POT)
    for name in (f".{DOC}.staging.abc123", f".{DOC}.trash.def456"):
        _age(pot_root / name)

    store.import_dir(
        pot_id=POT, slug=DOC, source_dir=_simple_dir(tmp_path / "v2", texts=("omega",))
    )

    assert [path.name for path in pot_root.iterdir()] == [DOC]
    assert store.get(
        pot_id=POT, resource_id=format_resource_id(DOC, "body", 0)
    ).text == ("omega")


def test_local_store_leaves_a_concurrent_imports_staging_directory_alone(tmp_path):
    # Sweeping by name would rmtree the staging tree of an import of the same
    # document that is still writing, and that import would go on to publish a
    # manifest naming chunk files it no longer has.
    store = LocalResourceStore(home=tmp_path / "home")
    store.import_dir(
        pot_id=POT, slug=DOC, source_dir=_simple_dir(tmp_path / "v1", texts=("alpha",))
    )
    pot_root = tmp_path / "home" / "resources" / pot_dir_name(POT)
    live = pot_root / f".{DOC}.staging.inflight"
    live.mkdir()
    (live / "written-so-far.txt").write_text("beta", encoding="utf-8")

    store.import_dir(
        pot_id=POT, slug=DOC, source_dir=_simple_dir(tmp_path / "v2", texts=("omega",))
    )

    assert (live / "written-so-far.txt").read_text(encoding="utf-8") == "beta"


def test_local_store_restores_a_document_lost_in_the_rename_window(tmp_path):
    # Killed between the two renames, the document's only copy is its trash
    # directory. Discarding it would drop bytes the graph still cites and
    # rewind the revision counter R7 hangs claim invalidation on.
    store = LocalResourceStore(home=tmp_path / "home")
    for version in ("v1", "v2"):
        store.import_dir(
            pot_id=POT,
            slug=DOC,
            source_dir=_simple_dir(
                tmp_path / version, texts=(version,), content_hash=f"body-{version}"
            ),
        )
    pot_root = tmp_path / "home" / "resources" / pot_dir_name(POT)
    os.rename(pot_root / DOC, pot_root / f".{DOC}.trash.interrupted")

    manifest = store.import_dir(
        pot_id=POT,
        slug=DOC,
        source_dir=_simple_dir(tmp_path / "v3", texts=("v3",), content_hash="body-v3"),
    )

    assert manifest.revision == 3
    assert manifest.sections_added == ()
    assert [path.name for path in pot_root.iterdir()] == [DOC]


def test_local_store_stores_chunk_text_byte_for_byte(tmp_path):
    # Universal-newline translation would rewrite the evidence this store
    # exists to hold, and silently break the script-computed content_hash.
    store = LocalResourceStore(home=tmp_path / "home")
    source = _simple_dir(tmp_path / "src", texts=("a\r\nb\rc",))

    store.import_dir(pot_id=POT, slug=DOC, source_dir=source)

    chunk = store.get(pot_id=POT, resource_id=format_resource_id(DOC, "body", 0))
    assert chunk.text == "a\r\nb\rc"
    assert chunk.chars == 6


def _age(path: Path) -> None:
    """Create ``path`` and backdate it well past the staleness cutoff."""
    path.mkdir()
    old = time.time() - 7 * 24 * 3600
    os.utime(path, (old, old))


def test_local_store_reports_an_unreadable_stored_manifest(tmp_path):
    store = LocalResourceStore(home=tmp_path / "home")
    store.import_dir(
        pot_id=POT, slug=DOC, source_dir=_simple_dir(tmp_path / "src", texts=("alpha",))
    )
    meta = tmp_path / "home" / "resources" / pot_dir_name(POT) / DOC / "meta.json"
    meta.write_text("{ not json", encoding="utf-8")

    with pytest.raises(ResourceStoreError) as exc:
        store.list(pot_id=POT, slug=DOC)

    assert exc.value.code == RESOURCE_MANIFEST_INVALID
    assert exc.value.recommended_next_action


def test_local_store_rejects_an_import_directory_without_a_manifest(tmp_path):
    store = LocalResourceStore(home=tmp_path / "home")
    empty = tmp_path / "empty"
    empty.mkdir()

    with pytest.raises(ResourceStoreError) as exc:
        store.import_dir(pot_id=POT, slug=DOC, source_dir=Path(empty))

    assert exc.value.code == RESOURCE_MANIFEST_INVALID
