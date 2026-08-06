"""Reusable public conformance checks for third-party graph backends."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable, Mapping, Sequence

from potpie_context_core.api import (
    ClaimQueryFilter,
    DEFAULT_GRAPH_DEFINITION,
    EdgeUpsert,
    EntityUpsert,
    MutationBatch,
    RESOURCE_CHUNK_MAX_CHARS,
    RESOURCE_SEQ_WIDTH,
    ResourceStoreError,
    format_resource_id,
)


def run_graph_backend_conformance(backend_factory: Callable[[], object]) -> None:
    """Exercise canonical writes, idempotency, invalidation, and pot isolation."""

    backend = backend_factory().bind_definition(DEFAULT_GRAPH_DEFINITION)
    caps = backend.capabilities()
    if not caps.mutation or not caps.claim_query:
        raise AssertionError("mutation and claim_query capabilities are mandatory")

    pot_a = "conformance:pot-a"
    pot_b = "conformance:pot-b"
    claim_key = "claim:conformance:one"
    batch = MutationBatch(
        entity_upserts=[
            EntityUpsert(entity_key="service:alpha", labels=("Service",)),
            EntityUpsert(entity_key="service:beta", labels=("Service",)),
        ],
        edge_upserts=[
            EdgeUpsert(
                edge_type="RELATED_TO",
                from_entity_key="service:alpha",
                to_entity_key="service:beta",
                properties={
                    "claim_key": claim_key,
                    "truth": "agent_claim",
                    "subgraph": "admin",
                    "fact": "alpha relates to beta",
                },
            )
        ],
    )
    first = backend.mutation.apply(batch, expected_pot_id=pot_a)
    second = backend.mutation.apply(batch, expected_pot_id=pot_a)
    if not first.ok or not second.ok:
        raise AssertionError("canonical mutation failed")

    rows_a = backend.claim_query.find_claims(ClaimQueryFilter(pot_id=pot_a))
    rows_b = backend.claim_query.find_claims(ClaimQueryFilter(pot_id=pot_b))
    if len([row for row in rows_a if row.claim_key == claim_key]) != 1:
        raise AssertionError("claim-key idempotency is not enforced")
    if rows_b:
        raise AssertionError("one pot can read another pot's claims")

    invalidated = backend.mutation.invalidate(
        pot_id=pot_a, claim_keys=(claim_key,), reason="conformance"
    )
    if invalidated != 1:
        raise AssertionError("claim invalidation did not affect exactly one claim")
    if backend.claim_query.find_claims(ClaimQueryFilter(pot_id=pot_a)):
        raise AssertionError("invalidated claims are visible by default")
    invalidated_rows = backend.claim_query.find_claims(
        ClaimQueryFilter(pot_id=pot_a, include_invalidated=True)
    )
    if len(invalidated_rows) != 1:
        raise AssertionError("invalidated claim history is not queryable")

    # A reset and every optional projection must stay partition-scoped.
    backend.mutation.apply(batch, expected_pot_id=pot_b)
    backend.mutation.reset_pot(pot_a)
    if not backend.claim_query.find_claims(ClaimQueryFilter(pot_id=pot_b)):
        raise AssertionError("resetting one pot changed another pot")

    if caps.inspection:
        foreign = backend.inspection.slice(
            pot_id=pot_a, filter_=ClaimQueryFilter(pot_id=pot_a)
        )
        if foreign.nodes or foreign.edges:
            raise AssertionError("inspection leaked reset pot state")
    if caps.analytics:
        counts_a = backend.analytics.counts(pot_a)
        counts_b = backend.analytics.counts(pot_b)
        if counts_a.get("claims", 0) != 0 or counts_b.get("claims", 0) < 1:
            raise AssertionError("analytics counts are not pot-isolated")
    if caps.snapshot:
        with TemporaryDirectory() as directory:
            destination = str(Path(directory) / "pot-b.json")
            manifest = backend.snapshot.export(pot_id=pot_b, destination=destination)
            if manifest.pot_id != pot_b or manifest.claim_count < 1:
                raise AssertionError("snapshot export did not preserve pot metadata")
            backend.snapshot.import_(pot_id="conformance:pot-c", source=destination)
            rows_c = backend.claim_query.find_claims(
                ClaimQueryFilter(pot_id="conformance:pot-c")
            )
            if not rows_c or any(row.pot_id != "conformance:pot-c" for row in rows_c):
                raise AssertionError("snapshot import is not target-pot isolated")


def write_import_directory(
    root: Path,
    sections: Sequence[Mapping[str, Any]],
    *,
    source_ref: str | None = None,
    source_kind: str | None = None,
) -> Path:
    """Materialize the directory an extraction script would emit.

    Each section is its ``meta.json`` entry verbatim, except that its chunks
    carry an extra ``text`` key: that text is written to
    ``<section>/<seq>.txt`` and stripped from the manifest, and a chunk with no
    ``seq`` takes its position. Everything else passes through untouched, so a
    caller can also build a deliberately invalid directory.
    """

    root.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    for section in sections:
        entry = {key: value for key, value in section.items() if key != "chunks"}
        refs: list[dict[str, Any]] = []
        for position, chunk in enumerate(section.get("chunks") or ()):
            ref = {key: value for key, value in chunk.items() if key != "text"}
            ref.setdefault("seq", position)
            refs.append(ref)
            text = chunk.get("text")
            if text is None:
                # A named chunk with no file — the missing-chunk case.
                continue
            section_dir = root / str(section.get("slug", ""))
            section_dir.mkdir(parents=True, exist_ok=True)
            name = f"{int(ref['seq']):0{RESOURCE_SEQ_WIDTH}d}.txt"
            (section_dir / name).write_text(text, encoding="utf-8")
        entry["chunks"] = refs
        entries.append(entry)
    payload = {
        "source_ref": source_ref,
        "source_kind": source_kind,
        "sections": entries,
    }
    (root / "meta.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return root


def run_resource_store_conformance(store_factory: Callable[[], object]) -> None:
    """Exercise import atomicity, batched reads, revisions, and pot isolation."""

    store = store_factory()
    pot_a = "conformance:pot-a"
    pot_b = "conformance:pot-b"
    doc = "q3-review"

    with TemporaryDirectory() as directory:
        root = Path(directory)
        first = write_import_directory(
            root / "v1",
            [
                {
                    "slug": "body",
                    "title": "Body",
                    "summary": "the document in full",
                    "ordinal": 0,
                    "content_hash": "body-1",
                    "chunks": [
                        {"label": "opening", "text": "alpha"},
                        {"label": "closing", "text": "omega"},
                    ],
                },
                {
                    "slug": "capacity",
                    "title": "Capacity",
                    "summary": "",
                    "ordinal": 1,
                    "content_hash": "capacity-1",
                    "chunks": [{"label": "limits", "text": "beta", "page": 3}],
                },
            ],
            source_ref="file:///conformance.pdf",
            source_kind="pdf",
        )
        manifest = store.import_dir(pot_id=pot_a, slug=doc, source_dir=first)
        if manifest.revision != 1:
            raise AssertionError("a first import must be revision 1")
        if {row.slug for row in manifest.sections} != {"body", "capacity"}:
            raise AssertionError("import did not record every section")
        if set(manifest.sections_added) != {"body", "capacity"}:
            raise AssertionError("a first import must report every section as added")
        if not any(row.summary_pending for row in manifest.sections):
            raise AssertionError("a section with no summary must import as pending")

        opening = format_resource_id(doc, "body", 0)
        closing = format_resource_id(doc, "body", 1)
        limits = format_resource_id(doc, "capacity", 0)
        chunk = store.get(pot_id=pot_a, resource_id=opening)
        if chunk.text != "alpha" or chunk.chars != 5 or chunk.revision != 1:
            raise AssertionError("get did not round-trip the stored chunk")
        if chunk.source_ref != "file:///conformance.pdf":
            raise AssertionError("get did not hydrate the document's source_ref")

        batch = store.get_many(pot_id=pot_a, resource_ids=(limits, opening, closing))
        if tuple(row.text for row in batch) != ("beta", "alpha", "omega"):
            raise AssertionError("get_many did not answer in the order requested")
        if batch[0].page != 3:
            raise AssertionError("get_many dropped the chunk's page metadata")

        labels = {
            ref.label
            for row in store.list(pot_id=pot_a, slug=doc)
            for ref in row.chunks
        }
        if labels != {"opening", "closing", "limits"}:
            raise AssertionError("list did not return every chunk label")

        # A refused import must leave the stored revision exactly as it was.
        oversized = write_import_directory(
            root / "oversized",
            [
                {
                    "slug": "body",
                    "title": "Body",
                    "summary": "one chunk over the cap",
                    "ordinal": 0,
                    "content_hash": "body-2",
                    "chunks": [
                        {"label": "huge", "text": "x" * (RESOURCE_CHUNK_MAX_CHARS + 1)}
                    ],
                }
            ],
        )
        try:
            store.import_dir(pot_id=pot_a, slug=doc, source_dir=oversized)
        except ResourceStoreError:
            pass
        else:
            raise AssertionError("an oversized chunk must be rejected at import")
        if store.get(pot_id=pot_a, resource_id=opening).text != "alpha":
            raise AssertionError("a rejected import damaged the stored document")

        # Re-import replaces the chunk set, bumps the revision, reports drift.
        second = write_import_directory(
            root / "v2",
            [
                {
                    "slug": "body",
                    "title": "Body",
                    "summary": "the document in full",
                    "ordinal": 0,
                    "content_hash": "body-1",
                    "chunks": [
                        {"label": "opening", "text": "alpha"},
                        {"label": "closing", "text": "omega"},
                    ],
                },
                {
                    "slug": "risks",
                    "title": "Risks",
                    "summary": "what could go wrong",
                    "ordinal": 1,
                    "content_hash": "risks-1",
                    "chunks": [{"label": "top risk", "text": "gamma"}],
                },
            ],
            source_ref="file:///conformance.pdf",
            source_kind="pdf",
        )
        replaced = store.import_dir(pot_id=pot_a, slug=doc, source_dir=second)
        if replaced.revision != 2:
            raise AssertionError("re-import must bump the revision")
        if replaced.sections_added != ("risks",):
            raise AssertionError("re-import did not report the added section")
        if replaced.sections_removed != ("capacity",):
            raise AssertionError("re-import did not report the removed section")
        if replaced.sections_kept != ("body",):
            raise AssertionError("an unchanged content_hash must report as kept")
        try:
            store.get(pot_id=pot_a, resource_id=limits)
        except ResourceStoreError:
            pass
        else:
            raise AssertionError("re-import left the prior revision's chunks readable")

        # Pots never see each other, and purging one leaves the other whole.
        store.import_dir(pot_id=pot_b, slug=doc, source_dir=first)
        try:
            store.get(pot_id=pot_b, resource_id=format_resource_id(doc, "risks", 0))
        except ResourceStoreError:
            pass
        else:
            raise AssertionError("one pot can read another pot's chunks")
        if store.get(pot_id=pot_b, resource_id=limits).text != "beta":
            raise AssertionError("the second pot did not get its own copy")
        store.purge_pot(pot_a)
        if store.get(pot_id=pot_b, resource_id=opening).text != "alpha":
            raise AssertionError("purging one pot removed another pot's resources")
        try:
            store.list(pot_id=pot_a, slug=doc)
        except ResourceStoreError:
            pass
        else:
            raise AssertionError("purge_pot left the pot's documents listable")

        # doctor's readiness probe: answers for a pot that has resources and
        # for one that was just purged, and never raises.
        ready = store.status(pot_id=pot_b)
        if not ready.ready or not ready.kind:
            raise AssertionError("a usable store must report itself ready and named")
        if ready.documents != 1:
            raise AssertionError("status did not count the pot's stored documents")
        if store.status(pot_id=pot_a).documents != 0:
            raise AssertionError("status counted documents in a purged pot")
        if store.status().documents is not None:
            raise AssertionError("status must not count documents without a pot")

        if store.delete(pot_id=pot_b, slug=doc) is not True:
            raise AssertionError("delete must report removing a stored document")
        if store.delete(pot_id=pot_b, slug=doc) is not False:
            raise AssertionError("deleting an absent document must be a no-op")
        if store.purge_pot(pot_a) is not False:
            raise AssertionError("purging an already-empty pot must be a no-op")


class GraphBackendConformanceMixin:
    """Pytest-compatible mixin for backend adapter test suites."""

    backend_factory: Callable[[], object]

    def test_graph_backend_conformance(self) -> None:
        run_graph_backend_conformance(self.backend_factory)


class ResourceStoreConformanceMixin:
    """Pytest-compatible mixin for resource-store adapter test suites."""

    store_factory: Callable[[], object]

    def test_resource_store_conformance(self) -> None:
        run_resource_store_conformance(self.store_factory)


__all__ = [
    "GraphBackendConformanceMixin",
    "ResourceStoreConformanceMixin",
    "run_graph_backend_conformance",
    "run_resource_store_conformance",
    "write_import_directory",
]
