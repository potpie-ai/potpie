"""Reusable public conformance checks for third-party graph backends."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable, Mapping, Sequence

from potpie_context_core.api import (
    ClaimQueryFilter,
    DEFAULT_GRAPH_DEFINITION,
    MATCH_MODE_DISABLED,
    MATCH_MODE_HYBRID,
    MATCH_MODE_LEXICAL,
    Chunk,
    ChunkRef,
    DocumentManifest,
    EdgeUpsert,
    EntityUpsert,
    MutationBatch,
    RESOURCE_CHUNK_MAX_CHARS,
    RESOURCE_SEQ_WIDTH,
    ResourceStoreError,
    SectionManifest,
    format_resource_id,
    parse_resource_id,
    read_import_files,
)
from potpie_context_core.ports.resource_index import SNIPPET_TARGET_CHARS


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

        # The same directory shipped as its contents — what the CLI sends,
        # because a host may not be able to read the caller's filesystem —
        # must import identically. A third pot keeps it a first import.
        pot_c = "conformance:pot-c"
        shipped = store.import_dir(
            pot_id=pot_c, slug=doc, files=read_import_files(first)
        )
        if shipped.revision != 1 or shipped.sections != manifest.sections:
            raise AssertionError("an import from files must match the directory import")
        if store.get(pot_id=pot_c, resource_id=limits).text != "beta":
            raise AssertionError("an import from files did not store the chunk text")
        if store.get(pot_id=pot_c, resource_id=limits).page != 3:
            raise AssertionError("an import from files dropped chunk metadata")

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


def build_conformance_document(
    *, pot_id: str = "conformance:pot-a", doc: str = "q3-review"
) -> tuple[DocumentManifest, tuple[Chunk, ...]]:
    """A two-section document with one deliberately unsummarized fact.

    ``ERR_QUOTA_EXCEEDED`` appears in a chunk body and in *no* section summary.
    That is the whole point of the corpus: it is the fact the summary-only
    index could never reach, so a profile that declares ``lexical`` and cannot
    find it has not implemented the capability it advertises.
    """
    bodies: dict[tuple[str, int], str] = {
        ("liability", 0): (
            "12. Limitation of Liability. In no event shall either party's "
            "aggregate liability exceed the fees paid in the twelve months "
            "preceding the claim."
        ),
        ("liability", 1): (
            "The cap on damages does not apply to indemnification obligations "
            "arising from third-party intellectual property claims."
        ),
        ("quotas", 0): (
            "Service quotas. When a tenant exceeds its provisioned throughput "
            "the API returns ERR_QUOTA_EXCEEDED with a retry-after header."
        ),
    }
    manifest = DocumentManifest(
        pot_id=pot_id,
        doc=doc,
        revision=1,
        source_ref="file:///conformance.pdf",
        source_kind="pdf",
        sections=(
            SectionManifest(
                slug="liability",
                title="12. Limitation of Liability",
                summary="How much either side can owe the other, and the carve-outs.",
                ordinal=0,
                content_hash="liability-1",
                chunks=(
                    ChunkRef(seq=0, label="cap on damages"),
                    ChunkRef(seq=1, label="IP indemnity carve-out"),
                ),
            ),
            SectionManifest(
                slug="quotas",
                title="7. Service Quotas",
                summary="Throughput ceilings and how the API reports one being hit.",
                ordinal=1,
                content_hash="quotas-1",
                chunks=(ChunkRef(seq=0, label="throughput ceiling"),),
            ),
        ),
    )
    chunks = tuple(
        Chunk(
            resource_id=format_resource_id(doc, section, seq),
            doc=doc,
            section=section,
            seq=seq,
            text=text,
            chars=len(text),
            revision=1,
            source_ref="file:///conformance.pdf",
        )
        for (section, seq), text in bodies.items()
    )
    return manifest, chunks


def run_resource_index_conformance(index_factory: Callable[[], object]) -> None:
    """Exercise capability honesty, the graph tie, drain, and pot isolation.

    Every assertion is gated on a *declared* capability, the way the graph
    backend suite gates on ``caps.inspection`` and friends. That is what makes
    the suite meaningful for a profile that implements one arm and not the
    other: ``none`` must pass it by honestly answering nothing, and
    ``sqlite_hybrid`` must pass it by actually retrieving — and neither is
    allowed to claim a capability it does not answer.
    """

    index = index_factory()
    caps = index.capabilities()
    pot_a = "conformance:pot-a"
    pot_b = "conformance:pot-b"
    manifest, chunks = build_conformance_document(pot_id=pot_a)

    # --- capability honesty -------------------------------------------------
    if caps.hybrid and not (caps.lexical and caps.semantic):
        raise AssertionError("a profile cannot declare hybrid without both arms")
    if index.status().profile != caps.profile:
        raise AssertionError("status and capabilities disagree about the profile")

    report = index.index_document(pot_id=pot_a, manifest=manifest, chunks=chunks)
    if caps.incremental and report.chunks != len(chunks):
        raise AssertionError("an incremental profile must index every chunk given")

    # A disabled profile is a legitimate implementation: it must answer, not
    # raise, and it must say ``disabled`` rather than look like an empty corpus.
    if not caps.implemented():
        empty = index.search(pot_id=pot_a, query="ERR_QUOTA_EXCEEDED", limit=5)
        if empty.hits or empty.match_mode != MATCH_MODE_DISABLED:
            raise AssertionError(
                "a profile declaring no capability must return zero hits "
                "labeled 'disabled'"
            )
        return

    # --- lexical: the fact no summary mentions ------------------------------
    if caps.lexical:
        found = index.search(pot_id=pot_a, query="ERR_QUOTA_EXCEEDED", limit=5)
        if not found.hits:
            raise AssertionError(
                "a lexical profile must find an exact token present in chunk "
                "text and absent from every section summary"
            )
        top = found.hits[0]
        if top.section != "quotas":
            raise AssertionError("the lexical arm ranked the wrong section first")
        if top.match_mode not in {MATCH_MODE_LEXICAL, MATCH_MODE_HYBRID}:
            raise AssertionError(
                f"unexpected match_mode for a term hit: {top.match_mode}"
            )

        # --- the graph tie, with no second retrieval call -------------------
        if top.document_key != f"document:{manifest.doc}":
            raise AssertionError("document_key is not derivable from the resource id")
        if top.section_key != f"docsection:{manifest.doc}:{top.section}":
            raise AssertionError("section_key is not derivable from the resource id")
        if parse_resource_id(top.resource_id).doc != manifest.doc:
            raise AssertionError("a hit's resource id does not round-trip")

    # --- snippets: a window, never the body ---------------------------------
    if caps.snippets:
        hit = index.search(pot_id=pot_a, query="liability", limit=5).hits
        if hit and any(len(row.snippet) > SNIPPET_TARGET_CHARS for row in hit):
            raise AssertionError("a snippet must be bounded, not a chunk body")

    # --- semantic: pending work drains, then paraphrase retrieves -----------
    if caps.semantic:
        drained = index.drain(pot_id=pot_a)
        if drained.remaining:
            raise AssertionError("one drain over a tiny corpus must clear the backlog")
        # Draining an empty backlog is a no-op, not an error: the loop calls
        # this on every wake-up.
        if index.drain(pot_id=pot_a).embedded:
            raise AssertionError("a second drain must find nothing pending")
        if index.status(pot_id=pot_a).pending_embeddings:
            raise AssertionError("status must agree with the drain about the backlog")
        paraphrase = index.search(
            pot_id=pot_a, query="how much can we be forced to pay", limit=5
        )
        if not paraphrase.hits:
            raise AssertionError(
                "a semantic profile must retrieve on a paraphrase that shares "
                "no distinctive term with the text"
            )
        if all(row.similarity is None for row in paraphrase.hits):
            raise AssertionError("a semantic hit must carry the similarity it scored")

    # --- pot isolation, the hardest invariant -------------------------------
    if index.search(pot_id=pot_b, query="ERR_QUOTA_EXCEEDED", limit=5).hits:
        raise AssertionError("one pot can read another pot's chunks")

    # --- adversarial input: query text is not a query language --------------
    for hostile in ("a:b", 'NEAR("x")', '"unbalanced', "col*umn", "-", "AND OR"):
        index.search(pot_id=pot_a, query=hostile, limit=3)

    # --- derived state: drop, re-derive, identical results ------------------
    if caps.incremental and caps.lexical:
        before = index.search(pot_id=pot_a, query="ERR_QUOTA_EXCEEDED", limit=5)
        if index.drop_document(pot_id=pot_a, slug=manifest.doc) is not True:
            raise AssertionError("dropping an indexed document must report success")
        if index.search(pot_id=pot_a, query="ERR_QUOTA_EXCEEDED", limit=5).hits:
            raise AssertionError("a dropped document must stop matching")
        if index.drop_document(pot_id=pot_a, slug=manifest.doc) is not False:
            raise AssertionError("dropping an absent document must be a no-op")
        index.index_document(pot_id=pot_a, manifest=manifest, chunks=chunks)
        if caps.semantic:
            # A rebuild restores the lexical postings inline and leaves the
            # vectors pending, exactly as an import does — so "identical
            # results" is a claim about the re-derived index once it has
            # finished draining, not about the instant after the write. Any
            # other reading would make the property untestable for a profile
            # whose whole design defers embedding.
            index.drain(pot_id=pot_a)
        after = index.search(pot_id=pot_a, query="ERR_QUOTA_EXCEEDED", limit=5)
        if [row.resource_id for row in after.hits] != [
            row.resource_id for row in before.hits
        ]:
            raise AssertionError("re-deriving the index changed its results")

    # --- teardown -----------------------------------------------------------
    index.purge_pot(pot_a)
    if index.search(pot_id=pot_a, query="ERR_QUOTA_EXCEEDED", limit=5).hits:
        raise AssertionError("purging a pot left its chunks searchable")


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


class ResourceIndexConformanceMixin:
    """Pytest-compatible mixin for resource-index adapter test suites."""

    index_factory: Callable[[], object]

    def test_resource_index_conformance(self) -> None:
        run_resource_index_conformance(self.index_factory)


__all__ = [
    "GraphBackendConformanceMixin",
    "ResourceIndexConformanceMixin",
    "ResourceStoreConformanceMixin",
    "build_conformance_document",
    "run_graph_backend_conformance",
    "run_resource_index_conformance",
    "run_resource_store_conformance",
    "write_import_directory",
]
