"""Integration tests for resource import graph bridge mutations."""

from __future__ import annotations

from pathlib import Path

import pytest

from potpie_context_engine.adapters.outbound.resources.local_chunk_store import (
    LocalResourceStore,
)
from potpie_context_engine.adapters.outbound.resources.graph_bridge import (
    document_entity_key,
    section_entity_key,
)
from potpie_context_engine.application.services.graph_service import DefaultGraphService
from potpie_context_engine.application.services.resource_service import ResourceService
from potpie_context_engine.testing import InMemoryGraphBackend
from potpie_context_engine.core.ports.claim_query import ClaimQueryFilter

pytestmark = pytest.mark.unit


def test_resource_ingest_writes_graph_claims(tmp_path: Path) -> None:
    home = tmp_path / "potpie-home"
    store = LocalResourceStore(home=home)
    backend = InMemoryGraphBackend()
    graph = DefaultGraphService(backend=backend)
    resources = ResourceService(graph=graph, store=store)

    source = tmp_path / "guide.md"
    source.write_text(
        "# Guide\n\nOverview.\n\n## Setup\n\nInstall dependencies.",
        encoding="utf-8",
    )
    pot_id = "pot-test"
    doc_slug = "guide"

    report = resources.ingest_file(
        pot_id=pot_id,
        doc_slug=doc_slug,
        source_path=source,
    )
    assert not report.errors
    assert report.graph_written

    labels = backend.claim_query.entity_labels(
        pot_id=pot_id,
        entity_keys=[document_entity_key(doc_slug)],
    )
    assert "Document" in labels.get(document_entity_key(doc_slug), ())

    section_key = section_entity_key(doc_slug, "setup")
    section_labels = backend.claim_query.entity_labels(
        pot_id=pot_id,
        entity_keys=[section_key],
    )
    assert "DocumentSection" in section_labels.get(section_key, ())

    rows = backend.claim_query.find_claims(
        ClaimQueryFilter(
            pot_id=pot_id,
            subject_key_in=(section_key,),
            predicate_in=("SECTION_OF", "RELATED_TO"),
            limit=10,
        )
    )
    predicates = {row.predicate for row in rows}
    assert "SECTION_OF" in predicates
    assert "RELATED_TO" in predicates


def test_section_keywords_enrich_description_but_not_summary() -> None:
    from potpie_context_engine.adapters.outbound.resources.graph_bridge import (
        build_import_mutations,
    )
    from potpie_context_engine.domain.resource_models import (
        ChunkRef,
        ResourceManifest,
        SectionManifest,
    )

    manifest = ResourceManifest(
        source_ref="file:///guide.pdf",
        source_kind="pdf",
        sections=[
            SectionManifest(
                slug="flash",
                title="Programming the flash",
                summary="How to reprogram the on-board flash.",
                ordinal=0,
                chunks=[ChunkRef(seq=0, label="flash")],
                keywords=["BOOTSEL", "UF2"],
            )
        ],
    )
    batch = build_import_mutations(
        pot_id="pot-test", doc_slug="guide", manifest=manifest
    )
    claim = next(op for op in batch["operations"] if op["op"] == "assert_claim")

    assert "Key terms: BOOTSEL, UF2" in claim["description"]
    assert "Key terms" not in claim["subject"]["summary"]
