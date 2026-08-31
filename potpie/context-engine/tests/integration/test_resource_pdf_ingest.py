"""End-to-end PDF ingest via degraded pypdf path (--allow-degraded)."""

from __future__ import annotations

from pathlib import Path

import pytest

from potpie_context_engine.adapters.outbound.resources.local_chunk_store import (
    LocalResourceStore,
)
from potpie_context_engine.application.services.graph_service import DefaultGraphService
from potpie_context_engine.application.services.resource_service import ResourceService
from potpie_context_engine.testing import InMemoryGraphBackend

pytestmark = [pytest.mark.integration, pytest.mark.documents_lite]

FIXTURE_PDF = (
    Path(__file__).resolve().parent.parent
    / "fixtures"
    / "resources"
    / "digital_one_page.pdf"
)


@pytest.fixture
def digital_pdf() -> Path:
    pytest.importorskip("pypdf")
    assert FIXTURE_PDF.is_file(), f"missing fixture: {FIXTURE_PDF}"
    return FIXTURE_PDF


def test_pdf_ingest_end_to_end(
    tmp_path: Path, digital_pdf: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from potpie_context_engine.adapters.outbound.resources.parsers import dispatch

    monkeypatch.setattr(
        dispatch,
        "documents_capabilities",
        lambda: {
            "pypdf": True,
            "pypdfium2": False,
            "docling": False,
            "rapidocr": False,
            "httpx": False,
        },
    )
    home = tmp_path / "potpie-home"
    store = LocalResourceStore(home=home)
    backend = InMemoryGraphBackend()
    graph = DefaultGraphService(backend=backend)
    resources = ResourceService(graph=graph, store=store)

    report = resources.ingest_file(
        pot_id="pot-pdf",
        doc_slug="install-guide",
        source_path=digital_pdf,
        allow_degraded=True,
    )
    assert not report.errors
    assert report.graph_written
    assert report.provenance_version == 0

    section_slug = report.sections_added[0] if report.sections_added else "page-1"
    chunk = resources.get_chunk(
        pot_id="pot-pdf",
        uri=f"potpie://res/install-guide/{section_slug}/0000",
    )
    assert "Install dependencies" in chunk.get("content", "")
    assert chunk.get("provenance") is None

    hits = resources.search_chunks(
        pot_id="pot-pdf",
        query="Install dependencies",
        limit=5,
    )
    assert hits
    assert hits[0]["doc_slug"] == "install-guide"
