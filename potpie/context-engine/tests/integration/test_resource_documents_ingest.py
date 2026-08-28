"""Integration tests for [documents] extra (Docling / RapidOCR).

Skipped when optional deps are not installed.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from potpie_context_engine.adapters.outbound.resources.capabilities import (
    documents_capabilities,
)
from potpie_context_engine.adapters.outbound.resources.local_chunk_store import (
    LocalResourceStore,
)
from potpie_context_engine.application.services.graph_service import DefaultGraphService
from potpie_context_engine.application.services.resource_service import ResourceService
from potpie_context_engine.testing import InMemoryGraphBackend

pytestmark = [pytest.mark.integration, pytest.mark.documents]

FIXTURE_PDF = (
    Path(__file__).resolve().parent.parent / "fixtures" / "resources" / "digital_one_page.pdf"
)


def _skip_without_documents() -> None:
    caps = documents_capabilities()
    if not caps["docling"]:
        pytest.skip("potpie[documents] not installed (docling missing)")


def test_docling_pdf_parse_mocked(tmp_path: Path) -> None:
    """Exercise docling code path with a mocked converter (no model download)."""
    _skip_without_documents()
    from potpie_context_engine.adapters.outbound.resources.parsers.pdf_docling import (
        parse_pdf_docling_to_staging,
    )

    class _FakeDocument:
        def export_to_markdown(self) -> str:
            return "## Intro\n\nInstall dependencies from the guide."

    class _FakeResult:
        document = _FakeDocument()

    class _FakeConverter:
        def convert(self, path: str) -> _FakeResult:
            return _FakeResult()

    out = tmp_path / "staging"
    with patch(
        "potpie_context_engine.adapters.outbound.resources.parsers.pdf_docling._build_converter",
        return_value=_FakeConverter(),
    ):
        manifest = parse_pdf_docling_to_staging(FIXTURE_PDF, out)
    assert manifest.source_kind == "pdf"
    assert any(s.slug == "intro" for s in manifest.sections)
    assert (out / "intro" / "0000.txt").is_file()


def test_image_ingest_with_mocked_enrichers(tmp_path: Path) -> None:
    caps = documents_capabilities()
    if not caps["rapidocr"] and not caps["docling"]:
        pytest.skip("potpie[documents] not installed")

    home = tmp_path / "home"
    store = LocalResourceStore(home=home)
    graph = DefaultGraphService(backend=InMemoryGraphBackend())
    resources = ResourceService(graph=graph, store=store)

    image = tmp_path / "chart.png"
    image.write_bytes(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde"
        b"\x00\x00\x00\x0cIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
    )

    with (
        patch(
            "potpie_context_engine.adapters.outbound.resources.parsers.image.ocr_image_text",
            return_value="Revenue chart Q4",
        ),
        patch(
            "potpie_context_engine.adapters.outbound.resources.parsers.image.caption_image_local",
            return_value="Bar chart showing quarterly revenue.",
        ),
    ):
        report = resources.ingest_file(
            pot_id="pot-img",
            doc_slug="chart",
            source_path=image,
        )

    assert not report.errors
    assert report.graph_written
    chunk = resources.get_chunk(
        pot_id="pot-img",
        uri="potpie://res/chart/visual/0",
    )
    assert "Revenue" in chunk.get("content", "") or "chart" in chunk.get("content", "").lower()
    assert chunk.get("ocr_text") == "Revenue chart Q4"
