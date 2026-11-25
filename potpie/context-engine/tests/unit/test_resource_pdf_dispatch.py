"""Unit tests for PDF parser dispatch."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from potpie_context_engine.adapters.outbound.resources.parsers.dispatch import (
    ParseOptions,
    parse_file_to_staging,
)

pytestmark = pytest.mark.unit


def test_pdf_dispatch_requires_docling_or_allow_degraded(tmp_path: Path) -> None:
    fixture = (
        Path(__file__).resolve().parent.parent
        / "fixtures"
        / "resources"
        / "digital_one_page.pdf"
    )
    out = tmp_path / "out"
    caps = {
        "pypdf": True,
        "pypdfium2": False,
        "docling": False,
        "rapidocr": False,
        "httpx": False,
    }
    with patch(
        "potpie_context_engine.adapters.outbound.resources.parsers.dispatch.documents_capabilities",
        return_value=caps,
    ):
        with pytest.raises(ImportError, match="requires Docling"):
            parse_file_to_staging(fixture, out)


def test_pdf_dispatch_uses_degraded_when_allowed(tmp_path: Path) -> None:
    pytest.importorskip("pypdf")
    fixture = (
        Path(__file__).resolve().parent.parent
        / "fixtures"
        / "resources"
        / "digital_one_page.pdf"
    )
    out = tmp_path / "out"
    caps = {
        "pypdf": True,
        "pypdfium2": False,
        "docling": False,
        "rapidocr": False,
        "httpx": False,
    }
    with patch(
        "potpie_context_engine.adapters.outbound.resources.parsers.dispatch.documents_capabilities",
        return_value=caps,
    ):
        manifest = parse_file_to_staging(
            fixture,
            out,
            options=ParseOptions(allow_degraded=True),
        )
    assert manifest.source_kind == "pdf"
    assert manifest.provenance_version == 0
    assert manifest.parser_tier.endswith("degraded")
    assert (out / "meta.json").is_file()


def test_pdf_dispatch_forwards_vision_provider_to_docling(tmp_path: Path) -> None:
    source = tmp_path / "mixed.pdf"
    out = tmp_path / "out"
    manifest = object()
    with (
        patch(
            "potpie_context_engine.adapters.outbound.resources.parsers.dispatch.documents_capabilities",
            return_value={"docling": True},
        ),
        patch(
            "potpie_context_engine.adapters.outbound.resources.parsers.pdf_docling_provenance.parse_pdf_docling_provenance_to_staging",
            return_value=manifest,
        ) as parse_docling,
    ):
        result = parse_file_to_staging(
            source, out, options=ParseOptions(vision_provider="local")
        )

    assert result is manifest
    parse_docling.assert_called_once_with(
        source, out, chunk_target=ParseOptions().chunk_target, vision_provider="local"
    )


def test_pdf_enrichment_reports_partial_picture_caption_coverage() -> None:
    from potpie_context_engine.adapters.outbound.resources.parsers.pdf_docling_provenance import (
        _enrichment_diagnostics,
    )
    from potpie_context_engine.core.resource_models import DocumentElementRecord

    pictures = [
        DocumentElementRecord(
            element_id="figure-1", element_type="picture", visual_caption="A diagram"
        ),
        DocumentElementRecord(element_id="figure-2", element_type="picture"),
    ]
    observed = _enrichment_diagnostics(pictures, vision_provider="local")
    assert observed["picture_enrichment"]["pictures_with_visual_caption"] == 1
    assert "partial_picture_visual_caption_coverage" in observed["picture_enrichment"]["warnings"]
