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
