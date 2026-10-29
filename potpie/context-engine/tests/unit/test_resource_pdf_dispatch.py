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


def test_pdf_dispatch_reports_docling_failure_or_uses_explicit_degraded_fallback(
    tmp_path: Path,
) -> None:
    fixture = (
        Path(__file__).resolve().parent.parent
        / "fixtures"
        / "resources"
        / "digital_one_page.pdf"
    )
    caps = {
        "pypdf": True,
        "pypdfium2": False,
        "docling": True,
        "rapidocr": True,
        "httpx": True,
    }
    degraded_manifest = type(
        "Manifest",
        (),
        {"parser_tier": "pypdf-degraded"},
    )()
    with (
        patch(
            "potpie_context_engine.adapters.outbound.resources.parsers.dispatch.documents_capabilities",
            return_value=caps,
        ),
        patch(
            "potpie_context_engine.adapters.outbound.resources.parsers."
            "pdf_docling_provenance.parse_pdf_docling_provenance_to_staging",
            side_effect=RuntimeError("layout model failed"),
        ),
        patch(
            "potpie_context_engine.adapters.outbound.resources.parsers.dispatch._parse_pdf_degraded",
            return_value=degraded_manifest,
        ) as degraded,
    ):
        result = parse_file_to_staging(
            fixture,
            tmp_path / "fallback",
            options=ParseOptions(allow_degraded=True),
        )
        assert result.parser_tier == "pypdf-degraded"
        degraded.assert_called_once()

        with pytest.raises(RuntimeError, match="Docling structured extraction failed"):
            parse_file_to_staging(
                fixture,
                tmp_path / "strict",
                options=ParseOptions(allow_degraded=False),
            )


def test_parse_rejects_non_positive_chunk_targets_and_keeps_hard_cap(
    tmp_path: Path,
) -> None:
    source = tmp_path / "guide.md"
    source.write_text("## Guide\n\n" + ("word " * 2000), encoding="utf-8")

    for invalid in (0, -1):
        with pytest.raises(ValueError, match="chunk target"):
            parse_file_to_staging(
                source,
                tmp_path / f"invalid-{invalid}",
                options=ParseOptions(chunk_target=invalid),
            )

    manifest = parse_file_to_staging(
        source,
        tmp_path / "extreme",
        options=ParseOptions(chunk_target=10**9),
    )
    for section in manifest.sections:
        for chunk in section.chunks:
            text = (tmp_path / "extreme" / section.slug / f"{chunk.seq:04d}.txt").read_text(
                encoding="utf-8"
            )
            assert len(text) <= 8000
