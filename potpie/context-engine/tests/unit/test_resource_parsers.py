"""Unit tests for markdown/text parsers."""

from __future__ import annotations

from pathlib import Path

import pytest

from potpie_context_engine.adapters.outbound.resources.parsers.dispatch import (
    ParseOptions,
    parse_file_to_staging,
)
from potpie_context_engine.adapters.outbound.resources.parsers.pdf_pypdf import (
    parse_pdf_pypdf,
)

pytestmark = pytest.mark.unit


def test_parse_markdown_to_staging(tmp_path: Path) -> None:
    source = tmp_path / "doc.md"
    source.write_text(
        "# Title\n\nIntro paragraph.\n\n## Section One\n\nFirst section body.\n\n## Section Two\n\nSecond section body.",
        encoding="utf-8",
    )
    out = tmp_path / "out"
    manifest = parse_file_to_staging(source, out)
    assert manifest.source_kind == "markdown"
    assert len(manifest.sections) >= 2
    assert (out / "meta.json").is_file()
    first = manifest.sections[0]
    assert (out / first.slug / "0000.txt").is_file()


def test_parse_pdf_to_staging(tmp_path: Path) -> None:
    pytest.importorskip("pypdf")
    fixture = Path(__file__).resolve().parent.parent / "fixtures" / "resources" / "digital_one_page.pdf"
    assert fixture.is_file()
    out = tmp_path / "pdf-out"
    manifest = parse_file_to_staging(
        fixture,
        out,
        options=ParseOptions(allow_degraded=True),
    )
    assert manifest.source_kind == "pdf"
    assert len(manifest.sections) == 1
    chunk_path = out / manifest.sections[0].slug / "0000.txt"
    assert chunk_path.is_file()
    assert "Install dependencies" in chunk_path.read_text(encoding="utf-8")


def test_parse_pdf_empty_raises_scanned_hint(tmp_path: Path) -> None:
    pytest.importorskip("pypdf")
    from pypdf import PdfWriter

    blank = tmp_path / "blank.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    writer.write(blank)
    with pytest.raises(ValueError, match="potpie\\[documents\\]"):
        parse_pdf_pypdf(blank)
