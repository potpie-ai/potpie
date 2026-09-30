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


def test_parse_pdf_to_staging(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Degraded (pypdf) path — pinned so a docling install does not reroute it."""
    pytest.importorskip("pypdf")
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
    fixture = (
        Path(__file__).resolve().parent.parent
        / "fixtures"
        / "resources"
        / "digital_one_page.pdf"
    )
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


def test_auto_summary_stays_prose_and_auto_keywords_carries_salient_terms() -> None:
    """Keywords are a structured field, not text bolted onto the summary; the
    distinctive terms living past the first 500 chars go into auto_keywords."""
    from potpie_context_engine.adapters.outbound.resources.parsers.markdown import (
        _auto_summary,
        auto_keywords,
    )

    lead = (
        "This section explains how the board is prepared for everyday use and "
        "covers the general workflow that most users follow when they first "
        "receive the device. "
    ) * 4  # ~600 chars of generic lead
    tail = (
        "Hold the BOOTSEL button during power-up and the device mounts as mass "
        "storage; drag a UF2 file onto the disk to reprogram the flash. The "
        "supported supply range is 1.8V to 5.5V via VSYS."
    )
    summary = _auto_summary("Programming the flash", lead + tail)
    keywords = auto_keywords("Programming the flash", lead + tail)

    assert "Programming the flash" in summary
    assert "Key terms" not in summary
    assert len(summary) <= 2000
    assert "BOOTSEL" in keywords
    assert "UF2" in keywords


def test_salient_terms_exclusion_is_token_based_not_substring() -> None:
    from potpie_context_engine.adapters.outbound.resources.parsers.markdown import (
        _salient_terms,
    )

    terms = _salient_terms(
        "The RAM subsystem uses RAM banks and RAM refresh cycles.",
        exclude="programming the flash",
    )
    assert "RAM" in terms


def test_salient_terms_upgrades_casing_when_acronym_form_dominates() -> None:
    from potpie_context_engine.adapters.outbound.resources.parsers.markdown import (
        _salient_terms,
    )

    terms = _salient_terms(
        "the bootsel pin works. Hold BOOTSEL now. Press BOOTSEL again.",
        exclude="",
    )
    assert "BOOTSEL" in terms
