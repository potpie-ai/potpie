"""pypdfium2 PDF text fallback ([documents] extra)."""

from __future__ import annotations

from pathlib import Path

from potpie_context_engine.adapters.outbound.resources.parsers.markdown import (
    ParsedSection,
    _manifest_from_parsed,
    _section_from_body,
    write_staging_from_parsed,
)
from potpie_context_engine.domain.resource_models import CHUNK_TARGET_DEFAULT, ResourceManifest

_DOCUMENTS_HINT = "pip install potpie[documents]"


def _require_pypdfium2() -> object:
    try:
        import pypdfium2 as pdfium
    except ImportError as exc:
        raise ImportError(
            f"PDF fallback requires pypdfium2. Install with: {_DOCUMENTS_HINT}"
        ) from exc
    return pdfium


def parse_pdf_pypdfium2(
    path: Path,
    *,
    chunk_target: int = CHUNK_TARGET_DEFAULT,
) -> tuple[ResourceManifest, dict[str, list[str]]]:
    pdfium = _require_pypdfium2()
    document = pdfium.PdfDocument(str(path))
    parsed_sections: list[ParsedSection] = []

    for page_index in range(len(document)):
        page = document[page_index]
        textpage = page.get_textpage()
        text = (textpage.get_text_range() or "").strip()
        if not text:
            continue
        page_num = page_index + 1
        parsed_sections.append(
            _section_from_body(
                title=f"Page {page_num}",
                ordinal=page_index,
                body=text,
                chunk_target=chunk_target,
            )
        )

    if not parsed_sections:
        raise ValueError(
            "no extractable text in PDF via pypdfium2; try potpie[documents] for OCR"
        )
    return _manifest_from_parsed(path, "pdf", parsed_sections)


def parse_pdf_pypdfium2_to_staging(
    source: Path,
    out_dir: Path,
    *,
    chunk_target: int = CHUNK_TARGET_DEFAULT,
) -> ResourceManifest:
    manifest, section_texts = parse_pdf_pypdfium2(source, chunk_target=chunk_target)
    write_staging_from_parsed(out_dir, manifest, section_texts)
    return manifest


__all__ = ["parse_pdf_pypdfium2", "parse_pdf_pypdfium2_to_staging"]
