"""PDF text extraction via pypdf (degraded path; bundled in potpie[documents])."""

from __future__ import annotations

from pathlib import Path

from potpie_context_engine.adapters.outbound.resources.parsers.markdown import (
    ParsedSection,
    _manifest_from_parsed,
    _section_from_body,
    write_staging_from_parsed,
)
from potpie_context_engine.domain.resource_models import (
    CHUNK_TARGET_DEFAULT,
    ResourceManifest,
)

_DOCUMENTS_HINT = "pip install potpie[documents]"
_SCANNED_PDF_HINT = (
    "no extractable text in PDF; scanned or layout-heavy PDFs need "
    "potpie[documents] (Docling + RapidOCR)"
)


def _require_pypdf() -> object:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ImportError(
            f"PDF parsing requires pypdf. Install with: {_DOCUMENTS_HINT}"
        ) from exc
    return PdfReader


def parse_pdf_pypdf(
    path: Path,
    *,
    chunk_target: int = CHUNK_TARGET_DEFAULT,
) -> tuple[ResourceManifest, dict[str, list[str]]]:
    PdfReader = _require_pypdf()
    reader = PdfReader(str(path))
    parsed_sections: list[ParsedSection] = []

    for page_index, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
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
        raise ValueError(_SCANNED_PDF_HINT)

    return _manifest_from_parsed(path, "pdf", parsed_sections)


def parse_pdf_to_staging(
    source: Path,
    out_dir: Path,
    *,
    chunk_target: int = CHUNK_TARGET_DEFAULT,
) -> ResourceManifest:
    manifest, section_texts = parse_pdf_pypdf(source, chunk_target=chunk_target)
    write_staging_from_parsed(out_dir, manifest, section_texts)
    return manifest


__all__ = ["parse_pdf_pypdf", "parse_pdf_to_staging"]
