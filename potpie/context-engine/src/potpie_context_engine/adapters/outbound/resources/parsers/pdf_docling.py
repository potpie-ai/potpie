"""Docling PDF parser — delegates to provenance-aware element ledger path."""

from __future__ import annotations

from pathlib import Path

from potpie_context_engine.adapters.outbound.resources.parsers.pdf_docling_provenance import (
    parse_pdf_docling_provenance,
    parse_pdf_docling_provenance_to_staging,
)
from potpie_context_engine.domain.resource_models import (
    CHUNK_TARGET_DEFAULT,
    ResourceManifest,
)


def parse_pdf_docling(
    path: Path,
    *,
    chunk_target: int = CHUNK_TARGET_DEFAULT,
) -> tuple[ResourceManifest, dict[str, list[str]]]:
    manifest, section_texts, _elements, _prov, _ocr = parse_pdf_docling_provenance(
        path,
        chunk_target=chunk_target,
    )
    return manifest, section_texts


def parse_pdf_docling_to_staging(
    source: Path,
    out_dir: Path,
    *,
    chunk_target: int = CHUNK_TARGET_DEFAULT,
) -> ResourceManifest:
    return parse_pdf_docling_provenance_to_staging(
        source,
        out_dir,
        chunk_target=chunk_target,
    )


__all__ = ["parse_pdf_docling", "parse_pdf_docling_to_staging"]
