"""Docling conversion for office formats and HTML ([documents] extra)."""

from __future__ import annotations

from pathlib import Path

from potpie_context_engine.adapters.outbound.resources.local_chunk_store import file_sha256
from potpie_context_engine.adapters.outbound.resources.parsers.markdown import (
    _auto_summary,
    _manifest_from_parsed,
    _parse_sections_from_markdown,
    write_staging_from_parsed,
)
from potpie_context_engine.adapters.outbound.resources.parsers.pdf_docling_provenance import (
    build_sections_from_docling,
)
from potpie_context_engine.domain.resource_models import (
    CHUNK_TARGET_DEFAULT,
    ResourceManifest,
)

_DOCUMENTS_INSTALL = "pip install potpie[documents]"
_PROVENANCE_VERSION = 2

_MIME_BY_SUFFIX = {
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".html": "text/html",
    ".htm": "text/html",
}


def _require_docling_converter() -> object:
    try:
        from docling.document_converter import DocumentConverter
    except ImportError as exc:
        raise ImportError(
            f"Docling conversion requires potpie[documents]. Install with: {_DOCUMENTS_INSTALL}"
        ) from exc
    return DocumentConverter()


def _source_kind_for_suffix(suffix: str) -> str:
    mapping = {
        ".docx": "docx",
        ".pptx": "pptx",
        ".xlsx": "xlsx",
        ".html": "html",
        ".htm": "html",
    }
    return mapping.get(suffix, "markdown")


def parse_docling_file(
    path: Path,
    *,
    chunk_target: int = CHUNK_TARGET_DEFAULT,
    staging_root: Path | None = None,
) -> tuple[
    ResourceManifest,
    dict[str, list[str]],
    list,
    dict,
    dict[str, list[str]],
]:
    suffix = path.suffix.lower()
    source_kind = _source_kind_for_suffix(suffix)
    mime_type = _MIME_BY_SUFFIX.get(suffix, "application/octet-stream")

    converter = _require_docling_converter()
    result = converter.convert(str(path))
    document = result.document
    artifacts_dir = staging_root / "artifacts" if staging_root else None

    provenance_version = 0
    parser_tier = "docling"
    elements: list = []
    provenance_map: dict = {}
    section_ocr: dict[str, list[str]] = {}

    try:
        parsed_sections, elements, provenance_map, section_ocr = build_sections_from_docling(
            document,
            chunk_target=chunk_target,
            artifacts_dir=artifacts_dir,
        )
    except Exception:
        parsed_sections = []

    if not parsed_sections:
        markdown = (document.export_to_markdown() or "").strip()
        if not markdown:
            raise ValueError(f"Docling produced no text for {path.name}")
        parsed_sections = _parse_sections_from_markdown(markdown, chunk_target)
        if not parsed_sections:
            raise ValueError(f"Docling markdown had no sections for {path.name}")
        provenance_version = 0
        parser_tier = "docling-markdown"

    manifest, section_texts = _manifest_from_parsed(path, source_kind, parsed_sections)
    content_hash = file_sha256(path)
    updated_sections = []
    for section in manifest.sections:
        body = "\n\n".join(section_texts.get(section.slug, []))
        updated_sections.append(
            section.model_copy(update={"summary": _auto_summary(section.title, body)})
        )
    if elements and provenance_map:
        provenance_version = _PROVENANCE_VERSION

    manifest = manifest.model_copy(
        update={
            "sections": updated_sections,
            "source_content_hash": content_hash,
            "mime_type": mime_type,
            "provenance_version": provenance_version,
            "parser_tier": parser_tier,
        }
    )
    return manifest, section_texts, elements, provenance_map, section_ocr


def parse_docling_file_to_staging(
    source: Path,
    out_dir: Path,
    *,
    chunk_target: int = CHUNK_TARGET_DEFAULT,
) -> ResourceManifest:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest, section_texts, elements, provenance_map, section_ocr = parse_docling_file(
        source,
        chunk_target=chunk_target,
        staging_root=out_dir,
    )
    write_staging_from_parsed(
        out_dir,
        manifest,
        section_texts,
        section_ocr=section_ocr,
        elements=elements or None,
        chunk_provenance=provenance_map or None,
    )
    return manifest


__all__ = ["parse_docling_file", "parse_docling_file_to_staging"]
