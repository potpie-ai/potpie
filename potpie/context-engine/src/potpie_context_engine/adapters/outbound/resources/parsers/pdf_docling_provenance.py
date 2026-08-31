"""Docling PDF parser with element ledger + provenance-aware chunking."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from potpie_context_engine.adapters.outbound.resources.local_chunk_store import (
    file_sha256,
)
from potpie_context_engine.adapters.outbound.resources.parsers.markdown import (
    _auto_summary,
    auto_keywords,
    _fit_section_chunks,
    _manifest_from_parsed,
    _slugify,
    _split_hard_cap,
    _split_oversized_section,
    ParsedSection,
    write_staging_from_parsed,
)
from potpie_context_engine.domain.resource_models import (
    CHUNK_TARGET_DEFAULT,
    ChunkProvenanceRecord,
    DocumentElementRecord,
    ResourceManifest,
    SECTION_CHUNK_MAX,
    text_sha256,
)

logger = logging.getLogger(__name__)

_DOCUMENTS_INSTALL = "pip install potpie[documents]"
_PROVENANCE_VERSION = 2
_SECTION_HEADER_LABELS = frozenset(
    {"section_header", "title", "SECTION_HEADER", "TITLE", "SectionHeaderItem"}
)
_SKIP_LABELS = frozenset(
    {"page_header", "page_footer", "PAGE_HEADER", "PAGE_FOOTER", "footnote", "FOOTNOTE"}
)


@dataclass(slots=True)
class _LedgerElement:
    record: DocumentElementRecord
    searchable_text: str = ""


@dataclass(slots=True)
class _StagedChunk:
    text: str
    provenance: list[ChunkProvenanceRecord] = field(default_factory=list)
    ocr_text: str = ""


def _require_docling_types() -> tuple[Any, Any, Any, Any, Any, Any]:
    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import (
            AcceleratorDevice,
            AcceleratorOptions,
            PdfPipelineOptions,
        )
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling_core.types.doc import (
            DocItemLabel,
            PictureItem,
            TableItem,
            TextItem,
        )
    except ImportError as exc:
        raise ImportError(
            f"Docling PDF parsing requires potpie[documents]. Install with: {_DOCUMENTS_INSTALL}"
        ) from exc
    return (
        DocumentConverter,
        PdfFormatOption,
        PdfPipelineOptions,
        InputFormat,
        AcceleratorOptions,
        AcceleratorDevice,
        DocItemLabel,
        TextItem,
        TableItem,
        PictureItem,
    )


def _build_converter() -> object:
    (
        DocumentConverter,
        PdfFormatOption,
        PdfPipelineOptions,
        InputFormat,
        AcceleratorOptions,
        AcceleratorDevice,
        *_,
    ) = _require_docling_types()
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True
    pipeline_options.generate_picture_images = True
    pipeline_options.accelerator_options = AcceleratorOptions(
        device=AcceleratorDevice.CPU,
    )
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        }
    )


def _label_name(item: object) -> str:
    label = getattr(item, "label", None)
    if label is None:
        return "text"
    if hasattr(label, "name"):
        return str(label.name)
    return str(label)


def _bbox_list(bbox: object | None) -> list[float] | None:
    if bbox is None:
        return None
    if hasattr(bbox, "as_tuple"):
        values = bbox.as_tuple()
    elif hasattr(bbox, "l"):
        values = (bbox.l, bbox.t, bbox.r, bbox.b)
    elif isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        values = bbox
    else:
        return None
    return [float(v) for v in values]


def _provenance_fields(
    item: object,
) -> tuple[int | None, list[float] | None, int | None, int | None]:
    page_number: int | None = None
    bbox: list[float] | None = None
    char_start: int | None = None
    char_end: int | None = None
    prov_list = getattr(item, "prov", None) or []
    if prov_list:
        prov = prov_list[0]
        page_number = getattr(prov, "page_no", None)
        bbox = _bbox_list(getattr(prov, "bbox", None))
        charspan = getattr(prov, "charspan", None)
        if charspan and len(charspan) >= 2:
            char_start = int(charspan[0])
            char_end = int(charspan[1])
    return page_number, bbox, char_start, char_end


def _element_text(item: object, document: object) -> str:
    if hasattr(item, "text") and item.text:
        return str(item.text).strip()
    if hasattr(item, "get_text"):
        try:
            return str(item.get_text()).strip()
        except TypeError:
            return str(item.get_text(document)).strip()
    return ""


def _try_ocr_picture(image_path: Path) -> str:
    try:
        from potpie_context_engine.adapters.outbound.resources.enrichers.ocr_rapidocr import (
            ocr_image_text,
        )

        return ocr_image_text(image_path).strip()
    except Exception as exc:
        logger.debug("picture OCR skipped: %s", exc)
        return ""


def _chunk_elements(
    elements: list[_LedgerElement],
    *,
    chunk_target: int,
) -> list[_StagedChunk]:
    if not elements:
        return []

    staged: list[_StagedChunk] = []
    current_text = ""
    current_prov: list[ChunkProvenanceRecord] = []

    def flush() -> None:
        nonlocal current_text, current_prov
        if current_text.strip():
            staged.append(
                _StagedChunk(text=current_text.strip(), provenance=list(current_prov))
            )
        current_text = ""
        current_prov = []

    for ledger in elements:
        text = ledger.searchable_text.strip()
        if not text:
            continue
        sep = "\n\n" if current_text else ""
        projected = f"{current_text}{sep}{text}" if current_text else text
        if current_text and len(projected) > chunk_target:
            flush()
            current_text = text
            current_prov = [
                ChunkProvenanceRecord(
                    element_id=ledger.record.element_id,
                    page_number=ledger.record.page_number,
                    bbox=ledger.record.bbox,
                    char_start=0,
                    char_end=len(text),
                )
            ]
        else:
            if sep:
                current_text = projected
            else:
                current_text = text
            current_prov.append(
                ChunkProvenanceRecord(
                    element_id=ledger.record.element_id,
                    page_number=ledger.record.page_number,
                    bbox=ledger.record.bbox,
                    char_start=0,
                    char_end=len(text),
                )
            )

    flush()

    normalized: list[_StagedChunk] = []
    for chunk in staged:
        parts = _split_hard_cap(chunk.text)
        for part in parts:
            normalized.append(
                _StagedChunk(
                    text=part,
                    provenance=list(chunk.provenance),
                    ocr_text=chunk.ocr_text,
                )
            )

    texts = [c.text for c in normalized]
    fitted = _fit_section_chunks(texts)
    if len(fitted) == len(normalized):
        return [
            _StagedChunk(
                text=fitted[i],
                provenance=normalized[i].provenance,
                ocr_text=normalized[i].ocr_text,
            )
            for i in range(len(fitted))
        ]

    refit: list[_StagedChunk] = []
    src_idx = 0
    src_offset = 0
    for fitted_text in fitted:
        prov: list[ChunkProvenanceRecord] = []
        remaining = fitted_text
        while remaining and src_idx < len(normalized):
            src = normalized[src_idx]
            src_text = src.text[src_offset:]
            if not src_text:
                src_idx += 1
                src_offset = 0
                continue
            take = min(len(remaining), len(src_text))
            if take > 0:
                for row in src.provenance:
                    if row not in prov:
                        prov.append(row)
                remaining = remaining[take:]
                src_offset += take
                if src_offset >= len(src.text):
                    src_idx += 1
                    src_offset = 0
            else:
                break
        refit.append(_StagedChunk(text=fitted_text, provenance=prov))
    return refit


def build_sections_from_docling(
    document: object,
    *,
    chunk_target: int,
    artifacts_dir: Path | None = None,
) -> tuple[
    list[ParsedSection],
    list[DocumentElementRecord],
    dict[tuple[str, int], list[ChunkProvenanceRecord]],
    dict[str, list[str]],
]:
    (
        _,
        _,
        _,
        _,
        _,
        _,
        DocItemLabel,
        TextItem,
        TableItem,
        PictureItem,
    ) = _require_docling_types()

    elements: list[DocumentElementRecord] = []
    section_titles: list[str] = []
    section_element_groups: list[list[_LedgerElement]] = []
    current_title = "body"
    current_group: list[_LedgerElement] = []
    element_index = 0
    table_counter = 0
    picture_counter = 0

    def start_section(title: str) -> None:
        nonlocal current_title, current_group
        if current_group:
            section_titles.append(current_title)
            section_element_groups.append(current_group)
        current_title = title.strip() or "body"
        current_group = []

    def add_ledger(
        record: DocumentElementRecord, searchable: str, ocr: str = ""
    ) -> None:
        current_group.append(
            _LedgerElement(
                record=record,
                searchable_text=searchable,
            )
        )

    for item, _level in document.iterate_items():
        label = _label_name(item)
        if label in _SKIP_LABELS:
            continue

        element_index += 1
        element_id = f"elem-{element_index:05d}"
        page_number, bbox, char_start, char_end = _provenance_fields(item)

        if isinstance(item, TableItem) or label.upper() == "TABLE":
            table_counter += 1
            csv_rel: str | None = None
            table_text = ""
            if artifacts_dir is not None:
                artifacts_dir.mkdir(parents=True, exist_ok=True)
                csv_path = artifacts_dir / f"table-{table_counter:04d}.csv"
                try:
                    # Probe for pandas up front: export_to_dataframe needs it,
                    # and a missing install should hit the except cleanly.
                    import pandas as pd  # noqa: F401

                    df = item.export_to_dataframe(doc=document)
                    df.to_csv(csv_path, index=False)
                    csv_rel = f"artifacts/table-{table_counter:04d}.csv"
                    table_text = df.to_csv(index=False).strip()
                except Exception as exc:
                    logger.debug("table CSV export failed: %s", exc)
                    try:
                        table_text = item.export_to_markdown(doc=document) or ""
                    except Exception:
                        table_text = _element_text(item, document)
                try:
                    img_path = artifacts_dir / f"table-{table_counter:04d}.png"
                    with img_path.open("wb") as fp:
                        item.get_image(document).save(fp, "PNG")
                except Exception:
                    pass
            else:
                try:
                    table_text = item.export_to_markdown(doc=document) or ""
                except Exception:
                    table_text = _element_text(item, document)

            record = DocumentElementRecord(
                element_id=element_id,
                element_type="table",
                text=table_text[:8000],
                artifact_ref=csv_rel,
                page_number=page_number,
                bbox=bbox,
                char_start=char_start,
                char_end=char_end,
                text_hash=text_sha256(table_text),
            )
            elements.append(record)
            searchable = table_text or f"[table {table_counter}]"
            add_ledger(record, searchable)
            continue

        if isinstance(item, PictureItem) or label.upper() == "PICTURE":
            picture_counter += 1
            img_rel: str | None = None
            ocr_text = ""
            caption = _element_text(item, document)
            if artifacts_dir is not None:
                artifacts_dir.mkdir(parents=True, exist_ok=True)
                img_path = artifacts_dir / f"img-{picture_counter:04d}.png"
                try:
                    with img_path.open("wb") as fp:
                        item.get_image(document).save(fp, "PNG")
                    img_rel = f"artifacts/img-{picture_counter:04d}.png"
                    ocr_text = _try_ocr_picture(img_path)
                except Exception as exc:
                    logger.debug("picture export failed: %s", exc)
            searchable = ocr_text or caption or f"[figure {picture_counter}]"
            record = DocumentElementRecord(
                element_id=element_id,
                element_type="picture",
                text=(caption or ocr_text or "")[:8000],
                artifact_ref=img_rel,
                page_number=page_number,
                bbox=bbox,
                char_start=char_start,
                char_end=char_end,
                text_hash=text_sha256(searchable),
            )
            elements.append(record)
            add_ledger(record, searchable)
            continue

        text = _element_text(item, document)
        is_header = label in _SECTION_HEADER_LABELS or label.upper() in {
            "SECTION_HEADER",
            "TITLE",
        }
        if is_header and text:
            start_section(text)
            record = DocumentElementRecord(
                element_id=element_id,
                element_type="section_header",
                text=text,
                page_number=page_number,
                bbox=bbox,
                char_start=char_start,
                char_end=char_end,
                text_hash=text_sha256(text),
            )
            elements.append(record)
            continue

        if not text:
            continue

        record = DocumentElementRecord(
            element_id=element_id,
            element_type=label.lower(),
            text=text,
            page_number=page_number,
            bbox=bbox,
            char_start=char_start,
            char_end=char_end,
            text_hash=text_sha256(text),
        )
        elements.append(record)
        add_ledger(record, text)

    if current_group or not section_element_groups:
        section_titles.append(current_title)
        section_element_groups.append(current_group)

    parsed_sections: list[ParsedSection] = []
    provenance_map: dict[tuple[str, int], list[ChunkProvenanceRecord]] = {}
    section_ocr: dict[str, list[str]] = {}

    for ordinal, (title, group) in enumerate(
        zip(section_titles, section_element_groups, strict=False)
    ):
        slug = _slugify(title, f"section-{ordinal}")
        staged = _chunk_elements(group, chunk_target=chunk_target)
        if not staged:
            continue
        chunks = [c.text for c in staged]
        body = "\n\n".join(chunks)
        parsed = ParsedSection(
            slug=slug,
            title=title[:200],
            ordinal=ordinal,
            text=body,
            chunks=chunks,
        )
        for part_idx, part in enumerate(_split_oversized_section(parsed)):
            parsed_sections.append(part)
            ocr_rows: list[str] = []
            base = part_idx * SECTION_CHUNK_MAX
            for seq in range(len(part.chunks)):
                src_idx = min(base + seq, len(staged) - 1)
                provenance_map[(part.slug, seq)] = list(staged[src_idx].provenance)
                ocr_rows.append(staged[src_idx].ocr_text)
            section_ocr[part.slug] = ocr_rows

    return parsed_sections, elements, provenance_map, section_ocr


def parse_pdf_docling_provenance(
    path: Path,
    *,
    chunk_target: int = CHUNK_TARGET_DEFAULT,
    staging_root: Path | None = None,
) -> tuple[
    ResourceManifest,
    dict[str, list[str]],
    list[DocumentElementRecord],
    dict[tuple[str, int], list[ChunkProvenanceRecord]],
    dict[str, list[str]],
]:
    converter = _build_converter()
    result = converter.convert(str(path))
    document = result.document
    artifacts_dir = staging_root / "artifacts" if staging_root else None

    parsed_sections, elements, provenance_map, section_ocr = (
        build_sections_from_docling(
            document,
            chunk_target=chunk_target,
            artifacts_dir=artifacts_dir,
        )
    )
    if not parsed_sections:
        raise ValueError("Docling produced no sections from element ledger")

    manifest, section_texts = _manifest_from_parsed(path, "pdf", parsed_sections)
    content_hash = file_sha256(path)
    mime_type = "application/pdf"
    updated_sections = []
    for section in manifest.sections:
        body = "\n\n".join(section_texts.get(section.slug, []))
        updated_sections.append(
            section.model_copy(
                update={
                    "summary": _auto_summary(section.title, body),
                    "keywords": auto_keywords(section.title, body),
                }
            )
        )
    manifest = manifest.model_copy(
        update={
            "sections": updated_sections,
            "source_content_hash": content_hash,
            "mime_type": mime_type,
            "provenance_version": _PROVENANCE_VERSION,
            "parser_tier": "docling",
        }
    )
    return manifest, section_texts, elements, provenance_map, section_ocr


def parse_pdf_docling_provenance_to_staging(
    source: Path,
    out_dir: Path,
    *,
    chunk_target: int = CHUNK_TARGET_DEFAULT,
) -> ResourceManifest:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest, section_texts, elements, provenance_map, section_ocr = (
        parse_pdf_docling_provenance(
            source,
            chunk_target=chunk_target,
            staging_root=out_dir,
        )
    )
    write_staging_from_parsed(
        out_dir,
        manifest,
        section_texts,
        section_ocr=section_ocr,
        elements=elements,
        chunk_provenance=provenance_map,
    )
    return manifest


__all__ = [
    "build_sections_from_docling",
    "parse_pdf_docling_provenance",
    "parse_pdf_docling_provenance_to_staging",
]
