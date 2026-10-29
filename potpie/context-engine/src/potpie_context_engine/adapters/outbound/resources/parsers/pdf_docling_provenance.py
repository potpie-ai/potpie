"""Docling PDF parser with element ledger + provenance-aware chunking."""

from __future__ import annotations

import csv
import io
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from potpie_context_engine.adapters.outbound.resources.local_chunk_store import (
    file_sha256,
)
from potpie_context_engine.adapters.outbound.resources.parsers.markdown import (
    _auto_summary,
    auto_keywords,
    _manifest_from_parsed,
    _slugify,
    _split_hard_cap,
    _split_oversized_section,
    ParsedSection,
    write_staging_from_parsed,
)
from potpie_context_engine.adapters.outbound.resources.parsers.cleaning import (
    normalize_extracted_text,
)
from potpie_context_engine.domain.resource_models import (
    CHUNK_HARD_CAP,
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
_PAGE_FURNITURE_LABELS = frozenset({"page_header", "page_footer"})
_LAYOUT_SENSITIVE_LABELS = frozenset({"code", "code_block"})


@dataclass(slots=True)
class _LedgerElement:
    record: DocumentElementRecord
    searchable_text: str = ""
    provenance: list[ChunkProvenanceRecord] = field(default_factory=list)


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


def _normalized_label(item: object) -> str:
    return _label_name(item).strip().lower()


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
) -> list[tuple[int | None, list[float] | None, int | None, int | None]]:
    fields: list[tuple[int | None, list[float] | None, int | None, int | None]] = []
    prov_list = getattr(item, "prov", None) or []
    for prov in prov_list:
        page_number = getattr(prov, "page_no", None)
        bbox = _bbox_list(getattr(prov, "bbox", None))
        char_start: int | None = None
        char_end: int | None = None
        charspan = getattr(prov, "charspan", None)
        if charspan and len(charspan) >= 2:
            char_start = int(charspan[0])
            char_end = int(charspan[1])
        fields.append((page_number, bbox, char_start, char_end))
    return fields


def _element_text(item: object, document: object) -> str:
    raw = ""
    if hasattr(item, "text") and item.text:
        raw = str(item.text)
    elif hasattr(item, "get_text"):
        try:
            raw = str(item.get_text())
        except TypeError:
            raw = str(item.get_text(document))
    return normalize_extracted_text(
        raw,
        preserve_layout=_normalized_label(item) in _LAYOUT_SENSITIVE_LABELS,
    )


def _split_markdown_row(line: str) -> list[str]:
    cells: list[str] = []
    cell: list[str] = []
    escaped = False
    for char in line.strip().strip("|"):
        if char == "|" and not escaped:
            cells.append("".join(cell).strip().replace(r"\|", "|"))
            cell = []
        else:
            cell.append(char)
        escaped = char == "\\" and not escaped
        if char != "\\":
            escaped = False
    cells.append("".join(cell).strip().replace(r"\|", "|"))
    return cells


def _table_matrix(
    table_text: str, *, table_format: str
) -> tuple[list[str], list[list[str]]]:
    lines = [line.strip() for line in table_text.splitlines() if line.strip()]
    if not lines:
        return [], []

    if table_format == "markdown":
        matrix = [_split_markdown_row(line) for line in lines]
        if len(matrix) > 1 and all(
            re.fullmatch(r":?-{3,}:?", cell.replace(" ", "")) for cell in matrix[1]
        ):
            del matrix[1]
    else:
        try:
            matrix = [
                [normalize_extracted_text(cell) for cell in row]
                for row in csv.reader(io.StringIO(table_text))
                if any(cell.strip() for cell in row)
            ]
        except csv.Error:
            matrix = []

    if not matrix:
        return [], []
    width = max(len(row) for row in matrix)
    headers = [
        (matrix[0][idx].strip() if idx < len(matrix[0]) else "") or f"column-{idx + 1}"
        for idx in range(width)
    ]
    rows = [row + [""] * (width - len(row)) for row in matrix[1:]]
    return headers, rows


def _table_searchable_chunks(
    table_text: str,
    *,
    table_number: int,
    chunk_target: int,
    table_format: str,
) -> list[str]:
    headers, rows = _table_matrix(table_text, table_format=table_format)
    if not headers:
        return _split_hard_cap(normalize_extracted_text(table_text))

    max_prefix = min(CHUNK_HARD_CAP // 3, max(128, chunk_target // 3))
    safe_headers = list(headers)
    if any(len(header) > max_prefix // 2 for header in headers):
        safe_headers = [
            header if len(header) <= max_prefix // 2 else f"column-{idx + 1}"
            for idx, header in enumerate(headers)
        ]
    prefix = f"Table {table_number} columns: {' | '.join(safe_headers)}"
    if len(prefix) > max_prefix:
        prefix = prefix[:max_prefix].rstrip()
    limit = min(CHUNK_HARD_CAP, max(chunk_target, len(prefix) + 64))
    chunks: list[str] = []
    current = prefix

    def flush() -> None:
        nonlocal current
        if current != prefix:
            chunks.append(current)
        current = prefix

    for row_number, row in enumerate(rows, start=1):
        row_prefix = f"Row {row_number}: "
        body_limit = max(1, limit - len(prefix) - len(row_prefix) - 1)
        fragments: list[str] = []
        packed_cells: list[str] = []

        def flush_cells() -> None:
            if packed_cells:
                fragments.append(f"{row_prefix}{' | '.join(packed_cells)}")
                packed_cells.clear()

        for column_index, (safe_header, value) in enumerate(
            zip(safe_headers, row, strict=False)
        ):
            header = safe_header
            label = f"{header}="
            cell = f"{label}{value}"
            projected = " | ".join([*packed_cells, cell])
            if len(projected) <= body_limit:
                packed_cells.append(cell)
                continue

            flush_cells()
            value_limit = body_limit - len(label)
            if value_limit < 1:
                # Long labels are replaced by a bounded column id above.
                label = f"column-{column_index + 1}="
                value_limit = max(1, body_limit - len(label))
            if len(value) <= value_limit:
                packed_cells.append(cell)
                continue
            start = 0
            while start < len(value):
                end = min(start + value_limit, len(value))
                if end < len(value):
                    window = value[start:end]
                    break_at = window.rfind(" ")
                    if break_at > value_limit // 3:
                        end = start + break_at
                fragments.append(f"{row_prefix}{label}{value[start:end].strip()}")
                start = end if end > start else start + 1
        flush_cells()

        for fragment in fragments:
            if len(current) + 1 + len(fragment) > limit:
                flush()
            current = f"{current}\n{fragment}"

    flush()
    if chunks:
        return chunks
    return [prefix]


def _repeated_page_furniture(
    items: list[tuple[object, object]], document: object
) -> set[tuple[str, str]]:
    pages_by_text: dict[tuple[str, str], set[int]] = defaultdict(set)
    for item, _level in items:
        label = _normalized_label(item)
        if label not in _PAGE_FURNITURE_LABELS:
            continue
        text = _element_text(item, document)
        for page_number, _, _, _ in _provenance_fields(item):
            if text and page_number is not None:
                pages_by_text[(label, text.casefold())].add(page_number)
    return {key for key, pages in pages_by_text.items() if len(pages) >= 2}


def _try_ocr_picture(image_path: Path) -> str:
    try:
        from potpie_context_engine.adapters.outbound.resources.enrichers.ocr_rapidocr import (
            ocr_image_text,
        )

        return ocr_image_text(image_path).strip()
    except Exception as exc:
        logger.debug("picture OCR skipped: %s", exc)
        return ""


def _try_vision_caption(image_path: Path, vision_provider: str) -> str:
    provider = (vision_provider or "local").strip().lower()
    if provider in {"", "none", "disabled"}:
        return ""
    try:
        if provider == "openai":
            from potpie_context_engine.adapters.outbound.resources.enrichers.vision_openai import (
                caption_image_openai,
            )

            return caption_image_openai(image_path).strip()
        if provider in {"local", "ollama"}:
            from potpie_context_engine.adapters.outbound.resources.enrichers.vision_httpx_ollama import (
                caption_image_local,
                ollama_reachable,
            )

            if not ollama_reachable():
                return ""
            return caption_image_local(image_path).strip()
    except Exception as exc:  # noqa: BLE001 - visual enrichment is optional
        logger.debug("picture vision caption skipped: %s", exc)
        return ""
    raise ValueError(
        f"unsupported vision provider: {vision_provider!r} (use local, openai, or none)"
    )


def _enrichment_diagnostics(
    elements: list[DocumentElementRecord], *, vision_provider: str
) -> dict[str, object]:
    """Summarize observed PDF enrichment without claiming provider success.

    OCR and vision adapters intentionally fail soft. Persisting the observed
    counts in the manifest makes a degraded ingest visible to callers and
    stress reports while keeping the raw extracted content unchanged.
    """
    pictures = [element for element in elements if element.element_type == "picture"]
    tables = [element for element in elements if element.element_type == "table"]
    normalized_provider = (vision_provider or "local").strip().lower()
    warnings: list[str] = []
    if pictures and not any(element.ocr_text.strip() for element in pictures):
        warnings.append("no_picture_ocr_observed")
    if (
        pictures
        and normalized_provider not in {"", "none", "disabled"}
        and not any(element.visual_caption.strip() for element in pictures)
    ):
        warnings.append("no_picture_visual_caption_observed")
    return {
        "picture_enrichment": {
            "pictures_total": len(pictures),
            "pictures_with_ocr": sum(bool(element.ocr_text.strip()) for element in pictures),
            "pictures_with_visual_caption": sum(
                bool(element.visual_caption.strip()) for element in pictures
            ),
            "vision_provider": normalized_provider or "local",
            "warnings": warnings,
        },
        "tables_total": len(tables),
    }


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
        parts = _split_hard_cap(text, min(chunk_target, CHUNK_HARD_CAP))
        ledger_provenance = ledger.provenance or [
            ChunkProvenanceRecord(
                element_id=ledger.record.element_id,
                page_number=ledger.record.page_number,
                bbox=ledger.record.bbox,
                char_start=ledger.record.char_start,
                char_end=ledger.record.char_end,
            )
        ]
        source_offset = 0
        for part in parts:
            sep = "\n\n" if current_text else ""
            projected = f"{current_text}{sep}{part}" if current_text else part
            if current_text and len(projected) > min(chunk_target, CHUNK_HARD_CAP):
                flush()
                projected = part
            current_text = projected
            part_start = source_offset
            part_end = min(source_offset + len(part), len(text))
            for provenance in ledger_provenance:
                has_charspan = (
                    provenance.char_start is not None
                    and provenance.char_end is not None
                    and provenance.char_end > provenance.char_start
                )
                if has_charspan and provenance.char_end <= part_start:
                    continue
                if has_charspan and provenance.char_start >= part_end:
                    continue
                current_prov.append(provenance)
            source_offset = part_end

    flush()

    return staged


def build_sections_from_docling(
    document: object,
    *,
    chunk_target: int,
    artifacts_dir: Path | None = None,
    vision_provider: str = "local",
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
        record: DocumentElementRecord,
        searchable: str,
        provenance: list[ChunkProvenanceRecord] | None = None,
    ) -> None:
        current_group.append(
            _LedgerElement(
                record=record,
                searchable_text=searchable,
                provenance=provenance
                or [
                    ChunkProvenanceRecord(
                        element_id=record.element_id,
                        page_number=record.page_number,
                        bbox=record.bbox,
                        char_start=0,
                        char_end=len(searchable),
                    )
                ],
            )
        )

    item_rows = list(document.iterate_items())
    repeated_furniture = _repeated_page_furniture(item_rows, document)

    for item, _level in item_rows:
        label = _label_name(item)
        normalized_label = label.strip().lower()
        if (
            normalized_label in _PAGE_FURNITURE_LABELS
            and (normalized_label, _element_text(item, document).casefold())
            in repeated_furniture
        ):
            continue

        element_index += 1
        element_id = f"elem-{element_index:05d}"
        item_provenance = _provenance_fields(item)
        page_number, bbox, char_start, char_end = (
            item_provenance[0] if item_provenance else (None, None, None, None)
        )
        provenance_records = [
            ChunkProvenanceRecord(
                element_id=(
                    element_id if prov_idx == 0 else f"{element_id}#{prov_idx + 1}"
                ),
                page_number=fields[0],
                bbox=fields[1],
                char_start=fields[2],
                char_end=fields[3],
            )
            for prov_idx, fields in enumerate(item_provenance)
        ]

        if isinstance(item, TableItem) or label.upper() == "TABLE":
            table_counter += 1
            csv_rel: str | None = None
            table_text = ""
            table_format = "markdown"
            if artifacts_dir is not None:
                artifacts_dir.mkdir(parents=True, exist_ok=True)
                csv_path = artifacts_dir / f"table-{table_counter:04d}.csv"
                try:
                    df = item.export_to_dataframe(doc=document)
                    df.to_csv(csv_path, index=False)
                    csv_rel = f"artifacts/table-{table_counter:04d}.csv"
                    table_text = normalize_extracted_text(df.to_csv(index=False) or "")
                    table_format = "csv"
                except Exception as exc:
                    logger.debug("table CSV export failed: %s", exc)
                    try:
                        table_text = normalize_extracted_text(
                            item.export_to_markdown(doc=document) or ""
                        )
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
                    table_text = normalize_extracted_text(
                        item.export_to_markdown(doc=document) or ""
                    )
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
                table_number=table_counter,
            )
            elements.append(record)
            searchable_chunks = _table_searchable_chunks(
                table_text or f"[table {table_counter}]",
                table_number=table_counter,
                chunk_target=chunk_target,
                table_format=table_format,
            )
            for searchable in searchable_chunks:
                add_ledger(record, searchable, provenance_records)
            continue

        if isinstance(item, PictureItem) or label.upper() == "PICTURE":
            picture_counter += 1
            img_rel: str | None = None
            ocr_text = ""
            caption = _element_text(item, document)
            visual_caption = ""
            if artifacts_dir is not None:
                artifacts_dir.mkdir(parents=True, exist_ok=True)
                img_path = artifacts_dir / f"img-{picture_counter:04d}.png"
                try:
                    with img_path.open("wb") as fp:
                        item.get_image(document).save(fp, "PNG")
                    img_rel = f"artifacts/img-{picture_counter:04d}.png"
                    ocr_text = _try_ocr_picture(img_path)
                    visual_caption = _try_vision_caption(img_path, vision_provider)
                except Exception as exc:
                    logger.debug("picture export failed: %s", exc)
            searchable_parts = [f"Figure {picture_counter}"]
            if caption:
                searchable_parts.append(f"Caption: {caption}")
            if visual_caption:
                searchable_parts.append(f"Visual caption: {visual_caption}")
            if ocr_text:
                searchable_parts.append(f"OCR: {ocr_text}")
            searchable = "\n".join(searchable_parts)
            record = DocumentElementRecord(
                element_id=element_id,
                element_type="picture",
                text=searchable[:8000],
                artifact_ref=img_rel,
                page_number=page_number,
                bbox=bbox,
                char_start=char_start,
                char_end=char_end,
                text_hash=text_sha256(searchable),
                caption=caption,
                ocr_text=ocr_text,
                visual_caption=visual_caption,
                figure_number=picture_counter,
            )
            elements.append(record)
            add_ledger(record, searchable, provenance_records)
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
        add_ledger(record, text, provenance_records)

    if current_group or not section_element_groups:
        section_titles.append(current_title)
        section_element_groups.append(current_group)

    parsed_sections: list[ParsedSection] = []
    provenance_map: dict[tuple[str, int], list[ChunkProvenanceRecord]] = {}
    section_ocr: dict[str, list[str]] = {}
    used_section_slugs: set[str] = set()

    def reserve_section_slug(base_slug: str) -> str:
        candidate = base_slug[:80].rstrip("-") or "section"
        suffix = 2
        while candidate in used_section_slugs:
            suffix_text = f"-{suffix}"
            candidate = f"{base_slug[:80 - len(suffix_text)].rstrip('-')}{suffix_text}"
            suffix += 1
        used_section_slugs.add(candidate)
        return candidate

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
            part_slug = reserve_section_slug(part.slug)
            if part_slug != part.slug:
                part = ParsedSection(
                    slug=part_slug,
                    title=part.title,
                    ordinal=part.ordinal,
                    text=part.text,
                    chunks=part.chunks,
                )
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
    vision_provider: str = "local",
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
            vision_provider=vision_provider,
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
            "ingestion_diagnostics": _enrichment_diagnostics(
                elements, vision_provider=vision_provider
            ),
        }
    )
    return manifest, section_texts, elements, provenance_map, section_ocr


def parse_pdf_docling_provenance_to_staging(
    source: Path,
    out_dir: Path,
    *,
    chunk_target: int = CHUNK_TARGET_DEFAULT,
    vision_provider: str = "local",
) -> ResourceManifest:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest, section_texts, elements, provenance_map, section_ocr = (
        parse_pdf_docling_provenance(
            source,
            chunk_target=chunk_target,
            staging_root=out_dir,
            vision_provider=vision_provider,
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
