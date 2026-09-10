"""Route source files to the correct tiered parser."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from potpie_context_engine.adapters.outbound.resources.capabilities import (
    documents_capabilities,
)
from potpie_context_engine.adapters.outbound.resources.local_chunk_store import (
    file_sha256,
)
from potpie_context_engine.adapters.outbound.resources.parsers.markdown import (
    parse_file_to_staging as parse_text_to_staging,
)
from potpie_context_engine.domain.resource_models import (
    CHUNK_TARGET_DEFAULT,
    ResourceManifest,
)

logger = logging.getLogger(__name__)

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"}
_OFFICE_SUFFIXES = {".docx", ".pptx", ".xlsx"}
_HTML_SUFFIXES = {".html", ".htm"}


@dataclass(slots=True)
class ParseOptions:
    chunk_target: int = CHUNK_TARGET_DEFAULT
    vision_provider: str = "local"
    prefer_docling: bool = True
    allow_degraded: bool = False


def _manifest_degraded(
    manifest: ResourceManifest,
    *,
    parser_tier: str,
    source_path: Path,
) -> ResourceManifest:
    return manifest.model_copy(
        update={
            "provenance_version": 0,
            "parser_tier": parser_tier,
            "source_content_hash": file_sha256(source_path),
            "mime_type": "application/pdf",
        }
    )


def _parse_pdf_degraded(
    source: Path, out_dir: Path, *, chunk_target: int
) -> ResourceManifest:
    import json

    caps = documents_capabilities()
    errors: list[str] = []

    if caps["pypdfium2"]:
        try:
            from potpie_context_engine.adapters.outbound.resources.parsers.pdf_pypdfium2 import (
                parse_pdf_pypdfium2_to_staging,
            )

            manifest = parse_pdf_pypdfium2_to_staging(
                source, out_dir, chunk_target=chunk_target
            )
            manifest = _manifest_degraded(
                manifest, parser_tier="pypdfium2-degraded", source_path=source
            )
            (out_dir / "meta.json").write_text(
                json.dumps(manifest.model_dump(), indent=2),
                encoding="utf-8",
            )
            return manifest
        except Exception as exc:
            logger.warning("pypdfium2 degraded PDF parse failed: %s", exc)
            errors.append(f"pypdfium2: {exc}")

    if caps["pypdf"]:
        from potpie_context_engine.adapters.outbound.resources.parsers.pdf_pypdf import (
            parse_pdf_to_staging,
        )

        manifest = parse_pdf_to_staging(source, out_dir, chunk_target=chunk_target)
        manifest = _manifest_degraded(
            manifest, parser_tier="pypdf-degraded", source_path=source
        )
        (out_dir / "meta.json").write_text(
            json.dumps(manifest.model_dump(), indent=2),
            encoding="utf-8",
        )
        return manifest

    detail = "; ".join(errors) if errors else "no degraded PDF parser available"
    raise ImportError(
        f"Degraded PDF parsing unavailable ({detail}). "
        "Install potpie[documents] for Docling provenance or ensure pypdf is installed."
    )


def _parse_pdf_to_staging(
    source: Path,
    out_dir: Path,
    *,
    chunk_target: int,
    allow_degraded: bool,
) -> ResourceManifest:
    caps = documents_capabilities()

    if caps["docling"]:
        from potpie_context_engine.adapters.outbound.resources.parsers.pdf_docling_provenance import (
            parse_pdf_docling_provenance_to_staging,
        )

        return parse_pdf_docling_provenance_to_staging(
            source, out_dir, chunk_target=chunk_target
        )

    if allow_degraded:
        return _parse_pdf_degraded(source, out_dir, chunk_target=chunk_target)

    raise ImportError(
        "PDF ingestion requires Docling (pip install potpie[documents]). "
        "Pass --allow-degraded for text-only ingest without provenance."
    )


def _parse_office_to_staging(
    source: Path, out_dir: Path, *, chunk_target: int
) -> ResourceManifest:
    caps = documents_capabilities()
    if not caps["docling"]:
        raise ImportError(
            f"Office format {source.suffix} requires potpie[documents] (Docling). "
            "Install with: pip install 'potpie[documents]'"
        )
    from potpie_context_engine.adapters.outbound.resources.parsers.docling_convert import (
        parse_docling_file_to_staging,
    )

    return parse_docling_file_to_staging(source, out_dir, chunk_target=chunk_target)


def _parse_html_to_staging(
    source: Path, out_dir: Path, *, chunk_target: int
) -> ResourceManifest:
    caps = documents_capabilities()
    if caps["docling"]:
        from potpie_context_engine.adapters.outbound.resources.parsers.docling_convert import (
            parse_docling_file_to_staging,
        )

        return parse_docling_file_to_staging(source, out_dir, chunk_target=chunk_target)

    from potpie_context_engine.adapters.outbound.resources.parsers.html import (
        parse_html_to_staging,
    )

    return parse_html_to_staging(source, out_dir, chunk_target=chunk_target)


def parse_file_to_staging(
    source: Path,
    out_dir: Path,
    *,
    chunk_target: int = CHUNK_TARGET_DEFAULT,
    vision_provider: str = "local",
    options: ParseOptions | None = None,
) -> ResourceManifest:
    opts = options or ParseOptions(
        chunk_target=chunk_target, vision_provider=vision_provider
    )
    suffix = source.suffix.lower()

    if suffix in {".md", ".markdown", ".txt", ".text"}:
        return parse_text_to_staging(source, out_dir, chunk_target=opts.chunk_target)

    if suffix == ".pdf":
        return _parse_pdf_to_staging(
            source,
            out_dir,
            chunk_target=opts.chunk_target,
            allow_degraded=opts.allow_degraded,
        )

    if suffix in _OFFICE_SUFFIXES:
        return _parse_office_to_staging(source, out_dir, chunk_target=opts.chunk_target)

    if suffix in _HTML_SUFFIXES:
        return _parse_html_to_staging(source, out_dir, chunk_target=opts.chunk_target)

    if suffix in _IMAGE_SUFFIXES:
        caps = documents_capabilities()
        if not (caps["rapidocr"] or caps["docling"]):
            raise ImportError(
                "image ingestion requires potpie[documents] (RapidOCR + optional vision)"
            )
        from potpie_context_engine.adapters.outbound.resources.parsers.image import (
            parse_image_to_staging,
        )

        return parse_image_to_staging(
            source,
            out_dir,
            vision_provider=opts.vision_provider,
        )

    raise ValueError(
        f"unsupported source format: {suffix or '(no extension)'}; "
        "supported: .md, .txt, .pdf, .html, .docx, .pptx, .xlsx, images (.png/.jpg/…)"
    )


__all__ = ["ParseOptions", "parse_file_to_staging"]
