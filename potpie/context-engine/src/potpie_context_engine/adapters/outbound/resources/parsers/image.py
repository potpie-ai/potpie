"""Standalone image ingestion — RapidOCR + optional vision caption."""

from __future__ import annotations

import shutil
from pathlib import Path

from potpie_context_engine.adapters.outbound.resources.enrichers.ocr_rapidocr import (
    ocr_image_text,
)
from potpie_context_engine.adapters.outbound.resources.enrichers.vision_httpx_ollama import (
    caption_image_local,
)
from potpie_context_engine.adapters.outbound.resources.enrichers.vision_openai import (
    caption_image_openai,
)
from potpie_context_engine.adapters.outbound.resources.parsers.markdown import (
    write_staging_from_parsed,
)
from potpie_context_engine.domain.resource_models import (
    CHUNK_HARD_CAP,
    ChunkRef,
    ResourceManifest,
    SectionManifest,
    text_sha256,
)

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"}
_SECTION_SLUG = "visual"


def _caption_image(path: Path, vision_provider: str) -> str:
    provider = vision_provider.strip().lower()
    if provider == "openai":
        return caption_image_openai(path)
    if provider in {"local", "ollama", ""}:
        return caption_image_local(path)
    raise ValueError(
        f"unsupported vision provider: {vision_provider!r} (use local or openai)"
    )


def parse_image(
    path: Path,
    *,
    vision_provider: str = "local",
) -> tuple[ResourceManifest, dict[str, list[str]], dict[str, list[str]]]:
    suffix = path.suffix.lower()
    if suffix not in _IMAGE_SUFFIXES:
        raise ValueError(f"unsupported image format: {suffix}")

    ocr_text = ocr_image_text(path)
    caption = _caption_image(path, vision_provider)

    parts: list[str] = []
    if caption:
        parts.append(f"Caption: {caption}")
    if ocr_text:
        parts.append(f"OCR: {ocr_text}")
    if not parts:
        parts.append(f"Image file {path.name} (no OCR or vision caption available)")

    content = "\n\n".join(parts)
    if len(content) > CHUNK_HARD_CAP:
        content = content[: CHUNK_HARD_CAP - 3] + "..."

    label = (caption or ocr_text or path.name)[:200]
    section = SectionManifest(
        slug=_SECTION_SLUG,
        title="Visual",
        summary=content[:2000],
        ordinal=0,
        content_hash=text_sha256(content),
        chunks=[ChunkRef(seq=0, label=label)],
    )
    manifest = ResourceManifest(
        source_ref=f"file://{path.resolve()}",
        source_kind="image",
        sections=[section],
    )
    section_texts = {_SECTION_SLUG: [content]}
    section_ocr = {_SECTION_SLUG: [ocr_text]}
    return manifest, section_texts, section_ocr


def parse_image_to_staging(
    source: Path,
    out_dir: Path,
    *,
    vision_provider: str = "local",
    artifact_name: str | None = None,
) -> ResourceManifest:
    manifest, section_texts, section_ocr = parse_image(
        source, vision_provider=vision_provider
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts = out_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    dest_name = artifact_name or f"source{source.suffix.lower()}"
    shutil.copy2(source, artifacts / dest_name)

    write_staging_from_parsed(
        out_dir,
        manifest,
        section_texts,
        section_ocr=section_ocr,
    )
    return manifest


__all__ = ["parse_image", "parse_image_to_staging"]
