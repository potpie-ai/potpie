"""HTML parser (stdlib tier 0; Docling path in docling_convert.py)."""

from __future__ import annotations

import re
from html import unescape
from pathlib import Path

from potpie_context_engine.adapters.outbound.resources.parsers.markdown import (
    _manifest_from_parsed,
    _parse_sections_from_markdown,
    write_staging_from_parsed,
)
from potpie_context_engine.domain.resource_models import (
    CHUNK_TARGET_DEFAULT,
    ResourceManifest,
)


def html_to_markdownish(raw_html: str) -> str:
    cleaned = re.sub(r"<script[^>]*>.*?</script>", "", raw_html, flags=re.I | re.S)
    cleaned = re.sub(r"<style[^>]*>.*?</style>", "", cleaned, flags=re.I | re.S)
    for level in range(1, 7):
        prefix = "##" if level <= 2 else "###"
        cleaned = re.sub(
            rf"<h{level}[^>]*>(.*?)</h{level}>",
            lambda m,
            p=prefix: f"\n\n{p} {unescape(re.sub(r'<[^>]+>', '', m.group(1))).strip()}\n\n",
            cleaned,
            flags=re.I | re.S,
        )
    cleaned = re.sub(r"<br\s*/?>", "\n", cleaned, flags=re.I)
    cleaned = re.sub(r"</(p|div|li|tr|table)>", "\n\n", cleaned, flags=re.I)
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    text = unescape(cleaned)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_html(
    path: Path,
    *,
    chunk_target: int = CHUNK_TARGET_DEFAULT,
) -> tuple[ResourceManifest, dict[str, list[str]]]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    markdownish = html_to_markdownish(raw)
    if not markdownish.strip():
        raise ValueError("HTML produced no extractable text")
    parsed_sections = _parse_sections_from_markdown(markdownish, chunk_target)
    if not parsed_sections:
        raise ValueError("HTML had no sections after extraction")
    return _manifest_from_parsed(path, "html", parsed_sections)


def parse_html_to_staging(
    source: Path,
    out_dir: Path,
    *,
    chunk_target: int = CHUNK_TARGET_DEFAULT,
) -> ResourceManifest:
    manifest, section_texts = parse_html(source, chunk_target=chunk_target)
    write_staging_from_parsed(out_dir, manifest, section_texts)
    return manifest


__all__ = ["parse_html", "parse_html_to_staging", "html_to_markdownish"]
