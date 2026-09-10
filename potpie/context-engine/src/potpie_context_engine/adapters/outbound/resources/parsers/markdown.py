"""Markdown and plain-text parsers (stdlib only, tier 0)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from potpie_context_engine.domain.resource_models import (
    CHUNK_HARD_CAP,
    CHUNK_TARGET_DEFAULT,
    ChunkProvenanceRecord,
    ChunkProvenanceSidecar,
    DocumentElementRecord,
    SECTION_CHUNK_MAX,
    ResourceManifest,
    SectionManifest,
    text_sha256,
)

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")


@dataclass(slots=True)
class ParsedSection:
    slug: str
    title: str
    ordinal: int
    text: str
    chunks: list[str]


def _slugify(title: str, fallback: str) -> str:
    raw = title.lower().strip()
    slug = re.sub(r"[^a-z0-9]+", "-", raw).strip("-")
    slug = re.sub(r"-{2,}", "-", slug)
    if not slug:
        slug = fallback
    return slug[:80]


def _pack_paragraphs(
    paragraphs: list[str], target: int = CHUNK_TARGET_DEFAULT
) -> list[str]:
    chunks: list[str] = []
    buf = ""
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if buf and len(buf) + len(para) + 2 > target:
            chunks.append(buf)
            buf = para
        else:
            buf = f"{buf}\n\n{para}" if buf else para
    if buf:
        chunks.append(buf)
    return chunks


def _split_hard_cap(text: str, cap: int = CHUNK_HARD_CAP) -> list[str]:
    if len(text) <= cap:
        return [text]
    parts: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + cap, len(text))
        if end < len(text):
            window = text[start:end]
            break_at = max(window.rfind("\n\n"), window.rfind("\n"), window.rfind(" "))
            if break_at > cap // 3:
                end = start + break_at
        piece = text[start:end].strip()
        if piece:
            parts.append(piece)
        start = end if end > start else end + 1
    return parts or [text[:cap]]


def _fit_section_chunks(chunks: list[str]) -> list[str]:
    expanded: list[str] = []
    for chunk in chunks:
        expanded.extend(_split_hard_cap(chunk))
    chunks = expanded
    while len(chunks) > SECTION_CHUNK_MAX:
        merged: list[str] = []
        for i in range(0, len(chunks), 2):
            if i + 1 < len(chunks):
                combined = f"{chunks[i]}\n\n{chunks[i + 1]}"
                if len(combined) <= CHUNK_HARD_CAP:
                    merged.append(combined)
                else:
                    merged.append(chunks[i])
                    merged.append(chunks[i + 1])
            else:
                merged.append(chunks[i])
        if len(merged) == len(chunks):
            break
        chunks = merged
    return chunks


def _split_oversized_section(parsed: ParsedSection) -> list[ParsedSection]:
    if len(parsed.chunks) <= SECTION_CHUNK_MAX:
        return [parsed]
    parts: list[ParsedSection] = []
    for part_idx in range(0, len(parsed.chunks), SECTION_CHUNK_MAX):
        chunk_slice = parsed.chunks[part_idx : part_idx + SECTION_CHUNK_MAX]
        text_slice = "\n\n".join(chunk_slice)
        part_num = part_idx // SECTION_CHUNK_MAX + 1
        title = parsed.title if part_num == 1 else f"{parsed.title} (part {part_num})"
        slug = (
            parsed.slug
            if part_num == 1
            else _slugify(title, f"{parsed.slug}-part-{part_num}")
        )
        parts.append(
            ParsedSection(
                slug=slug,
                title=title,
                ordinal=parsed.ordinal,
                text=text_slice,
                chunks=chunk_slice,
            )
        )
    return parts


def _section_from_body(
    *,
    title: str,
    ordinal: int,
    body: str,
    chunk_target: int,
) -> ParsedSection:
    slug = _slugify(title, f"section-{ordinal}")
    paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]
    chunks = _pack_paragraphs(paragraphs, target=chunk_target)
    if not chunks:
        chunks = [body]
    chunks = _fit_section_chunks(chunks)
    return ParsedSection(
        slug=slug, title=title, ordinal=ordinal, text=body, chunks=chunks
    )


_SUMMARY_STOPWORDS = frozenset(
    """a about above after again all also an and any are as at be because been
    before being below between both but by can could did do does doing down
    during each few for from further had has have having he her here hers him
    his how however if in into is it its itself just like may means might more
    most must not of off on once only onto or other our out over own same
    shall should since so some such than that the their theirs them then there
    these they this those through to too under until up upon use used uses
    using very was we were what when where which while who whom why will with
    would you your""".split()
)
_SUMMARY_TERM_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}")


def _salient_terms(text: str, exclude: str, limit: int = 12) -> list[str]:
    """Distinctive terms from the whole section, for the retrieval summary.

    Acronym-shaped tokens (an uppercase letter or digit past the first
    character) outrank plain words: they are the identifiers a searcher
    types, and they are the terms a leading excerpt most often misses.
    """
    counts: dict[str, list[object]] = {}
    for token in _SUMMARY_TERM_RE.findall(text):
        key = token.lower()
        if key in _SUMMARY_STOPWORDS:
            continue
        acronym_like = any(c.isupper() or c.isdigit() for c in token[1:])
        entry = counts.get(key)
        if entry is None:
            counts[key] = [1, token, acronym_like]
            continue
        entry[0] += 1
        if acronym_like and not entry[2]:
            entry[1] = token
            entry[2] = True
    excluded = {t.lower() for t in _SUMMARY_TERM_RE.findall(exclude)}
    ranked = sorted(counts.items(), key=lambda kv: (not kv[1][2], -kv[1][0], kv[0]))
    terms: list[str] = []
    for key, (_, original, _acronym) in ranked:
        if key in excluded:
            continue
        terms.append(original)
        if len(terms) >= limit:
            break
    return terms


def _lead_excerpt(text: str) -> str:
    excerpt = text.strip().replace("\n", " ")
    if len(excerpt) > 500:
        excerpt = excerpt[:497] + "..."
    return excerpt


def _auto_summary(title: str, text: str, max_len: int = 2000) -> str:
    summary = f"{title}: {_lead_excerpt(text)}".strip()
    return summary[:max_len]


def auto_keywords(title: str, text: str) -> list[str]:
    """Salient terms complementing the summary's lead excerpt.

    Carried as a structured section field (not appended to the summary
    prose) so the graph bridge can feed them to retrieval while display
    surfaces keep the plain summary.
    """
    return _salient_terms(text, exclude=f"{title} {_lead_excerpt(text)}")


def _parse_sections_from_markdown(text: str, chunk_target: int) -> list[ParsedSection]:
    sections: list[ParsedSection] = []
    current_title = "body"
    current_lines: list[str] = []
    ordinal = 0
    in_fence = False

    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_fence = not in_fence
            current_lines.append(line)
            continue
        if not in_fence:
            match = _HEADING_RE.match(line)
            if match and match.group(1) == "##":
                body = "\n".join(current_lines).strip()
                if body:
                    sections.append(
                        _section_from_body(
                            title=current_title,
                            ordinal=ordinal,
                            body=body,
                            chunk_target=chunk_target,
                        )
                    )
                    ordinal += 1
                current_title = match.group(2).strip()
                current_lines = []
                continue
        current_lines.append(line)

    body = "\n".join(current_lines).strip()
    if body or not sections:
        sections.append(
            _section_from_body(
                title=current_title,
                ordinal=ordinal,
                body=body or text,
                chunk_target=chunk_target,
            )
        )
    return sections


def parse_markdown(
    path: Path, *, chunk_target: int = CHUNK_TARGET_DEFAULT
) -> tuple[ResourceManifest, dict[str, list[str]]]:
    text = path.read_text(encoding="utf-8")
    parsed_sections = _parse_sections_from_markdown(text, chunk_target)
    return _manifest_from_parsed(path, "markdown", parsed_sections)


def parse_plain_text(
    path: Path, *, chunk_target: int = CHUNK_TARGET_DEFAULT
) -> tuple[ResourceManifest, dict[str, list[str]]]:
    text = path.read_text(encoding="utf-8")
    parsed = _section_from_body(
        title="body", ordinal=0, body=text, chunk_target=chunk_target
    )
    return _manifest_from_parsed(path, "text", [parsed])


def _unique_slug(slug: str, used: set[str]) -> str:
    if slug not in used:
        used.add(slug)
        return slug
    suffix = 2
    while True:
        stem = slug[: max(1, 80 - len(str(suffix)) - 1)].rstrip("-")
        candidate = f"{stem}-{suffix}"
        if candidate not in used:
            used.add(candidate)
            return candidate
        suffix += 1


def _manifest_from_parsed(
    path: Path,
    source_kind: str,
    parsed_sections: list[ParsedSection],
) -> tuple[ResourceManifest, dict[str, list[str]]]:
    manifest_sections: list[SectionManifest] = []
    section_texts: dict[str, list[str]] = {}
    used_slugs: set[str] = set()
    section_ordinal = 0
    for parsed in parsed_sections:
        for part in _split_oversized_section(parsed):
            slug = _unique_slug(part.slug, used_slugs)
            chunk_refs = []
            for idx, chunk_text in enumerate(part.chunks):
                label = chunk_text.split("\n", 1)[0][:200] or part.title
                chunk_refs.append({"seq": idx, "label": label})
            manifest_sections.append(
                SectionManifest(
                    slug=slug,
                    title=part.title[:200],
                    summary=_auto_summary(part.title, part.text),
                    keywords=auto_keywords(part.title, part.text),
                    ordinal=section_ordinal,
                    content_hash=text_sha256(part.text),
                    chunks=chunk_refs,
                )
            )
            section_texts[slug] = part.chunks
            section_ordinal += 1

    manifest = ResourceManifest(
        source_ref=f"file://{path.resolve()}",
        source_kind=source_kind,
        sections=manifest_sections,
    )
    return manifest, section_texts


def write_staging_from_parsed(
    path: Path,
    manifest: ResourceManifest,
    section_texts: dict[str, list[str]],
    *,
    section_ocr: dict[str, list[str]] | None = None,
    elements: list[DocumentElementRecord] | None = None,
    chunk_provenance: dict[tuple[str, int], list[ChunkProvenanceRecord]] | None = None,
) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "meta.json").write_text(
        json.dumps(manifest.model_dump(), indent=2),
        encoding="utf-8",
    )
    if elements:
        elements_path = path / "elements.jsonl"
        with elements_path.open("w", encoding="utf-8") as fh:
            for row in elements:
                fh.write(json.dumps(row.model_dump()) + "\n")
    ocr_map = section_ocr or {}
    prov_map = chunk_provenance or {}
    for section in manifest.sections:
        section_dir = path / section.slug
        section_dir.mkdir(parents=True, exist_ok=True)
        texts = section_texts.get(section.slug, [])
        ocr_texts = ocr_map.get(section.slug, [])
        for chunk_ref in section.chunks:
            idx = chunk_ref.seq
            text = texts[idx] if idx < len(texts) else ""
            (section_dir / f"{idx:04d}.txt").write_text(text, encoding="utf-8")
            ocr_text = ocr_texts[idx] if idx < len(ocr_texts) else ""
            if ocr_text.strip():
                (section_dir / f"{idx:04d}.ocr.txt").write_text(
                    ocr_text.strip(), encoding="utf-8"
                )
            prov_rows = prov_map.get((section.slug, idx))
            if prov_rows:
                sidecar = ChunkProvenanceSidecar(provenance=prov_rows)
                (section_dir / f"{idx:04d}.prov.json").write_text(
                    json.dumps(sidecar.model_dump(), indent=2),
                    encoding="utf-8",
                )
    return path


def parse_file_to_staging(
    source: Path,
    out_dir: Path,
    *,
    chunk_target: int = CHUNK_TARGET_DEFAULT,
) -> ResourceManifest:
    suffix = source.suffix.lower()
    if suffix in {".md", ".markdown"}:
        manifest, section_texts = parse_markdown(source, chunk_target=chunk_target)
    elif suffix in {".txt", ".text"}:
        manifest, section_texts = parse_plain_text(source, chunk_target=chunk_target)
    else:
        raise ValueError(f"unsupported text format: {suffix}")
    write_staging_from_parsed(out_dir, manifest, section_texts)
    return manifest


__all__ = [
    "ParsedSection",
    "parse_file_to_staging",
    "parse_markdown",
    "parse_plain_text",
    "write_staging_from_parsed",
]
