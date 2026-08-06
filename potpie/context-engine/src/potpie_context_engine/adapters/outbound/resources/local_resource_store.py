"""Local filesystem resource store — chunk bytes under the Potpie home.

Layout, per ``docs/context-graph/resources.md``::

    <home>/resources/<pot_dir>/<doc>/<section>/<seq>.txt
    <home>/resources/<pot_dir>/<doc>/meta.json

The on-disk ``meta.json`` is *not* the manifest the design rejected. That
decision ("Manifest lives on the graph nodes") is about the source of truth for
document *structure*, which stays in the graph and is read via
``graph describe document:<slug>``. This file is storage detail: R11 says
``resource get`` does no graph query, yet the CLI contract has ``get`` return
``revision`` and ``source_ref`` and ``list`` return chunk labels. Those come
from here. Nothing reads it to answer a structural question.

Beyond ``LocalResourceStore`` this module holds the pieces every
``ResourceStorePort`` implementation shares — reading and validating an import
directory, computing the revision diff, hydrating a ``Chunk`` — so a second
store (the in-memory stub today, S3 later) satisfies the same contract by
construction rather than by copying it.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import shutil
import tempfile
import time
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

from potpie_context_core.ports.resource_store import (
    RECOMMENDED_MAX_SECTION_CHUNKS,
    RESOURCE_CHUNK_MAX_CHARS,
    RESOURCE_CHUNK_TOO_LARGE,
    RESOURCE_LABEL_MAX_CHARS,
    RESOURCE_MANIFEST_INVALID,
    RESOURCE_NOT_FOUND,
    RESOURCE_SECTION_MISSING_CHUNK,
    RESOURCE_SEQ_WIDTH,
    RESOURCE_SUMMARY_MAX_CHARS,
    RESOURCE_TEXT_TOO_LARGE,
    Chunk,
    ChunkRef,
    DocumentManifest,
    ResourceId,
    ResourceStoreError,
    ResourceStoreStatus,
    SectionManifest,
    format_resource_id,
    parse_resource_id,
    require_resource_slug,
)
from potpie_context_engine.adapters.outbound.pots.local_pot_store import default_home

META_FILENAME = "meta.json"

# Anything outside this alphabet is folded to a hyphen when a pot id becomes a
# directory name.
_POT_DIR_UNSAFE_RE = re.compile(r"[^a-z0-9-]+")
_POT_DIR_BODY_MAX = 48
_POT_DIR_DIGEST_CHARS = 16

# A scratch directory younger than this may belong to an import that is running
# right now, so only older leftovers are swept. The daemon serializes RPC, but
# nothing serializes two direct in-process callers.
_SCRATCH_STALE_SECONDS = 3600.0


# --- pot directories --------------------------------------------------------


def pot_dir_name(pot_id: str) -> str:
    """Deterministic, collision-free directory name for ``pot_id``.

    Pot ids are opaque and routinely contain characters a path segment should
    not (``conformance:pot-a``), so the raw id cannot be a directory name.
    Sanitizing alone would collide — ``a:b`` and ``a/b`` both fold to ``a-b``
    — so the name is a readable sanitized body plus a sha256 prefix of the
    *raw* id. The digest makes the mapping injective; the body keeps
    ``<home>/resources`` browsable. Neither part can contain a separator or
    ``..``, so a pot id can never escape the resources root.
    """
    digest = hashlib.sha256(pot_id.encode("utf-8")).hexdigest()[:_POT_DIR_DIGEST_CHARS]
    body = _POT_DIR_UNSAFE_RE.sub("-", pot_id.strip().lower()).strip("-")
    body = body[:_POT_DIR_BODY_MAX].strip("-")
    return f"{body}-{digest}" if body else digest


# --- import directories -----------------------------------------------------


@dataclass(frozen=True, slots=True)
class SourceDocument:
    """A validated import directory: structure plus the chunk text it names."""

    source_ref: str | None
    source_kind: str | None
    sections: tuple[SectionManifest, ...]
    texts: Mapping[tuple[str, int], str]
    warnings: tuple[str, ...] = ()


def read_source_document(source_dir: Path) -> SourceDocument:
    """Read and fully validate the directory an extraction script produced.

    Everything that can be refused is refused here, before a store touches its
    own state: slugs, the ``meta.json`` shape, the required chunk ``label``,
    missing chunk files, the hard chunk-size cap, and the caps on the manifest
    text that becomes graph node properties. Oversized chunks are rejected
    rather than clamped so every stored chunk is uniformly safe to hand an
    agent, and an oversized ``summary`` is rejected because a section body
    pasted into it is a payload on its way into the graph (R1). Advisory
    problems — a section with no chunks, or with more than
    ``RECOMMENDED_MAX_SECTION_CHUNKS`` — become warnings.
    """
    root = Path(source_dir)
    raw = _read_json(root / META_FILENAME)
    raw_sections = raw.get("sections")
    if not isinstance(raw_sections, list) or not raw_sections:
        raise ResourceStoreError(
            RESOURCE_MANIFEST_INVALID,
            f"{META_FILENAME} must list at least one section",
            detail=str(root / META_FILENAME),
            recommended_next_action=(
                "Emit one section per real division of the source, or a single "
                "'body' section when it has none."
            ),
        )

    sections: list[SectionManifest] = []
    texts: dict[tuple[str, int], str] = {}
    warnings: list[str] = []
    seen: set[str] = set()
    for index, entry in enumerate(raw_sections):
        if not isinstance(entry, dict):
            raise ResourceStoreError(
                RESOURCE_MANIFEST_INVALID, f"section {index} is not an object"
            )
        slug = require_resource_slug(entry.get("slug"), kind="section")
        if slug in seen:
            raise ResourceStoreError(
                RESOURCE_MANIFEST_INVALID, f"duplicate section slug: {slug!r}"
            )
        seen.add(slug)
        summary = _text_field(
            entry, "summary", where=slug, max_chars=RESOURCE_SUMMARY_MAX_CHARS
        )
        refs = _read_chunk_refs(entry, slug=slug)
        for ref in refs:
            texts[(slug, ref.seq)] = _read_chunk_text(root / slug, slug=slug, ref=ref)
        sections.append(
            SectionManifest(
                slug=slug,
                title=_text_field(
                    entry, "title", where=slug, max_chars=RESOURCE_LABEL_MAX_CHARS
                ),
                summary=summary,
                ordinal=_int_field(entry, "ordinal", where=slug, default=index),
                content_hash=_text_field(
                    entry,
                    "content_hash",
                    where=slug,
                    max_chars=RESOURCE_LABEL_MAX_CHARS,
                ),
                chunks=refs,
                summary_pending=not summary.strip(),
            )
        )
        warnings.extend(_section_warnings(slug, refs))

    sections.sort(key=lambda section: (section.ordinal, section.slug))
    return SourceDocument(
        source_ref=_optional_text(raw, "source_ref"),
        source_kind=_optional_text(raw, "source_kind"),
        sections=tuple(sections),
        texts=texts,
        warnings=tuple(warnings),
    )


def build_import_manifest(
    *,
    pot_id: str,
    doc: str,
    source: SourceDocument,
    prior: DocumentManifest | None,
    source_ref: str | None = None,
    source_kind: str | None = None,
) -> DocumentManifest:
    """Turn a validated directory plus the prior revision into the import report.

    A section counts as *kept* only when its ``content_hash`` is non-empty and
    matches the stored one: an absent hash means "unknown", and unknown must
    read as changed so a re-summary pass is never skipped by accident.

    A kept section also keeps the summary the prior revision carried whenever
    the directory supplies none. Summaries are agent-authored, so re-running
    the extraction script over an unchanged source emits them empty; taking
    that verbatim would blank the index (R12) for exactly the content R14 says
    must not be re-summarized.
    """
    prior_sections = {row.slug: row for row in prior.sections} if prior else {}
    prior_hashes = {slug: row.content_hash for slug, row in prior_sections.items()}
    current = {row.slug: row.content_hash for row in source.sections}
    kept = tuple(
        sorted(
            slug
            for slug, digest in current.items()
            if digest and prior_hashes.get(slug) == digest
        )
    )
    unchanged = set(kept)
    return DocumentManifest(
        pot_id=pot_id,
        doc=doc,
        revision=prior.revision + 1 if prior else 1,
        source_ref=source_ref if source_ref is not None else source.source_ref,
        source_kind=source_kind if source_kind is not None else source.source_kind,
        sections=tuple(
            _with_carried_summary(row, prior_sections.get(row.slug))
            if row.slug in unchanged
            else row
            for row in source.sections
        ),
        sections_added=tuple(
            sorted(slug for slug in current if slug not in prior_hashes)
        ),
        sections_kept=kept,
        sections_removed=tuple(
            sorted(slug for slug in prior_hashes if slug not in current)
        ),
        warnings=source.warnings,
    )


def _with_carried_summary(
    section: SectionManifest, prior: SectionManifest | None
) -> SectionManifest:
    if prior is None or section.summary.strip() or not prior.summary.strip():
        return section
    return replace(section, summary=prior.summary, summary_pending=False)


# --- chunk lookup -----------------------------------------------------------


def find_chunk_ref(
    manifest: DocumentManifest, *, section: str, seq: int
) -> ChunkRef | None:
    """Locate one chunk in a stored manifest, or ``None`` if it names no such chunk."""
    for row in manifest.sections:
        if row.slug != section:
            continue
        for ref in row.chunks:
            if ref.seq == seq:
                return ref
    return None


def build_chunk(
    *, manifest: DocumentManifest, resource: ResourceId, ref: ChunkRef, text: str
) -> Chunk:
    """Hydrate the ``get`` response from stored text plus manifest provenance."""
    return Chunk(
        resource_id=format_resource_id(resource.doc, resource.section, resource.seq),
        doc=resource.doc,
        section=resource.section,
        seq=resource.seq,
        text=text,
        chars=len(text),
        revision=manifest.revision,
        source_ref=manifest.source_ref,
        page=ref.page,
        offset=ref.offset,
    )


def chunk_filename(seq: int) -> str:
    """``0``  ->  ``0000.txt`` — zero-padded so a section's chunks sort."""
    return f"{seq:0{RESOURCE_SEQ_WIDTH}d}.txt"


def chunk_not_found(resource_id: str) -> ResourceStoreError:
    """One phrasing of a chunk miss, shared by every store implementation."""
    return ResourceStoreError(
        RESOURCE_NOT_FOUND,
        f"no chunk stored for {resource_id}",
        recommended_next_action=(
            "Check the active pot, then run potpie resource list --doc <slug>."
        ),
    )


def document_not_found(doc: str) -> ResourceStoreError:
    """One phrasing of a document miss, shared by every store implementation."""
    return ResourceStoreError(
        RESOURCE_NOT_FOUND,
        f"no document stored as {doc!r}",
        recommended_next_action=(
            "Check the active pot, then run potpie resource import to add it."
        ),
    )


def section_not_found(doc: str, section: str) -> ResourceStoreError:
    """One phrasing of a section miss, shared by every store implementation."""
    return ResourceStoreError(
        RESOURCE_NOT_FOUND,
        f"document {doc!r} has no section {section!r}",
        recommended_next_action=(
            f"Run potpie resource list --doc {doc} to see its sections."
        ),
    )


# --- the store --------------------------------------------------------------


@dataclass(slots=True)
class LocalResourceStore:
    """Pot-scoped chunk storage on the local filesystem."""

    home: Path = field(default_factory=default_home)

    @property
    def _path(self) -> Path:
        return self.home / "resources"

    def _pot_root(self, pot_id: str) -> Path:
        return self._path / pot_dir_name(pot_id)

    # --- import -------------------------------------------------------------
    def import_dir(
        self,
        *,
        pot_id: str,
        slug: str,
        source_dir: Path,
        source_ref: str | None = None,
        source_kind: str | None = None,
    ) -> DocumentManifest:
        doc = require_resource_slug(slug, kind="document")
        # Everything that can fail, fails before the first rename: a rejected
        # import must leave the prior revision exactly as it was.
        source = read_source_document(Path(source_dir))
        pot_root = self._pot_root(pot_id)
        final = pot_root / doc
        pot_root.mkdir(parents=True, exist_ok=True)
        # Before the prior revision is read, not after: a crashed import left
        # it in a trash directory, and diffing against nothing would restart
        # the revision counter at 1.
        _recover_scratch(pot_root, doc, final=final)
        manifest = build_import_manifest(
            pot_id=pot_id,
            doc=doc,
            source=source,
            prior=_load_manifest(final, pot_id=pot_id, doc=doc),
            source_ref=source_ref,
            source_kind=source_kind,
        )

        # mkdtemp inside the pot root keeps staging on the same filesystem, so
        # the swap below is a rename and not a copy.
        staging = Path(tempfile.mkdtemp(dir=pot_root, prefix=f".{doc}.staging."))
        try:
            _write_document(staging, manifest, source.texts)
            _swap_into_place(staging=staging, final=final, doc=doc)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return manifest

    # --- read ---------------------------------------------------------------
    def get(self, *, pot_id: str, resource_id: str) -> Chunk:
        return self.get_many(pot_id=pot_id, resource_ids=(resource_id,))[0]

    def get_many(
        self, *, pot_id: str, resource_ids: tuple[str, ...]
    ) -> tuple[Chunk, ...]:
        manifests: dict[str, DocumentManifest] = {}
        chunks: list[Chunk] = []
        for resource_id in resource_ids:
            resource = parse_resource_id(resource_id)
            manifest = manifests.get(resource.doc)
            if manifest is None:
                # One meta.json read per document, however many chunks of it
                # the batch asks for.
                manifest = _load_manifest(
                    self._pot_root(pot_id) / resource.doc,
                    pot_id=pot_id,
                    doc=resource.doc,
                )
                if manifest is None:
                    raise chunk_not_found(resource_id)
                manifests[resource.doc] = manifest
            ref = find_chunk_ref(manifest, section=resource.section, seq=resource.seq)
            if ref is None:
                raise chunk_not_found(resource_id)
            path = (
                self._pot_root(pot_id)
                / resource.doc
                / resource.section
                / chunk_filename(resource.seq)
            )
            try:
                text = _read_text(path)
            except OSError as exc:
                raise chunk_not_found(resource_id) from exc
            chunks.append(
                build_chunk(manifest=manifest, resource=resource, ref=ref, text=text)
            )
        return tuple(chunks)

    def list(
        self, *, pot_id: str, slug: str, section: str | None = None
    ) -> tuple[SectionManifest, ...]:
        doc = require_resource_slug(slug, kind="document")
        manifest = _load_manifest(self._pot_root(pot_id) / doc, pot_id=pot_id, doc=doc)
        if manifest is None:
            raise document_not_found(doc)
        if section is None:
            return manifest.sections
        wanted = require_resource_slug(section, kind="section")
        rows = tuple(row for row in manifest.sections if row.slug == wanted)
        if not rows:
            raise section_not_found(doc, wanted)
        return rows

    # --- teardown -----------------------------------------------------------
    def delete(self, *, pot_id: str, slug: str) -> bool:
        doc = require_resource_slug(slug, kind="document")
        return _remove_tree(self._pot_root(pot_id) / doc)

    def purge_pot(self, pot_id: str) -> bool:
        return _remove_tree(self._pot_root(pot_id))

    # --- diagnostics --------------------------------------------------------
    def status(self, *, pot_id: str | None = None) -> ResourceStoreStatus:
        """Report writability of the resources root without creating it.

        The root is made on the first import, so its absence is normal and not
        a failure; what matters is whether the nearest ancestor that *does*
        exist can be written. Probing rather than creating keeps ``doctor``
        free of side effects — a diagnostic that leaves directories behind is
        one more thing to explain.
        """
        root = self._path
        probe = root
        while not probe.exists() and probe.parent != probe:
            probe = probe.parent
        ready = os.access(probe, os.W_OK | os.X_OK)
        return ResourceStoreStatus(
            kind="local",
            ready=ready,
            location=str(root),
            documents=_count_documents(self._pot_root(pot_id)) if pot_id else None,
            detail=None if ready else f"{probe} is not writable",
        )


# --- source-directory validation --------------------------------------------


def _read_json(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ResourceStoreError(
            RESOURCE_MANIFEST_INVALID,
            f"import directory has no {META_FILENAME}",
            detail=str(path),
            recommended_next_action=(
                "The extraction script must emit meta.json beside the section "
                "directories."
            ),
        ) from exc
    except OSError as exc:
        raise ResourceStoreError(
            RESOURCE_MANIFEST_INVALID, f"cannot read {path}: {exc}"
        ) from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ResourceStoreError(
            RESOURCE_MANIFEST_INVALID, f"{path} is not valid JSON: {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise ResourceStoreError(
            RESOURCE_MANIFEST_INVALID, f"{path} must contain a JSON object"
        )
    return data


def _read_chunk_refs(entry: dict[str, Any], *, slug: str) -> tuple[ChunkRef, ...]:
    raw = entry.get("chunks", [])
    if not isinstance(raw, list):
        raise ResourceStoreError(
            RESOURCE_MANIFEST_INVALID, f"section {slug!r}: chunks must be a list"
        )
    refs: list[ChunkRef] = []
    seen: set[int] = set()
    for position, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ResourceStoreError(
                RESOURCE_MANIFEST_INVALID,
                f"section {slug!r}: chunk {position} is not an object",
            )
        seq = _int_field(item, "seq", where=f"{slug} chunk {position}")
        if seq < 0:
            raise ResourceStoreError(
                RESOURCE_MANIFEST_INVALID,
                f"section {slug!r}: chunk seq must not be negative ({seq})",
            )
        if seq in seen:
            raise ResourceStoreError(
                RESOURCE_MANIFEST_INVALID,
                f"section {slug!r}: duplicate chunk seq {seq}",
            )
        seen.add(seq)
        label = _text_field(
            item,
            "label",
            where=f"{slug} chunk {seq}",
            max_chars=RESOURCE_LABEL_MAX_CHARS,
        )
        if not label.strip():
            # Required, not optional: the label is the agent's only signal for
            # choosing among a section's chunks.
            raise ResourceStoreError(
                RESOURCE_MANIFEST_INVALID,
                f"section {slug!r}: chunk {seq} has no label",
                recommended_next_action=(
                    "Derive each chunk's label from its leading heading or line."
                ),
            )
        refs.append(
            ChunkRef(
                seq=seq,
                label=label,
                page=_optional_int(item, "page", where=f"{slug} chunk {seq}"),
                offset=_optional_int(item, "offset", where=f"{slug} chunk {seq}"),
            )
        )
    refs.sort(key=lambda ref: ref.seq)
    return tuple(refs)


def _read_chunk_text(section_dir: Path, *, slug: str, ref: ChunkRef) -> str:
    path = section_dir / chunk_filename(ref.seq)
    try:
        text = _read_text(path)
    except FileNotFoundError as exc:
        raise ResourceStoreError(
            RESOURCE_SECTION_MISSING_CHUNK,
            f"section {slug!r} names chunk {ref.seq} but {path.name} is missing",
            detail=str(path),
        ) from exc
    except (OSError, UnicodeDecodeError) as exc:
        raise ResourceStoreError(
            RESOURCE_MANIFEST_INVALID,
            f"chunk {path} is not readable UTF-8 text: {exc}",
            recommended_next_action="Resources hold text only; binary is a non-goal.",
        ) from exc
    if len(text) > RESOURCE_CHUNK_MAX_CHARS:
        raise ResourceStoreError(
            RESOURCE_CHUNK_TOO_LARGE,
            f"section {slug!r} chunk {ref.seq} is {len(text)} chars, "
            f"over the {RESOURCE_CHUNK_MAX_CHARS} cap",
            detail=str(path),
            recommended_next_action=(
                "Split the chunk on a paragraph boundary and re-run the import."
            ),
        )
    return text


def _section_warnings(slug: str, refs: tuple[ChunkRef, ...]) -> list[str]:
    if not refs:
        return [f"section {slug!r} has no chunks"]
    if len(refs) > RECOMMENDED_MAX_SECTION_CHUNKS:
        return [
            (
                f"section {slug!r} has {len(refs)} chunks; sections should be 1-"
                f"{RECOMMENDED_MAX_SECTION_CHUNKS} so the agent can pick one by label"
            )
        ]
    return []


def _text_field(
    entry: dict[str, Any], key: str, *, where: str, max_chars: int | None = None
) -> str:
    value = entry.get(key, "")
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ResourceStoreError(
            RESOURCE_MANIFEST_INVALID, f"{where}: {key} must be a string"
        )
    if max_chars is not None and len(value) > max_chars:
        raise ResourceStoreError(
            RESOURCE_TEXT_TOO_LARGE,
            f"{where}: {key} is {len(value)} chars, over the {max_chars} cap",
            recommended_next_action=(
                "This text becomes a graph node property; the payload belongs "
                "in the section's chunks. Summarize instead of pasting."
            ),
        )
    return value


def _optional_text(entry: dict[str, Any], key: str) -> str | None:
    value = entry.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ResourceStoreError(
            RESOURCE_MANIFEST_INVALID, f"{META_FILENAME}: {key} must be a string"
        )
    return value or None


def _int_field(
    entry: dict[str, Any], key: str, *, where: str, default: int | None = None
) -> int:
    value = entry.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ResourceStoreError(
            RESOURCE_MANIFEST_INVALID, f"{where}: {key} must be an integer"
        )
    return value


def _optional_int(entry: dict[str, Any], key: str, *, where: str) -> int | None:
    if entry.get(key) is None:
        return None
    return _int_field(entry, key, where=where)


# --- on-disk manifest -------------------------------------------------------


def _load_manifest(doc_dir: Path, *, pot_id: str, doc: str) -> DocumentManifest | None:
    """Read a stored document's ``meta.json``, or ``None`` when absent.

    The diff and warning fields describe an import, not stored state, so they
    come back empty.
    """
    path = doc_dir / META_FILENAME
    if not path.is_file():
        return None
    try:
        data = _read_json(path)
        sections = tuple(
            SectionManifest(
                slug=row["slug"],
                title=row.get("title", ""),
                summary=row.get("summary", ""),
                ordinal=row["ordinal"],
                content_hash=row.get("content_hash", ""),
                chunks=tuple(
                    ChunkRef(
                        seq=chunk["seq"],
                        label=chunk["label"],
                        page=chunk.get("page"),
                        offset=chunk.get("offset"),
                    )
                    for chunk in row.get("chunks", ())
                ),
                summary_pending=bool(row.get("summary_pending", False)),
            )
            for row in data["sections"]
        )
        revision = int(data["revision"])
    except (KeyError, TypeError, ValueError, ResourceStoreError) as exc:
        # A store that cannot read its own manifest is recoverable but not by
        # guessing: say so, and say how.
        raise ResourceStoreError(
            RESOURCE_MANIFEST_INVALID,
            f"stored manifest for {doc!r} is unreadable: {exc}",
            detail=str(path),
            recommended_next_action=(
                f"Run potpie resource rm {doc} --confirm, then import again."
            ),
        ) from exc
    return DocumentManifest(
        pot_id=pot_id,
        doc=doc,
        revision=revision,
        source_ref=data.get("source_ref"),
        source_kind=data.get("source_kind"),
        sections=sections,
    )


def _manifest_to_json(manifest: DocumentManifest) -> dict[str, Any]:
    return {
        "doc": manifest.doc,
        "revision": manifest.revision,
        "source_ref": manifest.source_ref,
        "source_kind": manifest.source_kind,
        "sections": [
            {
                "slug": section.slug,
                "title": section.title,
                "summary": section.summary,
                "ordinal": section.ordinal,
                "content_hash": section.content_hash,
                "summary_pending": section.summary_pending,
                "chunks": [
                    {
                        "seq": ref.seq,
                        "label": ref.label,
                        "page": ref.page,
                        "offset": ref.offset,
                    }
                    for ref in section.chunks
                ],
            }
            for section in manifest.sections
        ],
    }


# --- atomic write -----------------------------------------------------------


def _write_document(
    staging: Path, manifest: DocumentManifest, texts: Mapping[tuple[str, int], str]
) -> None:
    for section in manifest.sections:
        section_dir = staging / section.slug
        section_dir.mkdir(parents=True, exist_ok=True)
        for ref in section.chunks:
            _write_text(
                section_dir / chunk_filename(ref.seq), texts[(section.slug, ref.seq)]
            )
        _sync_dir(section_dir)
    _write_text(
        staging / META_FILENAME,
        json.dumps(_manifest_to_json(manifest), indent=2, sort_keys=True),
    )
    _sync_dir(staging)


def _write_text(path: Path, text: str) -> None:
    """Write ``text`` verbatim and flush it to the platter.

    ``newline=""`` is what makes it verbatim: the default rewrites ``\\r\\n``
    and lone ``\\r`` to ``\\n``, and a store whose whole job is holding source
    evidence must not edit the evidence. The ``fsync`` is what makes the rename
    in :func:`_swap_into_place` a *publish* — without it the directory entry can
    reach disk ahead of the file contents, and a power loss leaves a document
    that lists every section and reads back empty.
    """
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def _read_text(path: Path) -> str:
    """Read a chunk exactly as stored — no universal-newline translation."""
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return handle.read()


def _sync_dir(path: Path) -> None:
    """Flush a directory entry so the names in it survive a power loss."""
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        # Not every platform lets a directory be opened; the file data is
        # already flushed, so degrade rather than fail the import.
        return
    try:
        os.fsync(fd)
    except OSError:
        return
    finally:
        os.close(fd)


def _swap_into_place(*, staging: Path, final: Path, doc: str) -> None:
    """Replace ``final`` with ``staging`` using two renames.

    ``os.replace(staging, final)`` is not enough: POSIX ``rename(2)`` refuses
    to replace a directory that is not empty. So the live document is renamed
    aside first, staging takes its place, and only then is the old copy
    removed. Anything raised in between — an ``OSError`` or a ``Ctrl-C`` —
    puts the live document straight back; a process killed outright leaves it
    in the trash directory, which the next import of this document restores
    before it reads the prior revision.
    """
    trash: Path | None = None
    if final.exists():
        trash = final.parent / f".{doc}.trash.{uuid.uuid4().hex}"
        os.rename(final, trash)
    try:
        os.rename(staging, final)
    except BaseException:
        if trash is not None:
            # Best effort: never mask the failure that got us here.
            with contextlib.suppress(OSError):
                os.rename(trash, final)
        raise
    _sync_dir(final.parent)
    if trash is not None:
        shutil.rmtree(trash, ignore_errors=True)


def _recover_scratch(pot_root: Path, doc: str, *, final: Path) -> None:
    """Undo a crashed import of ``doc``, then drop leftovers it is safe to drop.

    Two jobs, in order. First, if the live document is missing and a trash
    directory is present, the process died inside :func:`_swap_into_place`'s
    rename window: that trash *is* the document, so it goes back. Deleting it
    instead would discard the bytes the graph's ``retrieval_uri``s still point
    at and rewind ``revision`` to 1, which is the counter R7 hangs prior-
    revision claim invalidation on.

    Second, leftovers are swept by age, never by name alone. A directory named
    ``.<doc>.staging.*`` may belong to an import running right now, and
    deleting a live staging tree does not fail that import — it recreates the
    section directories it has not written yet and then publishes a manifest
    naming chunk files that no longer exist.
    """
    entries = [
        path
        for path in pot_root.iterdir()
        if path.name.startswith((f".{doc}.staging.", f".{doc}.trash."))
    ]
    if not final.exists():
        trash = [path for path in entries if path.name.startswith(f".{doc}.trash.")]
        newest = max(trash, key=_mtime, default=None)
        if newest is not None:
            with contextlib.suppress(OSError):
                os.rename(newest, final)
                entries.remove(newest)
    cutoff = time.time() - _SCRATCH_STALE_SECONDS
    for path in entries:
        if _mtime(path) < cutoff:
            shutil.rmtree(path, ignore_errors=True)


def _mtime(path: Path) -> float:
    """Modification time, or "now" when it cannot be read.

    An unreadable leftover reads as fresh so the sweep leaves it alone; the
    only cost is one more directory surviving until the next import.
    """
    try:
        return path.stat().st_mtime
    except OSError:
        return time.time()


def _count_documents(pot_root: Path) -> int:
    """Count a pot's stored documents — one readdir, no manifest reads.

    Scratch trees are dot-prefixed (``.<doc>.staging.``, ``.<doc>.trash.``), so
    skipping dotted names counts documents and nothing else.
    """
    try:
        return sum(
            1
            for path in pot_root.iterdir()
            if path.is_dir() and not path.name.startswith(".")
        )
    except OSError:
        return 0


def _remove_tree(path: Path) -> bool:
    if not path.exists():
        return False
    shutil.rmtree(path)
    return True


__all__ = [
    "LocalResourceStore",
    "META_FILENAME",
    "SourceDocument",
    "build_chunk",
    "build_import_manifest",
    "chunk_filename",
    "chunk_not_found",
    "document_not_found",
    "find_chunk_ref",
    "pot_dir_name",
    "read_source_document",
    "section_not_found",
]
