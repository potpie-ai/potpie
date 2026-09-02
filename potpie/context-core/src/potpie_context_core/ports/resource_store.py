"""``ResourceStorePort`` — where document payloads live.

The graph holds claims and pointers, never payloads. A document imported into
Potpie is therefore split in two: its *structure* (a ``Document`` owning
``DocumentSection`` entities) is upserted into the graph, and its *bytes* land
behind this port as chunk files — pot-scoped, sized to fit one tool-call
response, and reachable by a stable ``potpie://res/<doc>/<section>/<seq>`` id.

The port is the seam a cloud deployment swaps disk for object storage. The DTOs
live in ``potpie-context-core`` because the local daemon re-constructs them by
module path when a call crosses the RPC boundary (``potpie/daemon/rpc.py``
allows core classes only), so every field here must survive a
``cls(**decoded_fields)`` round trip: no ``init=False`` fields, and tuples
rather than sets for sequences.

See ``docs/context-graph/resources.md``.
"""

from __future__ import annotations

import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Protocol

from potpie_context_core.identity import is_valid_slug_body

# --- Contract constants -----------------------------------------------------

RESOURCE_URI_PREFIX = "potpie://res/"

# Chunks are sized so one fits comfortably in a tool-call response. The cap is
# enforced at import and never clamped at read, so every stored chunk is
# uniformly safe to hand an agent.
RESOURCE_CHUNK_TARGET_CHARS = 4000
RESOURCE_CHUNK_MAX_CHARS = 8000

# The manifest fields that become node properties and claim text. Bounding them
# is R1's enforcement point, not a style rule: import is the only step that
# reads the directory, so a section body pasted into ``summary`` gets into the
# graph permanently unless it is refused here. A summary is a paragraph and a
# title or label is a line, so both caps are far above honest use.
RESOURCE_SUMMARY_MAX_CHARS = 2000
RESOURCE_LABEL_MAX_CHARS = 200

# Sequence numbers are zero-padded so ids sort lexicographically: ``0000.txt``.
RESOURCE_SEQ_WIDTH = 4

# A source with no divisions of its own still gets exactly one section.
DEFAULT_SECTION_SLUG = "body"

# Search resolves to a section; from there an agent picks a chunk by label
# alone, so a section holding more than a handful of chunks is too coarse.
# Advisory: import warns, it does not reject.
RECOMMENDED_MAX_SECTION_CHUNKS = 5

# --- Error codes ------------------------------------------------------------
# Stable strings the CLI maps to exit codes and ``--json`` error payloads.

RESOURCE_SLUG_INVALID = "resource_slug_invalid"
RESOURCE_CHUNK_TOO_LARGE = "resource_chunk_too_large"
RESOURCE_ID_INVALID = "resource_id_invalid"
RESOURCE_NOT_FOUND = "resource_not_found"
RESOURCE_MANIFEST_INVALID = "resource_manifest_invalid"
RESOURCE_SECTION_MISSING_CHUNK = "resource_section_missing_chunk"
RESOURCE_TEXT_TOO_LARGE = "resource_text_too_large"
RESOURCE_IMPORT_INVALID = "resource_import_invalid"

#: Ceiling on the bytes one ``import`` may carry over the wire. Far above any
#: real document (a chunk is capped at 8,000 chars and a section holds 1-5 of
#: them) and low enough that ``resource import .`` in the wrong directory is
#: refused before it is serialised.
RESOURCE_IMPORT_MAX_BYTES = 64 * 1024 * 1024


class ResourceStoreError(ValueError):
    """A resource-store operation failed with a stable, reportable ``code``.

    Carries the four fields the CLI output contract needs — ``code``,
    ``message``, ``detail``, ``recommended_next_action`` — so an adapter never
    has to guess how to render a store failure.

    Subclassing ``ValueError`` is what makes those fields survive the daemon
    hop, the same reason ``UnknownGraphViewError`` does it: the daemon's error
    payload has a ``ValueError`` branch that forwards ``detail`` and
    ``recommended_next_action``, while anything else is logged as an unexpected
    daemon failure and reduced to a bare message. Every store failure here is a
    caller mistake — a bad slug, an oversized chunk, a chunk id that names
    nothing — so a validation error is also what it *is*.
    """

    def __init__(
        self,
        code: str,
        message: str,
        *,
        detail: str | None = None,
        recommended_next_action: str | None = None,
    ) -> None:
        self.code = code
        self.detail = detail
        self.recommended_next_action = recommended_next_action
        super().__init__(message)

    def __reduce__(self) -> tuple[Any, ...]:
        # ``BaseException.__reduce__`` would hand ``self.args`` — just the
        # message — back to a two-argument ``__init__``. Spell the round trip
        # out so copy and pickle work.
        return (
            self.__class__,
            (self.code, str(self)),
            {
                "detail": self.detail,
                "recommended_next_action": self.recommended_next_action,
            },
        )


# --- DTOs -------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ResourceId:
    """The three addressing segments of a parsed resource id."""

    doc: str
    section: str
    seq: int


@dataclass(frozen=True, slots=True)
class ChunkRef:
    """One chunk as its section names it.

    ``label`` is required, not optional: it is the agent's only signal for
    choosing among a section's chunks once search has landed on the section.
    """

    seq: int
    label: str
    page: int | None = None
    offset: int | None = None


@dataclass(frozen=True, slots=True)
class SectionManifest:
    """One division of a document — the searchable unit.

    ``summary`` is agent-authored and becomes the section's claim, so it is the
    entire index into these chunks. A two-pass ingest may import a section
    before its summary exists; that section carries ``summary_pending``.
    """

    slug: str
    title: str
    summary: str
    ordinal: int
    content_hash: str
    chunks: tuple[ChunkRef, ...] = ()
    summary_pending: bool = False


@dataclass(frozen=True, slots=True)
class DocumentManifest:
    """The stored shape of one document, plus the report of the import that
    produced it.

    ``sections`` is the durable part. The remaining fields describe the
    transition: ``sections_added`` are slugs new in this revision,
    ``sections_removed`` are slugs the prior revision had and this one does
    not, and ``sections_kept`` are slugs present in both whose
    ``content_hash`` is unchanged. A section present in both revisions with a
    *different* hash appears in ``sections`` and in none of the three tuples —
    that difference is how a caller tells changed from unchanged, and so
    re-summarizes only what actually moved.
    """

    pot_id: str
    doc: str
    revision: int
    source_ref: str | None
    source_kind: str | None
    sections: tuple[SectionManifest, ...] = ()
    sections_added: tuple[str, ...] = ()
    sections_kept: tuple[str, ...] = ()
    sections_removed: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ResourceStoreStatus:
    """Whether the store can take bytes right now, and where they land.

    ``potpie doctor`` is the caller. A store that cannot be written is
    invisible until an import fails, and an import — a whole extraction script
    plus a directory — is an expensive way to find out. ``location`` is
    human-facing (a path today, a bucket URI later) and never parsed;
    ``documents`` counts what the named pot holds, and is ``None`` when no pot
    was named or the store cannot count cheaply.
    """

    kind: str
    ready: bool
    location: str | None = None
    documents: int | None = None
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class Chunk:
    """One chunk's text plus the provenance a reader needs to cite it."""

    resource_id: str
    doc: str
    section: str
    seq: int
    text: str
    chars: int
    revision: int
    source_ref: str | None = None
    page: int | None = None
    offset: int | None = None


# --- Resource ids -----------------------------------------------------------


def require_resource_slug(value: object, *, kind: str = "document") -> str:
    """Return ``value`` if it is a legal ``--doc`` / ``--section`` slug.

    The grammar is the graph's own (``identity.is_valid_slug_body``), which
    already forbids ``/`` and ``..``. The redundant separator check is
    deliberate: these slugs become path segments in a store implementation, so
    traversal is checked rather than assumed.
    """
    if (
        not isinstance(value, str)
        or not is_valid_slug_body(value)
        or {"/", "\\"} & set(value)
        or ".." in value
    ):
        raise ResourceStoreError(
            RESOURCE_SLUG_INVALID,
            f"{kind} slug is not a valid slug: {value!r}",
            recommended_next_action=(
                "Use lowercase letters, digits, and single hyphens, e.g. q3-review."
            ),
        )
    return value


def format_resource_id(doc: str, section: str, seq: int) -> str:
    """Render the canonical ``potpie://res/<doc>/<section>/<seq>`` id."""
    require_resource_slug(doc, kind="document")
    require_resource_slug(section, kind="section")
    if not isinstance(seq, int) or isinstance(seq, bool) or seq < 0:
        raise ResourceStoreError(
            RESOURCE_ID_INVALID,
            f"chunk sequence must be a non-negative integer: {seq!r}",
        )
    return f"{RESOURCE_URI_PREFIX}{doc}/{section}/{seq:0{RESOURCE_SEQ_WIDTH}d}"


def parse_resource_id(resource_id: str) -> ResourceId:
    """Split a resource id back into its ``doc`` / ``section`` / ``seq``.

    Strict about the canonical form — the sequence segment must be the
    zero-padded rendering :func:`format_resource_id` produces — so parse and
    format round-trip exactly and no two ids address the same chunk.
    """
    if not isinstance(resource_id, str) or not resource_id.startswith(
        RESOURCE_URI_PREFIX
    ):
        raise ResourceStoreError(
            RESOURCE_ID_INVALID,
            f"resource id must start with {RESOURCE_URI_PREFIX!r}: {resource_id!r}",
            recommended_next_action=(
                "Use a chunk id from resource list or a search result's source refs."
            ),
        )
    segments = resource_id[len(RESOURCE_URI_PREFIX) :].split("/")
    if len(segments) != 3:
        raise ResourceStoreError(
            RESOURCE_ID_INVALID,
            f"resource id needs exactly doc/section/seq segments: {resource_id!r}",
        )
    doc, section, raw_seq = segments
    require_resource_slug(doc, kind="document")
    require_resource_slug(section, kind="section")
    if not (raw_seq.isascii() and raw_seq.isdigit()):
        raise ResourceStoreError(
            RESOURCE_ID_INVALID,
            f"chunk sequence must be ascii digits: {raw_seq!r}",
        )
    seq = int(raw_seq)
    if f"{seq:0{RESOURCE_SEQ_WIDTH}d}" != raw_seq:
        raise ResourceStoreError(
            RESOURCE_ID_INVALID,
            f"chunk sequence must be zero-padded to {RESOURCE_SEQ_WIDTH}: {raw_seq!r}",
        )
    return ResourceId(doc=doc, section=section, seq=seq)


# --- Import transport ---------------------------------------------------------
#
# An import is a directory on the *caller's* machine. The store may run inside
# a daemon on that machine or on a managed host that cannot see its filesystem,
# so the port takes the directory either as a path it can read (``source_dir``)
# or as the directory's contents (``files``). ``files`` is what crosses the
# wire; a store materialises it into a scratch directory and validates it
# exactly as it would a path, so the two forms cannot drift apart.

ImportFiles = Mapping[str, str]


def read_import_files(root: Path) -> dict[str, str]:
    """The directory an extraction script produced, as ``files`` for ``import_dir``.

    Every regular file under ``root``, keyed by its POSIX-relative path and
    decoded as UTF-8. Hidden entries (``.DS_Store``, editor swap files) are
    skipped: nothing an extraction script emits starts with a dot, and a host
    should not receive bytes the import will never read. Refused, in the
    store's own error shape: a directory that is not there or is empty, a file
    that is not UTF-8 text, and a tree over ``RESOURCE_IMPORT_MAX_BYTES``. The
    first two carry ``resource_manifest_invalid`` like a missing ``meta.json``;
    the rest are ``resource_import_invalid``, a transport fault.
    """
    base = Path(root)
    # A directory that is not there, or holds nothing, is the same mistake the
    # store reports for a missing ``meta.json`` — keep its code, so the repair
    # an agent learned for one applies to the other.
    if not base.is_dir():
        raise ResourceStoreError(
            RESOURCE_MANIFEST_INVALID,
            f"import directory not found: {base}",
            recommended_next_action=(
                "Point 'resource import' at the directory the extraction script "
                "wrote: meta.json plus <section>/<seq>.txt."
            ),
        )
    files: dict[str, str] = {}
    total = 0
    for path in sorted(base.rglob("*")):
        relative = path.relative_to(base)
        if any(part.startswith(".") for part in relative.parts):
            continue
        if not path.is_file():
            continue
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise ResourceStoreError(
                RESOURCE_IMPORT_INVALID, f"cannot read {path}: {exc}"
            ) from exc
        total += len(data)
        if total > RESOURCE_IMPORT_MAX_BYTES:
            raise ResourceStoreError(
                RESOURCE_IMPORT_INVALID,
                f"import directory exceeds {RESOURCE_IMPORT_MAX_BYTES} bytes: {base}",
                recommended_next_action=(
                    "Import one document per directory; split a very large "
                    "document into several documents."
                ),
            )
        try:
            files[relative.as_posix()] = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ResourceStoreError(
                RESOURCE_IMPORT_INVALID,
                f"{relative.as_posix()} is not UTF-8 text",
                detail=str(path),
                recommended_next_action=(
                    "Chunk files must be UTF-8 text; fix the extraction "
                    "script's encoding and re-run it."
                ),
            ) from exc
    if not files:
        raise ResourceStoreError(
            RESOURCE_MANIFEST_INVALID,
            f"import directory is empty: {base}",
            recommended_next_action=(
                "Run the extraction script first; it writes the meta.json and "
                "chunk files 'resource import' absorbs."
            ),
        )
    return files


def _safe_relative_path(name: str) -> Path:
    """``name`` as a path that stays inside the scratch directory, or a refusal.

    The mapping came over the wire, so a key is untrusted: an absolute path or
    a ``..`` segment would write outside the scratch tree, and a backslash is
    ambiguous across platforms.
    """
    if not name or name != name.strip() or "\\" in name:
        raise ResourceStoreError(
            RESOURCE_IMPORT_INVALID, f"invalid file name in import: {name!r}"
        )
    path = Path(name)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise ResourceStoreError(
            RESOURCE_IMPORT_INVALID, f"invalid file name in import: {name!r}"
        )
    return path


@contextmanager
def import_source(source_dir: Path | None, files: ImportFiles | None) -> Iterator[Path]:
    """The directory an import should read, whichever form it arrived in.

    Exactly one of ``source_dir`` / ``files`` must be given. ``files`` is
    written to a scratch directory that lives only for the block, so a store
    hands either form to the same directory validator and reads the texts into
    memory before the block ends.
    """
    if (source_dir is None) == (files is None):
        raise ResourceStoreError(
            RESOURCE_IMPORT_INVALID,
            "import_dir takes exactly one of source_dir or files",
        )
    if files is None:
        assert source_dir is not None
        yield Path(source_dir)
        return
    scratch = Path(tempfile.mkdtemp(prefix="potpie-import-"))
    try:
        for name, text in files.items():
            if not isinstance(text, str):
                raise ResourceStoreError(
                    RESOURCE_IMPORT_INVALID,
                    f"import file {name!r} must be text, got {type(text).__name__}",
                )
            target = scratch / _safe_relative_path(name)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
        yield scratch
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


# --- Port -------------------------------------------------------------------


class ResourceStorePort(Protocol):
    """Pot-scoped storage for document payloads the graph only points at."""

    def import_dir(
        self,
        *,
        pot_id: str,
        slug: str,
        source_dir: Path | None = None,
        files: ImportFiles | None = None,
        source_ref: str | None = None,
        source_kind: str | None = None,
    ) -> DocumentManifest:
        """Absorb a validated chunk directory as document ``slug``.

        The directory arrives in one of two forms, exactly one of them:
        ``source_dir``, a path this store can read itself, or ``files``, its
        contents keyed by POSIX-relative path (see :func:`read_import_files`).
        The CLI always sends ``files`` — the store may live in a daemon with
        another working directory, or on a managed host with no view of the
        caller's machine at all — and a job running beside the store may pass
        ``source_dir``. Both go through the same validation.

        Atomic: a directory that fails validation leaves any prior revision of
        the document exactly as it was. Re-importing the same slug replaces the
        chunk set and bumps ``revision``. ``source_ref`` / ``source_kind``
        override the values the directory's ``meta.json`` declares."""
        ...

    def get(self, *, pot_id: str, resource_id: str) -> Chunk:
        """Resolve one chunk id to its text — no graph query, no embedding."""
        ...

    def get_many(
        self, *, pot_id: str, resource_ids: tuple[str, ...]
    ) -> tuple[Chunk, ...]:
        """Resolve several chunk ids in one call, in the order given.

        The hot path: a daemon round trip costs far more than the bytes, so a
        multi-chunk read must not become a call per chunk."""
        ...

    def list(
        self, *, pot_id: str, slug: str, section: str | None = None
    ) -> tuple[SectionManifest, ...]:
        """Return a document's sections, with their chunk labels."""
        ...

    def delete(self, *, pot_id: str, slug: str) -> bool:
        """Remove one document's chunks. Deleting what is absent is a no-op."""
        ...

    def purge_pot(self, pot_id: str) -> bool:
        """Remove every resource the pot owns, for pot reset and teardown."""
        ...

    def status(self, *, pot_id: str | None = None) -> ResourceStoreStatus:
        """Report whether the store is usable, for ``potpie doctor``.

        Never raises. A store that cannot answer says so with ``ready=False``
        and a ``detail``, because a diagnostic that throws is useless in
        exactly the situation it exists for."""
        ...


__all__ = [
    "Chunk",
    "ChunkRef",
    "DEFAULT_SECTION_SLUG",
    "DocumentManifest",
    "ImportFiles",
    "RECOMMENDED_MAX_SECTION_CHUNKS",
    "RESOURCE_CHUNK_MAX_CHARS",
    "RESOURCE_CHUNK_TARGET_CHARS",
    "RESOURCE_CHUNK_TOO_LARGE",
    "RESOURCE_ID_INVALID",
    "RESOURCE_IMPORT_INVALID",
    "RESOURCE_IMPORT_MAX_BYTES",
    "RESOURCE_LABEL_MAX_CHARS",
    "RESOURCE_MANIFEST_INVALID",
    "RESOURCE_NOT_FOUND",
    "RESOURCE_SECTION_MISSING_CHUNK",
    "RESOURCE_SEQ_WIDTH",
    "RESOURCE_SLUG_INVALID",
    "RESOURCE_SUMMARY_MAX_CHARS",
    "RESOURCE_TEXT_TOO_LARGE",
    "RESOURCE_URI_PREFIX",
    "ResourceId",
    "ResourceStoreError",
    "ResourceStorePort",
    "ResourceStoreStatus",
    "SectionManifest",
    "format_resource_id",
    "import_source",
    "parse_resource_id",
    "read_import_files",
    "require_resource_slug",
]
