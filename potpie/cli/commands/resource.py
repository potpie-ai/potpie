"""Resource commands → ``HostShell.resources`` (``ResourceFacade``).

Document payloads: the bytes the graph only points at. An agent writes an
extraction script, the script emits a chunk directory, and ``resource import``
absorbs it; ``resource get`` resolves a chunk id straight to its text with no
graph query on the path. See ``docs/context-graph/resources.md``.

The store's failures carry their own stable codes (``resource_chunk_too_large``,
``resource_not_found``, …). :func:`_resource_contract` reports those verbatim
instead of flattening every one to ``validation_error``, because an agent
retries a bad slug and a too-large chunk differently. They all still exit ``1``:
each is a caller mistake, not an unavailable dependency.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence

import typer

from potpie.cli.commands._common import (
    contract,
    emit,
    fail,
    get_host,
    is_json,
    resolve_pot_id,
)
from potpie_context_core.errors import CapabilityNotImplemented
from potpie_context_core.ports.resource_store import (
    Chunk,
    SectionManifest,
    format_resource_id,
    read_import_files,
)
from potpie_context_core.resource_to_semantic import ResourceImportResult

resource_app = typer.Typer(
    help="Document payloads: import a chunk directory, read chunks, list, remove."
)

# A nested sub-app, the shape ``pot default`` and ``ledger sources`` already
# use. The index is an implementation detail of ``resource``, not a peer of it:
# nothing here is meaningful without documents, and promoting it to a root
# command group would advertise a fifth verb the four-tool contract does not
# have.
index_app = typer.Typer(help="The retrieval index over stored chunks.")
resource_app.add_typer(index_app, name="index")


@contextmanager
def _resource_contract() -> Iterator[None]:
    """The shared error boundary, plus the store's own error codes.

    A ``ResourceStoreError`` is a ``ValueError`` carrying ``code``; across the
    daemon hop it arrives as a plain ``ValueError`` with the same attribute
    re-attached. Either way the code reaches the ``--json`` envelope here, and
    anything without one falls through to ``contract()`` unchanged.
    """
    with contract():
        try:
            yield
        except CapabilityNotImplemented as exc:
            # These four verbs are advertised identically on every host —
            # `resource --help` is host-independent — so the refusal is the only
            # place a user can find out the feature is not there. It has to say
            # so in their terms: a host that does not serve the `resources`
            # surface answers "invalid RPC surface: resources", which names an
            # internal routing concept and nothing a user can act on.
            #
            # Still loud, and still exit 2. Unlike an enumeration, `resource get`
            # is aimed at a specific store; degrading it would mean answering
            # from somewhere the caller did not ask about.
            # Matched on the *surface* exactly, not on a prefix: the engine's own
            # gaps are named `resource_index.<profile>....` and mean something
            # much more specific ("switch the index profile"), which this
            # rewording would destroy.
            capability = str(getattr(exc, "capability", ""))
            if capability != "resources" and not capability.startswith("resources."):
                raise
            raise CapabilityNotImplemented(
                exc.capability,
                detail="this host does not serve document payloads",
                recommended_next_action=(
                    "run this against the local host: "
                    "'potpie --host local resource ...'"
                ),
            ) from exc
        except ValueError as exc:
            code = getattr(exc, "code", None)
            if not code:
                raise
            # No explicit `exit_code`: the code here comes from whatever the
            # raiser attached, so pinning the number pins it for codes that do
            # not exist yet. `exit_code_for` gives the same 1 for today's
            # `confirmation_required`/`daemon_error` and the right answer if one
            # of them ever becomes an unavailability.
            fail(
                code=str(code),
                message=str(exc),
                detail=getattr(exc, "detail", None),
                next_action=getattr(exc, "recommended_next_action", None),
            )


@resource_app.command("import")
def resource_import(
    directory: Path = typer.Argument(
        ...,
        help="Directory an extraction script produced: <section>/<seq>.txt + meta.json.",
    ),
    doc: str = typer.Option(..., "--doc", help="Document slug, e.g. q3-review."),
    source_ref: str = typer.Option(
        None, "--source-ref", help="Where the document came from; overrides meta.json."
    ),
    source_kind: str = typer.Option(
        None, "--source-kind", help="Format tag (pdf, spreadsheet, …)."
    ),
    pot: str = typer.Option(None, "--pot"),
) -> None:
    """Absorb a chunk directory as one document (atomic; replaces on re-import)."""
    with _resource_contract():
        host = get_host()
        pot_id = resolve_pot_id(host, pot)
        # The bytes travel with the call. The store may sit in a daemon with a
        # different working directory, or on a managed host with no view of
        # this machine at all, so a path would land somewhere else entirely —
        # or nowhere. Read the directory here and ship its contents; every
        # host then imports the same way.
        files = read_import_files(Path(directory).expanduser())
        try:
            result = host.resources.import_dir(
                pot_id=pot_id,
                slug=doc,
                files=files,
                source_ref=source_ref,
                source_kind=source_kind,
            )
        except Exception as exc:
            if not _host_predates_inline_import(exc):
                raise
            # A host built before ``files`` existed refuses the keyword itself.
            # That arrives as the daemon's generic error, which the contract
            # renders as "run potpie doctor"; the real repair is a host upgrade.
            raise CapabilityNotImplemented(
                "resource_import.inline_files",
                detail=(
                    "this host only imports from a directory it can read itself; "
                    "it predates inline import"
                ),
                recommended_next_action=(
                    "upgrade the host, or run the import against the local host: "
                    "'potpie --host local resource import ...'"
                ),
            ) from exc
        payload = _import_payload(result)
        emit(payload, human=_import_human(payload))


def _host_predates_inline_import(exc: BaseException) -> bool:
    """Did the host refuse the ``files`` keyword rather than the import?"""
    return "unexpected keyword argument 'files'" in str(exc)


@resource_app.command("get")
def resource_get(
    resource_ids: list[str] = typer.Argument(
        ..., help="One or more potpie://res/<doc>/<section>/<seq> ids."
    ),
    with_neighbors: bool = typer.Option(
        False,
        "--with-neighbors",
        help="Also return the chunks either side, within the same section.",
    ),
    pot: str = typer.Option(None, "--pot"),
) -> None:
    """Read chunk text by id — a file read, no graph query and no embedding."""
    with _resource_contract():
        host = get_host()
        pot_id = resolve_pot_id(host, pot)
        requested = tuple(resource_ids)
        chunks = host.resources.get(
            pot_id=pot_id, resource_ids=requested, with_neighbors=with_neighbors
        )
        if is_json():
            emit(
                {
                    "requested": list(requested),
                    "with_neighbors": with_neighbors,
                    "count": len(chunks),
                    "chunks": [_chunk_payload(row, requested) for row in chunks],
                },
                human="",
            )
            return
        # Deliberately not `emit`'s human block: this command's whole job is
        # returning stored text verbatim, and the shared formatter drops blank
        # lines and dims body copy, which would silently edit the evidence.
        for index, chunk in enumerate(chunks):
            if index:
                typer.echo("")
            typer.echo(_chunk_header(chunk, requested))
            typer.echo(chunk.text)


@resource_app.command("list")
def resource_list(
    doc: str = typer.Option(..., "--doc", help="Document slug."),
    section: str = typer.Option(None, "--section", help="Limit to one section."),
    pot: str = typer.Option(None, "--pot"),
) -> None:
    """List a document's sections with their chunk ids and labels."""
    with _resource_contract():
        host = get_host()
        pot_id = resolve_pot_id(host, pot)
        sections = host.resources.list(pot_id=pot_id, slug=doc, section=section)
        payload = {
            "doc": doc,
            "section_count": len(sections),
            "chunk_count": sum(len(row.chunks) for row in sections),
            "sections": [_section_payload(doc, row) for row in sections],
        }
        emit(payload, human=_list_human(payload))


@resource_app.command("rm")
def resource_rm(
    doc: str = typer.Argument(..., help="Document slug to remove."),
    confirm: bool = typer.Option(
        False, "--confirm", help="Required: removing a document's chunks is permanent."
    ),
    pot: str = typer.Option(None, "--pot"),
) -> None:
    """Remove one document's stored chunks from the active pot."""
    with _resource_contract():
        host = get_host()
        pot_id = resolve_pot_id(host, pot)
        if not confirm and not _confirmed_interactively(doc):
            fail(
                code="confirmation_required",
                message=f"removing document '{doc}' discards its stored chunks",
                next_action=f"re-run with 'potpie resource rm {doc} --confirm'",
            )
        result = host.resources.delete(pot_id=pot_id, slug=doc)
        removed = result.removed
        # ``graph_retracted`` reports the retraction's own result. It used to be
        # ``removed`` echoed back, which claimed a graph write had happened on
        # runs where none did.
        retracted = result.graph_retracted
        emit(
            {"doc": doc, "removed": removed, "graph_retracted": retracted},
            human=(
                f"removed document '{doc}' "
                + ("(chunks and section claims)" if retracted else "(chunks)")
                if removed
                else f"no document '{doc}' stored in this pot"
            ),
        )


# --- index ------------------------------------------------------------------


@index_app.command("status")
def resource_index_status(
    pot: str = typer.Option(None, "--pot"),
) -> None:
    """Profile, declared capabilities, counts, and outstanding embeddings."""
    with _resource_contract():
        host = get_host()
        pot_id = resolve_pot_id(host, pot)
        status = host.resources.index_status(pot_id=pot_id)
        payload = {
            "profile": status.profile,
            "ready": status.ready,
            # Declared capabilities, not a guess from the profile name: a
            # hybrid profile whose extension will not load reports itself
            # lexical here, which is the whole point of asking.
            "capabilities": list(status.capabilities),
            "match_mode": status.match_mode,
            "documents": status.documents,
            "chunks": status.chunks,
            "windows": status.windows,
            "pending_embeddings": status.pending_embeddings,
            "embedder": status.embedder,
            "dimensions": status.dimensions,
            "location": status.location,
            "replica": status.replica,
            "shared_store": status.shared_store,
            "detail": status.detail,
            "recommended_next_action": _index_next_action(status),
        }
        emit(payload, human=_index_status_human(payload))


@index_app.command("build")
def resource_index_build(
    doc: str = typer.Option(None, "--doc", help="Limit the drain to one document."),
    wait: bool = typer.Option(
        False, "--wait", help="Keep draining until nothing is pending."
    ),
    pot: str = typer.Option(None, "--pot"),
) -> None:
    """Embed pending chunks now instead of waiting for the background drain.

    Import returns before the vectors exist — that is deliberate, and the
    reason it takes seconds rather than minutes. This command is for the cases
    that cannot wait for a background loop: a CI step, a post-deploy hook, or a
    human who wants search working before the next command.
    """
    with _resource_contract():
        host = get_host()
        pot_id = resolve_pot_id(host, pot)
        # ``--doc`` narrows nothing today at the port level (pending work is
        # per-pot), so it is honoured by rebuilding that document's rows first
        # and then draining the pot. Rebuilding is what makes the flag mean
        # something on a document whose index rows are missing entirely.
        if doc:
            host.resources.index_rebuild(pot_id=pot_id, doc=doc)
        report = host.resources.index_build(pot_id=pot_id, wait=wait)
        payload = {
            "profile": report.profile,
            "doc": doc,
            "embedded": report.embedded,
            "remaining": report.remaining,
            "batches": report.batches,
            "elapsed_ms": report.elapsed_ms,
            "detail": report.detail,
        }
        emit(
            payload,
            human=(
                f"embedded {report.embedded} window(s) in {report.elapsed_ms}ms; "
                f"{report.remaining} pending"
                + (f"\n  ! {report.detail}" if report.detail else "")
            ),
        )


@index_app.command("rebuild")
def resource_index_rebuild(
    doc: str = typer.Option(None, "--doc", help="Rebuild one document only."),
    confirm: bool = typer.Option(
        False, "--confirm", help="Required: the index is dropped and re-derived."
    ),
    pot: str = typer.Option(None, "--pot"),
) -> None:
    """Drop the index and re-derive it from the stored files.

    The index is derived state, so this is its entire recovery story — there is
    no migration and no repair. It is safe by construction: the files are the
    source of truth and nothing here writes to them. ``--confirm`` is required
    only because re-embedding a corpus costs minutes, not because anything can
    be lost.
    """
    with _resource_contract():
        host = get_host()
        pot_id = resolve_pot_id(host, pot)
        if not confirm:
            fail(
                code="confirmation_required",
                message="rebuilding re-derives the whole index and re-embeds it",
                next_action=(
                    "re-run with 'potpie resource index rebuild --confirm'"
                    + (f" --doc {doc}" if doc else "")
                ),
            )
        reports = host.resources.index_rebuild(pot_id=pot_id, doc=doc)
        payload = {
            "documents": [
                {
                    "doc": report.doc,
                    "sections": report.sections,
                    "chunks": report.chunks,
                    "windows": report.windows,
                    "pending_embeddings": report.pending_embeddings,
                    "detail": report.detail,
                }
                for report in reports
            ],
            "document_count": len(reports),
            "chunk_count": sum(report.chunks for report in reports),
            "pending_embeddings": sum(report.pending_embeddings for report in reports),
            "recommended_next_action": (
                "Run 'potpie resource index build --wait' to embed now, or let the "
                "background drain finish."
                if any(report.pending_embeddings for report in reports)
                else 'Verify retrieval: potpie search "<a phrase>" --include resources'
            ),
        }
        emit(payload, human=_index_rebuild_human(payload))


def _index_next_action(status: Any) -> str:
    if not status.ready:
        return (
            "Set CONTEXT_ENGINE_RESOURCE_INDEX to a working profile, then run "
            "'potpie resource index rebuild --confirm'."
        )
    if status.pending_embeddings:
        return (
            f"{status.pending_embeddings} window(s) are not embedded yet; search is "
            "lexical until they are. Run 'potpie resource index build --wait' to "
            "finish now."
        )
    if not status.documents:
        return "Import a document: potpie resource import ./out --doc <slug>"
    return 'Verify retrieval: potpie search "<a phrase>" --include resources'


def _index_status_human(payload: dict[str, Any]) -> str:
    lines = [
        f"index: {payload['profile']} ready={payload['ready']} "
        f"mode={payload['match_mode']}",
        f"  capabilities: {', '.join(payload['capabilities']) or 'none'}",
        f"  documents: {payload['documents']}  chunks: {payload['chunks']}  "
        f"windows: {payload['windows']}",
        f"  pending embeddings: {payload['pending_embeddings']}",
    ]
    if payload["embedder"]:
        lines.append(f"  embedder: {payload['embedder']} ({payload['dimensions']}d)")
    if payload["location"]:
        lines.append(f"  location: {payload['location']}")
    if payload["shared_store"]:
        lines.append(f"  replica: {payload['replica']}")
    if payload["detail"]:
        lines.append(f"  ! {payload['detail']}")
    return "\n".join(lines)


def _index_rebuild_human(payload: dict[str, Any]) -> str:
    lines = [
        f"rebuilt {payload['document_count']} document(s), "
        f"{payload['chunk_count']} chunk(s)"
    ]
    for row in payload["documents"]:
        lines.append(
            f"  {row['doc']}: {row['chunks']} chunk(s), {row['windows']} window(s)"
            + (f" — {row['detail']}" if row["detail"] else "")
        )
    if payload["pending_embeddings"]:
        lines.append(f"  pending embeddings: {payload['pending_embeddings']}")
    return "\n".join(lines)


def _confirmed_interactively(doc: str) -> bool:
    """Ask, but only where a human can answer.

    ``--json`` means a script or an agent is driving and there is nobody to
    prompt; a non-tty means the same. Both must pass ``--confirm`` explicitly.
    """
    if is_json():
        return False
    from potpie.cli.ui.setup_wizard_ui import is_interactive_tty

    if not is_interactive_tty():
        return False
    from potpie.cli.ui.interactive_prompts import prompt_yes_no

    return bool(
        prompt_yes_no(f"Remove document '{doc}' and its stored chunks?", default=False)
    )


# --- payloads ---------------------------------------------------------------


def _import_payload(result: ResourceImportResult) -> dict[str, Any]:
    """Render the import report, with the changed sections spelled out.

    ``sections_changed`` is not a stored field: it is what is left after added,
    kept, and removed, and it is the answer to "what needs re-summarizing"
    (R14), so the CLI derives it rather than making every caller do the
    subtraction.
    """
    manifest = result.manifest
    accounted = {
        *manifest.sections_added,
        *manifest.sections_kept,
        *manifest.sections_removed,
    }
    changed = tuple(
        sorted(row.slug for row in manifest.sections if row.slug not in accounted)
    )
    pending = tuple(row.slug for row in manifest.sections if row.summary_pending)
    warnings = list(manifest.warnings)
    if pending:
        warnings.append(
            f"{len(pending)} section(s) imported without a summary: "
            f"{', '.join(pending)}. A section is found only by its summary, so "
            "read each one and write it before the document is searchable."
        )
    graph = _graph_payload(result)
    warnings.extend(graph["warnings"])
    index = _index_payload(result)
    warnings.extend(index["warnings"])
    return {
        "doc": manifest.doc,
        "revision": manifest.revision,
        "source_ref": manifest.source_ref,
        "source_kind": manifest.source_kind,
        "section_count": len(manifest.sections),
        "chunk_count": sum(len(row.chunks) for row in manifest.sections),
        "sections": [
            {
                "slug": row.slug,
                "title": row.title,
                "ordinal": row.ordinal,
                "chunk_count": len(row.chunks),
                "summary_pending": row.summary_pending,
            }
            for row in manifest.sections
        ],
        "sections_added": list(manifest.sections_added),
        "sections_kept": list(manifest.sections_kept),
        "sections_changed": list(changed),
        "sections_removed": list(manifest.sections_removed),
        "summary_pending": list(pending),
        "graph": graph,
        "index": index,
        "warnings": warnings,
        "recommended_next_action": _import_next_action(result, pending),
    }


def _index_payload(result: ResourceImportResult) -> dict[str, Any]:
    """Report the retrieval half — the part that makes the *text* findable.

    Deliberately not a warning when embeddings are outstanding. Lexical
    postings are written inline and vectors are drained in the background, so
    ``pending_embeddings > 0`` is the designed success shape of a fast import;
    treating it as a problem would train agents to wait for something they were
    never meant to wait for. A missing index, or one that failed to write, *is*
    a warning: search silently returns less.
    """
    report = result.index
    if report is None:
        return {
            "indexed": False,
            "profile": None,
            "warnings": [
                "chunks are stored, but no retrieval index is wired, so search "
                "cannot reach text that no section summary mentions."
            ],
        }
    payload: dict[str, Any] = {
        "indexed": report.detail is None,
        "profile": report.profile,
        "chunks": report.chunks,
        "windows": report.windows,
        "pending_embeddings": report.pending_embeddings,
        "warnings": [],
    }
    if report.detail:
        payload["detail"] = report.detail
        payload["warnings"] = [
            f"chunks are stored, but indexing reported: {report.detail}"
        ]
    return payload


def _graph_payload(result: ResourceImportResult) -> dict[str, Any]:
    """Report the structure half of the import — the part that makes it findable.

    A document whose bytes landed but whose graph write was rejected is the one
    failure mode an agent cannot see from the manifest: ``resource get`` keeps
    working while search returns nothing. So the outcome is reported as a field
    *and* as a warning, and a rejection carries the validator's own messages
    rather than a generic 'graph write failed'.
    """
    mutation = result.graph
    if mutation is None:
        return {
            "written": False,
            "status": "skipped",
            "entity_key": None,
            "warnings": [
                "chunks are stored, but no graph service was available to write "
                "the document's structure, so search cannot find it."
            ],
        }
    payload: dict[str, Any] = {
        # Readback-backed, not status-backed: a write that applies onto a
        # retracted claim reports 'applied' and is still invisible to search.
        "written": result.graph_written,
        "status": mutation.status,
        "entity_key": f"document:{result.manifest.doc}",
        "operations_applied": mutation.operations_applied,
        "claim_keys": list(mutation.claim_keys),
        "warnings": [],
    }
    if result.missing_claim_keys:
        payload["missing_claim_keys"] = list(result.missing_claim_keys)
        payload["warnings"] = [
            f"the graph write reported '{mutation.status}' but "
            f"{len(result.missing_claim_keys)} of {len(mutation.claim_keys)} claims "
            "cannot be read back, so this document is not fully findable."
        ]
    elif not payload["written"]:
        detail = mutation.detail or "; ".join(
            issue.message for issue in mutation.issues if issue.is_error
        )
        payload["warnings"] = [
            f"chunks are stored, but the graph write came back '{mutation.status}', "
            f"so search cannot find this document" + (f": {detail}" if detail else ".")
        ]
    return payload


def _import_next_action(result: ResourceImportResult, pending: Sequence[str]) -> str:
    """One next step, chosen by what is actually missing.

    Scope is the second check because it is the failure nobody notices: a
    document with no ``DOCUMENTS`` edge is findable by semantic luck alone, and
    import has no way to guess what the document is *about*. It fires only at
    zero live scope claims — recommending a link on a document that already has
    one made the signal unusable for telling linked from unlinked, which is the
    whole reason ``resources.md`` asks for it.

    ``scope_claim_count is None`` means nobody could look. Nudging then is the
    honest default: an unlinked document is the common case on a fresh import,
    and the cost of a redundant suggestion is lower than the cost of silence
    about a document nothing can find.
    """
    doc = result.manifest.doc
    if pending:
        return f"Write a summary for: {', '.join(pending)}"
    if result.scope_claim_count:
        return f'Verify retrieval: potpie search "<a phrase from {doc}>" --include docs'
    return (
        f"Link document:{doc} to what it covers with a DOCUMENTS claim "
        "(potpie graph propose, then potpie graph commit --verify), or it is "
        "findable by search alone."
    )


def _section_payload(doc: str, section: SectionManifest) -> dict[str, Any]:
    return {
        "slug": section.slug,
        "title": section.title,
        "ordinal": section.ordinal,
        "summary": section.summary,
        "summary_pending": section.summary_pending,
        "content_hash": section.content_hash,
        "chunks": [
            {
                # The id is the point of `list`: it is what `get` takes.
                "resource_id": format_resource_id(doc, section.slug, ref.seq),
                "seq": ref.seq,
                "label": ref.label,
                "page": ref.page,
                "offset": ref.offset,
            }
            for ref in section.chunks
        ],
    }


def _chunk_payload(chunk: Chunk, requested: Sequence[str]) -> dict[str, Any]:
    return {
        "resource_id": chunk.resource_id,
        "doc": chunk.doc,
        "section": chunk.section,
        "seq": chunk.seq,
        "text": chunk.text,
        "chars": chunk.chars,
        "revision": chunk.revision,
        "source_ref": chunk.source_ref,
        "page": chunk.page,
        "offset": chunk.offset,
        # False marks a chunk pulled in by --with-neighbors.
        "requested": chunk.resource_id in requested,
    }


# --- human rendering --------------------------------------------------------


def _import_human(payload: dict[str, Any]) -> str:
    counts = ", ".join(
        f"{len(payload[key])} {label}"
        for key, label in (
            ("sections_added", "added"),
            ("sections_changed", "changed"),
            ("sections_kept", "kept"),
            ("sections_removed", "removed"),
        )
        if payload[key]
    )
    graph = payload["graph"]
    index = payload["index"]
    lines = [
        f"imported {payload['doc']} revision {payload['revision']}",
        f"  sections: {payload['section_count']}" + (f" ({counts})" if counts else ""),
        f"  chunks: {payload['chunk_count']}",
        f"  graph: {graph['status']}"
        + (f" ({graph['entity_key']})" if graph["entity_key"] else ""),
        f"  index: {index['profile'] or 'none'}"
        + (
            f" ({index['chunks']} chunk(s)"
            + (
                f", {index['pending_embeddings']} embedding(s) pending)"
                if index.get("pending_embeddings")
                else ")"
            )
            if index["indexed"]
            else ""
        ),
    ]
    if payload["summary_pending"]:
        lines.append(f"  summary pending: {', '.join(payload['summary_pending'])}")
    lines.extend(f"  ! {warning}" for warning in payload["warnings"])
    return "\n".join(lines)


def _list_human(payload: dict[str, Any]) -> str:
    lines = [
        f"{payload['doc']}: {payload['section_count']} section(s), "
        f"{payload['chunk_count']} chunk(s)"
    ]
    for section in payload["sections"]:
        pending = " (summary pending)" if section["summary_pending"] else ""
        lines.append(f"  {section['slug']} — {section['title']}{pending}")
        for chunk in section["chunks"]:
            lines.append(f"    {chunk['resource_id']}  {chunk['label']}")
    return "\n".join(lines)


def _chunk_header(chunk: Chunk, requested: Sequence[str]) -> str:
    neighbor = "" if chunk.resource_id in requested else " [neighbor]"
    page = f", page {chunk.page}" if chunk.page is not None else ""
    return (
        f"{chunk.resource_id}{neighbor}  "
        f"({chunk.chars} chars, revision {chunk.revision}{page})"
    )


__all__ = ["resource_app"]
