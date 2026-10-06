"""``potpie resource`` — document payloads through the typed engine boundary.

Document payloads are the bytes the graph only points at. An agent writes an
extraction script, the script emits a chunk directory, and ``resource import``
absorbs it; ``resource get`` resolves a chunk id straight to its text with no
graph query on the path. See ``docs/context-graph/resources.md``.

Every command is one typed operation (``resource_import``, ``resource_get``,
``resource_list``, ``resource_rm``, ``resource_index_*``) dispatched through
``get_engine_client``, so the in-process runtime and the local daemon answer
identically. The store's failures keep their own stable codes
(``resource_chunk_too_large``, ``resource_not_found``, ...) instead of being
flattened to ``validation_error``, because an agent retries a bad slug and an
oversized chunk differently. They all exit ``1``: each is a caller mistake.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Iterator, Sequence

import typer

from potpie.cli.commands._common import (
    EXIT_VALIDATION,
    confirm_destructive_operation,
    contract,
    emit,
    fail,
    get_engine_client,
    get_root_runtime,
    is_json,
    resolve_pot_id,
    run_engine_operation,
)
from potpie_context_engine.core.cli_commands import join_command
from potpie_context_engine.core.ports.resource_store import (
    RESOURCE_GET_MAX_IDS,
    Chunk,
    ResourceStoreError,
    SectionManifest,
    format_resource_id,
    parse_resource_id,
    read_import_files,
)
from potpie_context_engine.core.resource_projection import (
    project_chunk,
    project_public_metadata,
)
from potpie_context_engine.core.resource_to_semantic import ResourceImportResult
from potpie_context_engine.requests import (
    ResourceGetRequest,
    ResourceImportRequest,
    ResourceIndexBuildRequest,
    ResourceIndexRebuildRequest,
    ResourceIndexStatusRequest,
    ResourceListRequest,
    ResourceRmRequest,
)

resource_app = typer.Typer(
    help="Document payloads: import a chunk directory, read chunks, list, remove."
)

_RESOURCE_OUTPUT_BUDGET_BYTES = 32_768

# A nested sub-app, the shape ``pot default`` already uses. The index is an
# implementation detail of ``resource``, not a peer of it: nothing here is
# meaningful without documents, and promoting it to a root command group would
# advertise a fifth verb the four-tool contract does not have.
index_app = typer.Typer(help="The retrieval index over stored chunks.")
resource_app.add_typer(index_app, name="index")


@contextmanager
def _resource_contract() -> Iterator[None]:
    """The shared error boundary, plus client-side store error codes.

    Failures from the engine already arrive typed, with the store's code. The
    one store error raised *here* is ``read_import_files`` refusing the
    directory before anything is dispatched; it carries its code too, and that
    code reaches the ``--json`` envelope instead of ``validation_error``.
    """
    with contract():
        try:
            yield
        except ResourceStoreError as exc:
            fail(
                code=exc.code,
                message=str(exc),
                detail=exc.detail,
                next_action=exc.recommended_next_action,
                exit_code=EXIT_VALIDATION,
            )


def _target_pot(pot: str | None) -> str:
    """The pot id every follow-up command names, resolved once up front."""
    return resolve_pot_id(get_root_runtime(), pot)


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
        pot_id = _target_pot(pot)
        # The directory is read here, on the caller's side, and its contents
        # travel with the call. The store may run in a daemon with another
        # working directory, so a path would be resolved somewhere else; and
        # the operation deliberately never reads the executing host's
        # filesystem on a caller's behalf.
        files = read_import_files(Path(directory).expanduser().resolve())
        result = run_engine_operation(
            get_engine_client(pot_id).resource_import(
                ResourceImportRequest(
                    doc=doc,
                    files=files,
                    source_ref=source_ref,
                    source_kind=source_kind,
                )
            )
        )
        payload = _import_payload(result)
        emit(payload, human=_import_human(payload))


@resource_app.command("get")
def resource_get(
    resource_ids: list[str] = typer.Argument(
        ...,
        help=f"One to {RESOURCE_GET_MAX_IDS} potpie://res/<doc>/<section>/<seq> ids.",
    ),
    with_neighbors: bool = typer.Option(
        False,
        "--with-neighbors",
        help="Also return the chunks either side, within the same section.",
    ),
    full: bool = typer.Option(
        False,
        "--full",
        help="Bypass the output budget; credential metadata remains redacted.",
    ),
    pot: str = typer.Option(None, "--pot"),
) -> None:
    """Read chunk text by id — a file read, no graph query and no embedding."""
    with _resource_contract():
        pot_id = _target_pot(pot)
        requested = tuple(resource_ids)
        result = run_engine_operation(
            get_engine_client(pot_id).resource_get(
                ResourceGetRequest(
                    resource_ids=requested, with_neighbors=with_neighbors
                )
            )
        )
        # ``success`` is every id resolved; anything else carries the per-id
        # outcomes, and only the failed ids need follow-up.
        batch = result if result.status != "success" else None
        chunks = tuple(project_chunk(chunk) for chunk in result.chunks)
        resolved_roots = (
            tuple(
                outcome.resource_id for outcome in batch.outcomes if outcome.chunk_ids
            )
            if batch
            else requested
        )
        receipt: dict[str, Any] = {}
        if batch:
            failures = [error for outcome in batch.outcomes for error in outcome.errors]
            first = failures[0] if failures else {}
            receipt = {
                "ok": False,
                "status": batch.status,
                "code": "resource_batch_partial"
                if chunks
                else first.get("code", "resource_not_found"),
                "message": "Only failed ids need correction or follow-up."
                if chunks
                else first.get("message", "No requested id resolved."),
                "detail": first.get("detail"),
                "recommended_next_action": first.get("recommended_next_action"),
                "outcomes": [asdict(outcome) for outcome in batch.outcomes],
            }
        payload = {
            **receipt,
            "requested": list(requested),
            "with_neighbors": with_neighbors,
            "count": len(chunks),
            "chunks": [_chunk_payload(row, resolved_roots) for row in chunks],
        }
        if not full:
            payload = _bound_resource_payload(payload, pot_id=pot_id)
        if is_json():
            emit(payload, human="")
            if batch:
                raise typer.Exit(code=EXIT_VALIDATION)
            return
        if batch:
            typer.echo(
                f"status={batch.status}; {len(chunks)} chunks returned. "
                "Only failed ids need follow-up."
            )
        # Deliberately not `emit`'s human block: this command's whole job is
        # returning projected text with its line breaks intact; the shared
        # formatter drops blank lines and dims body copy.
        for index, chunk in enumerate(payload["chunks"]):
            if index:
                typer.echo("")
            source = next(
                row for row in chunks if row.resource_id == chunk["resource_id"]
            )
            typer.echo(_chunk_header(source, resolved_roots))
            typer.echo(chunk["text"])
            if chunk.get("omitted_characters"):
                typer.echo(
                    f"  … {chunk['omitted_characters']} characters omitted by output budget"
                )
        if payload.get("omitted_chunk_count"):
            typer.echo(
                f"… {payload['omitted_chunk_count']} chunks omitted by output budget"
            )
        if payload.get("recommended_next_action"):
            typer.echo(f"Next: {payload['recommended_next_action']}")
        if batch:
            for outcome in receipt["outcomes"]:
                typer.echo(f"\n{outcome['resource_id']}: {outcome['status']}")
                for error in outcome["errors"]:
                    typer.echo(f"  {error['code']}: {error['message']}")
                    detail = error.get("detail")
                    for candidate in (
                        detail.get("candidates", ()) if isinstance(detail, dict) else ()
                    ):
                        typer.echo(f"  candidate: {candidate['resource_id']}")
                        if candidate.get("fetch_command"):
                            typer.echo(f"    {candidate['fetch_command']}")
                    if error.get("recommended_next_action"):
                        typer.echo(f"  Next: {error['recommended_next_action']}")
            raise typer.Exit(code=EXIT_VALIDATION)


@resource_app.command("list")
def resource_list(
    doc: str = typer.Option(..., "--doc", help="Document slug."),
    section: str = typer.Option(None, "--section", help="Limit to one section."),
    limit: int = typer.Option(10, "--limit", help="Maximum sections in the overview."),
    full: bool = typer.Option(
        False, "--full", help="Bypass the output budget for the selected sections."
    ),
    pot: str = typer.Option(None, "--pot"),
) -> None:
    """List a document's sections with their chunk ids and labels."""
    with _resource_contract():
        if limit < 1:
            raise ValueError("--limit must be >= 1")
        pot_id = _target_pot(pot)
        result = run_engine_operation(
            get_engine_client(pot_id).resource_list(
                ResourceListRequest(doc=doc, section=section)
            )
        )
        sections = result.sections
        shown = sections if section else sections[:limit]
        revisions = {row.revision for row in sections if row.revision is not None}
        revision = next(iter(revisions)) if len(revisions) == 1 else None
        payload = {
            "doc": doc,
            "section_count": len(sections),
            "chunk_count": sum(len(row.chunks) for row in sections),
            "returned_section_count": len(shown),
            "omitted_section_count": len(sections) - len(shown),
            "revision": revision,
            "sections": [
                _section_payload(doc, row, revision=row.revision) for row in shown
            ],
        }
        if len(sections) > len(shown):
            payload["recommended_next_action"] = join_command(
                [
                    "potpie",
                    "resource",
                    "list",
                    "--doc",
                    doc,
                    "--section",
                    sections[len(shown)].slug,
                    "--pot",
                    pot_id,
                ]
            )
        if not full:
            payload = _bound_resource_list_payload(payload, pot_id=pot_id)
        emit(payload, human=_list_human(payload))


@resource_app.command("rm")
def resource_rm(
    doc: str = typer.Argument(..., help="Document slug to remove."),
    confirm: bool = typer.Option(
        False, "--confirm", help="Required: removing a document's chunks is permanent."
    ),
    pot: str = typer.Option(None, "--pot"),
) -> None:
    """Remove one document's stored chunks (and its section claims) from a pot."""
    with _resource_contract():
        pot_id = _target_pot(pot)
        confirmation = confirm_destructive_operation(
            confirmed_by_flag=confirm,
            prompt=f"Remove document '{doc}' and its stored chunks from {pot_id}?",
            rerun_command=f"potpie resource rm {doc} --pot {pot_id} --confirm",
        )
        result = run_engine_operation(
            get_engine_client(pot_id).resource_rm(
                ResourceRmRequest(doc=doc), confirmation=confirmation
            )
        )
        removed = result.removed
        # ``graph_retracted`` is the retraction's own result, never ``removed``
        # echoed back: that claimed a graph write on runs where none happened.
        retracted = result.graph_retracted
        emit(
            {
                "doc": doc,
                "removed": removed,
                "graph_retracted": retracted,
                "review_required_claim_keys": list(result.review_required_claim_keys),
                "review_marker_errors": list(result.review_marker_errors),
            },
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
        pot_id = _target_pot(pot)
        status = run_engine_operation(
            get_engine_client(pot_id).resource_index_status(
                ResourceIndexStatusRequest()
            )
        )
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

    ``--wait`` drains one bounded batch per call until nothing is pending (or a
    batch embeds nothing, which means the embedder is failing), so no single
    call outlives the client's request deadline however large the backlog is.
    """
    with _resource_contract():
        pot_id = _target_pot(pot)
        client = get_engine_client(pot_id)
        # ``--doc`` first re-derives that document's index rows: pending work
        # is per pot, so this is what makes the flag mean something on a
        # document whose rows are missing entirely.
        report = run_engine_operation(
            client.resource_index_build(ResourceIndexBuildRequest(doc=doc or None))
        )
        embedded, batches, elapsed_ms = (
            report.embedded,
            report.batches,
            report.elapsed_ms,
        )
        while wait and report.remaining and report.embedded:
            report = run_engine_operation(
                client.resource_index_build(ResourceIndexBuildRequest())
            )
            embedded += report.embedded
            batches += report.batches
            elapsed_ms += report.elapsed_ms
        payload = {
            "profile": report.profile,
            "doc": doc,
            "embedded": embedded,
            "remaining": report.remaining,
            "batches": batches,
            "elapsed_ms": elapsed_ms,
            "detail": report.detail,
        }
        emit(
            payload,
            human=(
                f"embedded {embedded} window(s) in {elapsed_ms}ms; "
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
    be lost; that is also why this is not a destructive operation.
    """
    with _resource_contract():
        pot_id = _target_pot(pot)
        if not confirm:
            fail(
                code="confirmation_required",
                message="rebuilding re-derives the whole index and re-embeds it",
                next_action=(
                    "re-run with 'potpie resource index rebuild --confirm'"
                    + (f" --doc {doc}" if doc else "")
                ),
            )
        result = run_engine_operation(
            get_engine_client(pot_id).resource_index_rebuild(
                ResourceIndexRebuildRequest(doc=doc or None)
            )
        )
        reports = result.reports
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
            "Set a working profile with 'potpie config set resource_index "
            "<sqlite_hybrid|sqlite_fts>' (CONTEXT_ENGINE_RESOURCE_INDEX overrides "
            "it), then run 'potpie resource index rebuild --confirm'."
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
    warnings.extend(result.review_marker_errors)
    if pending:
        warnings.append(
            f"{len(pending)} section(s) imported without a summary: "
            f"{', '.join(pending)}. Summaries improve graph context retrieval; "
            "indexed chunk text is searchable with 'potpie search <phrase> --include resources'."
        )
    graph = _graph_payload(result)
    warnings.extend(graph["warnings"])
    index = _index_payload(result)
    warnings.extend(index["warnings"])
    return {
        # Every other write envelope carries `ok`; a consumer branching on the
        # same key it uses for the error shape had nothing to read here.
        "ok": True,
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
        "review_required_claim_keys": list(result.review_required_claim_keys),
        "review_marker_errors": list(result.review_marker_errors),
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


def _section_payload(
    doc: str, section: SectionManifest, *, revision: int
) -> dict[str, Any]:
    return project_public_metadata(
        {
            "slug": section.slug,
            "title": section.title,
            "ordinal": section.ordinal,
            "summary": section.summary,
            "summary_pending": section.summary_pending,
            "content_hash": section.content_hash,
            "chunks": [
                {
                    # The id is the point of `list`: it is what `get` takes.
                    "resource_id": format_resource_id(
                        doc, section.slug, ref.seq, revision=revision
                    ),
                    "seq": ref.seq,
                    "label": ref.label,
                    "page": ref.page,
                    "offset": ref.offset,
                }
                for ref in section.chunks
            ],
        }
    )


def _is_requested_chunk(chunk: Chunk, requested: Sequence[str]) -> bool:
    for resource_id in requested:
        try:
            parsed = parse_resource_id(resource_id)
        except ResourceStoreError:
            continue
        if (parsed.doc, parsed.section, parsed.seq) == (
            chunk.doc,
            chunk.section,
            chunk.seq,
        ) and parsed.revision in (None, chunk.revision):
            return True
    return False


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
        "requested": _is_requested_chunk(chunk, requested),
    }


def _bound_resource_payload(payload: dict[str, Any], *, pot_id: str) -> dict[str, Any]:
    def size(value: dict[str, Any]) -> int:
        return len(json.dumps(value, default=str, ensure_ascii=False).encode("utf-8"))

    if size(payload) <= _RESOURCE_OUTPUT_BUDGET_BYTES:
        return payload
    rows = list(payload["chunks"])
    original_text_lengths = {row["resource_id"]: len(row["text"]) for row in rows}
    selected: list[dict[str, Any]] = []
    omitted: list[str] = []
    partial: list[str] = []
    result = {
        **payload,
        "total_chunk_count": len(rows),
        "output_budget_bytes": _RESOURCE_OUTPUT_BUDGET_BYTES,
        "chunks": selected,
    }
    # Requested roots carry the answer. Neighbors are useful context but do
    # not consume the budget before their roots.
    for row in sorted(rows, key=lambda item: not item["requested"]):
        candidate = dict(row)
        if (
            size({**result, "chunks": [*selected, candidate]})
            <= _RESOURCE_OUTPUT_BUDGET_BYTES - 2_048
        ):
            selected.append(candidate)
            continue
        if row["requested"] and not selected:
            candidate["text"] = candidate["text"][:16_000]
            candidate["omitted_characters"] = len(row["text"]) - len(candidate["text"])
            selected.append(candidate)
            partial.append(row["resource_id"])
        else:
            omitted.append(row["resource_id"])
    result["count"] = len(selected)
    result["omitted_chunk_count"] = len(omitted)
    result["omitted_chunk_ids"] = omitted[:12]
    result["omitted_chunk_ids_count"] = max(0, len(omitted) - 12)
    while size(result) > _RESOURCE_OUTPUT_BUDGET_BYTES and selected:
        last = selected[-1]
        if last["text"]:
            cutoff = max(0, len(last["text"]) - 1_024)
            last["omitted_characters"] = (
                original_text_lengths[last["resource_id"]] - cutoff
            )
            last["text"] = last["text"][:cutoff]
            if last["resource_id"] not in partial:
                partial.append(last["resource_id"])
        else:
            omitted.append(selected.pop()["resource_id"])
            result["count"] = len(selected)
            result["omitted_chunk_count"] = len(omitted)
            result["omitted_chunk_ids"] = omitted[:12]
            result["omitted_chunk_ids_count"] = max(0, len(omitted) - 12)
    follow_up_id = (partial or omitted or [None])[0]
    if follow_up_id:
        result["recommended_next_action"] = join_command(
            [
                "potpie",
                "resource",
                "get",
                follow_up_id,
                "--full",
                "--pot",
                pot_id,
            ]
        )
    return result


def _bound_resource_list_payload(
    payload: dict[str, Any], *, pot_id: str
) -> dict[str, Any]:
    def size(value: dict[str, Any]) -> int:
        return len(json.dumps(value, ensure_ascii=False, default=str).encode("utf-8"))

    if size(payload) <= _RESOURCE_OUTPUT_BUDGET_BYTES:
        return payload
    sections = [dict(row) for row in payload["sections"]]
    result = {
        **payload,
        "sections": sections,
        "output_budget_bytes": _RESOURCE_OUTPUT_BUDGET_BYTES,
    }
    while len(sections) > 1 and size(result) > _RESOURCE_OUTPUT_BUDGET_BYTES - 1_024:
        sections.pop()
    omitted_fields = 0
    if size(result) > _RESOURCE_OUTPUT_BUDGET_BYTES - 1_024 and sections:
        section = sections[0]
        for key in ("title", "summary"):
            value = section.get(key)
            if isinstance(value, str) and len(value) > 1_000:
                section[key] = value[:1_000]
                omitted_fields += 1
        chunks = section.get("chunks") or []
        if len(chunks) > 12:
            section["chunks"] = chunks[:12]
            omitted_fields += 1
        for chunk in section.get("chunks") or []:
            if len(str(chunk.get("label") or "")) > 200:
                chunk["label"] = str(chunk["label"])[:200]
                omitted_fields += 1
    result["returned_section_count"] = len(sections)
    result["omitted_section_count"] = result["section_count"] - len(sections)
    result["omitted_field_count"] = omitted_fields
    if result["omitted_section_count"]:
        first_omitted = (
            payload["sections"][len(sections)]["slug"]
            if len(payload["sections"]) > len(sections)
            else None
        )
        if first_omitted:
            result["recommended_next_action"] = join_command(
                [
                    "potpie",
                    "resource",
                    "list",
                    "--doc",
                    payload["doc"],
                    "--section",
                    first_omitted,
                    "--full",
                    "--pot",
                    pot_id,
                ]
            )
    elif omitted_fields and sections:
        result["recommended_next_action"] = join_command(
            [
                "potpie",
                "resource",
                "list",
                "--doc",
                payload["doc"],
                "--section",
                sections[0]["slug"],
                "--full",
                "--pot",
                pot_id,
            ]
        )
    return result


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
    if payload.get("omitted_section_count"):
        lines.append(f"  … {payload['omitted_section_count']} sections omitted")
        lines.append(f"Next: {payload['recommended_next_action']}")
    elif payload.get("omitted_field_count"):
        lines.append(
            f"  … {payload['omitted_field_count']} section fields bounded by output budget"
        )
        lines.append(f"Next: {payload['recommended_next_action']}")
    return "\n".join(lines)


def _chunk_header(chunk: Chunk, requested: Sequence[str]) -> str:
    neighbor = "" if _is_requested_chunk(chunk, requested) else " [neighbor]"
    page = f", page {chunk.page}" if chunk.page is not None else ""
    return (
        f"{chunk.resource_id}{neighbor}  "
        f"({chunk.chars} chars, revision {chunk.revision}{page})"
    )


__all__ = ["resource_app"]
