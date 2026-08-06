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
    EXIT_VALIDATION,
    contract,
    emit,
    fail,
    get_host,
    is_json,
    resolve_pot_id,
)
from potpie_context_core.ports.resource_store import (
    Chunk,
    SectionManifest,
    format_resource_id,
)
from potpie_context_core.resource_to_semantic import ResourceImportResult

resource_app = typer.Typer(
    help="Document payloads: import a chunk directory, read chunks, list, remove."
)


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
        except ValueError as exc:
            code = getattr(exc, "code", None)
            if not code:
                raise
            fail(
                code=str(code),
                message=str(exc),
                detail=getattr(exc, "detail", None),
                next_action=getattr(exc, "recommended_next_action", None),
                exit_code=EXIT_VALIDATION,
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
        result = host.resources.import_dir(
            pot_id=pot_id,
            slug=doc,
            # The daemon runs with its own working directory, so a relative
            # path has to be resolved on this side of the hop or it lands
            # somewhere else entirely.
            source_dir=Path(directory).expanduser().resolve(),
            source_ref=source_ref,
            source_kind=source_kind,
        )
        payload = _import_payload(result)
        emit(payload, human=_import_human(payload))


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
        removed = host.resources.delete(pot_id=pot_id, slug=doc)
        emit(
            {"doc": doc, "removed": removed, "graph_retracted": removed},
            human=(
                f"removed document '{doc}' (chunks and section claims)"
                if removed
                else f"no document '{doc}' stored in this pot"
            ),
        )


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
        "warnings": warnings,
        "recommended_next_action": _import_next_action(manifest.doc, pending),
    }


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
        "written": mutation.ok and mutation.status == "applied",
        "status": mutation.status,
        "entity_key": f"document:{result.manifest.doc}",
        "operations_applied": mutation.operations_applied,
        "claim_keys": list(mutation.claim_keys),
        "warnings": [],
    }
    if not payload["written"]:
        detail = mutation.detail or "; ".join(
            issue.message for issue in mutation.issues if issue.is_error
        )
        payload["warnings"] = [
            f"chunks are stored, but the graph write came back '{mutation.status}', "
            f"so search cannot find this document" + (f": {detail}" if detail else ".")
        ]
    return payload


def _import_next_action(doc: str, pending: Sequence[str]) -> str:
    """One next step, chosen by what is actually missing.

    Scope is the default because it is the failure nobody notices: a document
    with no ``DOCUMENTS`` edge is findable by semantic luck alone, and import
    has no way to guess what the document is *about*.
    """
    if pending:
        return f"Write a summary for: {', '.join(pending)}"
    return (
        f"Link document:{doc} to what it covers with a DOCUMENTS claim "
        f"(potpie graph mutate), or it is findable by search alone."
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
    lines = [
        f"imported {payload['doc']} revision {payload['revision']}",
        f"  sections: {payload['section_count']}" + (f" ({counts})" if counts else ""),
        f"  chunks: {payload['chunk_count']}",
        f"  graph: {graph['status']}"
        + (f" ({graph['entity_key']})" if graph["entity_key"] else ""),
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
