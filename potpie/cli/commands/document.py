"""Document ingestion commands → the runtime's engine resources service."""

from __future__ import annotations

import json

import typer

from potpie.cli.commands._common import (
    contract,
    emit,
    fail,
    get_root_runtime,
    get_runtime,
    resolve_pot_id,
)
from potpie_context_engine.application.services.resource_service import ResourceService

document_app = typer.Typer(
    help="Ingest and fetch document chunks (PDF/MD/TXT/images).",
    no_args_is_help=True,
)


def _resource_service() -> ResourceService:
    runtime = get_runtime()
    resources = getattr(getattr(runtime, "engine", None), "resources", None)
    if resources is None:
        fail("document service is not available on this host")
    return resources


@document_app.command("ingest")
def document_ingest(
    path: str = typer.Argument(..., help="Source file (.md, .txt, .pdf, images)"),
    doc: str = typer.Option(..., "--doc", help="Document slug"),
    pot: str | None = typer.Option(None, "--pot"),
    source_ref: str | None = typer.Option(None, "--source-ref"),
    chunk_size: int = typer.Option(4000, "--chunk-size"),
    force: bool = typer.Option(False, "--force"),
    vision_provider: str = typer.Option(
        "local",
        "--vision-provider",
        help="Vision backend for images: local (Ollama) or openai",
    ),
    allow_degraded: bool = typer.Option(
        False,
        "--allow-degraded",
        help="Allow text-only PDF ingest without Docling provenance",
    ),
) -> None:
    with contract():
        pot_id = resolve_pot_id(get_root_runtime(), pot)
        report = _resource_service().ingest_file(
            pot_id=pot_id,
            doc_slug=doc,
            source_path=path,
            chunk_size=chunk_size,
            force=force,
            source_ref=source_ref,
            vision_provider=vision_provider,
            allow_degraded=allow_degraded,
        )
        emit(report.model_dump(), human=_format_import_report(report))


@document_app.command("show")
def document_show(
    uri: str = typer.Argument(..., help="Chunk URI potpie://res/<doc>/<section>/<seq>"),
    pot: str | None = typer.Option(None, "--pot"),
    with_neighbors: bool = typer.Option(False, "--with-neighbors"),
) -> None:
    with contract():
        pot_id = resolve_pot_id(get_root_runtime(), pot)
        payload = _resource_service().get_chunk(
            pot_id=pot_id,
            uri=uri,
            with_neighbors=with_neighbors,
        )
        emit(payload, human=_format_show(payload))


@document_app.command("list")
def document_list(
    pot: str | None = typer.Option(None, "--pot"),
    doc: str | None = typer.Option(None, "--doc", help="Filter by document slug"),
) -> None:
    with contract():
        pot_id = resolve_pot_id(get_root_runtime(), pot)
        docs = _resource_service().list_documents(pot_id=pot_id)
        if doc:
            docs = [row for row in docs if row.get("doc_slug") == doc]
        emit({"documents": docs}, human=_format_doc_list(docs))


@document_app.command("rm")
def document_rm(
    doc: str = typer.Argument(..., help="Document slug to remove"),
    pot: str | None = typer.Option(None, "--pot"),
    confirm: bool = typer.Option(False, "--confirm", help="Required to delete"),
) -> None:
    with contract():
        if not confirm:
            fail("pass --confirm to remove document chunks and registry rows")
        pot_id = resolve_pot_id(get_root_runtime(), pot)
        result = _resource_service().remove_document(pot_id=pot_id, doc_slug=doc)
        emit(result, human=f"removed document {doc}")


def _format_import_report(report) -> str:
    lines = [
        f"doc={report.doc_slug} graph_written={report.graph_written}",
        f"parser_tier={report.parser_tier} provenance_version={report.provenance_version}",
        f"added={report.sections_added} kept={report.sections_kept} "
        f"changed={report.sections_changed} removed={report.sections_removed}",
    ]
    if report.elements_added or report.elements_removed:
        lines.append(
            f"elements_added={report.elements_added} elements_removed={report.elements_removed}"
        )
    if report.summary_pending:
        lines.append(f"summary_pending: {', '.join(report.summary_pending)}")
    if report.recommended_next_action:
        lines.append(f"next: {report.recommended_next_action}")
    if report.errors:
        lines.append("errors:")
        for err in report.errors:
            lines.append(f"  {err.get('code')}: {err.get('message')}")
    return "\n".join(lines)


def _format_show(payload: dict) -> str:
    lines = [payload.get("content", "")]
    provenance = payload.get("provenance")
    if provenance:
        lines.append("")
        lines.append("provenance:")
        lines.append(json.dumps(provenance, indent=2))
    return "\n".join(lines)


def _format_doc_list(docs: list[dict]) -> str:
    if not docs:
        return "no ingested documents"
    return "\n".join(
        f"  {row.get('doc_slug')} ({row.get('section_count')} sections, rev {row.get('revision')})"
        for row in docs
    )


def register(app: typer.Typer) -> None:
    app.add_typer(document_app, name="document")


__all__ = ["document_app", "register"]
