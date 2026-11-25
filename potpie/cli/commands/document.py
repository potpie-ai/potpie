"""Document ingestion commands → the runtime's engine resources service."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from contextlib import contextmanager

import typer

from potpie.auth.adapters.credentials_store import (
    load_secure_secret,
    store_secure_secret,
)
from potpie.cli.commands._common import (
    contract,
    emit,
    fail,
    get_root_runtime,
    get_runtime,
    is_json,
    resolve_pot_id,
)
from potpie.config.contracts import ConfigService
from potpie.runtime.composition import build_local_config_service
from potpie_context_engine.adapters.outbound.resources import capabilities
from potpie_context_engine.adapters.outbound.resources.enrichers import (
    vision_httpx_ollama,
)
from potpie_context_engine.application.services.resource_service import ResourceService

document_app = typer.Typer(
    help="Ingest and fetch document chunks (PDF/MD/TXT/images).",
    no_args_is_help=True,
)

_FACTS_KEY_NAME = "document_facts_llm_api_key"
_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
_GEMINI_MODEL = "gemini-3.5-flash-lite"
_VISION_MODEL = "moondream"


@contextmanager
def _document_environment(
    config: ConfigService, *, vision_provider: str, extract_facts: bool
) -> Iterator[None]:
    """Apply saved settings only for this ingest; explicit environment wins."""
    desired: dict[str, str | None] = {}
    if vision_provider in {"local", "ollama"}:
        desired["POTPIE_VISION_MODEL"] = config.get("document.vision_model")
    if extract_facts:
        desired.update(
            {
                "POTPIE_FACTS_LLM_PROVIDER": config.get("document.facts_provider"),
                "POTPIE_FACTS_LLM_BASE_URL": config.get("document.facts_base_url"),
                "POTPIE_FACTS_LLM_MODEL": config.get("document.facts_model"),
                "POTPIE_FACTS_LLM_API_KEY": load_secure_secret(_FACTS_KEY_NAME),
            }
        )
    added = [
        name for name, value in desired.items() if value and name not in os.environ
    ]
    try:
        for name in added:
            os.environ[name] = str(desired[name])
        yield
    finally:
        for name in added:
            os.environ.pop(name, None)


@document_app.command("setup")
def document_setup() -> None:
    """Configure local PDF vision and Gemini facts once for later ingests."""
    with contract():
        if is_json():
            fail(
                code="interactive_required",
                message="document setup needs an interactive terminal",
                next_action="run 'potpie document setup' in your terminal",
            )
        caps = capabilities.documents_capabilities()
        missing = [
            name for name in ("docling", "rapidocr", "httpx") if not caps.get(name)
        ]
        if missing:
            fail(
                code="documents_extra_required",
                message=f"PDF setup needs {', '.join(missing)}",
                next_action="install potpie[documents] or run 'uv sync --extra documents'",
            )

        use_vision = typer.confirm(
            "Caption PDF images with a local Ollama vision model by default?",
            default=True,
        )
        config = build_local_config_service()
        vision_model = (
            typer.prompt(
                "Ollama vision model",
                default=config.get("document.vision_model") or _VISION_MODEL,
            ).strip()
            if use_vision
            else _VISION_MODEL
        )
        if use_vision and not vision_httpx_ollama.ollama_reachable():
            fail(
                code="ollama_unavailable",
                message="Ollama is not reachable at the configured OLLAMA_HOST",
                next_action="start Ollama, then rerun 'potpie document setup'",
            )
        if use_vision and not vision_httpx_ollama.ollama_model_available(vision_model):
            fail(
                code="vision_model_missing",
                message=f"{vision_model} is not installed in Ollama",
                next_action=f"run 'ollama pull {vision_model}', then rerun 'potpie document setup'",
            )
        if use_vision and not vision_httpx_ollama.ollama_model_supports_vision(
            vision_model
        ):
            fail(
                code="vision_model_not_capable",
                message=f"{vision_model} does not report Ollama vision capability",
                next_action="choose an installed Ollama vision model and rerun setup",
            )

        use_facts = typer.confirm(
            "Extract document facts with your Gemini API key by default?",
            default=True,
        )
        if use_facts:
            existing_key = load_secure_secret(_FACTS_KEY_NAME)
            key = typer.prompt(
                "Gemini API key"
                + (" (Enter to keep saved key)" if existing_key else ""),
                default="" if existing_key else None,
                show_default=False,
                hide_input=True,
            ).strip()
            if key:
                store_secure_secret(_FACTS_KEY_NAME, key, label="Gemini API key")
            elif not existing_key:
                fail(
                    code="gemini_key_required",
                    message="Gemini API key is required for fact extraction",
                )

        config.set("document.vision_provider", "local" if use_vision else "none")
        config.set("document.vision_model", vision_model)
        config.set("document.extract_facts", "on" if use_facts else "off")
        if use_facts:
            config.set("document.facts_provider", "openai")
            config.set("document.facts_model", _GEMINI_MODEL)
            config.set("document.facts_base_url", _GEMINI_BASE_URL)
        emit(
            {
                "vision_provider": "local" if use_vision else "none",
                "vision_model": vision_model if use_vision else None,
                "extract_facts": use_facts,
                "facts_model": _GEMINI_MODEL if use_facts else None,
                "gemini_key_saved": use_facts,
            },
            human="Document setup saved. Future 'potpie document ingest' commands use these defaults.",
        )


def _resource_service() -> ResourceService:
    runtime = get_runtime()
    resources = getattr(getattr(runtime, "engine", None), "resources", None)
    if resources is None:
        fail(
            code="resource_service_unavailable",
            message="document service is not available on this host",
            next_action="run the command with the context engine available",
            exit_code=2,
        )
    if isinstance(resources, ResourceService):
        return resources
    # Parsing remains local; the canonical facade owns revisioned storage,
    # indexing, and graph mutations for the resulting chunk directory.
    return ResourceService(facade=resources)


@document_app.command("ingest")
def document_ingest(
    path: str = typer.Argument(..., help="Source file (.md, .txt, .pdf, images)"),
    doc: str = typer.Option(..., "--doc", help="Document slug"),
    pot: str | None = typer.Option(None, "--pot"),
    source_ref: str | None = typer.Option(None, "--source-ref"),
    chunk_size: int = typer.Option(4000, "--chunk-size"),
    force: bool = typer.Option(False, "--force"),
    vision_provider: str | None = typer.Option(
        None,
        "--vision-provider",
        help="Vision captions: none, local (Ollama), or openai; uses saved setup default when omitted",
    ),
    allow_degraded: bool = typer.Option(
        False,
        "--allow-degraded",
        help="Allow text-only PDF ingest without Docling provenance",
    ),
    extract_facts: bool | None = typer.Option(
        None,
        "--extract-facts/--no-extract-facts",
        help="Use your configured API key or local Ollama endpoint to extract and link claims.",
    ),
) -> None:
    with contract():
        config = build_local_config_service()
        selected_vision = (
            vision_provider or config.get("document.vision_provider") or "none"
        )
        selected_facts = (
            config.get("document.extract_facts") == "on"
            if extract_facts is None
            else extract_facts
        )
        pot_id = resolve_pot_id(get_root_runtime(), pot)
        with _document_environment(
            config, vision_provider=selected_vision, extract_facts=selected_facts
        ):
            report = _resource_service().ingest_file(
                pot_id=pot_id,
                doc_slug=doc,
                source_path=path,
                chunk_size=chunk_size,
                force=force,
                source_ref=source_ref,
                vision_provider=selected_vision,
                allow_degraded=allow_degraded,
                extract_facts=selected_facts,
            )
        _emit_import_report(report)


@document_app.command("show")
def document_show(
    uri: str = typer.Argument(
        ..., help="Chunk URI potpie://res/<doc>/<section>/<seq>[@revN]"
    ),
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
    pictures = report.ingestion_diagnostics.get("picture_enrichment", {})
    if isinstance(pictures, dict) and pictures.get("pictures_total"):
        lines.append(
            f"figures={pictures['pictures_total']} "
            f"ocr={pictures.get('pictures_with_ocr', 0)} "
            f"vision_captions={pictures.get('pictures_with_visual_caption', 0)}"
        )
        if pictures.get("warnings"):
            lines.append(f"enrichment_warnings: {', '.join(pictures['warnings'])}")
    if report.facts_verified or report.facts_proposed or report.facts_suppressed:
        lines.append(
            f"facts_verified={report.facts_verified} facts_proposed={report.facts_proposed} "
            f"facts_suppressed={report.facts_suppressed}"
        )
    if report.fact_extraction_error:
        lines.append(f"fact_extraction_error: {report.fact_extraction_error}")
    if report.recommended_next_action:
        lines.append(f"next: {report.recommended_next_action}")
    if report.errors:
        lines.append("errors:")
        for err in report.errors:
            lines.append(f"  {err.get('code')}: {err.get('message')}")
    return "\n".join(lines)


def _emit_import_report(report) -> None:
    if report.errors or not report.graph_written:
        first = report.errors[0] if report.errors else {}
        code = first.get("code") or "resource_graph_write_failed"
        message = first.get("message") or "document ingest did not complete"
        fail(
            code=code,
            message=message,
            detail=report.model_dump(),
            next_action=report.recommended_next_action,
            exit_code=2 if code.startswith("resource_graph") else 1,
        )
    emit(report.model_dump(), human=_format_import_report(report))


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
