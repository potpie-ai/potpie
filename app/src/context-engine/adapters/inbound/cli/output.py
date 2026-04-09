"""CLI formatting, logging, and human-readable output for context-engine."""

from __future__ import annotations

import json
import logging
import os
import traceback
from dataclasses import dataclass, field
from typing import Any, Optional

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

# stderr for errors so stdout stays clean for piping JSON
_err = Console(stderr=True)
_out = Console()


def configure_cli_logging(verbose: bool) -> None:
    """Reduce Neo4j driver noise unless verbose or CONTEXT_ENGINE_VERBOSE_NEO4J=1."""
    env_verbose = os.getenv("CONTEXT_ENGINE_VERBOSE_NEO4J", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    noisy = verbose or env_verbose
    level = logging.DEBUG if noisy else logging.ERROR
    for name in ("neo4j", "neo4j.io", "neo4j.notifications"):
        logging.getLogger(name).setLevel(level)


@dataclass
class DoctorSnapshot:
    """Structured data for `doctor` command."""

    context_graph_enabled: bool
    neo4j_effective_set: bool
    neo4j_source: str  # "context_engine" | "legacy" | "missing"
    pot_maps_set: bool
    active_pot_id: Optional[str]
    potpie_api_key_env: bool
    potpie_stored_token: bool
    potpie_base_url: Optional[str]
    potpie_port_hint: Optional[str]
    database_url_set: bool
    github_token_set: bool
    potpie_health_ok: Optional[bool] = None
    potpie_health_message: Optional[str] = None
    summary_lines: list[str] = field(default_factory=list)


def emit_error(
    title: str,
    message: str,
    *,
    hint: Optional[str] = None,
    verbose: bool = False,
    exc: Optional[BaseException] = None,
) -> None:
    _err.print(f"[bold red]{title}[/bold red]")
    _err.print(f"[red]{message}[/red]")
    if hint:
        _err.print(f"[dim]{hint}[/dim]")
    if verbose and exc is not None:
        tb_text = "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        )
        _err.print(Syntax(tb_text, "python", theme="ansi_dark"))


def print_doctor_report(data: DoctorSnapshot, *, as_json: bool) -> None:
    if as_json:
        payload = {
            "context_graph_enabled": data.context_graph_enabled,
            "neo4j_effective_set": data.neo4j_effective_set,
            "neo4j_source": data.neo4j_source,
            "pot_maps_set": data.pot_maps_set,
            "active_pot_id": data.active_pot_id,
            "potpie_api_key_env": data.potpie_api_key_env,
            "potpie_stored_token": data.potpie_stored_token,
            "potpie_base_url": data.potpie_base_url,
            "potpie_port_hint": data.potpie_port_hint,
            "database_url_set": data.database_url_set,
            "github_token_set": data.github_token_set,
            "potpie_health_ok": data.potpie_health_ok,
            "potpie_health_message": data.potpie_health_message,
            "summary": data.summary_lines,
        }
        print(json.dumps(payload))
        return

    ctx_tbl = Table(show_header=False, box=None, padding=(0, 2))
    ctx_tbl.add_row("CONTEXT_GRAPH_ENABLED", str(data.context_graph_enabled))
    ctx_tbl.add_row("Neo4j URI (local)", "set" if data.neo4j_effective_set else "(missing)")
    ctx_tbl.add_row("Neo4j source", data.neo4j_source)
    if data.potpie_health_ok is not None:
        ctx_tbl.add_row(
            "GET /health",
            "ok" if data.potpie_health_ok else (data.potpie_health_message or "failed"),
        )
    _out.print(
        Panel(
            ctx_tbl,
            title="Context graph (CLI uses Potpie /api/v2/context; Neo4j local optional)",
            border_style="cyan",
        )
    )

    pot_tbl = Table(show_header=False, box=None, padding=(0, 2))
    pot_tbl.add_row(
        "Repo maps (CONTEXT_ENGINE_REPO_TO_POT / CONTEXT_ENGINE_POTS)",
        "set" if data.pot_maps_set else "(unset)",
    )
    pot_tbl.add_row("Active pot (`pot use`)", data.active_pot_id or "(unset)")
    pot_tbl.add_row("POTPIE_API_KEY in env", str(data.potpie_api_key_env))
    pot_tbl.add_row("Stored token (login)", str(data.potpie_stored_token))
    pot_tbl.add_row(
        "Potpie base URL",
        data.potpie_base_url or "(unset)",
    )
    if data.potpie_port_hint:
        pot_tbl.add_row("POTPIE_PORT hint", data.potpie_port_hint)
    _out.print(Panel(pot_tbl, title="Pot scope (CLI)", border_style="cyan"))

    other_tbl = Table(show_header=False, box=None, padding=(0, 2))
    other_tbl.add_row("Postgres URL env (local / other tools)", str(data.database_url_set))
    other_tbl.add_row("GITHUB_TOKEN / CONTEXT_ENGINE_GITHUB_TOKEN", str(data.github_token_set))
    _out.print(Panel(other_tbl, title="Other (sync / ledger)", border_style="dim"))

    for line in data.summary_lines:
        _out.print(line)


def _short_temporal_cell(value: Any) -> str:
    if value is None or value == "":
        return "—"
    s = str(value)
    if len(s) > 22:
        return s[:19] + "…"
    return s


def print_search_results(
    rows: list[dict[str, Any]], *, as_json: bool, with_temporal: bool = False
) -> None:
    if as_json:
        print(json.dumps(rows))
        return
    if not rows:
        _out.print("[dim]No results (empty graph or no matches for this query / pot).[/dim]")
        return
    table = Table(title="Search results", show_lines=False)
    table.add_column("#", style="dim", justify="right")
    table.add_column("name", max_width=36)
    table.add_column("summary", max_width=48)
    table.add_column("uuid", style="dim", max_width=38)
    if with_temporal:
        table.add_column("valid_at", style="dim", max_width=24)
        table.add_column("invalid_at", style="dim", max_width=24)
        table.add_column("created_at", style="dim", max_width=24)
    for i, row in enumerate(rows, start=1):
        name = str(row.get("name") or "")
        summary = str(row.get("summary") or row.get("fact") or "")
        if len(summary) > 200:
            summary = summary[:197] + "..."
        uid = str(row.get("uuid") or "")
        if with_temporal:
            table.add_row(
                str(i),
                name,
                summary,
                uid,
                _short_temporal_cell(row.get("valid_at")),
                _short_temporal_cell(row.get("invalid_at")),
                _short_temporal_cell(row.get("created_at")),
            )
        else:
            table.add_row(str(i), name, summary, uid)
    _out.print(table)


def print_ingest_result(out: dict[str, Any], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(out))
        return
    status = out.get("status")
    if status == "queued":
        _out.print(
            f"[green]Episode queued[/green] (async). event_id={out.get('event_id')} job_id={out.get('job_id')}"
        )
        return
    ep = out.get("episode_uuid")
    _out.print(f"[green]Episode ingested.[/green] episode_uuid={ep}")


def print_json_blob(data: dict[str, Any], *, as_json: bool) -> None:
    """Generic structured output for add / pot list / pot create."""
    if as_json:
        print(json.dumps(data))
        return
    _out.print(Syntax(json.dumps(data, indent=2), "json", theme="ansi_dark"))


def print_plain_line(message: str, *, as_json: bool, json_payload: Optional[dict[str, Any]] = None) -> None:
    if as_json and json_payload is not None:
        print(json.dumps(json_payload))
    else:
        _out.print(message)
