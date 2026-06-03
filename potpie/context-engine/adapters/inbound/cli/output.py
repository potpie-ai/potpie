"""CLI formatting, logging, and human-readable output for context-engine."""

from __future__ import annotations

import json
import logging
import os
import sys
import traceback
from dataclasses import dataclass, field
from typing import Any

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

# stderr for errors so stdout stays clean for piping JSON
_err = Console(stderr=True)
_out = Console()
_json_errors = False


def configure_error_output(*, as_json: bool) -> None:
    """Switch stderr errors between human Rich output and JSON lines."""
    global _json_errors
    _json_errors = as_json


def configure_cli_logging(verbose: bool) -> None:
    """Reduce graph driver noise unless verbose or graph-driver debug is enabled."""
    env_verbose = os.getenv(
        "CONTEXT_ENGINE_GRAPH_DRIVER_DEBUG", ""
    ).strip().lower() in (
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
    active_pot_id: str | None
    potpie_api_key_env: bool
    potpie_stored_token: bool
    potpie_base_url: str | None
    potpie_port_hint: str | None
    database_url_set: bool
    github_token_set: bool
    potpie_health_ok: bool | None = None
    potpie_health_message: str | None = None
    potpie_auth_ok: bool | None = None
    potpie_auth_message: str | None = None
    summary_lines: list[str] = field(default_factory=list)


def emit_error(
    title: str,
    message: str,
    *,
    hint: str | None = None,
    verbose: bool = False,
    exc: BaseException | None = None,
) -> None:
    if _json_errors:
        payload: dict[str, Any] = {
            "ok": False,
            "error": {
                "title": title,
                "message": message,
            },
        }
        if hint:
            payload["error"]["hint"] = hint
        if verbose and exc is not None:
            payload["error"]["traceback"] = "".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__)
            )
        print(json.dumps(payload), file=sys.stderr)
        return
    _err.print(f"[bold red]{title}[/bold red]")
    _err.print(f"[red]{message}[/red]")
    if hint:
        _err.print(f"[dim]{hint}[/dim]")
    if verbose and exc is not None:
        tb_text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
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
            "potpie_auth_ok": data.potpie_auth_ok,
            "potpie_auth_message": data.potpie_auth_message,
            "summary": data.summary_lines,
        }
        print(json.dumps(payload))
        return

    ctx_tbl = Table(show_header=False, box=None, padding=(0, 2))
    ctx_tbl.add_row("CONTEXT_GRAPH_ENABLED", str(data.context_graph_enabled))
    ctx_tbl.add_row(
        "Neo4j URI (local)", "set" if data.neo4j_effective_set else "(missing)"
    )
    ctx_tbl.add_row("Neo4j source", data.neo4j_source)
    if data.potpie_health_ok is not None:
        ctx_tbl.add_row(
            "GET /health",
            "ok" if data.potpie_health_ok else (data.potpie_health_message or "failed"),
        )
    if data.potpie_auth_ok is not None:
        ctx_tbl.add_row(
            "GET /api/v2/context/pots",
            "ok" if data.potpie_auth_ok else (data.potpie_auth_message or "failed"),
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
    other_tbl.add_row(
        "Postgres URL env (local / other tools)", str(data.database_url_set)
    )
    other_tbl.add_row(
        "GITHUB_TOKEN / CONTEXT_ENGINE_GITHUB_TOKEN", str(data.github_token_set)
    )
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


def _lifecycle_cli_tag(status: Any) -> str | None:
    """Short bracket tag for edge lifecycle (completed / unknown → no tag)."""
    if status is None or status == "":
        return None
    s = str(status).strip().lower()
    if s in ("completed", "unknown"):
        return None
    labels = {
        "planned": "[planned]",
        "proposed": "[planned]",
        "in_progress": "[active]",
        "deprecated": "[deprecated]",
        "decommissioned": "[decommissioned]",
    }
    return labels.get(s, f"[{s}]")


def _format_search_provenance_line(row: dict[str, Any]) -> str | None:
    """Human line: ``source: … • ref: YYYY-MM-DD • episode: shortuuid``."""
    parts: list[str] = []
    refs = row.get("source_refs")
    if isinstance(refs, list) and refs:
        labels = [str(x).strip() for x in refs if x is not None and str(x).strip()]
        if labels:
            parts.append("source: " + ", ".join(escape(lab) for lab in labels))
    rt = row.get("reference_time")
    if rt is not None and str(rt).strip():
        s = str(rt).strip()
        if "T" in s:
            date_part = s.split("T", 1)[0]
        else:
            date_part = s[:10] if len(s) >= 10 else s
        parts.append(f"ref: {escape(date_part)}")
    ep = row.get("episode_uuid")
    if ep:
        s = str(ep)
        short = s[:8] if len(s) >= 8 else s
        parts.append(f"episode: {escape(short)}")
    if not parts:
        return None
    return " • ".join(parts)


def _compact_valid_expired_line(row: dict[str, Any]) -> str | None:
    """One-line ``valid … • expired …`` when any temporal field is present."""
    va = row.get("valid_at")
    inv = row.get("invalid_at")
    if va is None and inv is None:
        return None
    return f"valid {_short_temporal_cell(va)} • expired {_short_temporal_cell(inv)}"


def print_search_results(
    rows: list[dict[str, Any]],
    *,
    as_json: bool,
    with_temporal: bool = False,
    show_provenance: bool = True,
) -> None:
    if as_json:
        print(json.dumps(rows))
        return
    if not rows:
        _out.print(
            "[dim]No results (empty graph or no matches for this query / pot).[/dim]"
        )
        return
    for i, row in enumerate(rows, start=1):
        name = str(row.get("name") or "(unnamed)")
        summary = str(row.get("summary") or row.get("fact") or "")
        cref = row.get("conflict_with_rows")
        if isinstance(cref, list) and cref:
            nums = ", ".join(str(int(x)) for x in cref if isinstance(x, int))
            if nums:
                summary = f"[!] conflict with row {nums} — {summary}".strip()
        tag = _lifecycle_cli_tag(row.get("lifecycle_status"))
        if tag:
            summary = f"{tag} {summary}".strip()
        if row.get("superseded_label"):
            summary = f"{row.get('superseded_label')} {summary}".strip()
        if len(summary) > 200:
            summary = summary[:197] + "..."
        uid = str(row.get("uuid") or "")
        lines = [escape(summary or "(no summary)")]
        if show_provenance:
            prov = _format_search_provenance_line(row)
            if prov:
                lines.append(f"[dim]{prov}[/dim]")
        lines.append(f"[dim]uuid:[/dim] {escape(uid)}")
        compact = _compact_valid_expired_line(row)
        if compact:
            lines.append(f"[dim]{escape(compact)}[/dim]")
        via = row.get("causal_via")
        if isinstance(via, dict):
            rel = via.get("relation") or "related"
            lines.append(f"[dim]↳ because:[/dim] {escape(str(rel))} (causal expansion)")
        if with_temporal:
            lines.append(
                "[dim]created_at:[/dim] "
                f"{escape(_short_temporal_cell(row.get('created_at')))}"
            )
        _out.print(
            Panel(
                "\n".join(lines),
                title=f"{i}. {escape(name)}",
                border_style="cyan",
            )
        )


def print_ingest_result(out: dict[str, Any], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(out))
        return
    status = out.get("status")
    if status == "reconciliation_rejected":
        ev = str(out.get("event_id") or "")
        short = ev[:8] if len(ev) >= 8 else ev
        _err.print(
            f"[yellow]Ingest rejected (reconciliation).[/yellow] event_id={short}"
        )
        for row in out.get("errors") or []:
            if isinstance(row, dict):
                ent = str(row.get("entity") or "")
                issue = str(row.get("issue") or "")
                _err.print(f"  {ent:<18} {issue}")
            else:
                _err.print(f"  {row}")
        _err.print(
            "[dim]Hint: widen ontology (see docs/context-graph/graph.md) "
            "or rephrase the episode.[/dim]"
        )
        return
    if status == "queued":
        event_id = out.get("event_id")
        _out.print(
            f"[green]Episode queued[/green] (async). event_id={event_id} job_id={out.get('job_id')}"
        )
        if event_id:
            _out.print(
                f"[dim]Next: potpie event wait {event_id}  or  potpie event show {event_id}[/dim]"
            )
        return
    ep = out.get("episode_uuid")
    ev = out.get("event_id")
    if ep:
        _out.print(f"[green]Episode ingested.[/green] episode_uuid={ep}")
    elif ev:
        _out.print(f"[green]Episode ingested.[/green] event_id={ev}")
    else:
        _out.print("[green]Episode ingested.[/green]")
    dgs = out.get("downgrades") or []
    if isinstance(dgs, list) and dgs:
        _out.print(
            f"[dim]{len(dgs)} downgrades applied "
            "(ontology soft-fail; see API downgrades / QualityIssue feed).[/dim]"
        )


def print_json_blob(data: dict[str, Any], *, as_json: bool) -> None:
    """Generic structured output for add / pot list / pot create."""
    if as_json:
        print(json.dumps(data))
        return
    _out.print(Syntax(json.dumps(data, indent=2), "json", theme="ansi_dark"))


def print_plain_line(
    message: str, *, as_json: bool, json_payload: dict[str, Any] | None = None
) -> None:
    if as_json and json_payload is not None:
        print(json.dumps(json_payload))
    else:
        _out.print(message)
