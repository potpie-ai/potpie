"""CLI formatting, logging, and human-readable output for context-engine."""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import traceback
from dataclasses import dataclass, field
from typing import Any

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.syntax import Syntax

from potpie_context_engine.adapters.inbound.cli.ui.brand import LOGO_COLOR, UI_MUTED_STYLE
from potpie_context_engine.adapters.inbound.cli.ui.format import (
    PANEL_BORDER,
    format_line,
    key_value_panel,
    print_line,
    print_structured_error,
    success_markup,
)

# stderr for errors so stdout stays clean for piping JSON
_err = Console(stderr=True)
_out = Console()
_json_errors = False


def configure_error_output(*, as_json: bool) -> None:
    """Switch stderr errors between human Rich output and JSON lines."""
    global _json_errors
    _json_errors = as_json


def configure_cli_logging(verbose: bool) -> None:
    """Reduce driver and HTTP client noise unless verbose or debug env is enabled."""
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
    # httpx logs every request at INFO; keep CLI auth/read output clean.
    http_level = logging.DEBUG if noisy else logging.WARNING
    for name in ("httpx", "httpcore"):
        logging.getLogger(name).setLevel(http_level)
    llm_level = logging.INFO if verbose else logging.WARNING
    for name in ("urllib3", "openai", "LiteLLM", "litellm"):
        logging.getLogger(name).setLevel(llm_level)


@dataclass
class DoctorSnapshot:
    """Structured data for `doctor` command."""

    context_graph_enabled: bool
    neo4j_effective_set: bool
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


def _error_code(title: str) -> str:
    """Stable machine code from a human title ('GitHub login failed' -> 'github_login_failed')."""
    slug = re.sub(r"[^a-z0-9]+", "_", title.strip().lower()).strip("_")
    return slug or "error"


def emit_error(
    title: str,
    message: str,
    *,
    hint: str | None = None,
    code: str | None = None,
    next_action: str | None = None,
    verbose: bool = False,
    exc: BaseException | None = None,
) -> None:
    """Emit a structured error.

    JSON mode mirrors the host-routed error contract (``commands/_common.fail``):
    ``{code, message, detail, recommended_next_action}`` — one schema across the
    whole CLI. Human mode keeps the rich title/message/hint rendering used by the
    interactive auth flows.
    """
    if _json_errors:
        payload: dict[str, Any] = {
            "code": code or _error_code(title),
            "message": message,
            "detail": hint,
            "recommended_next_action": next_action,
        }
        if verbose and exc is not None:
            payload["traceback"] = "".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__)
            )
        print(json.dumps(payload), file=sys.stderr)
        return
    print_structured_error(
        title=title,
        message=message,
        hint=hint,
        next_action=next_action,
        console=_err,
    )
    if verbose and exc is not None:
        tb_text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        _err.print(Syntax(tb_text, "python", theme="ansi_dark"))


def print_doctor_report(data: DoctorSnapshot, *, as_json: bool) -> None:
    if as_json:
        payload = {
            "context_graph_enabled": data.context_graph_enabled,
            "neo4j_effective_set": data.neo4j_effective_set,
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

    ctx_rows: list[tuple[str, str]] = [
        ("CONTEXT_GRAPH_ENABLED", str(data.context_graph_enabled)),
        (
            "Neo4j URI (local)",
            "set" if data.neo4j_effective_set else "(missing)",
        ),
    ]
    if data.potpie_health_ok is not None:
        ctx_rows.append(
            (
                "GET /health",
                "ok"
                if data.potpie_health_ok
                else (data.potpie_health_message or "failed"),
            )
        )
    if data.potpie_auth_ok is not None:
        ctx_rows.append(
            (
                "GET /api/v2/context/pots",
                "ok" if data.potpie_auth_ok else (data.potpie_auth_message or "failed"),
            )
        )
    _out.print(
        key_value_panel(
            "Context graph (CLI uses Potpie /api/v2/context; Neo4j local optional)",
            ctx_rows,
        )
    )

    pot_rows: list[tuple[str, str]] = [
        (
            "Repo maps (CONTEXT_ENGINE_REPO_TO_POT / CONTEXT_ENGINE_POTS)",
            "set" if data.pot_maps_set else "(unset)",
        ),
        ("Active pot (`pot use`)", data.active_pot_id or "(unset)"),
        ("POTPIE_API_KEY in env", str(data.potpie_api_key_env)),
        ("Stored token (login)", str(data.potpie_stored_token)),
        ("Potpie API URL", data.potpie_base_url or "(unset)"),
    ]
    _out.print(key_value_panel("Pot scope (CLI)", pot_rows))

    other_rows: list[tuple[str, str]] = [
        ("Postgres URL env (local / other tools)", str(data.database_url_set)),
        (
            "CONTEXT_ENGINE_GITHUB_TOKEN",
            str(data.github_token_set),
        ),
    ]
    _out.print(
        key_value_panel(
            "Other (sync / ledger)", other_rows, border_style=UI_MUTED_STYLE
        )
    )

    for line in data.summary_lines:
        _out.print(format_line(line))


def print_unified_status_report(
    data: DoctorSnapshot,
    *,
    as_json: bool,
    quick: bool,
    pot_id: str | None = None,
    pot_status: dict[str, Any] | None = None,
    pot_status_error: str | None = None,
) -> None:
    """CLI config/connectivity (doctor) plus optional ``POST /status`` for a pot."""
    if as_json:
        payload: dict[str, Any] = {
            "doctor": {
                "context_graph_enabled": data.context_graph_enabled,
                "neo4j_effective_set": data.neo4j_effective_set,
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
            },
            "quick": quick,
            "pot_id": pot_id,
            "pot_status": pot_status,
            "pot_status_error": pot_status_error,
        }
        print(json.dumps(payload))
        return

    print_doctor_report(data, as_json=False)
    if quick:
        _out.print(
            "[dim]Pot readiness skipped (--quick). Run `potpie status` without --quick.[/dim]"
        )
        return
    if pot_status is not None:
        _out.print()
        _out.print(
            Panel(
                Syntax(json.dumps(pot_status, indent=2), "json", theme="ansi_dark"),
                title=f"Pot readiness (POST /status){f' · {pot_id}' if pot_id else ''}",
                border_style=PANEL_BORDER,
            )
        )
    elif pot_status_error:
        _out.print()
        _out.print(
            f"[yellow]![/yellow] Pot readiness: [{UI_MUTED_STYLE}]{escape(pot_status_error)}[/{UI_MUTED_STYLE}]"
        )


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
    """Human line: ``source: … • ref: YYYY-MM-DD • mutation: shortuuid``."""
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
    mid = row.get("mutation_id")
    if mid:
        s = str(mid)
        short = s[:8] if len(s) >= 8 else s
        parts.append(f"mutation: {escape(short)}")
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
            f"[{UI_MUTED_STYLE}]No results (empty graph or no matches for this query / pot).[/{UI_MUTED_STYLE}]"
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
                title=f"[{LOGO_COLOR}]{i}.[/{LOGO_COLOR}] {escape(name)}",
                border_style=PANEL_BORDER,
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
            f"[yellow]![/yellow] Ingest rejected (reconciliation). "
            f"[{UI_MUTED_STYLE}]event_id={escape(short)}[/{UI_MUTED_STYLE}]"
        )
        for row in out.get("errors") or []:
            if isinstance(row, dict):
                ent = str(row.get("entity") or "")
                issue = str(row.get("issue") or "")
                _err.print(
                    f"  [{UI_MUTED_STYLE}]{escape(ent):<18}[/{UI_MUTED_STYLE}] {escape(issue)}"
                )
            else:
                _err.print(f"  [{UI_MUTED_STYLE}]{escape(str(row))}[/{UI_MUTED_STYLE}]")
        _err.print(
            f"[{UI_MUTED_STYLE}]Hint: widen ontology (see docs/context-graph/graph.md) "
            f"or rephrase the input.[/{UI_MUTED_STYLE}]"
        )
        return
    if status == "queued":
        event_id = out.get("event_id")
        _out.print(
            success_markup(
                f"Event queued (async). event_id={event_id} job_id={out.get('job_id')}"
            )
        )
        if event_id:
            _out.print(
                f"[{UI_MUTED_STYLE}]Next: potpie event wait {escape(str(event_id))}  "
                f"or  potpie event show {escape(str(event_id))}[/{UI_MUTED_STYLE}]"
            )
        return
    mid = out.get("mutation_id")
    ev = out.get("event_id")
    if mid:
        _out.print(success_markup(f"Mutations applied. mutation_id={mid}"))
    elif ev:
        _out.print(success_markup(f"Mutations applied. event_id={ev}"))
    else:
        _out.print(success_markup("Mutations applied."))
    dgs = out.get("downgrades") or []
    if isinstance(dgs, list) and dgs:
        _out.print(
            f"[{UI_MUTED_STYLE}]{len(dgs)} downgrades applied "
            f"(ontology soft-fail; see API downgrades / QualityIssue feed).[/{UI_MUTED_STYLE}]"
        )


def print_json_blob(data: dict[str, Any], *, as_json: bool) -> None:
    """Generic structured output for add / pot list / pot create."""
    if as_json:
        print(json.dumps(data))
        return
    _out.print(Syntax(json.dumps(data, indent=2), "json", theme="ansi_dark"))


def print_plain_line(
    message: str,
    *,
    as_json: bool,
    json_payload: dict[str, Any] | None = None,
    markup: bool = True,
    tone: str | None = None,
) -> None:
    if as_json and json_payload is not None:
        print(json.dumps(json_payload))
        return
    if not markup:
        _out.print(message, markup=False)
        return
    print_line(message, as_json=as_json, tone=tone, markup=True, console=_out)
