"""Graph + backend commands → ``HostShell.graph`` and the active ``GraphBackend``.

CLI code never touches a store directly; everything goes through the capability
ports. Unbuilt projections (semantic/inspection/analytics/snapshot on profiles
that lack them) surface as the structured not-implemented contract.
"""

from __future__ import annotations

import json
import re
import sys
import time
from contextlib import contextmanager
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from typing import Any

import typer

from bootstrap.observability_runtime import get_observability
from application.services.graph_workbench import (
    graph_error_envelope,
    graph_not_implemented_envelope,
    graph_success_envelope,
    new_graph_request_id,
    normalize_catalog_result,
    normalize_workbench_result,
)
from adapters.inbound.cli.commands._common import (
    EXIT_UNAVAILABLE,
    EXIT_VALIDATION,
    contract,
    emit,
    fail,
    get_host,
    is_json,
    json_error_formatter,
    resolve_pot_id,
)
from domain.errors import CapabilityNotImplemented
from domain.graph_contract import GRAPH_CONTRACT_VERSION as DATA_PLANE_CONTRACT_VERSION
from domain.graph_contract import ONTOLOGY_VERSION
from domain.graph_workbench import (
    GRAPH_WORKBENCH_COMMANDS,
    GraphUnsupported,
    GraphWorkbenchStatus,
)
from domain.ports.observability import SPAN_KIND_INTERNAL
from domain.graph_workbench_ontology import describe_contract
from domain.nudge import NUDGE_EVENT_HELP

graph_app = typer.Typer(help="Graph reads/admin via capability ports.")
inbox_app = typer.Typer(help="Pending graph-work inbox.")
quality_app = typer.Typer(help="Read-only graph quality reports.")
backend_app = typer.Typer(help="GraphBackend profile selection + health.")
timeline_app = typer.Typer(help="Timeline reads over the active project pot.")

graph_app.add_typer(inbox_app, name="inbox")
graph_app.add_typer(quality_app, name="quality")


class _GraphCliCommandContext:
    def __init__(self, command: str) -> None:
        self.command = command
        self.request_id = new_graph_request_id()
        self.pot_id: str | None = None
        self.subgraph_versions: dict[str, int] = {}
        self.telemetry_result = "ok"
        self.telemetry_error_code = "none"
        self.telemetry_attributes: dict[str, str] = {}

    def set_pot_id(self, pot_id: str | None) -> None:
        self.pot_id = pot_id

    def set_subgraph_versions(self, versions: Mapping[str, Any] | None) -> None:
        if not versions:
            return
        clean: dict[str, int] = {}
        for key, value in versions.items():
            try:
                clean[str(key)] = int(value)
            except (TypeError, ValueError):
                continue
        self.subgraph_versions = clean

    def format_error(self, payload: dict[str, Any]) -> dict[str, Any]:
        code = str(payload.get("code") or "error")
        self.mark_result(result=code, error_code=code)
        return graph_error_envelope(
            command=self.command,
            request_id=self.request_id,
            pot_id=self.pot_id,
            code=code,
            message=str(payload.get("message") or "Graph command failed."),
            detail=payload.get("detail"),
            subgraph_versions=self.subgraph_versions,
            recommended_next_action=payload.get("recommended_next_action"),
        ).to_dict()

    def mark_result(
        self,
        *,
        result: str,
        error_code: str = "none",
        attributes: Mapping[str, Any] | None = None,
    ) -> None:
        self.telemetry_result = result
        self.telemetry_error_code = error_code
        if attributes:
            for key, value in attributes.items():
                if value is None:
                    continue
                self.telemetry_attributes[str(key)] = str(value)


@contextmanager
def _graph_command(command: str):
    ctx = _GraphCliCommandContext(command)
    obs = get_observability()
    span_name, base_attrs = _graph_telemetry_shape(command)
    started_at = time.perf_counter()
    with obs.span(
        span_name,
        kind=SPAN_KIND_INTERNAL,
        attributes={
            **base_attrs,
            "command": command,
            "request_id": ctx.request_id,
        },
    ) as span:
        try:
            with json_error_formatter(ctx.format_error):
                with contract():
                    yield ctx
        except BaseException as exc:
            if ctx.telemetry_error_code == "none":
                if isinstance(exc, typer.Exit):
                    result = "ok" if (exc.exit_code in (None, 0)) else "exit"
                    ctx.mark_result(result=result, error_code="exit")
                else:
                    ctx.mark_result(
                        result="unexpected",
                        error_code=exc.__class__.__name__,
                    )
                    span.record_exception(exc)
                    span.set_error(repr(exc))
            raise
        finally:
            duration_ms = max((time.perf_counter() - started_at) * 1000.0, 0.0)
            attrs = {
                **base_attrs,
                **ctx.telemetry_attributes,
                "command": command,
                "request_id": ctx.request_id,
                "result": ctx.telemetry_result,
                "error_code": ctx.telemetry_error_code,
            }
            if ctx.pot_id:
                attrs["pot_id"] = ctx.pot_id
            span.set_attributes(attrs)
            if ctx.telemetry_error_code != "none":
                span.set_error(ctx.telemetry_error_code)
            _record_graph_command_telemetry(
                obs,
                command=command,
                duration_ms=duration_ms,
                attributes=attrs,
            )


def _graph_telemetry_shape(command: str) -> tuple[str, dict[str, str]]:
    raw = command.removeprefix("graph.").replace("-", "_")
    parts = raw.split(".")
    if not parts:
        return "graph.unknown", {}
    if parts[0] == "inbox":
        attrs = {"operation": parts[1] if len(parts) > 1 else "unknown"}
        return "graph.inbox", attrs
    if parts[0] == "quality":
        attrs = {"report": parts[1] if len(parts) > 1 else "summary"}
        return "graph.quality", attrs
    return f"graph.{parts[0]}", {}


def _record_graph_command_telemetry(
    obs,
    *,
    command: str,
    duration_ms: float,
    attributes: Mapping[str, str],
) -> None:
    _span_name, base_attrs = _graph_telemetry_shape(command)
    raw = command.removeprefix("graph.").replace("-", "_")
    root = raw.split(".", 1)[0] if raw else "unknown"
    metric_root = root
    metric_attrs = dict(base_attrs)
    metric_attrs.update(
        {
            key: value
            for key, value in attributes.items()
            if key
            in {
                "result",
                "error_code",
                "pot_id",
                "subgraph",
                "view",
                "risk",
                "status",
                "operation",
                "report",
                "backend_profile",
                "match_mode",
            }
        }
    )
    try:
        obs.counter(f"ce.graph.{metric_root}_total", 1, attributes=metric_attrs)
        obs.histogram(
            f"ce.graph.{metric_root}_ms",
            duration_ms,
            attributes=metric_attrs,
        )
    except Exception:  # noqa: BLE001 - observability must never fail a command
        pass


def _graph_telemetry_attributes(result: Mapping[str, Any]) -> dict[str, str]:
    attrs: dict[str, str] = {}
    for key in (
        "subgraph",
        "view",
        "risk",
        "status",
        "report",
        "action",
        "backend_profile",
        "match_mode",
    ):
        value = result.get(key)
        if value is not None:
            target = "operation" if key == "action" else key
            attrs[target] = str(value)
    backend = result.get("backend")
    if isinstance(backend, Mapping):
        if backend.get("profile") is not None:
            attrs["backend_profile"] = str(backend["profile"])
        if backend.get("ready") is not None:
            attrs["backend_ready"] = str(bool(backend["ready"])).lower()
    graph_service = result.get("graph_service")
    if (
        isinstance(graph_service, Mapping)
        and graph_service.get("match_mode") is not None
    ):
        attrs["match_mode"] = str(graph_service["match_mode"])
    return attrs


def _emit_graph_result(
    ctx: _GraphCliCommandContext,
    payload: Mapping[str, Any],
    *,
    human: str,
    warnings: tuple[str, ...] = (),
    unsupported: tuple[GraphUnsupported, ...] = (),
    recommended_next_action: str | None = None,
) -> None:
    result, versions, payload_warnings, payload_unsupported = (
        normalize_workbench_result(payload)
    )
    if versions:
        ctx.set_subgraph_versions(versions)
    merged_warnings = tuple(warnings) + payload_warnings
    merged_unsupported = tuple(unsupported) + payload_unsupported
    result_label = "ok"
    error_code = "none"
    if payload.get("ok", True) is False:
        result_label = str(payload.get("status") or _error_code_from_result(payload))
        error_code = _error_code_from_result(payload)
    ctx.mark_result(
        result=result_label,
        error_code=error_code,
        attributes=_graph_telemetry_attributes(result),
    )

    if payload.get("ok", True) is False:
        env = graph_error_envelope(
            command=ctx.command,
            request_id=ctx.request_id,
            pot_id=ctx.pot_id,
            code=_error_code_from_result(payload),
            message=_error_message_from_result(payload),
            detail=result or None,
            subgraph_versions=ctx.subgraph_versions,
            warnings=merged_warnings,
            unsupported=merged_unsupported,
            recommended_next_action=recommended_next_action
            or payload.get("recommended_next_action"),
        )
    else:
        env = graph_success_envelope(
            command=ctx.command,
            request_id=ctx.request_id,
            pot_id=ctx.pot_id,
            result=result,
            subgraph_versions=ctx.subgraph_versions,
            warnings=merged_warnings,
            unsupported=merged_unsupported,
            recommended_next_action=recommended_next_action
            or payload.get("recommended_next_action"),
    )
    emit(env.to_dict(), human=human)


def _emit_inbox_result(ctx: _GraphCliCommandContext, result) -> None:
    _emit_graph_result(ctx, result.to_dict(), human=_inbox_human(result))
    if not result.ok:
        raise typer.Exit(code=EXIT_VALIDATION)


def _emit_quality_result(ctx: _GraphCliCommandContext, result) -> None:
    _emit_graph_result(ctx, result.to_dict(), human=_quality_human(result))
    if not result.ok:
        raise typer.Exit(code=EXIT_VALIDATION)


def _emit_graph_not_implemented(
    ctx: _GraphCliCommandContext,
    *,
    detail: str | None = None,
    recommended_next_action: str | None = None,
) -> None:
    env = graph_not_implemented_envelope(
        command=ctx.command,
        request_id=ctx.request_id,
        pot_id=ctx.pot_id,
        detail=detail,
        recommended_next_action=recommended_next_action,
    )
    emit(env.to_dict(), human=detail or f"{ctx.command} is not implemented yet")
    raise typer.Exit(code=EXIT_UNAVAILABLE)


def _error_code_from_result(payload: Mapping[str, Any]) -> str:
    issues = payload.get("issues")
    if isinstance(issues, list) and issues:
        first = issues[0]
        if isinstance(first, Mapping) and first.get("code"):
            return str(first["code"])
    return str(payload.get("status") or "graph_command_failed")


def _error_message_from_result(payload: Mapping[str, Any]) -> str:
    if payload.get("detail"):
        return str(payload["detail"])
    issues = payload.get("issues")
    if isinstance(issues, list) and issues:
        first = issues[0]
        if isinstance(first, Mapping) and first.get("message"):
            return str(first["message"])
    status = payload.get("status")
    return f"Graph command failed with status {status!r}."


def _legacy_warning(command: str, replacement: str) -> tuple[str, ...]:
    return (
        f"{command} is a legacy transition command and is not part of the canonical V2 workbench command set; use {replacement}.",
    )


# --- Graph Surface Lite (V1.5) ----------------------------------------------


@graph_app.command("catalog")
def graph_catalog(
    task: str = typer.Option(None, "--task", help="(accepted, ignored in V1.5)"),
    subgraph: str = typer.Option(None, "--subgraph"),
    pot: str = typer.Option(None, "--pot"),
) -> None:
    """Discover the graph contract: versions, views, mutation ops, ontology."""
    from domain.ports.services.graph_service import GraphCatalogRequest

    with _graph_command("graph.catalog") as ctx:
        host = get_host()
        pot_id = resolve_pot_id(host, pot)
        ctx.set_pot_id(pot_id)
        result = host.graph.catalog(
            GraphCatalogRequest(pot_id=pot_id, task=task, subgraph=subgraph)
        )
        payload = normalize_catalog_result(result.to_dict(), task=task)
        human = (
            f"graph contract v2 / ontology {ONTOLOGY_VERSION} "
            f"(data-plane={payload['data_plane_graph_contract_version']}, match={payload['match_mode']})\n"
            f"commands: {', '.join(payload['commands'])}\n"
            f"views: {', '.join(v['name'] for v in payload['views'])}\n"
            f"mutation ops: {', '.join(payload['mutation_operations'])}\n"
            f"review-required: {', '.join(payload['review_required_operations'])}\n"
            f"deferred: {', '.join(payload['deferred_operations'])}"
        )
        _emit_graph_result(ctx, payload, human=human)


@graph_app.command("read")
def graph_read(
    subgraph: str = typer.Option(
        None, "--subgraph", help="Canonical graph subgraph, e.g. debugging"
    ),
    view: str = typer.Option(
        None, "--view", help="View name within --subgraph, e.g. prior_occurrences"
    ),
    query: str = typer.Option(None, "--query"),
    scope: str = typer.Option(None, "--scope", help="key:value[,key:value]"),
    current: bool = typer.Option(
        False,
        "--current",
        help="Resolve the pot from the current repo; does not add repo scope.",
    ),
    repo: str = typer.Option(
        None,
        "--repo",
        help="Optional repo scope (owner/repo, URL, or 'current'). Omit for project-wide timeline.",
    ),
    since: str = typer.Option(None, "--since", help="ISO instant lower bound."),
    until: str = typer.Option(None, "--until", help="ISO instant upper bound."),
    time_window: str = typer.Option(
        None,
        "--time-window",
        "--window",
        help="Relative lookback such as 24h, 7d, 2w. Ignored when --since is set.",
    ),
    environment: str = typer.Option(None, "--environment"),
    depth: int = typer.Option(
        None, "--depth", help="traversal depth (neighborhood views)"
    ),
    direction: str = typer.Option(None, "--direction", help="out | in | both"),
    limit: int = typer.Option(12, "--limit"),
    sort: str = typer.Option(
        "auto",
        "--sort",
        help="auto | score | occurred_at (timeline defaults to occurred_at)",
    ),
    dedupe: str = typer.Option(
        "auto", "--dedupe", help="auto | none | source_ref | activity"
    ),
    format_: str = typer.Option(
        "auto",
        "--format",
        help="auto | raw | events | table | jsonl (timeline defaults to events)",
    ),
    pot: str = typer.Option(None, "--pot"),
) -> None:
    """V2-style read over a named view (routes through the read trunk)."""
    from domain.ports.services.graph_service import GraphReadRequest

    with _graph_command("graph.read") as ctx:
        if not subgraph:
            raise ValueError("--subgraph is required")
        if not view:
            raise ValueError("--view is required")
        if "." in view:
            raise ValueError(
                "graph read now requires --subgraph <name> --view <view>; "
                f"got fully-qualified view {view!r}"
            )
        host = get_host()
        pot_id = resolve_pot_id(host, pot)
        ctx.set_pot_id(pot_id)
        del current  # pot resolution already considers the current working tree.
        since_dt, until_dt = _resolve_time_bounds(
            since=since, until=until, window=time_window
        )
        parsed_scope = _parse_scope(scope)
        if repo:
            parsed_scope["repo"] = _resolve_repo_scope(repo)
        effective_format = _effective_requested_format(
            subgraph=subgraph, view=view, requested=format_
        )
        read_limit = _service_limit_for_read(
            subgraph=subgraph,
            view=view,
            format_=effective_format,
            requested_limit=limit,
        )
        result = host.graph.read(
            GraphReadRequest(
                pot_id=pot_id,
                subgraph=subgraph,
                view=view,
                query=query,
                scope=parsed_scope,
                environment=environment,
                since=since_dt,
                until=until_dt,
                depth=depth,
                direction=direction,
                limit=read_limit,
                freshness_preference="fresh"
                if _is_timeline_view(f"{subgraph}.{view}") and not query
                else "balanced",
            )
        )
        _emit_graph_read(
            ctx, result, format_=format_, sort=sort, dedupe=dedupe, event_limit=limit
        )


@timeline_app.command("recent")
def timeline_recent(
    query: str = typer.Option(None, "--query"),
    since: str = typer.Option(None, "--since", help="ISO instant lower bound."),
    until: str = typer.Option(None, "--until", help="ISO instant upper bound."),
    time_window: str = typer.Option(
        None,
        "--time-window",
        "--window",
        help="Relative lookback such as 24h, 7d, 2w. Ignored when --since is set.",
    ),
    service: str = typer.Option(
        None,
        "--service",
        help="Optional service scope. Omit for project-wide timeline across repos.",
    ),
    limit: int = typer.Option(12, "--limit"),
    format_: str = typer.Option(
        "auto", "--format", help="auto | events | table | raw | jsonl"
    ),
    pot: str = typer.Option(None, "--pot"),
) -> None:
    """Recent project events from the active/current pot, across all repo sources."""
    from domain.ports.services.graph_service import GraphReadRequest

    with contract():
        host = get_host()
        pot_id = resolve_pot_id(host, pot)
        since_dt, until_dt = _resolve_time_bounds(
            since=since, until=until, window=time_window
        )
        scope = {"service": service} if service else {}
        read_limit = _service_limit_for_read(
            subgraph="recent_changes",
            view="timeline",
            format_="events",
            requested_limit=limit,
        )
        result = host.graph.read(
            GraphReadRequest(
                pot_id=pot_id,
                subgraph="recent_changes",
                view="timeline",
                query=query,
                scope=scope,
                since=since_dt,
                until=until_dt,
                limit=read_limit,
                freshness_preference="fresh" if not query else "balanced",
            )
        )
        _emit_read(
            result,
            format_=format_,
            sort="occurred_at",
            dedupe="source_ref",
            event_limit=limit,
        )


@graph_app.command("search-entities")
def graph_search_entities(
    query_arg: str = typer.Argument(None, help="text to match entities/claims against"),
    query: str = typer.Option(None, "--query", help="text to match entities/claims against"),
    type_: str = typer.Option(None, "--type", help="entity label filter, e.g. Service"),
    predicate: str = typer.Option(None, "--predicate"),
    subgraph: str = typer.Option(None, "--subgraph"),
    scope: str = typer.Option(None, "--scope", help="key:value[,key:value]"),
    truth: str = typer.Option(None, "--truth"),
    since: str = typer.Option(None, "--since", help="ISO instant lower bound."),
    until: str = typer.Option(None, "--until", help="ISO instant upper bound."),
    environment: str = typer.Option(None, "--environment"),
    external_id: str = typer.Option(None, "--external-id"),
    limit: int = typer.Option(10, "--limit"),
    pot: str = typer.Option(None, "--pot"),
) -> None:
    """Narrow entity/claim lookup for identity resolution before a write."""
    from domain.ports.services.graph_service import GraphEntitySearchRequest

    with _graph_command("graph.search-entities") as ctx:
        effective_query = query or query_arg
        if not effective_query:
            raise ValueError("query is required")
        host = get_host()
        pot_id = resolve_pot_id(host, pot)
        ctx.set_pot_id(pot_id)
        since_dt, until_dt = _resolve_time_bounds(
            since=since, until=until, window=None
        )
        result = host.graph.search_entities(
            GraphEntitySearchRequest(
                pot_id=pot_id,
                query=effective_query,
                type=type_,
                predicate=predicate,
                subgraph=subgraph,
                scope=_parse_scope(scope),
                truth=truth,
                since=since_dt,
                until=until_dt,
                environment=environment,
                external_id=external_id,
                limit=limit,
            )
        )
        payload = result.to_dict()
        human = (
            "\n".join(
                f"  • [{', '.join(e['labels']) or '?'}] {e['key']} (score={e['score']:.2f})"
                for e in payload["entities"]
            )
            or "(no matching entities)"
        )
        _emit_graph_result(ctx, payload, human=human)


@graph_app.command("mutate")
def graph_mutate(
    file: str = typer.Option(
        None, "--file", help="mutation JSON path; omit to read stdin"
    ),
    dry_run: bool = typer.Option(False, "--dry-run"),
    allow_review_required: bool = typer.Option(False, "--allow-review-required"),
    approved_by: str = typer.Option(
        None, "--approved-by", help="user-ref for medium-risk approval"
    ),
    pot: str = typer.Option(None, "--pot"),
) -> None:
    """Legacy transition wrapper over graph propose + commit."""
    with _graph_command("graph.mutate") as ctx:
        host = get_host()
        pot_id = resolve_pot_id(host, pot)
        ctx.set_pot_id(pot_id)
        payload = _load_json(file)
        proposal = host.graph_workbench.propose(payload, pot_id=pot_id)
        legacy_warning = _legacy_warning(
            "graph.mutate", "graph.propose and graph.commit"
        )
        if dry_run or not proposal.ok:
            _emit_graph_result(
                ctx,
                proposal.to_dict(),
                human=_proposal_human(proposal),
                warnings=legacy_warning,
            )
            if not proposal.ok:
                raise typer.Exit(code=EXIT_VALIDATION)
            return
        if proposal.status == "review_required" and not (
            allow_review_required and approved_by
        ):
            _emit_graph_result(
                ctx,
                proposal.to_dict(),
                human=_proposal_human(proposal),
                warnings=legacy_warning,
                recommended_next_action=(
                    f"Review the plan, then run `potpie graph commit {proposal.plan_id} "
                    "--approved-by <user-ref> --json` when policy allows."
                ),
            )
            return

        result = host.graph_workbench.commit(
            proposal.plan_id,
            pot_id=pot_id,
            approved_by=approved_by if allow_review_required else None,
        )
        _emit_graph_result(
            ctx,
            result.to_dict(),
            human=_commit_human(result),
            warnings=legacy_warning,
            recommended_next_action=(
                "Use `potpie graph propose --file <mutation.json> --json` followed "
                "by `potpie graph commit <plan_id> --json`."
            ),
        )
        if not result.ok:
            raise typer.Exit(code=EXIT_VALIDATION)


# Static, schema-only mutation skeletons (Stage 5 ergonomics). These are
# helpers for harnesses writing mutation JSON by hand — placeholders only,
# never values read from the repo. Keys reference the canonical ontology so
# the contract tests can pin them against ENTITY_TYPES / EDGE_TYPES.
_MUTATION_TEMPLATES: dict[str, dict[str, Any]] = {
    "repo-baseline": {
        "pot_id": "<pot-id>",
        "idempotency_key": "baseline:<owner>/<repo>:v1",
        "created_by": {"surface": "cli", "harness": "<harness>"},
        "operations": [
            {
                "op": "upsert_entity",
                "subject": {
                    "key": "repo:<host>/<owner>/<repo>",
                    "type": "Repository",
                    "name": "<repo>",
                    "summary": "<one-line repo purpose>",
                    "description": "<retrieval card: what the repo is, app type, synonyms a searcher would type>",
                },
            },
            {
                "op": "assert_claim",
                "subject": {"key": "repo:<host>/<owner>/<repo>", "type": "Repository"},
                "predicate": "PROVIDES",
                "object": {
                    "key": "feature:<feature-slug>",
                    "type": "Feature",
                    "name": "<feature name>",
                    "summary": "<one-line capability>",
                    "description": "<retrieval card: what it does, user-facing synonyms, scope>",
                },
                "truth": "source_observation",
                "confidence": 0.9,
                "description": "<where the source says this — e.g. README features section>",
                "evidence": [
                    {
                        "source_ref": "repo:<owner>/<repo>#README",
                        "authority": "repository_metadata",
                    }
                ],
            },
            {
                "op": "link_entities",
                "subject": {"key": "service:<service-slug>", "type": "Service"},
                "predicate": "DEFINED_IN",
                "object": {"key": "repo:<host>/<owner>/<repo>", "type": "Repository"},
                "truth": "source_observation",
                "confidence": 0.9,
                "description": "<which manifest/workflow defines the service>",
                "evidence": [
                    {
                        "source_ref": "repo:<owner>/<repo>#package.json",
                        "authority": "repository_metadata",
                    }
                ],
            },
        ],
    },
    "feature": {
        "pot_id": "<pot-id>",
        "operations": [
            {
                "op": "assert_claim",
                "subject": {"key": "service:<service-slug>", "type": "Service"},
                "predicate": "PROVIDES",
                "object": {
                    "key": "feature:<feature-slug>",
                    "type": "Feature",
                    "name": "<feature name>",
                    "summary": "<one-line capability>",
                    "description": "<retrieval card with synonyms and scope>",
                },
                "truth": "source_observation",
                "confidence": 0.9,
                "description": "<evidence summary>",
                "evidence": [
                    {"source_ref": "<ref>", "authority": "repository_metadata"}
                ],
            }
        ],
    },
    "preference": {
        "pot_id": "<pot-id>",
        "idempotency_key": "preference:<owner>/<repo>:<preference-slug>",
        "created_by": {"surface": "cli", "harness": "<harness>"},
        "operations": [
            {
                "op": "assert_claim",
                "subject": {
                    "key": "preference:<preference-slug>",
                    "type": "Preference",
                    "name": "<short preference name>",
                    "summary": "<one-line prescription>",
                    "description": "<retrieval card: when it applies, synonyms, scope>",
                    "properties": {
                        "policy_kind": "<error_handling|logging|testing|library_choice|file_structure>",
                        "prescription": "<specific guidance an agent should follow>",
                        "strength": "<hard|strong|normal|weak>",
                        "audience": "<repo|service|path|language>",
                    },
                },
                "predicate": "POLICY_APPLIES_TO",
                "object": {"key": "repo:<host>/<owner>/<repo>", "type": "Repository"},
                "truth": "preference",
                "confidence": 0.9,
                "description": "<where the preference is stated>",
                "extra": {
                    "repo": "<host>/<owner>/<repo>",
                    "service": "<service-slug>",
                    "file_path": "<optional/path/or/directory>",
                    "language": "<language>",
                },
                "evidence": [{"source_ref": "<ref>", "authority": "user_statement"}],
            }
        ],
    },
    "preference-policy": {
        "pot_id": "<pot-id>",
        "idempotency_key": "preference:<owner>/<repo>:<preference-slug>",
        "created_by": {"surface": "cli", "harness": "<harness>"},
        "operations": [
            {
                "op": "assert_claim",
                "subject": {
                    "key": "preference:<preference-slug>",
                    "type": "Preference",
                    "name": "<short preference name>",
                    "summary": "<one-line prescription>",
                    "description": "<retrieval card: task words, scope, synonyms>",
                    "properties": {
                        "policy_kind": "<error_handling|logging|testing|library_choice|file_structure>",
                        "prescription": "<specific guidance an agent should follow>",
                        "strength": "<hard|strong|normal|weak>",
                        "audience": "<repo|service|path|language>",
                    },
                },
                "predicate": "POLICY_APPLIES_TO",
                "object": {
                    "key": "code:<repo-or-service>:<path-or-symbol>",
                    "type": "CodeAsset",
                },
                "truth": "preference",
                "confidence": 0.9,
                "description": "<evidence summary and why this policy applies to this scope>",
                "extra": {
                    "repo": "<host>/<owner>/<repo>",
                    "service": "<service-slug>",
                    "file_path": "<path/or/directory>",
                    "language": "<language>",
                },
                "evidence": [{"source_ref": "<ref>", "authority": "user_statement"}],
            }
        ],
    },
    "infra-snapshot": {
        "pot_id": "<pot-id>",
        "idempotency_key": "infra:<owner>/<repo>:<service-slug>:<environment>",
        "created_by": {"surface": "cli", "harness": "<harness>"},
        "operations": [
            {
                "op": "link_entities",
                "subject": {"key": "service:<service-slug>", "type": "Service"},
                "predicate": "DEPLOYED_TO",
                "object": {"key": "environment:<environment>", "type": "Environment"},
                "truth": "source_observation",
                "confidence": 0.95,
                "environment": "<environment>",
                "description": "<source showing service runs in this environment>",
                "evidence": [
                    {"source_ref": "<ref>", "authority": "repository_metadata"}
                ],
            },
            {
                "op": "link_entities",
                "subject": {"key": "service:<service-slug>", "type": "Service"},
                "predicate": "USES_ADAPTER",
                "object": {
                    "key": "adapter:<domain>:<adapter-slug>",
                    "type": "Adapter",
                    "summary": "<adapter/provider selected in this env>",
                    "description": "<retrieval card: adapter, provider, backend, env>",
                },
                "truth": "source_observation",
                "confidence": 0.9,
                "environment": "<environment>",
                "description": "<source showing which adapter/backend is selected>",
                "evidence": [
                    {"source_ref": "<ref>", "authority": "repository_metadata"}
                ],
            },
            {
                "op": "link_entities",
                "subject": {"key": "service:<service-slug>", "type": "Service"},
                "predicate": "DEPLOYED_WITH",
                "object": {
                    "key": "deployment_target:<environment>:<target-slug>",
                    "type": "DeploymentTarget",
                    "summary": "<deployment target/mechanism>",
                    "description": "<retrieval card: platform, workload, deploy mechanism>",
                },
                "truth": "source_observation",
                "confidence": 0.9,
                "environment": "<environment>",
                "description": "<source showing deployment target/mechanism>",
                "evidence": [
                    {"source_ref": "<ref>", "authority": "repository_metadata"}
                ],
            },
            {
                "op": "link_entities",
                "subject": {"key": "service:<service-slug>", "type": "Service"},
                "predicate": "CONFIGURES",
                "object": {
                    "key": "config:<service-or-env>:<config-name>",
                    "type": "ConfigVariable",
                    "summary": "<config variable>",
                    "description": "<retrieval card: env var/config key and behavior it selects>",
                },
                "truth": "source_observation",
                "confidence": 0.8,
                "environment": "<environment>",
                "description": "<source showing this config affects the service/adapter>",
                "evidence": [
                    {"source_ref": "<ref>", "authority": "repository_metadata"}
                ],
            },
        ],
    },
    "bug-fix": {
        "pot_id": "<pot-id>",
        "idempotency_key": "bug-fix:<bug-slug>:<fix-hash>",
        "created_by": {"surface": "cli", "harness": "<harness>"},
        "operations": [
            {
                "op": "assert_claim",
                "subject": {
                    "key": "bug_pattern:<bug-slug>",
                    "type": "BugPattern",
                    "name": "<symptom name>",
                    "summary": "<one-line symptom>",
                    "description": "<retrieval card: error text, symptoms, synonyms, where it shows up>",
                    "properties": {
                        "symptom_signature": "<stable symptom/error signature>",
                    },
                },
                "predicate": "REPRODUCES",
                "object": {"key": "service:<service-slug>", "type": "Service"},
                "truth": "agent_claim",
                "confidence": 0.8,
                "description": "<how the bug manifests>",
            },
            {
                "op": "assert_claim",
                "subject": {
                    "key": "fix:<fix-hash>",
                    "type": "Fix",
                    "summary": "<one-line fix>",
                    "description": "<retrieval card: what fixed it, files touched, verification>",
                    "properties": {
                        "fix_steps": "<what changed or what to do>",
                        "verification_status": "<verified|unverified|failed>",
                    },
                },
                "predicate": "RESOLVED",
                "object": {"key": "bug_pattern:<bug-slug>", "type": "BugPattern"},
                "truth": "agent_claim",
                "confidence": 0.8,
                "description": "<fix summary + verification status>",
            },
            {
                "op": "assert_claim",
                "subject": {
                    "key": "activity:<source>:<verification-id>",
                    "type": "Activity",
                },
                "predicate": "VERIFIED",
                "object": {"key": "fix:<fix-hash>", "type": "Fix"},
                "truth": "agent_claim",
                "confidence": 0.8,
                "description": "<how the fix was verified, or what remains unverified>",
            },
        ],
    },
    "decision": {
        "pot_id": "<pot-id>",
        "operations": [
            {
                "op": "assert_claim",
                "subject": {
                    "key": "decision:<decision-hash>",
                    "type": "Decision",
                    "name": "<decision title>",
                    "summary": "<one-line decision>",
                    "description": "<retrieval card: rationale, alternatives rejected, synonyms>",
                },
                "predicate": "DECIDED",
                "object": {"key": "repo:<host>/<owner>/<repo>", "type": "Repository"},
                "truth": "user_decision",
                "confidence": 1.0,
                "description": "<who decided and why>",
                "evidence": [{"source_ref": "<ref>", "authority": "user_statement"}],
            }
        ],
    },
    "timeline-event": {
        "pot_id": "<pot-id>",
        "operations": [
            {
                "op": "append_event",
                "verb": "<verb e.g. merged_pr|deployed|decided>",
                "occurred_at": "<ISO-8601 timestamp>",
                "description": "<what happened, written for timeline recall>",
                "actor": {"key": "person:<handle>", "type": "Person"},
                "targets": [{"key": "service:<service-slug>", "type": "Service"}],
                "evidence": [{"source_ref": "<ref>", "authority": "external_system"}],
            }
        ],
    },
    "timeline-change": {
        "pot_id": "<pot-id>",
        "idempotency_key": "timeline:<source>:<id>",
        "created_by": {"surface": "cli", "harness": "<harness>"},
        "operations": [
            {
                "op": "append_event",
                "verb": "<verb e.g. merged_pr|linear_done|deployed|incident>",
                "occurred_at": "<ISO-8601 timestamp>",
                "description": "<what changed, source title, affected behavior, regression keywords>",
                "actor": {"key": "person:<handle>", "type": "Person"},
                "targets": [
                    {"key": "service:<service-slug>", "type": "Service"},
                    {
                        "key": "code:<repo-or-service>:<path-or-symbol>",
                        "type": "CodeAsset",
                    },
                ],
                "mentions": [{"key": "feature:<feature-slug>", "type": "Feature"}],
                "evidence": [{"source_ref": "<ref>", "authority": "external_system"}],
            }
        ],
    },
}


@graph_app.command("mutation-template")
def graph_mutation_template(
    kind: str = typer.Option(
        "repo-baseline",
        "--kind",
        help=f"template kind: {' | '.join(sorted(_MUTATION_TEMPLATES))}",
    ),
) -> None:
    """Print a schema-only mutation skeleton for `graph propose`.

    Pure schema helper: emits placeholders for the harness to fill from
    sources it has actually read. It never inspects the repository or infers
    graph facts.
    """
    with _graph_command("graph.mutation-template") as ctx:
        template = _MUTATION_TEMPLATES.get(kind.strip().lower())
        if template is None:
            fail(
                code="unknown_template_kind",
                message=f"Unknown mutation template kind {kind!r}.",
                next_action=f"pick one of: {', '.join(sorted(_MUTATION_TEMPLATES))}",
            )
        rendered = json.dumps(template, indent=2)
        emit(
            graph_success_envelope(
                command=ctx.command,
                request_id=ctx.request_id,
                pot_id=ctx.pot_id,
                result={"kind": kind, "template": template},
                warnings=_legacy_warning(
                    "graph.mutation-template", "graph.describe mutation examples"
                ),
                recommended_next_action=(
                    "Use `potpie graph describe <subgraph> --examples --json` once "
                    "describe is implemented."
                ),
            ).to_dict(),
            human=rendered,
        )


@graph_app.command("nudge")
def graph_nudge(
    event: str = typer.Option(
        ...,
        "--event",
        help=NUDGE_EVENT_HELP,
    ),
    session: str = typer.Option(
        ..., "--session", help="harness session id (dedup key)"
    ),
    path: str = typer.Option(
        None, "--path", help="file path scope (PreToolUse Write/Edit)"
    ),
    scope: str = typer.Option(None, "--scope", help="key:value[,key:value]"),
    query: str = typer.Option(
        None, "--query", help="symptom/intent text (test_failed etc.)"
    ),
    limit: int = typer.Option(5, "--limit", help="max injected items"),
    pot: str = typer.Option(None, "--pot"),
) -> None:
    """Event→action policy brain: inject ranked context, prompt a write, or stay silent.

    Deterministic and free — reads via the local embedder, never calls a model.
    Hooks forward their event + path here and inject the result.
    """
    from domain.nudge import GraphNudgeRequest

    with _graph_command("graph.nudge") as ctx:
        host = get_host()
        pot_id = resolve_pot_id(host, pot)
        ctx.set_pot_id(pot_id)
        result = host.nudge.nudge(
            GraphNudgeRequest(
                pot_id=pot_id,
                event=event,
                session_id=session,
                scope=_parse_scope(scope),
                path=path,
                query=query,
                limit=limit,
            )
        )
        _emit_graph_result(
            ctx,
            result.to_dict(),
            human=_nudge_human(result),
            warnings=_legacy_warning("graph.nudge", "the installed hook adapter"),
            recommended_next_action=(
                "Hooks should read the `result` object from this workbench envelope."
            ),
        )


@graph_app.command("status")
def graph_status(pot: str = typer.Option(None, "--pot")) -> None:
    with _graph_command("graph.status") as ctx:
        host = get_host()
        pot_id = resolve_pot_id(host, pot)
        ctx.set_pot_id(pot_id)
        dp = host.graph.data_plane_status(pot_id)
        versions = {"_global": int(dict(dp.counts).get("claims", 0))}
        ctx.set_subgraph_versions(versions)
        payload = _graph_status_payload(host, pot_id, dp)
        recommended = None
        if not dp.backend_ready:
            recommended = (
                "Run `potpie backend doctor` to inspect graph backend readiness "
                "and capability-specific failures."
            )
        _emit_graph_result(
            ctx,
            payload,
            human=f"backend={dp.backend_profile} ready={dp.backend_ready} counts={dict(dp.counts)}",
            recommended_next_action=recommended,
        )


@graph_app.command("describe")
def graph_describe(
    subgraph: str = typer.Argument(None),
    view: str = typer.Option(None, "--view"),
    examples: bool = typer.Option(False, "--examples"),
    pot: str = typer.Option(None, "--pot"),
) -> None:
    with _graph_command("graph.describe") as ctx:
        _set_optional_pot(ctx, pot)
        payload = describe_contract(
            subgraph=subgraph,
            view=view,
            include_examples=examples,
        )
        subgraph_name = payload["subgraph"]["name"]
        described = payload["view"]["name"] if view else subgraph_name
        described_view = described.split(".", 1)[1] if "." in described else described
        _emit_graph_result(
            ctx,
            payload,
            human=_describe_human(payload),
            recommended_next_action=(
                f"Use `potpie graph read --subgraph {subgraph_name} --view {described_view} --json` after choosing a scope."
                if view
                else "Use `potpie graph describe <subgraph> --view <view> --json` for one backed view."
            ),
        )


@graph_app.command("neighborhood")
def graph_neighborhood(
    entity: str = typer.Option(None, "--entity"),
    predicate: str = typer.Option(None, "--predicate"),
    depth: int = typer.Option(2, "--depth"),
    direction: str = typer.Option("both", "--direction"),
    limit: int = typer.Option(50, "--limit"),
    pot: str = typer.Option(None, "--pot"),
) -> None:
    with _graph_command("graph.neighborhood") as ctx:
        if not entity:
            raise ValueError("--entity is required")
        normalized_direction = (direction or "both").strip().lower()
        if normalized_direction not in {"out", "in", "both"}:
            raise ValueError("--direction must be one of: out, in, both")
        if depth < 1:
            raise ValueError("--depth must be >= 1")
        if limit < 1:
            raise ValueError("--limit must be >= 1")
        host = get_host()
        _require_backend_capability(
            host,
            capability="inspection",
            method="neighborhood",
            command="graph neighborhood",
        )
        pot_id = resolve_pot_id(host, pot)
        ctx.set_pot_id(pot_id)
        predicates = _parse_predicates(predicate)
        sl = host.backend.inspection.neighborhood(
            pot_id=pot_id,
            entity_key=entity,
            depth=depth,
            direction=normalized_direction,
            predicates=predicates,
            limit=limit,
        )
        payload = {
            "entity_key": entity,
            "depth": depth,
            "direction": normalized_direction,
            "predicates": list(predicates),
            "nodes": [
                {
                    "key": n.key,
                    "labels": list(n.labels),
                    "properties": dict(n.properties),
                }
                for n in sl.nodes
            ],
            "edges": [
                {
                    "predicate": e.predicate,
                    "from": e.from_key,
                    "to": e.to_key,
                    "properties": dict(e.properties),
                }
                for e in sl.edges
            ],
            "truncated": sl.truncated,
        }
        _emit_graph_result(
            ctx,
            payload,
            human=f"{len(sl.nodes)} nodes, {len(sl.edges)} edges around {entity}",
        )


@graph_app.command("propose")
def graph_propose(
    file: str = typer.Option(
        None, "--file", help="mutation JSON path; omit to read stdin"
    ),
    ttl: str = typer.Option("1h", "--ttl", help="plan expiry such as 30m, 1h, 2d"),
    pot: str = typer.Option(None, "--pot"),
) -> None:
    with _graph_command("graph.propose") as ctx:
        host = get_host()
        pot_id = resolve_pot_id(host, pot)
        ctx.set_pot_id(pot_id)
        payload = _load_json(file)
        result = host.graph_workbench.propose(
            payload,
            pot_id=pot_id,
            ttl_seconds=_parse_ttl_seconds(ttl),
        )
        _emit_graph_result(
            ctx,
            result.to_dict(),
            human=_proposal_human(result),
        )
        if not result.ok:
            raise typer.Exit(code=EXIT_VALIDATION)


@graph_app.command("commit")
def graph_commit(
    plan_id: str = typer.Argument(None),
    approved_by: str = typer.Option(
        None, "--approved-by", help="user-ref for medium-risk approval"
    ),
    pot: str = typer.Option(None, "--pot"),
) -> None:
    with _graph_command("graph.commit") as ctx:
        if not plan_id:
            fail(code="validation_error", message="plan_id is required")
        host = get_host()
        pot_id = resolve_pot_id(host, pot)
        ctx.set_pot_id(pot_id)
        result = host.graph_workbench.commit(
            plan_id,
            pot_id=pot_id,
            approved_by=approved_by,
        )
        _emit_graph_result(
            ctx,
            result.to_dict(),
            human=_commit_human(result),
        )
        if not result.ok:
            raise typer.Exit(code=EXIT_VALIDATION)


@graph_app.command("history")
def graph_history(
    entity: str = typer.Option(None, "--entity"),
    claim: str = typer.Option(None, "--claim"),
    subgraph: str = typer.Option(None, "--subgraph"),
    plan: str = typer.Option(None, "--plan"),
    mutation: str = typer.Option(None, "--mutation"),
    since: str = typer.Option(None, "--since"),
    until: str = typer.Option(None, "--until"),
    limit: int = typer.Option(50, "--limit"),
    pot: str = typer.Option(None, "--pot"),
) -> None:
    with _graph_command("graph.history") as ctx:
        host = get_host()
        pot_id = resolve_pot_id(host, pot)
        ctx.set_pot_id(pot_id)
        since_dt, until_dt = _resolve_time_bounds(
            since=since, until=until, window=None
        )
        result = host.graph_workbench.history(
            pot_id=pot_id,
            entity_key=entity,
            claim_key=claim,
            subgraph=subgraph,
            plan_id=plan,
            mutation_id=mutation,
            since=since_dt,
            until=until_dt,
            limit=limit,
        )
        _emit_graph_result(
            ctx,
            result.to_dict(),
            human=_history_human(result),
        )


@inbox_app.command("add")
def graph_inbox_add(
    summary: str = typer.Option(None, "--summary"),
    details: str = typer.Option(None, "--details"),
    evidence: list[str] | None = typer.Option(None, "--evidence"),
    source_ref: list[str] | None = typer.Option(None, "--source-ref"),
    subgraph: list[str] | None = typer.Option(None, "--subgraph"),
    created_by: str = typer.Option(None, "--created-by"),
    pot: str = typer.Option(None, "--pot"),
) -> None:
    with _graph_command("graph.inbox.add") as ctx:
        host = get_host()
        pot_id = resolve_pot_id(host, pot)
        ctx.set_pot_id(pot_id)
        result = host.graph_workbench.inbox_add(
            pot_id=pot_id,
            summary=summary,
            details=details,
            evidence=tuple(evidence or ()),
            source_refs=tuple(source_ref or ()),
            suspected_subgraphs=tuple(subgraph or ()),
            created_by=_parse_created_by(created_by),
        )
        _emit_inbox_result(ctx, result)


@inbox_app.command("list")
def graph_inbox_list(
    status: list[str] | None = typer.Option(None, "--status"),
    claimed_by: str = typer.Option(None, "--claimed-by"),
    subgraph: str = typer.Option(None, "--subgraph"),
    source_ref: str = typer.Option(None, "--source-ref"),
    since: str = typer.Option(None, "--since"),
    until: str = typer.Option(None, "--until"),
    limit: int = typer.Option(50, "--limit"),
    pot: str = typer.Option(None, "--pot"),
) -> None:
    with _graph_command("graph.inbox.list") as ctx:
        host = get_host()
        pot_id = resolve_pot_id(host, pot)
        ctx.set_pot_id(pot_id)
        since_dt, until_dt = _resolve_time_bounds(
            since=since, until=until, window=None
        )
        result = host.graph_workbench.inbox_list(
            pot_id=pot_id,
            status=tuple(status or ()),
            claimed_by=claimed_by,
            suspected_subgraph=subgraph,
            source_ref=source_ref,
            since=since_dt,
            until=until_dt,
            limit=limit,
        )
        _emit_inbox_result(ctx, result)


@inbox_app.command("show")
def graph_inbox_show(
    item_id: str = typer.Argument(None),
    pot: str = typer.Option(None, "--pot"),
) -> None:
    with _graph_command("graph.inbox.show") as ctx:
        if not item_id:
            raise ValueError("item_id is required")
        host = get_host()
        pot_id = resolve_pot_id(host, pot)
        ctx.set_pot_id(pot_id)
        result = host.graph_workbench.inbox_show(pot_id=pot_id, item_id=item_id)
        _emit_inbox_result(ctx, result)


@inbox_app.command("claim")
def graph_inbox_claim(
    item_id: str = typer.Argument(None),
    claimed_by: str = typer.Option(None, "--by", "--claimed-by"),
    pot: str = typer.Option(None, "--pot"),
) -> None:
    with _graph_command("graph.inbox.claim") as ctx:
        if not item_id:
            raise ValueError("item_id is required")
        host = get_host()
        pot_id = resolve_pot_id(host, pot)
        ctx.set_pot_id(pot_id)
        result = host.graph_workbench.inbox_claim(
            pot_id=pot_id,
            item_id=item_id,
            claimed_by=claimed_by,
        )
        _emit_inbox_result(ctx, result)


@inbox_app.command("mark-applied")
def graph_inbox_mark_applied(
    item_id: str = typer.Argument(None),
    plan: str = typer.Option(None, "--plan"),
    mutation: str = typer.Option(None, "--mutation"),
    closed_by: str = typer.Option(None, "--by", "--closed-by"),
    pot: str = typer.Option(None, "--pot"),
) -> None:
    with _graph_command("graph.inbox.mark-applied") as ctx:
        if not item_id:
            raise ValueError("item_id is required")
        host = get_host()
        pot_id = resolve_pot_id(host, pot)
        ctx.set_pot_id(pot_id)
        result = host.graph_workbench.inbox_mark_applied(
            pot_id=pot_id,
            item_id=item_id,
            closed_by=closed_by,
            linked_plan_id=plan,
            linked_mutation_id=mutation,
        )
        _emit_inbox_result(ctx, result)


@inbox_app.command("mark-rejected")
def graph_inbox_mark_rejected(
    item_id: str = typer.Argument(None),
    reason: str = typer.Option(None, "--reason"),
    closed_by: str = typer.Option(None, "--by", "--closed-by"),
    pot: str = typer.Option(None, "--pot"),
) -> None:
    with _graph_command("graph.inbox.mark-rejected") as ctx:
        if not item_id:
            raise ValueError("item_id is required")
        host = get_host()
        pot_id = resolve_pot_id(host, pot)
        ctx.set_pot_id(pot_id)
        result = host.graph_workbench.inbox_mark_rejected(
            pot_id=pot_id,
            item_id=item_id,
            closed_by=closed_by,
            rejection_reason=reason,
        )
        _emit_inbox_result(ctx, result)


@inbox_app.command("close")
def graph_inbox_close(
    item_id: str = typer.Argument(None),
    plan: str = typer.Option(None, "--plan"),
    mutation: str = typer.Option(None, "--mutation"),
    reason: str = typer.Option(None, "--reason"),
    closed_by: str = typer.Option(None, "--by", "--closed-by"),
    pot: str = typer.Option(None, "--pot"),
) -> None:
    with _graph_command("graph.inbox.close") as ctx:
        if not item_id:
            raise ValueError("item_id is required")
        host = get_host()
        pot_id = resolve_pot_id(host, pot)
        ctx.set_pot_id(pot_id)
        result = host.graph_workbench.inbox_close(
            pot_id=pot_id,
            item_id=item_id,
            closed_by=closed_by,
            linked_plan_id=plan,
            linked_mutation_id=mutation,
            rejection_reason=reason,
        )
        _emit_inbox_result(ctx, result)


@quality_app.command("summary")
def graph_quality_summary(
    pot: str = typer.Option(None, "--pot"),
) -> None:
    _run_quality_report(command="graph.quality.summary", report="summary", pot=pot)


@quality_app.command("duplicate-candidates")
def graph_quality_duplicate_candidates(
    subgraph: str = typer.Option(None, "--subgraph"),
    limit: int = typer.Option(50, "--limit"),
    pot: str = typer.Option(None, "--pot"),
) -> None:
    _run_quality_report(
        command="graph.quality.duplicate-candidates",
        report="duplicate-candidates",
        pot=pot,
        subgraph=subgraph,
        limit=limit,
    )


@quality_app.command("stale-facts")
def graph_quality_stale_facts(
    subgraph: str = typer.Option(None, "--subgraph"),
    limit: int = typer.Option(50, "--limit"),
    pot: str = typer.Option(None, "--pot"),
) -> None:
    _run_quality_report(
        command="graph.quality.stale-facts",
        report="stale-facts",
        pot=pot,
        subgraph=subgraph,
        limit=limit,
    )


@quality_app.command("conflicting-claims")
def graph_quality_conflicting_claims(
    subgraph: str = typer.Option(None, "--subgraph"),
    limit: int = typer.Option(50, "--limit"),
    pot: str = typer.Option(None, "--pot"),
) -> None:
    _run_quality_report(
        command="graph.quality.conflicting-claims",
        report="conflicting-claims",
        pot=pot,
        subgraph=subgraph,
        limit=limit,
    )


@quality_app.command("orphan-entities")
def graph_quality_orphan_entities(
    subgraph: str = typer.Option(None, "--subgraph"),
    limit: int = typer.Option(50, "--limit"),
    pot: str = typer.Option(None, "--pot"),
) -> None:
    _run_quality_report(
        command="graph.quality.orphan-entities",
        report="orphan-entities",
        pot=pot,
        subgraph=subgraph,
        limit=limit,
    )


@quality_app.command("low-confidence")
def graph_quality_low_confidence(
    subgraph: str = typer.Option(None, "--subgraph"),
    limit: int = typer.Option(50, "--limit"),
    threshold: float = typer.Option(0.5, "--threshold"),
    pot: str = typer.Option(None, "--pot"),
) -> None:
    _run_quality_report(
        command="graph.quality.low-confidence",
        report="low-confidence",
        pot=pot,
        subgraph=subgraph,
        limit=limit,
        confidence_threshold=threshold,
    )


@quality_app.command("projection-drift")
def graph_quality_projection_drift(
    subgraph: str = typer.Option(None, "--subgraph"),
    limit: int = typer.Option(50, "--limit"),
    pot: str = typer.Option(None, "--pot"),
) -> None:
    _run_quality_report(
        command="graph.quality.projection-drift",
        report="projection-drift",
        pot=pot,
        subgraph=subgraph,
        limit=limit,
    )


def _run_quality_report(
    *,
    command: str,
    report: str,
    pot: str | None,
    subgraph: str | None = None,
    limit: int = 50,
    confidence_threshold: float = 0.5,
) -> None:
    with _graph_command(command) as ctx:
        host = get_host()
        pot_id = resolve_pot_id(host, pot)
        ctx.set_pot_id(pot_id)
        result = host.graph_workbench.quality(
            pot_id=pot_id,
            report=report,
            subgraph=subgraph,
            limit=limit,
            confidence_threshold=confidence_threshold,
        )
        _emit_quality_result(ctx, result)


@graph_app.command("inspect")
def graph_inspect(
    entity_key: str = typer.Argument(None),
    depth: int = typer.Option(2, "--depth"),
    pot: str = typer.Option(None, "--pot"),
) -> None:
    with _graph_command("graph.inspect") as ctx:
        if not entity_key:
            raise ValueError("entity_key is required")
        host = get_host()
        _require_backend_capability(
            host,
            capability="inspection",
            method="neighborhood",
            command="graph inspect",
        )
        pot_id = resolve_pot_id(host, pot)
        ctx.set_pot_id(pot_id)
        sl = host.backend.inspection.neighborhood(
            pot_id=pot_id, entity_key=entity_key, depth=depth
        )
        _emit_graph_result(
            ctx,
            {
                "nodes": [{"key": n.key, "labels": list(n.labels)} for n in sl.nodes],
                "edges": [
                    {"predicate": e.predicate, "from": e.from_key, "to": e.to_key}
                    for e in sl.edges
                ],
            },
            human=f"{len(sl.nodes)} nodes, {len(sl.edges)} edges around {entity_key}",
            warnings=_legacy_warning("graph.inspect", "graph.neighborhood"),
            recommended_next_action="Use `potpie graph neighborhood --entity <key> --json` once implemented.",
        )


@graph_app.command("export")
def graph_export(
    file: str = typer.Argument(None), pot: str = typer.Option(None, "--pot")
) -> None:
    with _graph_command("graph.export") as ctx:
        if not file:
            raise ValueError("file is required")
        host = get_host()
        _require_backend_capability(
            host,
            capability="snapshot",
            method="export",
            command="graph export",
        )
        pot_id = resolve_pot_id(host, pot)
        ctx.set_pot_id(pot_id)
        manifest = host.backend.snapshot.export(pot_id=pot_id, destination=file)
        _emit_graph_result(
            ctx,
            {"location": manifest.location, "claims": manifest.claim_count},
            human=f"exported {manifest.claim_count} claims → {manifest.location}",
        )


@graph_app.command("import")
def graph_import(
    file: str = typer.Argument(None), pot: str = typer.Option(None, "--pot")
) -> None:
    with _graph_command("graph.import") as ctx:
        if not file:
            raise ValueError("file is required")
        host = get_host()
        _require_backend_capability(
            host,
            capability="snapshot",
            method="import_",
            command="graph import",
        )
        pot_id = resolve_pot_id(host, pot)
        ctx.set_pot_id(pot_id)
        manifest = host.backend.snapshot.import_(pot_id=pot_id, source=file)
        _emit_graph_result(
            ctx,
            {"location": manifest.location, "claims": manifest.claim_count},
            human=f"imported {manifest.claim_count} claims from {manifest.location}",
        )


@graph_app.command("repair")
def graph_repair(
    semantic_index: bool = typer.Option(False, "--semantic-index"),
    entity_summaries: bool = typer.Option(False, "--entity-summaries"),
    all_: bool = typer.Option(False, "--all"),
    pot: str = typer.Option(None, "--pot"),
) -> None:
    with _graph_command("graph.repair") as ctx:
        host = get_host()
        pot_id = resolve_pot_id(host, pot)
        ctx.set_pot_id(pot_id)
        targets = []
        if not all_:
            if semantic_index:
                targets.append("semantic_index")
            if entity_summaries:
                targets.append("entity_summaries")
        report = host.backend.analytics.repair(pot_id, targets=targets)
        _emit_graph_result(
            ctx,
            {"targets": list(report.targets), "repaired": dict(report.repaired)},
            human=report.detail or f"repaired {dict(report.repaired)}",
        )


@backend_app.command("list")
def backend_list() -> None:
    with contract():
        from adapters.outbound.graph.backends import KNOWN_PROFILES

        active = get_host().backend.profile
        emit(
            {"profiles": list(KNOWN_PROFILES), "active": active},
            human="\n".join(
                f"{'*' if p == active else ' '} {p}" for p in KNOWN_PROFILES
            ),
        )


@backend_app.command("status")
def backend_status() -> None:
    with contract():
        host = get_host()
        caps = host.backend.capabilities()
        emit(
            {"profile": host.backend.profile, "capabilities": list(caps.implemented())},
            human=f"{host.backend.profile}: {', '.join(caps.implemented())}",
        )


@backend_app.command("use")
def backend_use(profile: str) -> None:
    # TODO(stage-N): persist the selected profile to local config; the host
    # rebuilds per process from $CONTEXT_ENGINE_BACKEND today.
    with contract():
        emit(
            {"profile": profile, "persisted": False},
            human=f"(advisory) set CONTEXT_ENGINE_BACKEND={profile}; persistence is TODO",
        )


@backend_app.command("doctor")
def backend_doctor() -> None:
    with contract():
        host = get_host()
        pot = host.pots.active_pot()
        readiness = host.backend.mutation.readiness(pot.pot_id if pot else "")
        emit(
            {
                "profile": readiness.profile,
                "ready": readiness.ready,
                "capability_ready": dict(readiness.capability_ready),
                "detail": readiness.detail,
            },
            human=f"{readiness.profile} ready={readiness.ready} {dict(readiness.capability_ready)}",
        )


# --- Graph Surface Lite helpers ---------------------------------------------


def _set_optional_pot(ctx: _GraphCliCommandContext, pot: str | None) -> None:
    host = get_host()
    if pot:
        ctx.set_pot_id(resolve_pot_id(host, pot))
        return
    active = _safe(lambda: host.pots.active_pot(), None)
    if active is not None:
        ctx.set_pot_id(getattr(active, "pot_id", None))


def _graph_status_payload(host: Any, pot_id: str, dp) -> dict[str, Any]:
    caps = _safe(lambda: host.backend.capabilities(), None)
    implemented = list(caps.implemented()) if caps is not None else []
    status = GraphWorkbenchStatus(
        host={
            "kind": getattr(host, "profile", "local"),
            "liveness": "ok",
        },
        pot={
            "id": pot_id,
            "selected_backend": dp.backend_profile,
        },
        graph_service={
            "graph_contract_version": "v2",
            "data_plane_graph_contract_version": DATA_PLANE_CONTRACT_VERSION,
            "ontology_version": ONTOLOGY_VERSION,
            "supported_commands": list(GRAPH_WORKBENCH_COMMANDS),
            "reader_backed_includes": list(dp.reader_backed_includes),
            "validator_ready": True,
            "match_mode": dp.match_mode,
        },
        backend={
            "profile": dp.backend_profile,
            "ready": dp.backend_ready,
            "detail": dp.detail,
            "readiness_command": "potpie backend doctor",
            "recommended_next_action": (
                "Run `potpie backend doctor` to inspect graph backend readiness "
                "and capability-specific failures."
            )
            if not dp.backend_ready
            else None,
            "implemented_capabilities": implemented,
            "counts": dict(dp.counts),
            "freshness": dict(dp.freshness),
        },
        ledger={"status": "not_inspected"},
        skills={"status": "not_inspected"},
        quality=dict(dp.quality),
    )
    return status.to_dict()


def _describe_human(payload: Mapping[str, Any]) -> str:
    subgraph = payload.get("subgraph")
    if not isinstance(subgraph, Mapping):
        return "graph contract"
    view = payload.get("view")
    if isinstance(view, Mapping):
        filters = ", ".join(str(v) for v in view.get("supported_filters", ())) or "-"
        relations = ", ".join(str(v) for v in view.get("inline_relations", ())) or "-"
        return (
            f"{view.get('name')} ({view.get('result_shape')})\n"
            f"purpose: {view.get('purpose')}\n"
            f"filters: {filters}\n"
            f"relations: {relations}"
        )
    views = subgraph.get("views", ())
    view_names = ", ".join(str(v.get("name")) for v in views if isinstance(v, Mapping))
    relation_names = ", ".join(
        str(r.get("name"))
        for r in subgraph.get("relation_types", ())
        if isinstance(r, Mapping)
    )
    return (
        f"{subgraph.get('name')}: {subgraph.get('purpose')}\n"
        f"views: {view_names}\n"
        f"relations: {relation_names}"
    )


def _safe(fn, default):
    try:
        return fn()
    except Exception:  # noqa: BLE001
        return default


def _require_backend_capability(
    host: Any,
    *,
    capability: str,
    method: str,
    command: str,
) -> None:
    caps = host.backend.capabilities()
    if bool(getattr(caps, capability, False)):
        return
    profile = getattr(caps, "profile", getattr(host.backend, "profile", "unknown"))
    raise CapabilityNotImplemented(
        f"graph.{profile}.{capability}.{method}",
        detail=f"{command} is not supported by the active '{profile}' backend",
        recommended_next_action=(
            "run 'potpie backend status' to inspect capabilities, or switch to "
            f"a backend that implements {capability}"
        ),
    )


def _parse_scope(scope: str | None) -> dict[str, str]:
    if not scope:
        return {}
    out: dict[str, str] = {}
    for pair in scope.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if ":" not in pair:
            raise ValueError(
                f"invalid --scope entry {pair!r}; expected key:value pairs"
            )
        key, value = pair.split(":", 1)
        key = key.strip()
        if not key:
            raise ValueError(
                f"invalid --scope entry {pair!r}; scope keys must not be empty"
            )
        value = value.strip()
        if not value:
            raise ValueError(
                f"invalid --scope entry {pair!r}; scope values must not be empty"
            )
        out[key] = value
    return out


def _parse_created_by(value: str | None) -> dict[str, Any]:
    clean = value.strip() if isinstance(value, str) else ""
    if not clean:
        return {"surface": "cli"}
    if clean.startswith("{"):
        try:
            parsed = json.loads(clean)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid --created-by JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError("--created-by JSON must be an object")
        parsed.setdefault("surface", "cli")
        return parsed
    return {"surface": "cli", "actor": clean}


def _parse_predicates(predicate: str | None) -> tuple[str, ...]:
    if not predicate:
        return ()
    out: list[str] = []
    for raw in predicate.split(","):
        value = raw.strip().upper()
        if value:
            out.append(value)
    return tuple(dict.fromkeys(out))


def _load_json(file: str | None) -> dict:
    """Load a mutation payload from a file or stdin."""
    if file:
        try:
            with open(file, encoding="utf-8") as fh:
                raw = fh.read()
        except OSError as exc:
            raise ValueError(f"cannot read mutation file {file!r}: {exc}") from exc
    else:
        raw = sys.stdin.read()
    if not raw.strip():
        raise ValueError(
            "empty mutation payload (provide --file or pipe JSON on stdin)"
        )
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in mutation payload: {exc}") from exc


def _emit_graph_read(
    ctx: _GraphCliCommandContext,
    result,
    *,
    format_: str,
    sort: str,
    dedupe: str,
    event_limit: int | None = None,
) -> None:
    if not is_json():
        _emit_read(
            result, format_=format_, sort=sort, dedupe=dedupe, event_limit=event_limit
        )
        return

    normalized_format = _effective_read_format(result, format_)
    if normalized_format == "jsonl":
        rows = _timeline_events(result, sort=sort, dedupe=dedupe, limit=event_limit)
        if not rows:
            rows = _raw_item_rows(result)
        payload = _read_payload(
            result,
            format_="raw",
            sort=sort,
            dedupe=dedupe,
            event_limit=event_limit,
        )
        payload["read_shape"] = "jsonl"
        payload["rows"] = rows
        payload["row_count"] = len(rows)
    else:
        payload = _read_payload(
            result,
            format_=normalized_format,
            sort=sort,
            dedupe=dedupe,
            event_limit=event_limit,
        )
    _emit_graph_result(
        ctx,
        payload,
        human=_read_human(
            result,
            format_=normalized_format,
            sort=sort,
            dedupe=dedupe,
            event_limit=event_limit,
        ),
    )


def _emit_read(
    result, *, format_: str, sort: str, dedupe: str, event_limit: int | None = None
) -> None:
    normalized_format = _effective_read_format(result, format_)
    if normalized_format == "jsonl":
        rows = _timeline_events(result, sort=sort, dedupe=dedupe, limit=event_limit)
        if not rows:
            rows = _raw_item_rows(result)
        for row in rows:
            typer.echo(json.dumps(row, default=str))
        return
    payload = _read_payload(
        result,
        format_=normalized_format,
        sort=sort,
        dedupe=dedupe,
        event_limit=event_limit,
    )
    emit(
        payload,
        human=_read_human(
            result,
            format_=normalized_format,
            sort=sort,
            dedupe=dedupe,
            event_limit=event_limit,
        ),
    )


def _read_payload(
    result,
    *,
    format_: str = "raw",
    sort: str = "auto",
    dedupe: str = "auto",
    event_limit: int | None = None,
) -> dict:
    payload = result.to_dict()
    if format_ in ("events", "table"):
        events = _timeline_events(result, sort=sort, dedupe=dedupe, limit=event_limit)
        payload["read_shape"] = "events"
        payload["events"] = events
        payload["event_count"] = len(events)
        payload["freshness"] = _timeline_freshness(events)
    return payload


def _read_human(
    result,
    *,
    format_: str = "raw",
    sort: str = "auto",
    dedupe: str = "auto",
    event_limit: int | None = None,
) -> str:
    if format_ in ("events", "table"):
        return _timeline_human(result, sort=sort, dedupe=dedupe, event_limit=event_limit)
    payload = result.to_dict()
    items = payload.get("items", [])
    lines = [
        f"view={payload.get('view')} backed={payload.get('backed')} "
        f"items={len(items)} quality={payload.get('quality', {}).get('status')}"
    ]
    for item in items[:10]:
        fact = item.get("summary") or item.get("entity_key") or ""
        lines.append(f"  • [{item.get('entity_type') or '?'}] {fact}")
    return "\n".join(lines)


def _raw_item_rows(result) -> list[dict[str, Any]]:
    return list(result.to_dict().get("items", []))


def _effective_read_format(result, requested: str) -> str:
    value = (requested or "auto").strip().lower()
    if value not in {"auto", "raw", "events", "table", "jsonl"}:
        raise ValueError("--format must be one of: auto, raw, events, table, jsonl")
    if value == "auto":
        return "events" if _is_timeline_view(result.to_dict().get("view")) else "raw"
    return value


def _effective_requested_format(*, subgraph: str, view: str, requested: str) -> str:
    value = (requested or "auto").strip().lower()
    if value == "auto":
        return "events" if _is_timeline_view(f"{subgraph}.{view}") else "raw"
    return value


def _service_limit_for_read(
    *, subgraph: str, view: str, format_: str, requested_limit: int
) -> int:
    if _is_timeline_view(f"{subgraph}.{view}") and format_ in {
        "events",
        "table",
        "jsonl",
    }:
        return min(max(requested_limit * 8, 40), 200)
    return requested_limit


def _is_timeline_view(view: str | None) -> bool:
    return str(view or "").strip() == "recent_changes.timeline"


def _timeline_events(
    result,
    *,
    sort: str = "auto",
    dedupe: str = "auto",
    limit: int | None = None,
) -> list[dict[str, Any]]:
    payload = result.to_dict()
    if not _is_timeline_view(payload.get("view")):
        return []
    dedupe_mode = _normalize_dedupe(dedupe)
    by_key: dict[str, dict[str, Any]] = {}
    ordered: list[dict[str, Any]] = []
    for item in payload.get("items", []):
        for event in _events_from_item(item):
            key = _event_dedupe_key(event, mode=dedupe_mode)
            if key is not None and key in by_key:
                existing = by_key[key]
                if float(event.get("score") or 0.0) > float(
                    existing.get("score") or 0.0
                ):
                    existing.update(event)
                continue
            ordered.append(event)
            if key is not None:
                by_key[key] = event
    events = _sort_events(ordered, sort=sort)
    return events[:limit] if limit is not None and limit >= 0 else events


def _events_from_item(item: Mapping[str, Any]) -> list[dict[str, Any]]:
    payload = dict(item)
    relations = payload.get("relations")
    item_score = float(payload.get("score") or 0.0)
    if isinstance(relations, list):
        return [
            _event_from_relation(rel, item_score=item_score)
            for rel in relations
            if isinstance(rel, Mapping) and _relation_has_timeline_fact(rel)
        ]
    claim = payload.get("claim") if isinstance(payload.get("claim"), Mapping) else {}
    if claim.get("source_refs") or payload.get("source_refs") or payload.get("summary"):
        flat_payload = {**claim, **payload}
        return [_event_from_flat_payload(flat_payload, item_score=item_score)]
    return []


def _relation_has_timeline_fact(rel: Mapping[str, Any]) -> bool:
    return bool(
        rel.get("fact") or rel.get("source_refs") or _activity_key_from_relation(rel)
    )


def _event_from_relation(
    rel: Mapping[str, Any], *, item_score: float
) -> dict[str, Any]:
    fact = _str(rel.get("fact"))
    source_refs = _string_list(rel.get("source_refs"))
    related_entity = (
        rel.get("related_entity")
        if isinstance(rel.get("related_entity"), Mapping)
        else {}
    )
    return {
        "activity_key": _activity_key_from_relation(rel),
        "occurred_at": _event_occurred_at(rel, fact=fact),
        "source_refs": source_refs,
        "fact": fact,
        "predicate": _str(rel.get("predicate")),
        "actor_key": _actor_key_from_relation(rel),
        "target_key": _target_key_from_relation(rel),
        "related_key": _str(rel.get("related_key")),
        "related_name": _str(related_entity.get("name")),
        "truth": _str(rel.get("truth")),
        "evidence_strength": _str(rel.get("evidence_strength")),
        "source_system": _str(rel.get("source_system")),
        "score": float(rel.get("score") or item_score or 0.0),
    }


def _event_from_flat_payload(
    payload: Mapping[str, Any], *, item_score: float
) -> dict[str, Any]:
    fact = _str(payload.get("fact") or payload.get("summary"))
    return {
        "activity_key": _str(payload.get("activity_key")),
        "occurred_at": _event_occurred_at(payload, fact=fact),
        "source_refs": _string_list(payload.get("source_refs")),
        "fact": fact,
        "predicate": _str(payload.get("predicate")),
        "actor_key": None,
        "target_key": _str(payload.get("object_key")),
        "related_key": _str(payload.get("object_key")),
        "related_name": None,
        "truth": _str(payload.get("truth")),
        "evidence_strength": _str(payload.get("evidence_strength")),
        "source_system": _str(payload.get("source_system")),
        "score": float(item_score or 0.0),
    }


def _activity_key_from_relation(rel: Mapping[str, Any]) -> str | None:
    predicate = _str(rel.get("predicate")).upper()
    from_key = _str(rel.get("from_key"))
    to_key = _str(rel.get("to_key"))
    if predicate in {"TOUCHED", "MENTIONS"}:
        return from_key
    if predicate in {"PERFORMED", "AUTHORED"}:
        return to_key
    return from_key if from_key.startswith("activity:") else to_key


def _actor_key_from_relation(rel: Mapping[str, Any]) -> str | None:
    predicate = _str(rel.get("predicate")).upper()
    if predicate in {"PERFORMED", "AUTHORED"}:
        return _str(rel.get("from_key")) or None
    return None


def _target_key_from_relation(rel: Mapping[str, Any]) -> str | None:
    predicate = _str(rel.get("predicate")).upper()
    if predicate in {"TOUCHED", "MENTIONS"}:
        return _str(rel.get("to_key")) or None
    return None


def _event_occurred_at(
    payload: Mapping[str, Any], *, fact: str | None = None
) -> str | None:
    props = (
        payload.get("properties")
        if isinstance(payload.get("properties"), Mapping)
        else {}
    )
    for value in (
        payload.get("occurred_at"),
        props.get("occurred_at"),
    ):
        if isinstance(value, str) and value.strip():
            return value.strip()
    if fact:
        m = re.search(r"\bon (\d{4}-\d{2}-\d{2})\b", fact)
        if m:
            return m.group(1)
    value = payload.get("valid_at")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _normalize_dedupe(value: str) -> str:
    mode = (value or "auto").strip().lower()
    if mode == "auto":
        return "source_ref"
    if mode not in {"none", "source_ref", "activity"}:
        raise ValueError("--dedupe must be one of: auto, none, source_ref, activity")
    return mode


def _event_dedupe_key(event: Mapping[str, Any], *, mode: str) -> str | None:
    if mode == "none":
        return None
    if mode == "activity":
        return _str(event.get("activity_key")) or None
    refs = _string_list(event.get("source_refs"))
    if refs:
        return "source_ref:" + "|".join(sorted(refs))
    return _str(event.get("activity_key")) or _str(event.get("fact")) or None


def _sort_events(events: list[dict[str, Any]], *, sort: str) -> list[dict[str, Any]]:
    mode = (sort or "auto").strip().lower()
    if mode == "auto":
        mode = "occurred_at"
    if mode not in {"score", "occurred_at"}:
        raise ValueError("--sort must be one of: auto, score, occurred_at")
    if mode == "score":
        return sorted(events, key=lambda e: float(e.get("score") or 0.0), reverse=True)
    return sorted(
        events,
        key=lambda e: (
            _parse_sort_dt(e.get("occurred_at")),
            float(e.get("score") or 0.0),
        ),
        reverse=True,
    )


def _timeline_freshness(events: list[Mapping[str, Any]]) -> dict[str, Any]:
    dates = [e.get("occurred_at") for e in events if e.get("occurred_at")]
    return {
        "latest_event_at": max(dates) if dates else None,
        "source_refs_count": len(
            {ref for e in events for ref in _string_list(e.get("source_refs"))}
        ),
        "local_worktree_included": False,
        "note": "Timeline reads recorded graph events for the whole pot/project across repo sources; uncommitted local changes are not included unless recorded.",
    }


def _timeline_human(
    result, *, sort: str, dedupe: str, event_limit: int | None = None
) -> str:
    events = _timeline_events(result, sort=sort, dedupe=dedupe, limit=event_limit)
    payload = result.to_dict()
    lines = [
        f"view={payload.get('view')} events={len(events)} "
        f"quality={payload.get('quality', {}).get('status')}",
        "scope=project-wide pot timeline across registered repo sources; local uncommitted worktree is not included",
    ]
    for event in events[:20]:
        refs = ", ".join(_string_list(event.get("source_refs"))) or "no-source-ref"
        when = event.get("occurred_at") or "unknown-date"
        fact = event.get("fact") or event.get("activity_key") or "(no fact)"
        lines.append(f"  • {when} [{refs}] {fact}")
    return "\n".join(lines)


def _resolve_time_bounds(
    *, since: str | None, until: str | None, window: str | None
) -> tuple[datetime | None, datetime | None]:
    until_dt = _parse_instant(until) if until else None
    since_dt = _parse_instant(since) if since else None
    if since_dt is not None:
        return since_dt, until_dt
    if window:
        end = until_dt or datetime.now(timezone.utc)
        return end - _parse_duration(window), until_dt
    return None, until_dt


def _parse_instant(value: str) -> datetime:
    raw = value.strip()
    if not raw:
        raise ValueError("timestamp must be non-empty")
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise ValueError(f"invalid timestamp {value!r}; expected ISO 8601") from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _parse_duration(value: str) -> timedelta:
    m = re.fullmatch(r"\s*(\d+)\s*([mhdw])\s*", value.strip().lower())
    if not m:
        raise ValueError("--time-window must look like 30m, 24h, 7d, or 2w")
    amount = int(m.group(1))
    unit = m.group(2)
    if unit == "m":
        return timedelta(minutes=amount)
    if unit == "h":
        return timedelta(hours=amount)
    if unit == "d":
        return timedelta(days=amount)
    return timedelta(weeks=amount)


def _parse_ttl_seconds(value: str) -> int:
    try:
        ttl = _parse_duration(value)
    except ValueError as exc:
        raise ValueError("--ttl must look like 30m, 1h, 7d, or 2w") from exc
    seconds = int(ttl.total_seconds())
    if seconds <= 0:
        raise ValueError("--ttl must be positive")
    return seconds


def _parse_sort_dt(value: Any) -> datetime:
    if isinstance(value, str) and value.strip():
        raw = value.strip()
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw):
            raw = raw + "T00:00:00+00:00"
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(raw)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return datetime.min.replace(tzinfo=timezone.utc)


def _resolve_repo_scope(repo: str) -> str:
    value = repo.strip()
    if not value:
        raise ValueError("--repo must be non-empty")
    if value == "current":
        remote = _current_repo_remote_for_scope()
        if remote:
            return remote
        raise ValueError("--repo current requires a git remote.origin.url")
    return _normalize_repo_for_scope(value)


def _current_repo_remote_for_scope() -> str | None:
    import subprocess

    try:
        proc = subprocess.run(
            ["git", "config", "--get", "remote.origin.url"],
            capture_output=True,
            text=True,
            timeout=1,
            check=False,
        )
    except Exception:  # noqa: BLE001
        return None
    if proc.returncode != 0:
        return None
    return _normalize_repo_for_scope(proc.stdout.strip())


def _normalize_repo_for_scope(value: str) -> str:
    text = value.strip()
    if text.endswith(".git"):
        text = text[:-4]
    if text.startswith("git@"):
        text = text[4:].replace(":", "/", 1)
    elif "://" in text:
        from urllib.parse import urlparse

        parsed = urlparse(text)
        path = parsed.path.strip("/")
        if parsed.netloc and path:
            text = f"{parsed.netloc}/{path}"
    return text.strip("/").lower()


def _str(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, (list, tuple)):
        return [v for v in value if isinstance(v, str) and v]
    return []


def _nudge_human(result) -> str:
    if result.silent:
        return f"silent (event={result.event}): {result.detail or 'nothing to inject'}"
    lines = [f"event={result.event}"]
    if result.inject_context:
        lines.append(result.inject_context)
    if result.instruction:
        lines.append(f"instruction: {result.instruction}")
    if result.injected_keys:
        lines.append(f"(injected {len(result.injected_keys)} item(s))")
    return "\n".join(lines)


def _mutate_human(result) -> str:
    if result.would_apply is not None:
        head = (
            f"{result.status}: would_apply={result.would_apply} risk={result.risk} "
            f"accepted={result.operations_accepted} preview={dict(result.preview or {})}"
        )
    else:
        head = (
            f"{result.status}: risk={result.risk} "
            f"auto_committed={result.auto_committed} "
            f"applied={result.operations_applied} {dict(result.mutations_applied)}"
        )
    lines = [head]
    for issue in result.issues:
        marker = "error" if issue.is_error else "warn"
        lines.append(f"  [{marker}] {issue.code}: {issue.message}")
    return "\n".join(lines)


def _proposal_human(result) -> str:
    lines = [
        (
            f"{result.status}: plan_id={result.plan_id} risk={result.risk} "
            f"auto_applicable={result.auto_applicable}"
        )
    ]
    if result.diff:
        lines.append(f"diff: {result.diff.to_dict()}")
    for issue in getattr(result, "issues", ()):
        code = issue.get("code") if isinstance(issue, Mapping) else None
        message = issue.get("message") if isinstance(issue, Mapping) else None
        if code or message:
            lines.append(f"  [issue] {code}: {message}")
    return "\n".join(lines)


def _commit_human(result) -> str:
    head = f"{result.status}: plan_id={result.plan_id} risk={result.risk}"
    if result.mutation_id:
        head += f" mutation_id={result.mutation_id}"
    lines = [head]
    if result.detail:
        lines.append(result.detail)
    return "\n".join(lines)


def _history_human(result) -> str:
    payload = result.to_dict()
    entries = payload.get("entries", [])
    lines = [f"history: entries={len(entries)} filters={payload.get('filters', {})}"]
    detail = payload.get("detail")
    if detail:
        lines.append(str(detail))
    for entry in entries[:10]:
        when = entry.get("occurred_at") or "unknown-time"
        kind = entry.get("kind") or "entry"
        status = entry.get("status") or "-"
        summary = entry.get("summary") or entry.get("id")
        lines.append(f"  • {when} [{kind}:{status}] {summary}")
    return "\n".join(lines)


def _inbox_human(result) -> str:
    payload = result.to_dict()
    if payload.get("item"):
        item = payload["item"]
        head = (
            f"inbox {payload.get('action')}: "
            f"{item.get('item_id')} status={item.get('status')}"
        )
        detail = payload.get("detail")
        return "\n".join([head, str(detail)] if detail else [head])
    items = payload.get("items", [])
    lines = [f"inbox {payload.get('action')}: items={len(items)}"]
    detail = payload.get("detail")
    if detail:
        lines.append(str(detail))
    for item in items[:10]:
        lines.append(
            f"  {item.get('item_id')} [{item.get('status')}] {item.get('summary')}"
        )
    return "\n".join(lines)


def _quality_human(result) -> str:
    payload = result.to_dict()
    findings = payload.get("findings", [])
    lines = [
        f"quality {payload.get('report')}: status={payload.get('status')} findings={len(findings)}"
    ]
    detail = payload.get("detail")
    if detail:
        lines.append(str(detail))
    for finding in findings[:10]:
        lines.append(
            f"  {finding.get('finding_id')} [{finding.get('severity')}] "
            f"{finding.get('summary')}"
        )
    return "\n".join(lines)


__all__ = ["backend_app", "graph_app", "timeline_app"]
