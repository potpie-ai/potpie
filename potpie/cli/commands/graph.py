"""Graph + backend commands → ``HostShell.graph`` and the active ``GraphBackend``.

CLI code never touches a store directly; everything goes through the capability
ports. Unbuilt projections (semantic/inspection/analytics/snapshot on profiles
that lack them) surface as the structured not-implemented contract.
"""

from __future__ import annotations

import json
from typing import Any

import typer
from application.services.graph_workbench import (
    graph_success_envelope,
    normalize_catalog_result,
)
from domain.errors import CapabilityNotImplemented
from domain.nudge import NUDGE_EVENT_HELP

from potpie.cli.commands._common import (
    EXIT_VALIDATION,
    contract,
    emit,
    empty_pot_warnings,
    fail,
    get_host,
    pot_scope_human,
    resolve_pot_id,
)
from potpie.cli.commands.graph_bulk import (
    _bulk_commit_summary,
    _bulk_chunk_entry,
    _bulk_human,
    _bulk_issues_from_proposal,
    _bulk_next_action,
    _bulk_proposal_summary,
    _bulk_run_status,
    _build_bulk_chunks,
    _bulk_skipped_chunk,
    _load_bulk_mutation_payloads,
    _load_json,
    _new_bulk_run_payload,
    _write_bulk_manifest,
)
from potpie.cli.commands.graph_common import (
    _parse_created_by,
    _parse_predicates,
    _parse_scope,
    _parse_ttl_seconds,
    _resolve_repo_scope,
    _safe,
)
from potpie.cli.commands.graph_read import (
    _effective_requested_format,
    _emit_read,
    _is_timeline_view,
    _resolve_time_bounds,
    _service_limit_for_read,
    _with_read_context,
)
from potpie.cli.commands.graph_runtime import (
    _emit_graph_read,
    _emit_graph_result,
    _emit_inbox_result,
    _emit_quality_result,
    _graph_command,
    _legacy_warning,
)
from potpie.cli.commands.graph_templates import _MUTATION_TEMPLATES
from potpie.cli.commands.graph_render import (
    _catalog_human,
    _catalog_payload_for_profile,
    _commit_human,
    _data_plane_status_payload,
    _describe_human,
    _graph_status_payload,
    _history_human,
    _neighborhood_human,
    _neighborhood_relation,
    _nudge_human,
    _proposal_human,
)

graph_app = typer.Typer(help="Graph reads/admin via capability ports.")
inbox_app = typer.Typer(help="Pending graph-work inbox.")
quality_app = typer.Typer(help="Read-only graph quality reports.")
bulk_app = typer.Typer(help="Chunked semantic graph mutation application.")
backend_app = typer.Typer(help="GraphBackend profile selection + health.")
timeline_app = typer.Typer(help="Timeline reads over the active project pot.")

graph_app.add_typer(inbox_app, name="inbox")
graph_app.add_typer(quality_app, name="quality")
graph_app.add_typer(bulk_app, name="bulk")


















# --- Graph Surface Lite (V1.5) ----------------------------------------------


@graph_app.command("catalog")
def graph_catalog(
    task: str = typer.Option(None, "--task", help="(accepted, ignored in V1.5)"),
    subgraph: str = typer.Option(None, "--subgraph"),
    profile: str = typer.Option(
        "full",
        "--profile",
        help="full | read",
    ),
    format_: str = typer.Option(
        "auto",
        "--format",
        help="auto | table",
    ),
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
        payload = _catalog_payload_for_profile(payload, profile=profile)
        human = _catalog_human(payload, format_=format_)
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
    source_ref: list[str] | None = typer.Option(
        None,
        "--source-ref",
        help="Exact claim source ref such as github:owner/repo#issue/123.",
    ),
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
    detail: str = typer.Option(
        "compact",
        "--detail",
        help="compact | full",
    ),
    relations: str = typer.Option(
        "summary",
        "--relations",
        help="summary | full",
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
                source_refs=tuple(source_ref or ()),
                since=since_dt,
                until=until_dt,
                depth=depth,
                direction=direction,
                limit=read_limit,
                detail=detail,
                relations=relations,
                freshness_preference=(
                    "fresh"
                    if _is_timeline_view(f"{subgraph}.{view}") and not query
                    else "balanced"
                ),
            )
        )
        _emit_graph_read(
            ctx,
            result,
            format_=format_,
            sort=sort,
            dedupe=dedupe,
            event_limit=limit,
            human_prefix=pot_scope_human(host, pot_id),
            warnings=empty_pot_warnings(host, pot_id),
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
    detail: str = typer.Option(
        "compact",
        "--detail",
        help="compact | full",
    ),
    relations: str = typer.Option(
        "summary",
        "--relations",
        help="summary | full",
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
                detail=detail,
                relations=relations,
                freshness_preference="fresh" if not query else "balanced",
            )
        )
        _emit_read(
            result,
            format_=format_,
            sort="occurred_at",
            dedupe="source_ref",
            event_limit=limit,
            human_prefix=pot_scope_human(host, pot_id),
            warnings=empty_pot_warnings(host, pot_id),
        )


@graph_app.command("search-entities")
def graph_search_entities(
    query_arg: str = typer.Argument(None, help="text to match entities/claims against"),
    query: str = typer.Option(
        None, "--query", help="text to match entities/claims against"
    ),
    type_: str = typer.Option(None, "--type", help="entity label filter, e.g. Service"),
    predicate: str = typer.Option(None, "--predicate"),
    subgraph: str = typer.Option(None, "--subgraph"),
    scope: str = typer.Option(None, "--scope", help="key:value[,key:value]"),
    truth: str = typer.Option(None, "--truth"),
    source_system: str = typer.Option(None, "--source-system"),
    source_family: str = typer.Option(None, "--source-family"),
    since: str = typer.Option(None, "--since", help="ISO instant lower bound."),
    until: str = typer.Option(None, "--until", help="ISO instant upper bound."),
    environment: str = typer.Option(None, "--environment"),
    external_id: str = typer.Option(None, "--external-id"),
    source_ref: list[str] | None = typer.Option(
        None,
        "--source-ref",
        help="Exact claim source ref such as github:owner/repo#issue/123.",
    ),
    limit: int = typer.Option(10, "--limit"),
    supporting_claims: int = typer.Option(
        0,
        "--supporting-claims",
        help="number of supporting claims per entity to include in JSON",
    ),
    pot: str = typer.Option(None, "--pot"),
) -> None:
    """Narrow entity/claim lookup for identity resolution before a write."""
    from domain.ports.services.graph_service import GraphEntitySearchRequest

    with _graph_command("graph.search-entities") as ctx:
        effective_query = query or query_arg
        if not effective_query:
            raise ValueError("query is required")
        if supporting_claims < 0:
            raise ValueError("--supporting-claims must be >= 0")
        host = get_host()
        pot_id = resolve_pot_id(host, pot)
        ctx.set_pot_id(pot_id)
        since_dt, until_dt = _resolve_time_bounds(since=since, until=until, window=None)
        result = host.graph.search_entities(
            GraphEntitySearchRequest(
                pot_id=pot_id,
                query=effective_query,
                type=type_,
                predicate=predicate,
                subgraph=subgraph,
                scope=_parse_scope(scope),
                truth=truth,
                source_system=source_system,
                source_family=source_family,
                since=since_dt,
                until=until_dt,
                environment=environment,
                external_id=external_id,
                source_refs=tuple(source_ref or ()),
                limit=limit,
                supporting_claims=supporting_claims,
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
        warnings = empty_pot_warnings(host, pot_id)
        _emit_graph_result(
            ctx,
            payload,
            human=_with_read_context(
                human,
                human_prefix=pot_scope_human(host, pot_id),
                warnings=(),
            ),
            warnings=warnings,
            recommended_next_action=warnings[0] if warnings else None,
        )


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
        warnings = empty_pot_warnings(host, pot_id)
        recommended = None
        if not dp.backend_ready:
            recommended = (
                "Run `potpie backend doctor` to inspect graph backend readiness "
                "and capability-specific failures."
            )
        elif warnings:
            recommended = warnings[0]
        elif payload.get("health_status") not in {None, "ok"}:
            recommended = (
                payload.get("quality", {}).get("recommended_next_action")
                or "Run `potpie graph quality summary --json` to inspect graph health."
            )
        _emit_graph_result(
            ctx,
            payload,
            human=(
                f"{pot_scope_human(host, pot_id)}\n"
                f"backend={dp.backend_profile} ready={dp.backend_ready} "
                f"counts={dict(dp.counts)} health={payload.get('health_status')}"
            ),
            warnings=warnings,
            recommended_next_action=recommended,
        )


@graph_app.command("describe")
def graph_describe(
    subgraph: str = typer.Argument(None),
    view: str = typer.Option(None, "--view"),
    examples: bool = typer.Option(False, "--examples"),
    pot: str = typer.Option(None, "--pot"),
) -> None:
    from domain.ports.services.graph_service import GraphDescribeRequest

    with _graph_command("graph.describe") as ctx:
        _set_optional_pot(ctx, pot)
        payload = get_host().graph.describe(
            GraphDescribeRequest(
                subgraph=subgraph,
                view=view,
                include_examples=examples,
            )
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
    entity: str = typer.Option(..., "--entity"),
    predicate: str = typer.Option(None, "--predicate"),
    depth: int = typer.Option(2, "--depth"),
    direction: str = typer.Option("both", "--direction"),
    limit: int = typer.Option(50, "--limit"),
    detail: str = typer.Option("summary", "--detail", help="summary | full"),
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
        detail_mode = (detail or "summary").strip().lower()
        if detail_mode not in {"summary", "full"}:
            raise ValueError("--detail must be one of: summary, full")
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
        relations = [_neighborhood_relation(edge) for edge in sl.edges]
        payload = {
            "entity_key": entity,
            "depth": depth,
            "direction": normalized_direction,
            "predicates": list(predicates),
            "detail": detail_mode,
            "node_count": len(sl.nodes),
            "relation_count": len(relations),
            "relations": relations,
            "truncated": sl.truncated,
        }
        if detail_mode == "full":
            payload["nodes"] = [
                {
                    "key": n.key,
                    "labels": list(n.labels),
                    "properties": dict(n.properties),
                }
                for n in sl.nodes
            ]
            payload["edges"] = [
                {
                    "predicate": e.predicate,
                    "from": e.from_key,
                    "to": e.to_key,
                    "properties": dict(e.properties),
                }
                for e in sl.edges
            ]
        _emit_graph_result(
            ctx,
            payload,
            human=_neighborhood_human(payload),
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
    verify: bool = typer.Option(
        False,
        "--verify",
        help="read back committed claim keys and run post-commit quality checks",
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
            verify=verify,
        )
        _emit_graph_result(
            ctx,
            result.to_dict(),
            human=_commit_human(result),
        )
        if not result.ok:
            raise typer.Exit(code=EXIT_VALIDATION)
        if verify and result.verification is not None and not result.verification.ok:
            raise typer.Exit(code=EXIT_VALIDATION)


@bulk_app.command("apply")
def graph_bulk_apply(
    file: str = typer.Option(
        None,
        "--file",
        help="mutation JSON/NDJSON path; omit to read stdin",
    ),
    chunk_size: int = typer.Option(
        100,
        "--chunk-size",
        help="semantic operations per proposed chunk",
    ),
    start_chunk: int = typer.Option(
        1,
        "--start-chunk",
        help="1-based chunk index to start from when resuming",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="propose chunks but do not commit them",
    ),
    continue_on_error: bool = typer.Option(
        False,
        "--continue-on-error",
        help="attempt remaining chunks after a failed proposal or commit",
    ),
    verify: bool = typer.Option(
        False,
        "--verify",
        help="include graph data-plane status after the run",
    ),
    manifest: str = typer.Option(
        None,
        "--manifest",
        help="write a JSON run manifest after each attempted chunk",
    ),
    idempotency_key: str = typer.Option(
        None,
        "--idempotency-key",
        help="base idempotency key; chunk suffixes are added when needed",
    ),
    ttl: str = typer.Option("1h", "--ttl", help="plan expiry such as 30m, 1h, 2d"),
    approved_by: str = typer.Option(
        None, "--approved-by", help="user-ref for review-required chunks"
    ),
    pot: str = typer.Option(None, "--pot"),
) -> None:
    """Apply many semantic mutations through propose/commit chunks.

    This is an orchestration helper for agent-authored semantic mutations. It
    does not scan sources or infer facts; it keeps high-volume writes on the
    same validated workbench path as ordinary graph updates.
    """
    with _graph_command("graph.bulk.apply") as ctx:
        if chunk_size < 1:
            raise ValueError("--chunk-size must be >= 1")
        if start_chunk < 1:
            raise ValueError("--start-chunk must be >= 1")

        host = get_host()
        pot_id = resolve_pot_id(host, pot)
        ctx.set_pot_id(pot_id)
        ttl_seconds = _parse_ttl_seconds(ttl)
        source_payloads = _load_bulk_mutation_payloads(file)
        chunks = _build_bulk_chunks(
            source_payloads,
            chunk_size=chunk_size,
            idempotency_key=idempotency_key,
        )
        if start_chunk > len(chunks):
            raise ValueError(
                f"--start-chunk {start_chunk} is beyond {len(chunks)} chunks"
            )

        run = _new_bulk_run_payload(
            pot_id=pot_id,
            chunks_total=len(chunks),
            operations_total=sum(len(chunk["operations"]) for chunk in chunks),
            chunk_size=chunk_size,
            dry_run=dry_run,
            start_chunk=start_chunk,
            manifest=manifest,
        )
        ok = True

        for chunk in chunks:
            index = int(chunk["index"])
            if index < start_chunk:
                run["chunks"].append(_bulk_skipped_chunk(chunk))
                continue

            entry = _bulk_chunk_entry(chunk)
            run["chunks_attempted"] += 1
            run["operations_attempted"] += entry["operation_count"]
            proposal = host.graph_workbench.propose(
                chunk["payload"],
                pot_id=pot_id,
                ttl_seconds=ttl_seconds,
            )
            entry["proposal"] = _bulk_proposal_summary(proposal)
            entry["plan_id"] = proposal.plan_id

            if not proposal.ok:
                ok = False
                entry["ok"] = False
                entry["status"] = "proposal_failed"
                run["issues"].extend(_bulk_issues_from_proposal(proposal, index))
                run["chunks"].append(entry)
                _write_bulk_manifest(manifest, run)
                if not continue_on_error:
                    break
                continue

            if dry_run:
                entry["ok"] = True
                entry["status"] = proposal.status
                run["chunks_validated"] += 1
                run["operations_validated"] += entry["operation_count"]
                run["chunks"].append(entry)
                _write_bulk_manifest(manifest, run)
                continue

            if proposal.status == "review_required" and not approved_by:
                ok = False
                entry["ok"] = False
                entry["status"] = "review_required"
                run["issues"].append(
                    {
                        "code": "approval_required",
                        "message": (
                            f"chunk {index} requires approval; rerun with "
                            "--approved-by <user-ref> or inspect the plan"
                        ),
                        "severity": "error",
                        "chunk": index,
                    }
                )
                run["chunks"].append(entry)
                _write_bulk_manifest(manifest, run)
                if not continue_on_error:
                    break
                continue

            commit = host.graph_workbench.commit(
                proposal.plan_id,
                pot_id=pot_id,
                approved_by=approved_by,
            )
            entry["commit"] = _bulk_commit_summary(commit)
            entry["ok"] = bool(commit.ok)
            entry["status"] = commit.status
            if commit.ok:
                run["chunks_committed"] += 1
                run["operations_committed"] += entry["operation_count"]
                if commit.mutation_id:
                    entry["mutation_id"] = commit.mutation_id
            else:
                ok = False
                run["issues"].append(
                    {
                        "code": str(commit.status or "commit_failed"),
                        "message": commit.detail or "chunk commit failed",
                        "severity": "error",
                        "chunk": index,
                    }
                )
            run["chunks"].append(entry)
            _write_bulk_manifest(manifest, run)
            if not commit.ok and not continue_on_error:
                break

        run["ok"] = ok
        run["status"] = _bulk_run_status(run, dry_run=dry_run, ok=ok)
        if verify:
            status = host.graph.data_plane_status(pot_id)
            run["verification"] = _data_plane_status_payload(status)

        _write_bulk_manifest(manifest, run)
        _emit_graph_result(
            ctx,
            run,
            human=_bulk_human(run),
            recommended_next_action=_bulk_next_action(run),
        )
        if not ok:
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
        since_dt, until_dt = _resolve_time_bounds(since=since, until=until, window=None)
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
        since_dt, until_dt = _resolve_time_bounds(since=since, until=until, window=None)
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
    entity_key: str = typer.Argument(...),
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
    file: str = typer.Argument(...), pot: str = typer.Option(None, "--pot")
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
    file: str = typer.Argument(...), pot: str = typer.Option(None, "--pot")
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


def _set_optional_pot(ctx: Any, pot: str | None) -> None:
    host = get_host()
    if pot:
        ctx.set_pot_id(resolve_pot_id(host, pot))
        return
    active = _safe(lambda: host.pots.active_pot(), None)
    if active is not None:
        ctx.set_pot_id(getattr(active, "pot_id", None))












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










































































































































__all__ = ["backend_app", "graph_app", "timeline_app"]
