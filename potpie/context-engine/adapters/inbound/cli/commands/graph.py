"""Graph + backend commands → ``HostShell.graph`` and the active ``GraphBackend``.

CLI code never touches a store directly; everything goes through the capability
ports. Unbuilt projections (semantic/inspection/analytics/snapshot on profiles
that lack them) surface as the structured not-implemented contract.
"""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from typing import Any

import typer

from adapters.inbound.cli.commands._common import (
    EXIT_VALIDATION,
    contract,
    emit,
    fail,
    get_host,
    resolve_pot_id,
)
from domain.errors import CapabilityNotImplemented
from domain.nudge import NUDGE_EVENT_HELP

graph_app = typer.Typer(help="Graph reads/admin via capability ports.")
backend_app = typer.Typer(help="GraphBackend profile selection + health.")
timeline_app = typer.Typer(help="Timeline reads over the active project pot.")


# --- Graph Surface Lite (V1.5) ----------------------------------------------


@graph_app.command("catalog")
def graph_catalog(
    task: str = typer.Option(None, "--task", help="(accepted, ignored in V1.5)"),
    subgraph: str = typer.Option(None, "--subgraph"),
    pot: str = typer.Option(None, "--pot"),
) -> None:
    """Discover the graph contract: versions, views, mutation ops, ontology."""
    from domain.ports.services.graph_service import GraphCatalogRequest

    with contract():
        host = get_host()
        pot_id = resolve_pot_id(host, pot)
        result = host.graph.catalog(
            GraphCatalogRequest(pot_id=pot_id, task=task, subgraph=subgraph)
        )
        payload = result.to_dict()
        human = (
            f"graph contract {payload['graph_contract_version']} / ontology "
            f"{payload['ontology_version']} (match={payload['match_mode']})\n"
            f"commands: {', '.join(payload['commands'])}\n"
            f"views: {', '.join(v['name'] for v in payload['views'])}\n"
            f"mutation ops: {', '.join(payload['mutation_operations'])}\n"
            f"review-required: {', '.join(payload['review_required_operations'])}\n"
            f"deferred: {', '.join(payload['deferred_operations'])}"
        )
        emit(payload, human=human)


@graph_app.command("read")
def graph_read(
    view: str = typer.Option(
        ..., "--view", help="<subgraph>.<view>, e.g. bugs.prior_occurrences"
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

    with contract():
        host = get_host()
        pot_id = resolve_pot_id(host, pot)
        del current  # pot resolution already considers the current working tree.
        since_dt, until_dt = _resolve_time_bounds(
            since=since, until=until, window=time_window
        )
        parsed_scope = _parse_scope(scope)
        if repo:
            parsed_scope["repo"] = _resolve_repo_scope(repo)
        effective_format = _effective_requested_format(view=view, requested=format_)
        read_limit = _service_limit_for_read(
            view=view, format_=effective_format, requested_limit=limit
        )
        env = host.graph.read(
            GraphReadRequest(
                pot_id=pot_id,
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
                if _is_timeline_view(view) and not query
                else "balanced",
            )
        )
        _emit_read(env, format_=format_, sort=sort, dedupe=dedupe, event_limit=limit)


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
            view="recent_changes.timeline", format_="events", requested_limit=limit
        )
        env = host.graph.read(
            GraphReadRequest(
                pot_id=pot_id,
                view="recent_changes.timeline",
                query=query,
                scope=scope,
                since=since_dt,
                until=until_dt,
                limit=read_limit,
                freshness_preference="fresh" if not query else "balanced",
            )
        )
        _emit_read(
            env,
            format_=format_,
            sort="occurred_at",
            dedupe="source_ref",
            event_limit=limit,
        )


@graph_app.command("search-entities")
def graph_search_entities(
    query: str = typer.Argument(..., help="text to match entities/claims against"),
    type_: str = typer.Option(None, "--type", help="entity label filter, e.g. Service"),
    predicate: str = typer.Option(None, "--predicate"),
    environment: str = typer.Option(None, "--environment"),
    limit: int = typer.Option(10, "--limit"),
    pot: str = typer.Option(None, "--pot"),
) -> None:
    """Narrow entity/claim lookup for identity resolution before a write."""
    from domain.ports.services.graph_service import GraphEntitySearchRequest

    with contract():
        host = get_host()
        pot_id = resolve_pot_id(host, pot)
        result = host.graph.search_entities(
            GraphEntitySearchRequest(
                pot_id=pot_id,
                query=query,
                type=type_,
                predicate=predicate,
                environment=environment,
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
        emit(payload, human=human)


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
    """Validate and directly apply semantic graph mutations."""
    from domain.semantic_mutations import (
        SemanticMutationParseError,
        SemanticMutationRequest,
    )

    with contract():
        host = get_host()
        pot_id = resolve_pot_id(host, pot)
        payload = _load_json(file)
        try:
            request = SemanticMutationRequest.parse(
                payload,
                pot_id=pot_id,
                dry_run=dry_run,
                allow_review_required=allow_review_required,
                approved_by=approved_by,
            )
        except SemanticMutationParseError as exc:
            fail(code="invalid_mutation_payload", message=str(exc))
            return
        result = host.graph.mutate(request)
        emit(result.to_dict(), human=_mutate_human(result))
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
    """Print a schema-only mutation skeleton for `graph mutate`.

    Pure schema helper: emits placeholders for the harness to fill from
    sources it has actually read. It never inspects the repository or infers
    graph facts.
    """
    with contract():
        template = _MUTATION_TEMPLATES.get(kind.strip().lower())
        if template is None:
            fail(
                code="unknown_template_kind",
                message=f"Unknown mutation template kind {kind!r}.",
                next_action=f"pick one of: {', '.join(sorted(_MUTATION_TEMPLATES))}",
            )
        rendered = json.dumps(template, indent=2)
        emit(
            {"kind": kind, "template": template},
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

    with contract():
        host = get_host()
        pot_id = resolve_pot_id(host, pot)
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
        emit(result.to_dict(), human=_nudge_human(result))


@graph_app.command("status")
def graph_status(pot: str = typer.Option(None, "--pot")) -> None:
    with contract():
        host = get_host()
        pot_id = resolve_pot_id(host, pot)
        dp = host.graph.data_plane_status(pot_id)
        emit(
            {
                "backend": dp.backend_profile,
                "ready": dp.backend_ready,
                "counts": dict(dp.counts),
                "freshness": dict(dp.freshness),
                "quality": dict(dp.quality),
            },
            human=f"backend={dp.backend_profile} ready={dp.backend_ready} counts={dict(dp.counts)}",
        )


@graph_app.command("inspect")
def graph_inspect(
    entity_key: str,
    depth: int = typer.Option(2, "--depth"),
    pot: str = typer.Option(None, "--pot"),
) -> None:
    with contract():
        host = get_host()
        _require_backend_capability(
            host,
            capability="inspection",
            method="neighborhood",
            command="graph inspect",
        )
        pot_id = resolve_pot_id(host, pot)
        sl = host.backend.inspection.neighborhood(
            pot_id=pot_id, entity_key=entity_key, depth=depth
        )
        emit(
            {
                "nodes": [{"key": n.key, "labels": list(n.labels)} for n in sl.nodes],
                "edges": [
                    {"predicate": e.predicate, "from": e.from_key, "to": e.to_key}
                    for e in sl.edges
                ],
            },
            human=f"{len(sl.nodes)} nodes, {len(sl.edges)} edges around {entity_key}",
        )


@graph_app.command("export")
def graph_export(file: str, pot: str = typer.Option(None, "--pot")) -> None:
    with contract():
        host = get_host()
        _require_backend_capability(
            host,
            capability="snapshot",
            method="export",
            command="graph export",
        )
        pot_id = resolve_pot_id(host, pot)
        manifest = host.backend.snapshot.export(pot_id=pot_id, destination=file)
        emit(
            {"location": manifest.location, "claims": manifest.claim_count},
            human=f"exported {manifest.claim_count} claims → {manifest.location}",
        )


@graph_app.command("import")
def graph_import(file: str, pot: str = typer.Option(None, "--pot")) -> None:
    with contract():
        host = get_host()
        _require_backend_capability(
            host,
            capability="snapshot",
            method="import_",
            command="graph import",
        )
        pot_id = resolve_pot_id(host, pot)
        manifest = host.backend.snapshot.import_(pot_id=pot_id, source=file)
        emit(
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
    with contract():
        host = get_host()
        pot_id = resolve_pot_id(host, pot)
        targets = []
        if not all_:
            if semantic_index:
                targets.append("semantic_index")
            if entity_summaries:
                targets.append("entity_summaries")
        report = host.backend.analytics.repair(pot_id, targets=targets)
        emit(
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


def _emit_read(
    env, *, format_: str, sort: str, dedupe: str, event_limit: int | None = None
) -> None:
    normalized_format = _effective_read_format(env, format_)
    if normalized_format == "jsonl":
        rows = _timeline_events(env, sort=sort, dedupe=dedupe, limit=event_limit)
        if not rows:
            rows = _raw_item_rows(env)
        for row in rows:
            typer.echo(json.dumps(row, default=str))
        return
    payload = _read_payload(
        env,
        format_=normalized_format,
        sort=sort,
        dedupe=dedupe,
        event_limit=event_limit,
    )
    emit(
        payload,
        human=_read_human(
            env,
            format_=normalized_format,
            sort=sort,
            dedupe=dedupe,
            event_limit=event_limit,
        ),
    )


def _read_payload(
    env,
    *,
    format_: str = "raw",
    sort: str = "auto",
    dedupe: str = "auto",
    event_limit: int | None = None,
) -> dict:
    from domain.graph_contract import GRAPH_CONTRACT_VERSION, ONTOLOGY_VERSION

    envelope = env.to_dict()
    meta = dict(envelope.get("metadata", {}))
    payload = {
        "ok": True,
        "graph_contract_version": meta.get(
            "graph_contract_version", GRAPH_CONTRACT_VERSION
        ),
        "ontology_version": meta.get("ontology_version", ONTOLOGY_VERSION),
        "view": meta.get("view"),
        "subgraph": meta.get("subgraph"),
        "backed": meta.get("backed"),
        "match_mode": meta.get("match_mode"),
        "subgraph_versions": meta.get("subgraph_versions", {}),
        "inline_relations": meta.get("inline_relations", []),
        "read_shape": meta.get("read_shape", "flat_claims"),
        "inline_relation_count": meta.get("inline_relation_count", 0),
        "pot_id": envelope["pot_id"],
        "intent": envelope["intent"],
        "overall_confidence": envelope["overall_confidence"],
        "items": envelope["items"],
        "coverage": envelope["coverage"],
        "unsupported_includes": envelope["unsupported_includes"],
        "as_of": envelope["as_of"],
    }
    if format_ in ("events", "table"):
        events = _timeline_events(env, sort=sort, dedupe=dedupe, limit=event_limit)
        payload["read_shape"] = "events"
        payload["events"] = events
        payload["event_count"] = len(events)
        payload["freshness"] = _timeline_freshness(events)
    return payload


def _read_human(
    env,
    *,
    format_: str = "raw",
    sort: str = "auto",
    dedupe: str = "auto",
    event_limit: int | None = None,
) -> str:
    if format_ in ("events", "table"):
        return _timeline_human(env, sort=sort, dedupe=dedupe, event_limit=event_limit)
    meta = dict(env.metadata)
    lines = [
        f"view={meta.get('view')} backed={meta.get('backed')} "
        f"items={len(env.items)} confidence={env.overall_confidence}"
    ]
    for item in env.items[:10]:
        payload = dict(item.payload)
        fact = payload.get("fact") or payload.get("summary") or ""
        entity = payload.get("entity")
        if not fact and isinstance(entity, dict):
            fact = f"{entity.get('key')} relations={payload.get('relation_count', 0)}"
        lines.append(f"  • [{item.include}] {fact}")
    for unsup in env.unsupported_includes:
        lines.append(f"  ! {unsup.name}: {unsup.reason}")
    return "\n".join(lines)


def _raw_item_rows(env) -> list[dict[str, Any]]:
    return [
        {"include": i.include, "score": i.score, "payload": dict(i.payload)}
        for i in env.items
    ]


def _effective_read_format(env, requested: str) -> str:
    value = (requested or "auto").strip().lower()
    if value not in {"auto", "raw", "events", "table", "jsonl"}:
        raise ValueError("--format must be one of: auto, raw, events, table, jsonl")
    if value == "auto":
        return "events" if _is_timeline_view(dict(env.metadata).get("view")) else "raw"
    return value


def _effective_requested_format(*, view: str, requested: str) -> str:
    value = (requested or "auto").strip().lower()
    if value == "auto":
        return "events" if _is_timeline_view(view) else "raw"
    return value


def _service_limit_for_read(*, view: str, format_: str, requested_limit: int) -> int:
    if _is_timeline_view(view) and format_ in {"events", "table", "jsonl"}:
        return min(max(requested_limit * 8, 40), 200)
    return requested_limit


def _is_timeline_view(view: str | None) -> bool:
    return str(view or "").strip() == "recent_changes.timeline"


def _timeline_events(
    env,
    *,
    sort: str = "auto",
    dedupe: str = "auto",
    limit: int | None = None,
) -> list[dict[str, Any]]:
    meta = dict(env.metadata)
    if not _is_timeline_view(meta.get("view")):
        return []
    dedupe_mode = _normalize_dedupe(dedupe)
    by_key: dict[str, dict[str, Any]] = {}
    ordered: list[dict[str, Any]] = []
    for item in env.items:
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


def _events_from_item(item) -> list[dict[str, Any]]:
    payload = dict(item.payload)
    relations = payload.get("relations")
    if isinstance(relations, list):
        return [
            _event_from_relation(rel, item_score=item.score)
            for rel in relations
            if isinstance(rel, Mapping) and _relation_has_timeline_fact(rel)
        ]
    if payload.get("activity_key") or payload.get("fact") or payload.get("source_refs"):
        return [_event_from_flat_payload(payload, item_score=item.score)]
    return []


def _relation_has_timeline_fact(rel: Mapping[str, Any]) -> bool:
    return bool(
        rel.get("fact") or rel.get("source_refs") or _activity_key_from_relation(rel)
    )


def _event_from_relation(
    rel: Mapping[str, Any], *, item_score: float
) -> dict[str, Any]:
    claim = rel.get("claim") if isinstance(rel.get("claim"), Mapping) else {}
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
        "score": float(claim.get("score") or item_score or 0.0),
    }


def _event_from_flat_payload(
    payload: Mapping[str, Any], *, item_score: float
) -> dict[str, Any]:
    fact = _str(payload.get("fact"))
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
    env, *, sort: str, dedupe: str, event_limit: int | None = None
) -> str:
    events = _timeline_events(env, sort=sort, dedupe=dedupe, limit=event_limit)
    meta = dict(env.metadata)
    lines = [
        f"view={meta.get('view')} pot={env.pot_id} events={len(events)} confidence={env.overall_confidence}",
        "scope=project-wide pot timeline across registered repo sources; local uncommitted worktree is not included",
    ]
    for event in events[:20]:
        refs = ", ".join(_string_list(event.get("source_refs"))) or "no-source-ref"
        when = event.get("occurred_at") or "unknown-date"
        fact = event.get("fact") or event.get("activity_key") or "(no fact)"
        lines.append(f"  • {when} [{refs}] {fact}")
    for unsup in env.unsupported_includes:
        lines.append(f"  ! {unsup.name}: {unsup.reason}")
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


__all__ = ["backend_app", "graph_app", "timeline_app"]
