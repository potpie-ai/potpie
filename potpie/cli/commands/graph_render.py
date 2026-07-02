"""Human and JSON payload render helpers for graph CLI commands."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from domain.errors import CapabilityNotImplemented
from domain.graph_contract import GRAPH_CONTRACT_VERSION as DATA_PLANE_CONTRACT_VERSION
from domain.graph_contract import ONTOLOGY_VERSION
from domain.graph_workbench import GRAPH_WORKBENCH_COMMANDS, GraphWorkbenchStatus

from potpie.cli.commands._common import pot_scope_info
from potpie.cli.commands.graph_common import _safe, _str, _string_list

def _graph_status_payload(host: Any, pot_id: str, dp: Any) -> dict[str, Any]:
    caps = _safe(lambda: host.backend.capabilities(), None)
    implemented = list(caps.implemented()) if caps is not None else []
    pot_info = pot_scope_info(host, pot_id)
    quality_summary = _graph_status_quality_summary(host, pot_id, dp)
    health_status = str(quality_summary.get("health_status") or "unknown")
    status = GraphWorkbenchStatus(
        host={
            "kind": getattr(host, "profile", "local"),
            "liveness": "ok",
        },
        pot={
            "id": pot_id,
            "name": pot_info.get("name"),
            "source_count": pot_info.get("source_count", 0),
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
                (
                    "Run `potpie backend doctor` to inspect graph backend readiness "
                    "and capability-specific failures."
                )
                if not dp.backend_ready
                else None
            ),
            "implemented_capabilities": implemented,
            "counts": dict(dp.counts),
            "freshness": dict(dp.freshness),
        },
        ledger={"status": "not_inspected"},
        skills={"status": "not_inspected"},
        quality=quality_summary,
    )
    payload = status.to_dict()
    payload["health_status"] = health_status
    return payload


def _graph_status_quality_summary(
    host: Any, pot_id: str, dp: Any
) -> dict[str, Any]:
    fallback = _data_plane_quality_summary(dp)
    workbench = getattr(host, "graph_workbench", None)
    if workbench is None or not getattr(workbench, "quality", None):
        return fallback
    try:
        result = workbench.quality(
            pot_id=pot_id,
            report="summary",
            subgraph=None,
            limit=20,
            confidence_threshold=0.5,
        )
    except CapabilityNotImplemented as exc:
        fallback["health_status"] = "unavailable"
        fallback["status"] = "unavailable"
        fallback["detail"] = str(exc)
        return fallback
    except Exception as exc:
        fallback["health_status"] = "unavailable"
        fallback["status"] = "unavailable"
        fallback["detail"] = f"quality summary unavailable: {exc}"
        return fallback

    body = result.to_dict()
    metrics = dict(body.get("metrics") or {})
    reports = dict(metrics.get("quality_reports") or {})
    compact_reports = {
        name: {
            "status": report.get("status"),
            "finding_count": report.get("finding_count", 0),
            "severity_counts": dict(report.get("severity_counts") or {}),
        }
        for name, report in reports.items()
        if isinstance(report, Mapping)
    }
    counts_value = metrics.get("counts")
    counts = counts_value if isinstance(counts_value, Mapping) else {}
    out = {
        "status": body.get("status"),
        "health_status": body.get("status"),
        "source": "quality_summary",
        "claim_count": counts.get("claims", dict(dp.counts).get("claims")),
        "quality_counts": dict(metrics.get("quality_counts") or {}),
        "total_findings": metrics.get("total_findings", body.get("finding_count", 0)),
        "reports": compact_reports,
    }
    if body.get("recommended_next_action"):
        out["recommended_next_action"] = body["recommended_next_action"]
    return out


def _data_plane_quality_summary(dp: Any) -> dict[str, Any]:
    quality = dict(getattr(dp, "quality", {}) or {})
    status = str(quality.get("status") or "unknown")
    claim_count = quality.get(
        "claim_count", dict(getattr(dp, "counts", {}) or {}).get("claims")
    )
    return {
        **quality,
        "status": status,
        "health_status": status,
        "source": "data_plane",
        "claim_count": claim_count,
        "quality_counts": {},
        "reports": {},
        "total_findings": quality.get("open_conflicts", 0),
    }


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


def _data_plane_status_payload(status: Any) -> dict[str, Any]:
    return {
        "pot_id": status.pot_id,
        "backend_profile": status.backend_profile,
        "backend_ready": status.backend_ready,
        "reader_backed_includes": list(status.reader_backed_includes),
        "counts": dict(status.counts),
        "freshness": dict(status.freshness),
        "quality": dict(status.quality),
        "match_mode": status.match_mode,
        "detail": status.detail,
    }


def _neighborhood_relation(edge: Any) -> dict[str, Any]:
    props = dict(edge.properties or {})
    source_refs = _string_list(props.get("source_refs"))
    if not source_refs and props.get("source_ref"):
        source_refs = [_str(props.get("source_ref"))]
    score = props.get("semantic_similarity")
    if score is None:
        score = props.get("score")
    return {
        "predicate": edge.predicate,
        "from": edge.from_key,
        "to": edge.to_key,
        "from_key": edge.from_key,
        "to_key": edge.to_key,
        "fact": _str(
            props.get("fact") or props.get("description") or props.get("summary")
        ),
        "source_refs": source_refs,
        "truth": _str(props.get("truth")),
        "environment": _str(props.get("environment")),
        "score": float(score) if isinstance(score, (int, float)) else None,
        "claim_key": _str(props.get("claim_key")),
        "source_system": _str(props.get("source_system")),
    }


def _neighborhood_human(payload: Mapping[str, Any]) -> str:
    relations = payload.get("relations") or ()
    lines = [
        (
            f"entity={payload.get('entity_key')} relations={len(relations)} "
            f"nodes={payload.get('node_count')} detail={payload.get('detail')}"
        )
    ]
    for rel in list(relations)[:20]:
        if not isinstance(rel, Mapping):
            continue
        refs = ", ".join(_string_list(rel.get("source_refs"))) or "no-source-ref"
        fact = rel.get("fact") or f"{rel.get('from')} -> {rel.get('to')}"
        lines.append(f"  • {rel.get('predicate')} [{refs}] {fact}")
    return "\n".join(lines)


def _catalog_payload_for_profile(
    payload: Mapping[str, Any], *, profile: str
) -> dict[str, Any]:
    mode = (profile or "full").strip().lower()
    if mode not in {"full", "read"}:
        raise ValueError("--profile must be one of: full, read")
    result = dict(payload)
    result["profile"] = mode
    if mode == "full":
        return result

    read_commands = [
        command
        for command in result.get("commands", ())
        if command
        in {
            "status",
            "catalog",
            "describe",
            "search-entities",
            "read",
            "neighborhood",
        }
    ]
    result["commands"] = read_commands
    result["views"] = [_compact_catalog_view(view) for view in result.get("views", ())]
    if "task_ranking" in result:
        result["task_ranking"] = [
            _compact_catalog_ranking(entry, rank=index + 1)
            for index, entry in enumerate(result.get("task_ranking", ()))
        ]
    for key in (
        "admin_commands",
        "legacy_commands",
        "mutation_operations",
        "review_required_operations",
        "deferred_operations",
        "entity_types",
        "predicates",
        "transition",
    ):
        result.pop(key, None)
    return result


def _compact_catalog_view(view: Mapping[str, Any]) -> dict[str, Any]:
    out = {
        key: view[key]
        for key in (
            "name",
            "subgraph",
            "view",
            "backed",
            "description",
            "result_shape",
            "required_scope",
            "required_any_scope",
            "supported_filters",
        )
        if key in view
    }
    if "subgraph" in out and "view" in out:
        out["next_read"] = (
            f"potpie graph read --subgraph {out['subgraph']} --view {out['view']}"
        )
    return out


def _compact_catalog_ranking(entry: Mapping[str, Any], *, rank: int) -> dict[str, Any]:
    out: dict[str, Any] = {"rank": rank}
    for key in ("view", "subgraph", "score", "matched_terms"):
        if key in entry:
            out[key] = entry[key]
    reason = entry.get("reason") or entry.get("why")
    if reason:
        out["reason"] = reason
    return out


def _catalog_human(payload: Mapping[str, Any], *, format_: str) -> str:
    mode = (format_ or "auto").strip().lower()
    if mode not in {"auto", "table"}:
        raise ValueError("--format must be one of: auto, table")
    if mode == "table" or payload.get("profile") == "read":
        lines = [
            f"graph catalog profile={payload.get('profile', 'full')} "
            f"match={payload.get('match_mode')}"
        ]
        task = payload.get("task")
        if task:
            lines.append(f"task={task}")
        rankings = payload.get("task_ranking") or ()
        if rankings:
            lines.append("rank | score | view | reason")
            lines.append("--- | --- | --- | ---")
            for entry in rankings[:8]:
                reason = str(entry.get("reason") or "")
                lines.append(
                    f"{entry.get('rank')} | {entry.get('score')} | "
                    f"{entry.get('view')} | {reason}"
                )
        lines.append("view | backed | filters")
        lines.append("--- | --- | ---")
        for view in payload.get("views", ()):
            filters = ", ".join(view.get("supported_filters") or ()) or "-"
            lines.append(
                f"{view.get('name')} | {str(bool(view.get('backed'))).lower()} | {filters}"
            )
        return "\n".join(lines)

    return (
        f"graph contract v2 / ontology {ONTOLOGY_VERSION} "
        f"(data-plane={payload['data_plane_graph_contract_version']}, match={payload['match_mode']})\n"
        f"commands: {', '.join(payload['commands'])}\n"
        f"views: {', '.join(v['name'] for v in payload['views'])}\n"
        f"mutation ops: {', '.join(payload['mutation_operations'])}\n"
        f"review-required: {', '.join(payload['review_required_operations'])}\n"
        f"deferred: {', '.join(payload['deferred_operations'])}"
    )


def _nudge_human(result: Any) -> str:
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


def _mutate_human(result: Any) -> str:
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


def _proposal_human(result: Any) -> str:
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


def _commit_human(result: Any) -> str:
    head = f"{result.status}: plan_id={result.plan_id} risk={result.risk}"
    if result.mutation_id:
        head += f" mutation_id={result.mutation_id}"
    lines = [head]
    verification = getattr(result, "verification", None)
    if verification is not None:
        lines.append(
            "verification: "
            f"status={verification.status} "
            f"readback={verification.readback_count}/{len(verification.claim_keys)} "
            f"quality={verification.quality_status}"
        )
        if verification.quality_regressions:
            lines.append(
                f"quality_regressions={dict(verification.quality_regressions)}"
            )
        if verification.missing_claim_keys:
            lines.append(f"missing_claim_keys={list(verification.missing_claim_keys)}")
    if result.detail:
        lines.append(result.detail)
    return "\n".join(lines)


def _history_human(result: Any) -> str:
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


def _inbox_human(result: Any) -> str:
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


def _quality_human(result: Any) -> str:
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
