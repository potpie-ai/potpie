"""Render the graph catalog returned by the selected host, including its ontology."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from potpie_context_core.graph_contract import ONTOLOGY_VERSION


def _ontology_lines(payload: Mapping[str, Any]) -> list[str]:
    if payload.get("profile") == "read":
        return [
            (
                "For ingestion and writes, use graph catalog --profile full "
                "for entity types, predicates, and allowed endpoints."
            )
        ]

    lines = [
        "",
        "entity | key prefix | identity | scope | description",
        "--- | --- | --- | --- | ---",
    ]
    for entity in payload.get("entity_types", ()):
        lines.append(
            f"{entity['label']} | {entity['key_prefix']}: | {entity['identity_policy']} | "
            f"{str(bool(entity.get('scope'))).lower()} | {entity.get('description', '')}"
        )
    lines.extend(
        [
            "",
            "predicate | allowed subject -> object | required properties | description",
            "--- | --- | --- | ---",
        ]
    )
    for predicate in payload.get("predicates", ()):
        pairs = "; ".join(f"{a} -> {b}" for a, b in predicate["allowed_pairs"])
        required = ", ".join(predicate.get("required_properties", ())) or "-"
        lines.append(
            f"{predicate['name']} | {pairs} | {required} | {predicate.get('description', '')}"
        )
    lines.append(
        "@Scope = entity types marked scope=true; @Activity = activity types; "
        "* = any endpoint. Templates are examples, not the full ontology."
    )
    return lines


def render_catalog(payload: Mapping[str, Any], *, format_: str) -> str:
    mode = (format_ or "auto").strip().lower()
    if mode not in {"auto", "table"}:
        raise ValueError("--format must be one of: auto, table")
    if mode == "table" or payload.get("profile") == "read":
        lines = [
            (
                f"graph catalog profile={payload.get('profile', 'full')} "
                f"match={payload.get('match_mode')}"
            )
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
        lines.extend(_ontology_lines(payload))
        return "\n".join(lines)

    lines = [
        (
            f"graph contract v2 / ontology {payload.get('ontology_version', ONTOLOGY_VERSION)} "
            f"(data-plane={payload['data_plane_graph_contract_version']}, match={payload['match_mode']})"
        ),
        f"commands: {', '.join(payload['commands'])}",
        f"views: {', '.join(v['name'] for v in payload['views'])}",
        f"mutation ops: {', '.join(payload['mutation_operations'])}",
        f"review-required: {', '.join(payload['review_required_operations'])}",
        f"deferred: {', '.join(payload['deferred_operations'])}",
    ]
    support = payload.get("admin_command_support")
    if isinstance(support, Mapping) and support:
        lines.append(
            "admin: "
            + ", ".join(
                command if support[command] else f"{command} (unavailable)"
                for command in sorted(support)
            )
        )
    lines.extend(_ontology_lines(payload))
    return "\n".join(lines)
