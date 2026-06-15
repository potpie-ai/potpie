"""Shared Graph V2 workbench envelope assembly."""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from typing import Any

from domain.graph_contract import GRAPH_CONTRACT_VERSION as DATA_PLANE_CONTRACT_VERSION
from domain.graph_workbench import (
    GRAPH_WORKBENCH_ADMIN_COMMANDS,
    GRAPH_WORKBENCH_COMMANDS,
    GRAPH_WORKBENCH_CONTRACT_VERSION,
    GRAPH_WORKBENCH_LEGACY_COMMANDS,
    GraphCommandEnvelope,
    GraphCommandError,
    GraphUnsupported,
    GraphUnsupportedResult,
)
from domain.graph_workbench_ontology import ranked_catalog_views

_CROSS_CUTTING_RESULT_KEYS = frozenset(
    {
        "ok",
        "graph_contract_version",
        "ontology_version",
        "subgraph_versions",
        "pot_id",
        "warnings",
        "unsupported",
        "recommended_next_action",
    }
)


def new_graph_request_id() -> str:
    return f"req:{uuid.uuid4().hex}"


def graph_success_envelope(
    *,
    command: str,
    request_id: str,
    pot_id: str | None,
    result: Mapping[str, Any] | None = None,
    subgraph_versions: Mapping[str, int] | None = None,
    warnings: tuple[str, ...] | list[str] = (),
    unsupported: tuple[GraphUnsupported, ...] | list[GraphUnsupported] = (),
    recommended_next_action: str | Mapping[str, Any] | None = None,
) -> GraphCommandEnvelope:
    return GraphCommandEnvelope(
        ok=True,
        command=command,
        request_id=request_id,
        pot_id=pot_id,
        result=dict(result or {}),
        subgraph_versions=dict(subgraph_versions or {}),
        warnings=tuple(warnings),
        unsupported=tuple(unsupported),
        recommended_next_action=recommended_next_action,
    )


def graph_error_envelope(
    *,
    command: str,
    request_id: str,
    pot_id: str | None,
    code: str,
    message: str,
    detail: Any = None,
    subgraph_versions: Mapping[str, int] | None = None,
    warnings: tuple[str, ...] | list[str] = (),
    unsupported: tuple[GraphUnsupported, ...] | list[GraphUnsupported] = (),
    recommended_next_action: str | Mapping[str, Any] | None = None,
) -> GraphCommandEnvelope:
    return GraphCommandEnvelope(
        ok=False,
        command=command,
        request_id=request_id,
        pot_id=pot_id,
        result=None,
        subgraph_versions=dict(subgraph_versions or {}),
        warnings=tuple(warnings),
        unsupported=tuple(unsupported),
        recommended_next_action=recommended_next_action,
        error=GraphCommandError(code=code, message=message, detail=detail),
    )


def graph_not_implemented_envelope(
    *,
    command: str,
    request_id: str,
    pot_id: str | None,
    detail: str | None = None,
    recommended_next_action: str | None = None,
) -> GraphCommandEnvelope:
    message = f"{command} is not implemented yet"
    return graph_error_envelope(
        command=command,
        request_id=request_id,
        pot_id=pot_id,
        code="not_implemented",
        message=message,
        detail=detail,
        unsupported=(
            GraphUnsupported(
                name=command,
                reason="not_implemented",
                detail=detail,
            ),
        ),
        recommended_next_action=recommended_next_action,
    )


def graph_not_implemented_result(command: str, *, detail: str | None = None) -> dict:
    return GraphUnsupportedResult(
        status="not_implemented",
        command=command,
        detail=detail,
    ).to_dict()


def normalize_workbench_result(
    payload: Mapping[str, Any],
) -> tuple[
    dict[str, Any], dict[str, int], tuple[str, ...], tuple[GraphUnsupported, ...]
]:
    """Move cross-cutting fields from a legacy result into envelope fields."""
    result = dict(payload)
    subgraph_versions = _mapping_of_ints(result.pop("subgraph_versions", {}))
    warnings = _string_tuple(result.pop("warnings", ()))
    unsupported = _unsupported_tuple(result.pop("unsupported", ()))
    unsupported += _unsupported_tuple(result.pop("unsupported_includes", ()))
    for key in _CROSS_CUTTING_RESULT_KEYS:
        result.pop(key, None)
    return result, subgraph_versions, warnings, unsupported


def normalize_catalog_result(
    payload: Mapping[str, Any],
    *,
    task: str | None = None,
) -> dict[str, Any]:
    """Project the V1.5 data-plane catalog as a V2 workbench catalog body."""
    result = dict(payload)
    data_plane_version = result.pop(
        "graph_contract_version", DATA_PLANE_CONTRACT_VERSION
    )
    result.pop("ontology_version", None)
    result.pop("ok", None)
    result["data_plane_graph_contract_version"] = data_plane_version
    result["workbench_graph_contract_version"] = GRAPH_WORKBENCH_CONTRACT_VERSION
    result["commands"] = list(GRAPH_WORKBENCH_COMMANDS)
    result["admin_commands"] = list(GRAPH_WORKBENCH_ADMIN_COMMANDS)
    result["legacy_commands"] = list(GRAPH_WORKBENCH_LEGACY_COMMANDS)
    views, ranking = ranked_catalog_views(
        result.get("views", ()),
        task,
    )
    result["views"] = views
    if task:
        result["task"] = task
        result["task_ranking"] = ranking
    result["transition"] = {
        "legacy_commands_callable": True,
        "mutate_replacement": "propose + commit",
        "inspect_replacement": "neighborhood",
    }
    return result


def _mapping_of_ints(value: Any) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    out: dict[str, int] = {}
    for key, raw in value.items():
        try:
            out[str(key)] = int(raw)
        except (TypeError, ValueError):
            continue
    return out


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (list, tuple)):
        return tuple(str(v) for v in value if v is not None)
    return (str(value),)


def _unsupported_tuple(value: Any) -> tuple[GraphUnsupported, ...]:
    if value is None:
        return ()
    if isinstance(value, GraphUnsupported):
        return (value,)
    if isinstance(value, Mapping):
        value = (value,)
    if not isinstance(value, (list, tuple)):
        return (GraphUnsupported(name=str(value), reason="unsupported"),)
    out: list[GraphUnsupported] = []
    for item in value:
        if isinstance(item, GraphUnsupported):
            out.append(item)
            continue
        if isinstance(item, Mapping):
            out.append(
                GraphUnsupported(
                    name=str(item.get("name") or item.get("include") or "unknown"),
                    reason=str(item.get("reason") or item.get("code") or "unsupported"),
                    detail=item.get("detail"),
                )
            )
            continue
        out.append(GraphUnsupported(name=str(item), reason="unsupported"))
    return tuple(out)


__all__ = [
    "graph_error_envelope",
    "graph_not_implemented_envelope",
    "graph_not_implemented_result",
    "graph_success_envelope",
    "new_graph_request_id",
    "normalize_catalog_result",
    "normalize_workbench_result",
]
