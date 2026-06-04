"""Read-only reconciliation tools backed by ``ContextGraphPort``.

Bounded tool set for the Ingestion Agent to inspect current graph state,
look up entities/facts, and surface prior events/conflicts before emitting a
reconciliation plan. All tools are pot-scoped and read-only — mutation still
flows through ``ContextGraphPort.apply_plan``.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from domain.graph_query import (
    ContextGraphQuery,
    preset_change_history,
    preset_file_owners,
    preset_graph_overview,
    preset_semantic_search,
)
from domain.ports.context_graph import ContextGraphPort
from domain.ports.reconciliation_tools import ReconciliationToolsPort, ToolDescriptor
from domain.reconciliation import ReconciliationRequest

logger = logging.getLogger(__name__)


_TOOLS: tuple[ToolDescriptor, ...] = (
    ToolDescriptor(
        name="context_search",
        category="context_lookup",
        description=(
            "Semantic search across the pot's existing facts. Use to find "
            "decisions, incidents, services, features, or documents that "
            "may already capture what this event describes."
        ),
        json_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "free-text query"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 25, "default": 8},
                "node_labels": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "optional canonical labels to constrain results",
                },
            },
            "required": ["query"],
        },
    ),
    ToolDescriptor(
        name="context_recent_changes",
        category="context_lookup",
        description=(
            "Recent change history for a file, function, or PR in this pot. "
            "Use to see what has been touched or discussed recently near "
            "the target of the current event."
        ),
        json_schema={
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "function_name": {"type": "string"},
                "pr_number": {"type": "integer"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 25, "default": 10},
            },
        },
    ),
    ToolDescriptor(
        name="context_file_owners",
        category="context_lookup",
        description="Owners / reviewers inferred for a file in this pot.",
        json_schema={
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5},
            },
            "required": ["file_path"],
        },
    ),
    ToolDescriptor(
        name="context_graph_overview",
        category="context_lookup",
        description=(
            "High-level readiness/size signal for this pot (entity/edge "
            "counts, recent activity). Use to decide whether the graph "
            "already has relevant memory."
        ),
        json_schema={
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 20},
            },
        },
    ),
)


def _run_query(graph: ContextGraphPort, query: ContextGraphQuery) -> dict[str, Any]:
    """Run a query synchronously.

    The agent may invoke tools from inside a running event loop (pydantic-deep
    runs under ``asyncio.run``). ``ContextGraphPort.query()`` handles this by
    raising only for answer queries when a loop is already running; the
    evidence-leg queries used by these tools (semantic_search, change_history,
    owners, graph_overview) execute on the sync path.
    """
    try:
        result = graph.query(query)
    except RuntimeError as exc:
        # Sync path refused to run in a live loop; fall back to async via thread.
        import concurrent.futures as _cf

        def _inner() -> Any:
            return asyncio.run(graph.query_async(query))

        try:
            with _cf.ThreadPoolExecutor(max_workers=1) as pool:
                result = pool.submit(_inner).result()
        except Exception as inner_exc:
            logger.exception("reconciliation tool async fallback failed")
            return {"error": str(inner_exc), "kind": "error", "source_error": str(exc)}
    except Exception as exc:
        logger.exception("reconciliation tool query failed")
        return {"error": str(exc), "kind": "error"}
    return result.model_dump()


class ContextGraphReconciliationTools(ReconciliationToolsPort):
    """Bounded read-only tools over ``ContextGraphPort``."""

    def __init__(self, context_graph: ContextGraphPort) -> None:
        self._graph = context_graph

    def list_tools(self, request: ReconciliationRequest) -> list[ToolDescriptor]:
        # Current request context is not used to narrow the catalog in v1;
        # pot scoping happens inside ``execute_read_tool``.
        del request
        return list(_TOOLS)

    def execute_read_tool(
        self,
        request: ReconciliationRequest,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        if not self._graph.enabled:
            return {"error": "context_graph_disabled", "kind": "error"}
        pot_id = request.pot_id
        repo_name = request.repo_name
        args = arguments or {}
        if tool_name == "context_search":
            q = str(args.get("query") or "").strip()
            if not q:
                return {"error": "query_required", "kind": "error"}
            limit = int(args.get("limit") or 8)
            labels_arg = args.get("node_labels") or []
            node_labels = [str(x) for x in labels_arg] if isinstance(labels_arg, list) else []
            return _run_query(
                self._graph,
                preset_semantic_search(
                    pot_id=pot_id,
                    query=q,
                    limit=max(1, min(limit, 25)),
                    repo_name=repo_name,
                    node_labels=node_labels or None,
                ),
            )
        if tool_name == "context_recent_changes":
            file_path = args.get("file_path")
            function_name = args.get("function_name")
            pr_number = args.get("pr_number")
            limit = int(args.get("limit") or 10)
            if not any([file_path, function_name, pr_number]):
                return {
                    "error": "one_of_file_path_function_name_pr_number_required",
                    "kind": "error",
                }
            return _run_query(
                self._graph,
                preset_change_history(
                    pot_id=pot_id,
                    file_path=file_path,
                    function_name=function_name,
                    pr_number=int(pr_number) if pr_number else None,
                    repo_name=repo_name,
                    limit=max(1, min(limit, 25)),
                ),
            )
        if tool_name == "context_file_owners":
            file_path = args.get("file_path")
            if not file_path:
                return {"error": "file_path_required", "kind": "error"}
            limit = int(args.get("limit") or 5)
            return _run_query(
                self._graph,
                preset_file_owners(
                    pot_id=pot_id,
                    file_path=str(file_path),
                    repo_name=repo_name,
                    limit=max(1, min(limit, 10)),
                ),
            )
        if tool_name == "context_graph_overview":
            limit = int(args.get("limit") or 20)
            return _run_query(
                self._graph,
                preset_graph_overview(
                    pot_id=pot_id,
                    limit=max(1, min(limit, 50)),
                ),
            )
        return {"error": f"unknown_tool:{tool_name}", "kind": "error"}


def build_initial_context_snapshot(
    tools: ReconciliationToolsPort,
    request: ReconciliationRequest,
    *,
    semantic_seed: str | None = None,
) -> dict[str, Any]:
    """Prefetch a bounded baseline snapshot the agent sees before any tool call."""
    out: dict[str, Any] = {}
    try:
        out["graph_overview"] = tools.execute_read_tool(
            request, "context_graph_overview", {"limit": 20}
        )
    except Exception:
        logger.exception("graph_overview snapshot failed")
        out["graph_overview"] = {"error": "snapshot_failed", "kind": "error"}
    if semantic_seed:
        try:
            out["semantic_seed"] = tools.execute_read_tool(
                request,
                "context_search",
                {"query": semantic_seed, "limit": 6},
            )
        except Exception:
            logger.exception("semantic_seed snapshot failed")
            out["semantic_seed"] = {"error": "snapshot_failed", "kind": "error"}
    # Best-effort: use event source_id as a secondary search seed (surfaces
    # prior events that touched the same source ref).
    source_id = request.event.source_id or request.event.source_event_id
    if source_id:
        try:
            out["source_ref_hits"] = tools.execute_read_tool(
                request,
                "context_search",
                {"query": str(source_id), "limit": 5},
            )
        except Exception:
            logger.exception("source_ref_hits snapshot failed")
            out["source_ref_hits"] = {"error": "snapshot_failed", "kind": "error"}
    return out


__all__ = [
    "ContextGraphReconciliationTools",
    "build_initial_context_snapshot",
]
