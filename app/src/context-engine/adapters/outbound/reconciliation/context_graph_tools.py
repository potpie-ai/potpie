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
    preset_context_search,
    preset_reader_lookup,
)
from domain.ports.context_graph import ContextGraphPort
from domain.ports.reconciliation_tools import ReconciliationToolsPort, ToolDescriptor
from domain.reconciliation import ReconciliationRequest

logger = logging.getLogger(__name__)


# Tool name → ReadOrchestrator include family. ``None`` is the generic,
# intent-routed search (the orchestrator expands the request's intent into
# reader families). Every non-None value MUST be an include the orchestrator
# backs; ``test_read_tools_match_orchestrator`` pins this so the agent tool
# surface can never drift from the readers again.
READ_TOOL_INCLUDE: dict[str, str | None] = {
    "context_search": None,
    "context_coding_preferences": "coding_preferences",
    "context_infra_topology": "infra_topology",
    "context_timeline": "timeline",
    "context_prior_bugs": "prior_bugs",
}


_TOOLS: tuple[ToolDescriptor, ...] = (
    ToolDescriptor(
        name="context_search",
        category="context_lookup",
        description=(
            "Generic lookup across the pot's existing memory. Routes by the "
            "task's intent into the project's readers (coding preferences, "
            "infra topology, timeline, prior bugs). Start here, then narrow "
            "with a targeted tool below."
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
        name="context_coding_preferences",
        category="context_lookup",
        description=(
            "Coding conventions, patterns, libraries, and rules that apply to "
            "this pot (optionally scoped to a file). Use before proposing how "
            "something should be written or structured."
        ),
        json_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "optional free-text focus"},
                "file_path": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 25, "default": 8},
            },
        },
    ),
    ToolDescriptor(
        name="context_infra_topology",
        category="context_lookup",
        description=(
            "Services, datastores, dependencies, environments, and deployment "
            "topology known for this pot. Use to see what infrastructure "
            "already exists before adding or changing it."
        ),
        json_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "optional free-text focus"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 20},
            },
        },
    ),
    ToolDescriptor(
        name="context_timeline",
        category="context_lookup",
        description=(
            "Recent changes and activity in this pot, optionally scoped to a "
            "file, function, or PR. Use to see what has been touched or "
            "discussed recently near the target of the current event."
        ),
        json_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "file_path": {"type": "string"},
                "function_name": {"type": "string"},
                "pr_number": {"type": "integer"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 25, "default": 10},
            },
        },
    ),
    ToolDescriptor(
        name="context_prior_bugs",
        category="context_lookup",
        description=(
            "Prior occurrences of a symptom and the fixes / decisions that "
            "resolved them. Use when an event describes a bug, incident, or "
            "regression to check whether it has happened before."
        ),
        json_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "symptom / error to match"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 25, "default": 8},
            },
        },
    ),
)


def _run_query(graph: ContextGraphPort, query: ContextGraphQuery) -> dict[str, Any]:
    """Run a query synchronously.

    The agent may invoke tools from inside a running event loop (pydantic-deep
    runs under ``asyncio.run``). ``ContextGraphPort.query()`` handles this by
    raising only for answer queries when a loop is already running; the
    retrieve-goal queries used by these tools (the generic search and the P9
    reader lookups) execute on the sync path.
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
        if tool_name not in READ_TOOL_INCLUDE:
            return {"error": f"unknown_tool:{tool_name}", "kind": "error"}
        args = arguments or {}
        pot_id = request.pot_id
        repo_name = request.repo_name
        query = str(args.get("query") or "").strip()
        limit = max(1, min(int(args.get("limit") or 12), 50))
        include = READ_TOOL_INCLUDE[tool_name]

        if include is None:  # generic, intent-routed search
            if not query:
                return {"error": "query_required", "kind": "error"}
            labels_arg = args.get("node_labels") or []
            node_labels = (
                [str(x) for x in labels_arg] if isinstance(labels_arg, list) else []
            )
            graph_query = preset_context_search(
                pot_id=pot_id,
                query=query,
                repo_name=repo_name,
                node_labels=node_labels or None,
                limit=limit,
            )
        else:  # targeted single-reader lookup
            pr_number = args.get("pr_number")
            graph_query = preset_reader_lookup(
                pot_id=pot_id,
                include=include,
                query=query or None,
                repo_name=repo_name,
                file_path=args.get("file_path"),
                function_name=args.get("function_name"),
                pr_number=int(pr_number) if pr_number else None,
                limit=limit,
            )
        return _run_query(self._graph, graph_query)


def build_initial_context_snapshot(
    tools: ReconciliationToolsPort,
    request: ReconciliationRequest,
    *,
    semantic_seed: str | None = None,
) -> dict[str, Any]:
    """Prefetch a bounded baseline snapshot the agent sees before any tool call."""
    out: dict[str, Any] = {}
    try:
        out["infra_topology"] = tools.execute_read_tool(
            request, "context_infra_topology", {"limit": 20}
        )
    except Exception:
        logger.exception("infra_topology snapshot failed")
        out["infra_topology"] = {"error": "snapshot_failed", "kind": "error"}
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


# Public, port-agnostic view of the read-tool catalog. The reconciliation
# agent reaches these via ``ReconciliationToolsPort``; the read-side query
# agent consumes the same descriptors directly so both agents stay on one
# tool surface.
READ_TOOL_DESCRIPTORS: tuple[ToolDescriptor, ...] = _TOOLS


__all__ = [
    "ContextGraphReconciliationTools",
    "build_initial_context_snapshot",
    "READ_TOOL_DESCRIPTORS",
    "READ_TOOL_INCLUDE",
]
