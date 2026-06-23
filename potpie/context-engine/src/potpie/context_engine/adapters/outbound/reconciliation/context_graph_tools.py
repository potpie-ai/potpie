"""Read-only reconciliation tools backed by canonical ``GraphService``.

Bounded tool set for the Ingestion Agent to inspect current graph state,
look up entities/facts, and surface prior events/conflicts before emitting a
reconciliation plan. All tools are pot-scoped and read-only.
"""

from __future__ import annotations

import logging
from typing import Any

from potpie.context_engine.application.services.envelope_builder import envelope_to_dict
from potpie.context_engine.domain.llm_reconciliation import ReconciliationRequest
from potpie.context_engine.domain.ports.agent_context import ResolveRequest
from potpie.context_engine.domain.ports.reconciliation_tools import ReconciliationToolsPort, ToolDescriptor
from potpie.context_engine.domain.ports.services.graph_service import GraphService

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
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 50,
                    "default": 20,
                },
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
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 25,
                    "default": 10,
                },
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


def _run_query(graph: GraphService, request: ResolveRequest) -> dict[str, Any]:
    """Run a bounded read through the canonical graph service."""
    try:
        env = graph.resolve(request)
    except Exception as exc:
        logger.exception("reconciliation tool query failed")
        return {"error": str(exc), "kind": "error"}
    return {
        "kind": "resolve",
        "goal": "retrieve",
        "strategy": "auto",
        "result": envelope_to_dict(env),
        "meta": {"path": "resolve"},
    }


def _scope(**fields: Any) -> dict[str, Any]:
    return {key: value for key, value in fields.items() if value not in (None, [], ())}


class ContextGraphReconciliationTools(ReconciliationToolsPort):
    """Bounded read-only tools over canonical ``GraphService``."""

    def __init__(self, graph: GraphService) -> None:
        self._graph = graph

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
        backend = getattr(self._graph, "backend", None)
        if not bool(getattr(backend, "enabled", True)):
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
            resolve_request = ResolveRequest(
                pot_id=pot_id,
                task=query,
                scope=_scope(repo_name=repo_name),
                max_items=limit,
            )
        else:  # targeted single-reader lookup
            pr_number = args.get("pr_number")
            resolve_request = ResolveRequest(
                pot_id=pot_id,
                task=query or None,
                include=(include,),
                scope=_scope(
                    repo_name=repo_name,
                    file_path=args.get("file_path"),
                    function_name=args.get("function_name"),
                    pr_number=int(pr_number) if pr_number else None,
                ),
                max_items=limit,
            )
        return _run_query(self._graph, resolve_request)


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
