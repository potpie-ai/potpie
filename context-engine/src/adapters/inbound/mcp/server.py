"""MCP server: expose read-only context queries (same logic as HTTP query routes)."""

from __future__ import annotations

import logging

from mcp.server.fastmcp import FastMCP

from adapters.outbound.graphiti.episodic import GraphitiEpisodicAdapter
from adapters.outbound.neo4j.structural import Neo4jStructuralAdapter
from adapters.outbound.settings_env import EnvContextEngineSettings
from application.use_cases.query_context import (
    get_change_history,
    get_decisions,
    get_file_owners,
    search_project_context,
)

logger = logging.getLogger(__name__)
mcp = FastMCP("context-engine")
_settings = EnvContextEngineSettings()
_structural = Neo4jStructuralAdapter(_settings)
_episodic = GraphitiEpisodicAdapter(_settings)


@mcp.tool()
def context_get_change_history(
    project_id: str,
    file_path: str | None = None,
    function_name: str | None = None,
    limit: int = 10,
) -> list[dict]:
    """Return PR-linked change history for a project (Neo4j structural graph)."""
    if not _settings.is_enabled():
        return []
    return get_change_history(
        _structural,
        project_id,
        function_name=function_name,
        file_path=file_path,
        limit=limit,
    )


@mcp.tool()
def context_get_file_owners(project_id: str, file_path: str, limit: int = 5) -> list[dict]:
    """Return likely file owners from PR touch history."""
    if not _settings.is_enabled():
        return []
    return get_file_owners(_structural, project_id, file_path, limit)


@mcp.tool()
def context_get_decisions(
    project_id: str,
    file_path: str | None = None,
    function_name: str | None = None,
    limit: int = 20,
) -> list[dict]:
    """Return design decisions linked to code nodes."""
    if not _settings.is_enabled():
        return []
    return get_decisions(
        _structural,
        project_id,
        file_path=file_path,
        function_name=function_name,
        limit=limit,
    )


@mcp.tool()
def context_search(
    project_id: str,
    query: str,
    limit: int = 8,
    node_labels: str | None = None,
) -> list[dict]:
    """Semantic search over Graphiti episodic entities. node_labels: comma-separated optional."""
    labels = None
    if node_labels:
        labels = [x.strip() for x in node_labels.split(",") if x.strip()]
    return search_project_context(
        _episodic,
        project_id,
        query,
        limit=limit,
        node_labels=labels,
    )


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
