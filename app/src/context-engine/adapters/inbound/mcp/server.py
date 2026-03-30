"""MCP server: expose read-only context queries (same logic as HTTP query routes)."""

from __future__ import annotations

import logging

from mcp.server.fastmcp import FastMCP

from adapters.inbound.mcp.project_access import assert_mcp_pot_allowed
from adapters.outbound.graphiti.episodic import GraphitiEpisodicAdapter
from adapters.outbound.neo4j.structural import Neo4jStructuralAdapter
from adapters.outbound.settings_env import EnvContextEngineSettings
from application.use_cases.query_context import (
    get_change_history,
    get_decisions,
    get_file_owners,
    get_pr_diff,
    get_pr_review_context,
    search_pot_context,
)

logger = logging.getLogger(__name__)
mcp = FastMCP("context-engine")
_settings = EnvContextEngineSettings()
_structural = Neo4jStructuralAdapter(_settings)
_episodic = GraphitiEpisodicAdapter(_settings)


@mcp.tool()
def context_get_change_history(
    pot_id: str,
    file_path: str | None = None,
    function_name: str | None = None,
    limit: int = 10,
    repo_name: str | None = None,
) -> list[dict]:
    """Return PR-linked change history for a pot (Neo4j structural graph)."""
    assert_mcp_pot_allowed(pot_id)
    if not _settings.is_enabled():
        return []
    return get_change_history(
        _structural,
        pot_id,
        function_name=function_name,
        file_path=file_path,
        limit=limit,
        repo_name=repo_name,
    )


@mcp.tool()
def context_get_file_owners(
    pot_id: str, file_path: str, limit: int = 5, repo_name: str | None = None
) -> list[dict]:
    """Return likely file owners from PR touch history."""
    assert_mcp_pot_allowed(pot_id)
    if not _settings.is_enabled():
        return []
    return get_file_owners(_structural, pot_id, file_path, limit, repo_name=repo_name)


@mcp.tool()
def context_get_decisions(
    pot_id: str,
    file_path: str | None = None,
    function_name: str | None = None,
    limit: int = 20,
    repo_name: str | None = None,
) -> list[dict]:
    """Return design decisions linked to code nodes."""
    assert_mcp_pot_allowed(pot_id)
    if not _settings.is_enabled():
        return []
    return get_decisions(
        _structural,
        pot_id,
        file_path=file_path,
        function_name=function_name,
        limit=limit,
        repo_name=repo_name,
    )


@mcp.tool()
def context_get_pr_review_context(
    pot_id: str, pr_number: int, repo_name: str | None = None
) -> dict:
    """Return a PR's title/summary plus linked review-thread discussions (Decision nodes)."""
    assert_mcp_pot_allowed(pot_id)
    if not _settings.is_enabled():
        return {
            "found": False,
            "pr_number": pr_number,
            "pr_title": None,
            "pr_summary": None,
            "review_threads": [],
        }
    return get_pr_review_context(_structural, pot_id, pr_number, repo_name=repo_name)


@mcp.tool()
def context_get_pr_diff(
    pot_id: str,
    pr_number: int,
    file_path: str | None = None,
    limit: int = 30,
    repo_name: str | None = None,
) -> list[dict]:
    """Return file-level PR diff excerpts captured during ingestion."""
    assert_mcp_pot_allowed(pot_id)
    if not _settings.is_enabled():
        return []
    return get_pr_diff(
        _structural,
        pot_id,
        pr_number,
        file_path=file_path,
        limit=limit,
        repo_name=repo_name,
    )


def _mcp_search_pot_context(
    pot_id: str,
    query: str,
    limit: int = 8,
    node_labels: str | None = None,
    repo_name: str | None = None,
) -> list[dict]:
    assert_mcp_pot_allowed(pot_id)
    labels = None
    if node_labels:
        labels = [x.strip() for x in node_labels.split(",") if x.strip()]
    return search_pot_context(
        _episodic,
        pot_id,
        query,
        limit=limit,
        node_labels=labels,
        repo_name=repo_name,
    )


@mcp.tool()
def context_search(
    pot_id: str,
    query: str,
    limit: int = 8,
    node_labels: str | None = None,
    repo_name: str | None = None,
) -> list[dict]:
    """Semantic search over Graphiti episodic entities. node_labels: comma-separated optional."""
    return _mcp_search_pot_context(
        pot_id, query, limit=limit, node_labels=node_labels, repo_name=repo_name
    )


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
