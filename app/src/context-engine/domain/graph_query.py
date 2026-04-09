"""Unified graph query kind (episodic + structural + resolve)."""

from __future__ import annotations

from enum import StrEnum


class GraphQueryKind(StrEnum):
    """All read-side graph query modes exposed via one API / CLI."""

    SEMANTIC_SEARCH = "semantic_search"
    CHANGE_HISTORY = "change_history"
    FILE_OWNERS = "file_owners"
    DECISIONS = "decisions"
    PR_REVIEW_CONTEXT = "pr_review_context"
    PR_DIFF = "pr_diff"
    PROJECT_GRAPH = "project_graph"
    RESOLVE_CONTEXT = "resolve_context"
