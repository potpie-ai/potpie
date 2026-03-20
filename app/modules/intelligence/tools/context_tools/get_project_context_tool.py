"""Agent tool: get_project_context — search the context graph for a project."""

import asyncio
from typing import Any, List, Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)

GET_PROJECT_CONTEXT_DESCRIPTION = """Retrieve project context (recent PRs, design decisions, commits) from the context graph.
Use this before asking MCQs or generating a spec to ground the spec in recent changes and decisions.
Inputs:
- project_id: The project/repository ID (UUID from context).
- query: Optional search query (e.g. feature name, area of code).
- branch: Optional branch name to bias results.
- file_paths: Optional list of file paths to bias results.
- limit: Max number of context items (default 10).
Returns a list of context items with fact, name, and created_at."""


class GetProjectContextInput(BaseModel):
    """Input schema for get_project_context tool."""

    project_id: str = Field(description="The project/repository ID (UUID)")
    query: Optional[str] = Field(None, description="What to search for")
    branch: Optional[str] = Field(None, description="Optional branch to bias results")
    file_paths: Optional[List[str]] = Field(None, description="Optional file paths to bias results")
    limit: Optional[int] = Field(10, description="Max number of context items to return")


class GetProjectContextTool:
    """Tool that queries the context graph (Graphiti) for a project."""

    name = "Get Project Context"
    description = GET_PROJECT_CONTEXT_DESCRIPTION

    def __init__(self, db: Any, user_id: str):
        self.db = db
        self.user_id = user_id

    async def arun(
        self,
        project_id: str,
        query: Optional[str] = None,
        branch: Optional[str] = None,
        file_paths: Optional[List[str]] = None,
        limit: Optional[int] = 10,
    ) -> List[dict]:
        """Async entry (used when agent runs in async context)."""
        return await asyncio.to_thread(
            self.run, project_id, query, branch, file_paths, limit
        )

    def run(
        self,
        project_id: str,
        query: Optional[str] = None,
        branch: Optional[str] = None,
        file_paths: Optional[List[str]] = None,
        limit: Optional[int] = 10,
    ) -> List[dict]:
        """Sync entry: query Graphiti and return normalized context items."""
        from app.core.config_provider import config_provider
        from app.modules.context_graph.graphiti_client import ContextGraphClient

        if not config_provider.get_context_graph_config().get("enabled"):
            return []

        parts = []
        if branch:
            parts.append(f"branch:{branch}")
        if query and str(query).strip():
            parts.append(str(query).strip())
        if file_paths:
            parts.extend(file_paths)
        # When no query: use a broad fallback so Graphiti returns recent PR/commit context.
        # Searching by project_id (UUID) would match nothing in episode text.
        search_text = " ".join(parts) if parts else "pull request merged commit recent changes"
        limit = limit or 10

        client = ContextGraphClient()
        try:
            edges = asyncio.run(client.search(project_id, search_text, limit))
        except Exception as e:
            logger.warning("get_project_context search failed: %s", e)
            return []

        out = []
        for e in edges or []:
            out.append({
                "fact": getattr(e, "fact", str(e)),
                "name": getattr(e, "name", ""),
                "created_at": str(getattr(e, "created_at", "")),
            })
        return out


def get_project_context_tool(sql_db: Any, user_id: str) -> StructuredTool:
    """Build the StructuredTool for get_project_context."""
    tool = GetProjectContextTool(sql_db, user_id)
    return StructuredTool.from_function(
        coroutine=tool.arun,
        func=tool.run,
        name="get_project_context",
        description=GET_PROJECT_CONTEXT_DESCRIPTION,
        args_schema=GetProjectContextInput,
    )
