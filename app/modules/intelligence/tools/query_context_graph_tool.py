"""query_context_graph — stub tool for the debug agent.

Queries the context-graph service for code locations relevant to a
natural-language description.  The real context-graph service is under
separate development; this file ships the stub so:

  1. The tool contract (args schema + output schema) is locked in from day one.
  2. The debug agent's prompt can reference the tool name today.
  3. The future implementer can replace the stub body with the real service call
     (and update the factory signature in tool_service.py to take db/user_id if
     needed) without changing any agent consumers.

When ``available=False``, the agent must fall back to the discovery tools
listed in ``debug_hypothesis_contract.DISCOVERY_PRIORITY_ORDER`` (e.g.
``ask_knowledge_graph_queries``, ``search_text``, ``get_code_file_structure``).
"""

from typing import List, Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Pydantic models — output schema
# ---------------------------------------------------------------------------


class ContextGraphResult(BaseModel):
    """A single result entry returned by the context-graph service."""

    file: str
    symbol: Optional[str] = None
    snippet: Optional[str] = None  # a small code excerpt for the agent to read
    score: float = Field(..., ge=0.0, le=1.0, description="Relevance score, 0.0–1.0.")


class QueryContextGraphOutput(BaseModel):
    """Output shape for query_context_graph.

    ``available`` is ``False`` for the stub; ``True`` once the real service
    is wired.  ``results`` is always an empty list from the stub.
    ``message`` is a human-readable hint for the agent when ``available`` is
    ``False``; ``None`` when the real service returns hits.
    """

    available: bool
    results: List[ContextGraphResult]
    message: Optional[str] = None


# ---------------------------------------------------------------------------
# Input schema
# ---------------------------------------------------------------------------


class QueryContextGraphInput(BaseModel):
    query: str = Field(
        description=(
            "Natural-language description of what to find, e.g. "
            "'where is payment timeout handled'."
        )
    )
    project_id: str = Field(
        description="Project ID — same value the rest of the agent uses."
    )
    limit: int = Field(
        default=10,
        description="Maximum number of results to return (default 10).",
    )


# ---------------------------------------------------------------------------
# Stub implementation
# ---------------------------------------------------------------------------

_STUB_MESSAGE = (
    "context graph service not yet wired; agent should fall back to "
    "ask_knowledge_graph_queries / search_text / search_bash / file structure tools"
)


def query_context_graph(
    query: str,
    project_id: str,
    limit: int = 10,
) -> dict:
    """Stub implementation — returns available=False unconditionally.

    Args:
        query: Natural-language description of what to find.
        project_id: Project identifier.
        limit: Maximum results requested (ignored by stub).

    Returns:
        Serialised ``QueryContextGraphOutput`` with ``available=False``.
    """
    logger.debug(
        "query_context_graph called (stub) query={!r} project_id={!r} limit={}",
        query,
        project_id,
        limit,
    )
    return QueryContextGraphOutput(
        available=False,
        results=[],
        message=_STUB_MESSAGE,
    ).model_dump()


# ---------------------------------------------------------------------------
# LangChain StructuredTool factory
# ---------------------------------------------------------------------------


def query_context_graph_tool() -> StructuredTool:
    """Create the query_context_graph StructuredTool.

    No db/user_id required — the stub performs no I/O.
    """
    return StructuredTool.from_function(
        func=query_context_graph,
        name="query_context_graph",
        description=(
            "Query the context-graph service for code locations relevant to a "
            "natural-language description (e.g. 'where is payment timeout handled'). "
            "Returns a ranked list of file/symbol/snippet results with relevance scores. "
            "IMPORTANT: the service is not yet wired — the tool returns available=False "
            "today. When available=False, fall back to ask_knowledge_graph_queries, "
            "search_text, search_bash, get_code_file_structure, or fetch_file as "
            "directed by the discovery priority order."
        ),
        args_schema=QueryContextGraphInput,
    )
