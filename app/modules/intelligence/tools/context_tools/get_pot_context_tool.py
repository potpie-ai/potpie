from typing import Any, Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.modules.context_graph.wiring import build_container_for_user_session
from domain.graph_query import preset_semantic_search


class GetPotContextInput(BaseModel):
    pot_id: str = Field(
        description="Context graph pot scope id (Graphiti group_id for this workspace)"
    )
    query: str = Field(description="Natural language query for context graph search")
    limit: int = Field(default=8, description="Max results")
    node_labels: Optional[list[str]] = Field(
        default=None,
        description="Optional label filters, e.g. ['PullRequest', 'Decision', 'Issue']",
    )


class GetPotContextTool:
    def __init__(self, sql_db: Session, user_id: str):
        self.sql_db = sql_db
        self.user_id = user_id
        self._container = build_container_for_user_session(sql_db, user_id)
        self._settings = self._container.settings
        self._context_graph = self._container.context_graph

    def _assert_pot_access(self, pot_id: str) -> None:
        if self._container.pots.resolve_pot(pot_id) is None:
            raise ValueError("Pot scope not found for user")

    async def arun(
        self,
        pot_id: str,
        query: str,
        limit: int = 8,
        node_labels: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        self._assert_pot_access(pot_id)
        if not self._settings.is_enabled() or self._context_graph is None:
            return []
        out = await self._context_graph.query_async(
            preset_semantic_search(
                pot_id=pot_id,
                query=query,
                limit=max(1, min(limit, 50)),
                node_labels=node_labels,
            )
        )
        return list(out.result or [])

    def run(
        self,
        pot_id: str,
        query: str,
        limit: int = 8,
        node_labels: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        self._assert_pot_access(pot_id)
        if not self._settings.is_enabled() or self._context_graph is None:
            return []
        out = self._context_graph.query(
            preset_semantic_search(
                pot_id=pot_id,
                query=query,
                limit=max(1, min(limit, 50)),
                node_labels=node_labels,
            )
        )
        return list(out.result or [])


def get_pot_context_tool(sql_db: Session, user_id: str) -> StructuredTool:
    instance = GetPotContextTool(sql_db, user_id)
    return StructuredTool.from_function(
        coroutine=instance.arun,
        func=instance.run,
        name="get_pot_context",
        description="Semantic context search over pot-scoped Graphiti entities with optional label filters.",
        args_schema=GetPotContextInput,
    )
