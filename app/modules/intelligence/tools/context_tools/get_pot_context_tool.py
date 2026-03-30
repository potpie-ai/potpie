from typing import Any, Optional

from adapters.outbound.graphiti.episodic import GraphitiEpisodicAdapter
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.modules.context_graph.wiring import PotpieContextEngineSettings
from app.modules.projects.projects_model import Project
from application.use_cases.query_context import search_pot_context, search_pot_context_async


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
        self._episodic = GraphitiEpisodicAdapter(PotpieContextEngineSettings())

    def _assert_pot_access(self, pot_id: str) -> None:
        project = self.sql_db.query(Project).filter(Project.id == pot_id).first()
        if not project or project.user_id != self.user_id:
            raise ValueError("Pot scope not found for user")

    async def arun(
        self,
        pot_id: str,
        query: str,
        limit: int = 8,
        node_labels: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        self._assert_pot_access(pot_id)
        return await search_pot_context_async(
            self._episodic,
            pot_id,
            query,
            limit=max(1, min(limit, 50)),
            node_labels=node_labels,
        )

    def run(
        self,
        pot_id: str,
        query: str,
        limit: int = 8,
        node_labels: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        self._assert_pot_access(pot_id)
        return search_pot_context(
            self._episodic,
            pot_id,
            query,
            limit=max(1, min(limit, 50)),
            node_labels=node_labels,
        )


def get_pot_context_tool(sql_db: Session, user_id: str) -> StructuredTool:
    instance = GetPotContextTool(sql_db, user_id)
    return StructuredTool.from_function(
        coroutine=instance.arun,
        func=instance.run,
        name="get_pot_context",
        description="Semantic context search over pot-scoped Graphiti entities with optional label filters.",
        args_schema=GetPotContextInput,
    )
