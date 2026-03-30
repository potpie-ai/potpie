import asyncio
from typing import Any, Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.modules.context_graph.wiring import PotpieContextEngineSettings
from app.modules.projects.projects_model import Project
from adapters.outbound.neo4j.structural import Neo4jStructuralAdapter
from application.use_cases.query_context import get_decisions as ce_get_decisions


class GetDecisionsInput(BaseModel):
    pot_id: str = Field(description="Context graph pot scope id (UUID)")
    file_path: Optional[str] = Field(default=None, description="Optional file path filter")
    function_name: Optional[str] = Field(default=None, description="Optional function/class filter")
    limit: int = Field(default=20, description="Max decisions to return")


class GetDecisionsTool:
    def __init__(self, sql_db: Session, user_id: str):
        self.sql_db = sql_db
        self.user_id = user_id
        self._settings = PotpieContextEngineSettings()
        self._structural = Neo4jStructuralAdapter(self._settings)

    def _assert_pot_access(self, pot_id: str) -> None:
        project = self.sql_db.query(Project).filter(Project.id == pot_id).first()
        if not project or project.user_id != self.user_id:
            raise ValueError("Pot scope not found for user")

    async def arun(
        self,
        pot_id: str,
        file_path: Optional[str] = None,
        function_name: Optional[str] = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self.run, pot_id, file_path, function_name, limit)

    def run(
        self,
        pot_id: str,
        file_path: Optional[str] = None,
        function_name: Optional[str] = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        self._assert_pot_access(pot_id)
        if not self._settings.is_enabled():
            return []
        return ce_get_decisions(
            self._structural,
            pot_id,
            file_path=file_path,
            function_name=function_name,
            limit=limit,
        )


def get_decisions_tool(sql_db: Session, user_id: str) -> StructuredTool:
    instance = GetDecisionsTool(sql_db, user_id)
    return StructuredTool.from_function(
        coroutine=instance.arun,
        func=instance.run,
        name="get_decisions",
        description=(
            "Get design decisions: from code nodes (HAS_DECISION) and from PR review threads / "
            "PR conversation linked in the context graph. Use file_path to narrow to a path; "
            "omit filters for a mix of code-linked and PR-linked decisions."
        ),
        args_schema=GetDecisionsInput,
    )
