import asyncio
from typing import Any, Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.modules.context_graph.wiring import build_container_for_user_session
from domain.graph_query import preset_decisions


class GetDecisionsInput(BaseModel):
    pot_id: str = Field(description="Context graph pot scope id (UUID)")
    file_path: Optional[str] = Field(default=None, description="Optional file path filter")
    function_name: Optional[str] = Field(default=None, description="Optional function/class filter")
    limit: int = Field(default=20, description="Max decisions to return")


class GetDecisionsTool:
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
        if not self._settings.is_enabled() or self._context_graph is None:
            return []
        out = self._context_graph.query(
            preset_decisions(
                pot_id=pot_id,
                file_path=file_path,
                function_name=function_name,
                limit=limit,
            )
        )
        return list(out.result or [])


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
