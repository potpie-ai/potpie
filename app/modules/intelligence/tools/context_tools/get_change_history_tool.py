import asyncio
from typing import Any, Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.modules.context_graph.wiring import build_container_for_user_session
from domain.graph_query import preset_change_history


class GetChangeHistoryInput(BaseModel):
    pot_id: str = Field(description="Context graph pot scope id (UUID)")
    function_name: Optional[str] = Field(default=None, description="Optional function/class name")
    file_path: Optional[str] = Field(default=None, description="Optional repository-relative file path")
    limit: int = Field(default=10, description="Max records to return")


class GetChangeHistoryTool:
    name = "Get Change History"

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
        function_name: Optional[str] = None,
        file_path: Optional[str] = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self.run, pot_id, function_name, file_path, limit)

    def run(
        self,
        pot_id: str,
        function_name: Optional[str] = None,
        file_path: Optional[str] = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        self._assert_pot_access(pot_id)
        if not self._settings.is_enabled() or self._context_graph is None:
            return []
        out = self._context_graph.query(
            preset_change_history(
                pot_id=pot_id,
                function_name=function_name,
                file_path=file_path,
                limit=limit,
            )
        )
        return list(out.result or [])


def get_change_history_tool(sql_db: Session, user_id: str) -> StructuredTool:
    instance = GetChangeHistoryTool(sql_db, user_id)
    return StructuredTool.from_function(
        coroutine=instance.arun,
        func=instance.run,
        name="get_change_history",
        description=(
            "Get change history for a function or file, including PRs, issue links, and decisions."
        ),
        args_schema=GetChangeHistoryInput,
    )
