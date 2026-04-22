import asyncio
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.modules.context_graph.wiring import build_container_for_user_session
from domain.graph_query import preset_file_owners


class GetFileOwnerInput(BaseModel):
    pot_id: str = Field(description="Context graph pot scope id (UUID)")
    file_path: str = Field(description="Repository-relative file path")
    limit: int = Field(default=5, description="Max owners to return")


class GetFileOwnerTool:
    def __init__(self, sql_db: Session, user_id: str):
        self.sql_db = sql_db
        self.user_id = user_id
        self._container = build_container_for_user_session(sql_db, user_id)
        self._settings = self._container.settings
        self._context_graph = self._container.context_graph

    def _assert_pot_access(self, pot_id: str) -> None:
        if self._container.pots.resolve_pot(pot_id) is None:
            raise ValueError("Pot scope not found for user")

    async def arun(self, pot_id: str, file_path: str, limit: int = 5) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self.run, pot_id, file_path, limit)

    def run(self, pot_id: str, file_path: str, limit: int = 5) -> list[dict[str, Any]]:
        self._assert_pot_access(pot_id)
        if not self._settings.is_enabled() or self._context_graph is None:
            return []
        out = self._context_graph.query(
            preset_file_owners(pot_id=pot_id, file_path=file_path, limit=limit)
        )
        return list(out.result or [])


def get_file_owner_tool(sql_db: Session, user_id: str) -> StructuredTool:
    instance = GetFileOwnerTool(sql_db, user_id)
    return StructuredTool.from_function(
        coroutine=instance.arun,
        func=instance.run,
        name="get_file_owner",
        description="Get likely file owners by PR touch history and recency.",
        args_schema=GetFileOwnerInput,
    )
