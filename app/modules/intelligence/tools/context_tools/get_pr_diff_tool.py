import asyncio
from typing import Any, Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.modules.context_graph.wiring import build_container_for_user_session
from domain.graph_query import preset_pr_diff


class GetPrDiffInput(BaseModel):
    pot_id: str = Field(description="Context graph pot scope id (UUID)")
    pr_number: int = Field(ge=1, description="GitHub pull request number")
    file_path: Optional[str] = Field(
        default=None,
        description="Optional repository-relative file path filter",
    )
    limit: int = Field(default=30, description="Max file diff rows to return")


class GetPrDiffTool:
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
        pr_number: int,
        file_path: Optional[str] = None,
        limit: int = 30,
    ) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self.run, pot_id, pr_number, file_path, limit)

    def run(
        self,
        pot_id: str,
        pr_number: int,
        file_path: Optional[str] = None,
        limit: int = 30,
    ) -> list[dict[str, Any]]:
        self._assert_pot_access(pot_id)
        if not self._settings.is_enabled() or self._context_graph is None:
            return []
        out = self._context_graph.query(
            preset_pr_diff(
                pot_id=pot_id,
                pr_number=pr_number,
                file_path=file_path,
                limit=limit,
            )
        )
        return list(out.result or [])


def get_pr_diff_tool(sql_db: Session, user_id: str) -> StructuredTool:
    instance = GetPrDiffTool(sql_db, user_id)
    return StructuredTool.from_function(
        coroutine=instance.arun,
        func=instance.run,
        name="get_pr_diff",
        description=(
            "Get file-level diff snippets for a PR from context graph structural edges. "
            "Returns file_path, patch_excerpt, additions, deletions, and status."
        ),
        args_schema=GetPrDiffInput,
    )
