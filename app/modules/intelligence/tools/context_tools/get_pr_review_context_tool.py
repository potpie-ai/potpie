import asyncio
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.modules.context_graph.wiring import build_container_for_user_session
from domain.graph_query import preset_pr_review_context


class GetPrReviewContextInput(BaseModel):
    pot_id: str = Field(description="Context graph pot scope id (UUID)")
    pr_number: int = Field(ge=1, description="GitHub pull request number")


class GetPrReviewContextTool:
    def __init__(self, sql_db: Session, user_id: str):
        self.sql_db = sql_db
        self.user_id = user_id
        self._container = build_container_for_user_session(sql_db, user_id)
        self._settings = self._container.settings
        self._context_graph = self._container.context_graph

    def _assert_pot_access(self, pot_id: str) -> None:
        if self._container.pots.resolve_pot(pot_id) is None:
            raise ValueError("Pot scope not found for user")

    async def arun(self, pot_id: str, pr_number: int) -> dict[str, Any]:
        return await asyncio.to_thread(self.run, pot_id, pr_number)

    def run(self, pot_id: str, pr_number: int) -> dict[str, Any]:
        self._assert_pot_access(pot_id)
        if not self._settings.is_enabled() or self._context_graph is None:
            return {
                "found": False,
                "pr_number": pr_number,
                "pr_title": None,
                "pr_summary": None,
                "review_threads": [],
            }
        out = self._context_graph.query(
            preset_pr_review_context(pot_id=pot_id, pr_number=pr_number)
        )
        return dict(out.result or {})


def get_pr_review_context_tool(sql_db: Session, user_id: str) -> StructuredTool:
    instance = GetPrReviewContextTool(sql_db, user_id)
    return StructuredTool.from_function(
        coroutine=instance.arun,
        func=instance.run,
        name="get_pr_review_context",
        description=(
            "For a specific merged PR number, return title/summary plus structured discussions: "
            "line-level review threads and the main PR conversation (issue comments). Use for why/"
            "rationale questions when you know or can infer the PR # (e.g. from get_pot_context)."
        ),
        args_schema=GetPrReviewContextInput,
    )
