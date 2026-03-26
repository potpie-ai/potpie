import asyncio
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.modules.context_graph.wiring import PotpieContextEngineSettings
from app.modules.projects.projects_model import Project
from adapters.outbound.neo4j.structural import Neo4jStructuralAdapter
from application.use_cases.query_context import get_pr_review_context as ce_get_pr_review_context


class GetPrReviewContextInput(BaseModel):
    project_id: str = Field(description="Project ID (UUID)")
    pr_number: int = Field(ge=1, description="GitHub pull request number")


class GetPrReviewContextTool:
    def __init__(self, sql_db: Session, user_id: str):
        self.sql_db = sql_db
        self.user_id = user_id
        self._settings = PotpieContextEngineSettings()
        self._structural = Neo4jStructuralAdapter(self._settings)

    def _assert_project_access(self, project_id: str) -> None:
        project = self.sql_db.query(Project).filter(Project.id == project_id).first()
        if not project or project.user_id != self.user_id:
            raise ValueError("Project not found for user")

    async def arun(self, project_id: str, pr_number: int) -> dict[str, Any]:
        return await asyncio.to_thread(self.run, project_id, pr_number)

    def run(self, project_id: str, pr_number: int) -> dict[str, Any]:
        self._assert_project_access(project_id)
        if not self._settings.is_enabled():
            return {
                "found": False,
                "pr_number": pr_number,
                "pr_title": None,
                "pr_summary": None,
                "review_threads": [],
            }
        return ce_get_pr_review_context(self._structural, project_id, pr_number)


def get_pr_review_context_tool(sql_db: Session, user_id: str) -> StructuredTool:
    instance = GetPrReviewContextTool(sql_db, user_id)
    return StructuredTool.from_function(
        coroutine=instance.arun,
        func=instance.run,
        name="get_pr_review_context",
        description=(
            "For a specific merged PR number, return title/summary plus structured discussions: "
            "line-level review threads and the main PR conversation (issue comments). Use for why/"
            "rationale questions when you know or can infer the PR # (e.g. from get_project_context)."
        ),
        args_schema=GetPrReviewContextInput,
    )
