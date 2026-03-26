import asyncio
from typing import Any, Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.modules.context_graph.wiring import PotpieContextEngineSettings
from app.modules.projects.projects_model import Project
from adapters.outbound.neo4j.structural import Neo4jStructuralAdapter
from application.use_cases.query_context import get_pr_diff as ce_get_pr_diff


class GetPrDiffInput(BaseModel):
    project_id: str = Field(description="Project ID (UUID)")
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
        self._settings = PotpieContextEngineSettings()
        self._structural = Neo4jStructuralAdapter(self._settings)

    def _assert_project_access(self, project_id: str) -> None:
        project = self.sql_db.query(Project).filter(Project.id == project_id).first()
        if not project or project.user_id != self.user_id:
            raise ValueError("Project not found for user")

    async def arun(
        self,
        project_id: str,
        pr_number: int,
        file_path: Optional[str] = None,
        limit: int = 30,
    ) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self.run, project_id, pr_number, file_path, limit)

    def run(
        self,
        project_id: str,
        pr_number: int,
        file_path: Optional[str] = None,
        limit: int = 30,
    ) -> list[dict[str, Any]]:
        self._assert_project_access(project_id)
        if not self._settings.is_enabled():
            return []
        return ce_get_pr_diff(
            self._structural,
            project_id,
            pr_number,
            file_path=file_path,
            limit=limit,
        )


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
