import asyncio
from typing import Any, Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.modules.context_graph.wiring import PotpieContextEngineSettings
from app.modules.projects.projects_model import Project
from adapters.outbound.neo4j.structural import Neo4jStructuralAdapter
from application.use_cases.query_context import get_change_history as ce_get_change_history


class GetChangeHistoryInput(BaseModel):
    project_id: str = Field(description="Project ID (UUID)")
    function_name: Optional[str] = Field(default=None, description="Optional function/class name")
    file_path: Optional[str] = Field(default=None, description="Optional repository-relative file path")
    limit: int = Field(default=10, description="Max records to return")


class GetChangeHistoryTool:
    name = "Get Change History"

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
        function_name: Optional[str] = None,
        file_path: Optional[str] = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self.run, project_id, function_name, file_path, limit)

    def run(
        self,
        project_id: str,
        function_name: Optional[str] = None,
        file_path: Optional[str] = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        self._assert_project_access(project_id)
        if not self._settings.is_enabled():
            return []
        return ce_get_change_history(
            self._structural,
            project_id,
            function_name=function_name,
            file_path=file_path,
            limit=limit,
        )


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
