from typing import Any, Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.modules.context_graph.graphiti_client import ContextGraphClient
from app.modules.projects.projects_model import Project


class GetProjectContextInput(BaseModel):
    project_id: str = Field(description="Project ID (UUID)")
    query: str = Field(description="Natural language query for context graph search")
    limit: int = Field(default=8, description="Max results")
    node_labels: Optional[list[str]] = Field(
        default=None,
        description="Optional label filters, e.g. ['PullRequest', 'Decision', 'Issue']",
    )


class GetProjectContextTool:
    def __init__(self, sql_db: Session, user_id: str):
        self.sql_db = sql_db
        self.user_id = user_id
        self.client = ContextGraphClient()

    def _assert_project_access(self, project_id: str) -> None:
        project = self.sql_db.query(Project).filter(Project.id == project_id).first()
        if not project or project.user_id != self.user_id:
            raise ValueError("Project not found for user")

    @staticmethod
    def _results_to_rows(results: list[Any]) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for item in results:
            output.append(
                {
                    "uuid": str(getattr(item, "uuid", "")),
                    "name": getattr(item, "name", None),
                    "summary": getattr(item, "summary", None),
                    "fact": getattr(item, "fact", None),
                }
            )
        return output

    async def arun(
        self,
        project_id: str,
        query: str,
        limit: int = 8,
        node_labels: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        self._assert_project_access(project_id)
        if not self.client.enabled:
            return []
        results = await self.client.search_async(
            project_id=project_id,
            query=query,
            limit=max(1, min(limit, 50)),
            node_labels=node_labels,
        )
        return self._results_to_rows(results)

    def run(
        self,
        project_id: str,
        query: str,
        limit: int = 8,
        node_labels: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        self._assert_project_access(project_id)
        if not self.client.enabled:
            return []
        results = self.client.search(
            project_id=project_id,
            query=query,
            limit=max(1, min(limit, 50)),
            node_labels=node_labels,
        )
        return self._results_to_rows(results)


def get_project_context_tool(sql_db: Session, user_id: str) -> StructuredTool:
    instance = GetProjectContextTool(sql_db, user_id)
    return StructuredTool.from_function(
        coroutine=instance.arun,
        func=instance.run,
        name="get_project_context",
        description="Semantic context search over project Graphiti entities with optional label filters.",
        args_schema=GetProjectContextInput,
    )
