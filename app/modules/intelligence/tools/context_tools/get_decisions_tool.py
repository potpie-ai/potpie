import asyncio
from typing import Any, Optional

from langchain_core.tools import StructuredTool
from neo4j import GraphDatabase
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.config_provider import config_provider
from app.modules.projects.projects_model import Project


class GetDecisionsInput(BaseModel):
    project_id: str = Field(description="Project ID (UUID)")
    file_path: Optional[str] = Field(default=None, description="Optional file path filter")
    function_name: Optional[str] = Field(default=None, description="Optional function/class filter")
    limit: int = Field(default=20, description="Max decisions to return")


class GetDecisionsTool:
    def __init__(self, sql_db: Session, user_id: str):
        self.sql_db = sql_db
        self.user_id = user_id
        neo4j = config_provider.get_neo4j_config()
        self.driver = GraphDatabase.driver(
            neo4j.get("uri"),
            auth=(neo4j.get("username"), neo4j.get("password")),
        )

    def _assert_project_access(self, project_id: str) -> None:
        project = self.sql_db.query(Project).filter(Project.id == project_id).first()
        if not project or project.user_id != self.user_id:
            raise ValueError("Project not found for user")

    async def arun(
        self,
        project_id: str,
        file_path: Optional[str] = None,
        function_name: Optional[str] = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self.run, project_id, file_path, function_name, limit)

    def run(
        self,
        project_id: str,
        file_path: Optional[str] = None,
        function_name: Optional[str] = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        self._assert_project_access(project_id)

        query = """
        MATCH (n:NODE {repoId: $project_id})
        WHERE ($file_path IS NULL OR n.file_path = $file_path)
          AND ($function_name IS NULL OR toLower(coalesce(n.name, '')) CONTAINS toLower($function_name))
        OPTIONAL MATCH (n)-[:HAS_DECISION]->(d:Entity)
        OPTIONAL MATCH (n)-[:MODIFIED_IN]->(pr:Entity)
        WHERE d IS NOT NULL AND 'Decision' IN labels(d)
        RETURN DISTINCT
            coalesce(d.decision_made, d.name, d.summary) AS decision_made,
            coalesce(d.alternatives_rejected, '') AS alternatives_rejected,
            coalesce(d.rationale, d.summary, '') AS rationale,
            coalesce(pr.pr_number, pr.number) AS pr_number
        LIMIT $limit
        """
        with self.driver.session() as session:
            res = session.run(
                query,
                project_id=project_id,
                file_path=file_path,
                function_name=function_name,
                limit=max(1, min(limit, 100)),
            )
            return [record.data() for record in res]


def get_decisions_tool(sql_db: Session, user_id: str) -> StructuredTool:
    instance = GetDecisionsTool(sql_db, user_id)
    return StructuredTool.from_function(
        coroutine=instance.arun,
        func=instance.run,
        name="get_decisions",
        description="Get design decisions linked to code nodes in a file/function scope.",
        args_schema=GetDecisionsInput,
    )
