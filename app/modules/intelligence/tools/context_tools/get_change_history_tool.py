import asyncio
from typing import Any, Optional

from langchain_core.tools import StructuredTool
from neo4j import GraphDatabase
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.config_provider import config_provider
from app.modules.projects.projects_model import Project
from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)


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

        query = """
        MATCH (n:NODE {repoId: $project_id})
        WHERE ($file_path IS NULL OR n.file_path = $file_path)
          AND ($function_name IS NULL OR toLower(coalesce(n.name, '')) CONTAINS toLower($function_name))
        OPTIONAL MATCH (n)-[:MODIFIED_IN]->(pr:Entity)
        WHERE 'PullRequest' IN labels(pr)
        OPTIONAL MATCH (pr)-[:Fixes]->(iss:Entity)
        OPTIONAL MATCH (n)-[:HAS_DECISION]->(dec:Entity)
        WITH n, pr,
             collect(DISTINCT coalesce(iss.issue_number, iss.number, iss.name)) AS fixed_issues,
             collect(DISTINCT coalesce(dec.decision_made, dec.name, dec.summary)) AS decisions
        WHERE pr IS NOT NULL
        RETURN coalesce(pr.pr_number, pr.number) AS pr_number,
               coalesce(pr.title, pr.name) AS title,
               coalesce(pr.why_summary, pr.summary, '') AS why_summary,
               coalesce(pr.change_type, '') AS change_type,
               coalesce(pr.feature_area, '') AS feature_area,
               fixed_issues,
               decisions
        ORDER BY pr_number DESC
        LIMIT $limit
        """
        with self.driver.session() as session:
            res = session.run(
                query,
                project_id=project_id,
                function_name=function_name,
                file_path=file_path,
                limit=max(1, min(limit, 100)),
            )
            return [record.data() for record in res]


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
