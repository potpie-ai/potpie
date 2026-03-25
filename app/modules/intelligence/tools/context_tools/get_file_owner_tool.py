import asyncio
from typing import Any

from langchain_core.tools import StructuredTool
from neo4j import GraphDatabase
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.config_provider import config_provider
from app.modules.projects.projects_model import Project


class GetFileOwnerInput(BaseModel):
    project_id: str = Field(description="Project ID (UUID)")
    file_path: str = Field(description="Repository-relative file path")
    limit: int = Field(default=5, description="Max owners to return")


class GetFileOwnerTool:
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

    async def arun(self, project_id: str, file_path: str, limit: int = 5) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self.run, project_id, file_path, limit)

    def run(self, project_id: str, file_path: str, limit: int = 5) -> list[dict[str, Any]]:
        self._assert_project_access(project_id)
        query = """
        MATCH (f:FILE {repoId: $project_id, file_path: $file_path})-[:TOUCHED_BY]->(pr:Entity)
        WHERE 'PullRequest' IN labels(pr)
        OPTIONAL MATCH (pr)-[:AuthoredBy]->(dev:Entity)
        WITH coalesce(dev.github_login, dev.name, pr.author, 'unknown') AS github_login,
             count(DISTINCT pr) AS pr_count,
             max(coalesce(pr.merged_at, pr.updated_at, pr.created_at)) AS last_touched
        RETURN github_login, pr_count, last_touched
        ORDER BY pr_count DESC, last_touched DESC
        LIMIT $limit
        """
        with self.driver.session() as session:
            res = session.run(
                query,
                project_id=project_id,
                file_path=file_path,
                limit=max(1, min(limit, 50)),
            )
            return [record.data() for record in res]


def get_file_owner_tool(sql_db: Session, user_id: str) -> StructuredTool:
    instance = GetFileOwnerTool(sql_db, user_id)
    return StructuredTool.from_function(
        coroutine=instance.arun,
        func=instance.run,
        name="get_file_owner",
        description="Get likely file owners by PR touch history and recency.",
        args_schema=GetFileOwnerInput,
    )
