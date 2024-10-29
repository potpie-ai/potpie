from langchain_core.tools import StructuredTool
from pydantic import BaseModel
from sqlalchemy.orm import Session
import asyncio

from app.modules.github.github_service import GithubService


class RepoStructureRequest(BaseModel):
    repo_id: str


class RepoStructureService:
    def __init__(self, db: Session):
        self.github_service = GithubService(db)

    def fetch_repo_structure(self, repo_id: str) -> str:
        return self.github_service.get_project_structure(repo_id)
        
    async def arun(self, repo_id: str) -> str:
        return await self.fetch_repo_structure(repo_id)


def get_code_file_structure_tool(db: Session) -> StructuredTool:
    return StructuredTool(
        name="get_code_file_structure",
        description="Retrieve the hierarchical file structure of a specified repository.",
        coroutine=RepoStructureService(db).arun,
        args_schema=RepoStructureRequest,
    )
