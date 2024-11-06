from typing import List
import asyncio

from langchain_core.tools import StructuredTool
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.modules.github.github_service import GithubService
from app.modules.intelligence.tools.tool_schema import ToolParameter


class GetCodeFileStructureToolRequest(BaseModel):
    project_id: str


class GetCodeFileStructureTool:
    name = "get_code_file_structure"
    description = """Retrieve the hierarchical file structure of a specified repository.
        :param project_id: string, the repository ID (UUID) to get the file structure for.

            example:
            {
                "project_id": "550e8400-e29b-41d4-a716-446655440000"
            }
            
        Returns string containing the hierarchical file structure.
        """

    def __init__(self, db: Session):
        self.github_service = GithubService(db)


    async def fetch_repo_structure(self, project_id: str) -> str:
        return await self.github_service.get_project_structure_async(project_id)

    async def arun(self, project_id: str) -> str:
        return await self.fetch_repo_structure(project_id)

    def run(self, project_id: str) -> str:
        return asyncio.run(self.fetch_repo_structure(project_id))


def get_code_file_structure_tool(db: Session) -> StructuredTool:
    return StructuredTool(
        name="get_code_file_structure",
        description="Retrieve the hierarchical file structure of a specified repository.",
        coroutine=GetCodeFileStructureTool(db).arun,
        func=GetCodeFileStructureTool(db).run,
        args_schema=GetCodeFileStructureToolRequest,
    )
