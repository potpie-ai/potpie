from typing import List

from langchain_core.tools import StructuredTool
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.modules.github.github_service import GithubService
from app.modules.intelligence.tools.tool_schema import ToolParameter


class GetCodeFileStructureToolRequest(BaseModel):
    project_id: str


class GetCodeFileStructureTool:
    name = "get_code_file_structure"
    description = "Retrieve the hierarchical file structure of a specified repository."

    def __init__(self, db: Session):
        self.github_service = GithubService(db)

    def fetch_repo_structure(self, project_id: str) -> str:
        return self.github_service.get_project_structure(project_id)

    async def arun(self, project_id: str) -> str:
        return await self.fetch_repo_structure(project_id)

    def run(self, project_id: str) -> str:
        return self.fetch_repo_structure(project_id)

    @staticmethod
    def get_parameters() -> List[ToolParameter]:
        return [
            ToolParameter(
                name="project_id",
                type="string",
                description="The repository ID (UUID)",
                required=True,
            )
        ]


def get_code_file_structure_tool(db: Session) -> StructuredTool:
    return StructuredTool(
        name="get_code_file_structure",
        description="Retrieve the hierarchical file structure of a specified repository.",
        coroutine=GetCodeFileStructureTool(db).arun,
        func=GetCodeFileStructureTool(db).run,
        args_schema=GetCodeFileStructureToolRequest,
    )
