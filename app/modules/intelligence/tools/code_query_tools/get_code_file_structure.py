import asyncio
from typing import Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.modules.code_provider.code_provider_service import CodeProviderService


class RepoStructureRequest(BaseModel):
    project_id: str
    path: Optional[str] = None


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
        self.github_service = CodeProviderService(db)

    async def fetch_repo_structure(
        self, project_id: str, path: Optional[str] = None
    ) -> str:
        return await self.github_service.get_project_structure_async(project_id, path)

    async def arun(self, project_id: str, path: Optional[str] = None) -> str:
        try:
            return await self.fetch_repo_structure(project_id, path)
        except:
            return "error fetching data"

    def run(self, project_id: str, path: Optional[str] = None) -> str:
        try:
            return asyncio.run(self.fetch_repo_structure(project_id, path))
        except:
            return "error fetching data"


def get_code_file_structure_tool(db: Session) -> StructuredTool:
    return StructuredTool(
        name="get_code_file_structure",
        description="""Retrieve the hierarchical file structure of a specified repository or subdirectory in a repository. Expecting 'project_id' as a required input and an optional 'path' to specify a subdirectory. If no path is provided, it will assume the root by default.
        For input :
        ```
            dir_name
                subdir_name
                    ...
                filename.extension
        ```
        the path for the subdir_name should be dir_name/subdir_name""",
        coroutine=GetCodeFileStructureTool(db).arun,
        func=GetCodeFileStructureTool(db).run,
        args_schema=RepoStructureRequest,
    )
