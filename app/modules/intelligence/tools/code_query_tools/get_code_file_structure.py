import asyncio
from typing import Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.modules.code_provider.code_provider_service import CodeProviderService


class RepoStructureRequest(BaseModel):
    repo_id: str
    path: Optional[str] = None


class RepoStructureService:
    def __init__(self, db: Session):
        self.github_service = CodeProviderService(db)

    async def fetch_repo_structure(
        self, repo_id: str, path: Optional[str] = None
    ) -> str:
        return await self.github_service.get_project_structure_async(repo_id, path)

    async def run(self, repo_id: str, path: Optional[str] = None) -> str:
        return await self.fetch_repo_structure(repo_id, path)

    def run_tool(self, repo_id: str, path: Optional[str] = None) -> str:
        return asyncio.run(self.fetch_repo_structure(repo_id, path))


def get_code_file_structure_tool(db: Session) -> StructuredTool:
    return StructuredTool(
        name="Get Code File Structure",
        description="""Retrieve the hierarchical file structure of a specified repository or subdirectory in a repository. Expecting 'repo_id' as a required input and an optional 'path' to specify a subdirectory. If no path is provided, it will assume the root by default.
        For input :
        ```
            dir_name
                subdir_name
                    ...
                filename.extension
        ```
        the path for the subdir_name should be dir_name/subdir_name""",
        coroutine=RepoStructureService(db).run,
        func=RepoStructureService(db).run_tool,
        args_schema=RepoStructureRequest,
    )
