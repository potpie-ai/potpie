import asyncio
import logging
import os
from typing import Dict, Optional

from fastapi import HTTPException
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from app.core.database import get_db
from app.modules.code_provider.code_provider_service import CodeProviderService
from app.modules.projects.projects_service import ProjectService


class PatchExtractionInput(BaseModel):
    project_id: str = Field(
        ..., description="The ID of the project being evaluated, this is a UUID."
    )
    base_branch: Optional[str] = Field(
        None, description="The base branch to compare against, this is a branch name. If none is provided, the default branch will be used."
    )


class PatchExtractionResponse(BaseModel):
    patches: Dict[str, str] = Field(..., description="Dictionary of file patches")


class PatchExtractionTool:
    name = "Get patches"
    description = """Extracts only the patch dictionary from a Git repository.
        :param project_id: string, the ID of the project being evaluated (UUID).
        :param base_branch: string, (optional) the base branch to compare against, this is a branch name. If none is provided, the default branch will be used.

            example:
            {
                "project_id": "550e8400-e29b-41d4-a716-446655440000"
            }

        Returns dictionary containing:
        - patches: Dict[str, str] - file patches
        """

    def __init__(self, sql_db, user_id):
        self.sql_db = sql_db
        self.user_id = user_id

    async def get_patches(self, project_id: str, base_branch: Optional[str] = None) -> PatchExtractionResponse:
        patches_dict = {}
        project_details = await ProjectService(self.sql_db).get_project_from_db_by_id(
            project_id
        )

        if project_details is None:
            raise HTTPException(status_code=400, detail="Project Details not found.")

        if project_details["user_id"] != self.user_id:
            raise ValueError(
                f"Project id {project_id} not found for user {self.user_id}"
            )

        repo_name = project_details["project_name"]
        branch_name = project_details["branch_name"]
        repo_path = project_details["repo_path"]
        
        # Use CodeProviderService to get patches
        code_service = CodeProviderService(self.sql_db)
        
        try:
            # Check if it's a local repository first
            if repo_path and os.path.exists(repo_path) and os.path.isdir(repo_path):
                # Use local service for local repositories
                if code_service.local_service:
                    patches_dict = code_service.local_service.get_local_repo_diff(
                        repo_path, branch_name, base_branch
                    )
                else:
                    raise Exception("Local repository service not available")
            else:
                # Use GitHub service for remote repositories
                if code_service.github_service:
                    github, _, _ = code_service.github_service.get_github_repo_details(
                        repo_name
                    )
                    repo = github.get_repo(repo_name)
                    base_branch = base_branch if base_branch else repo.default_branch
                    git_diff = repo.compare(base_branch, branch_name)
                    if git_diff.files == []:
                        git_diff = repo.compare(branch_name, base_branch)

                    patches_dict = {
                        file.filename: file.patch for file in git_diff.files if file.patch
                    }
                else:
                    raise Exception("GitHub service not configured and no local repository found")
            
            return PatchExtractionResponse(patches=patches_dict)
            
        except Exception as e:
            logging.error(f"Error extracting patches for project_id: {project_id}, error: {str(e)}")
            raise HTTPException(
                status_code=400, detail=f"Error while fetching patches: {str(e)}"
            )

    async def arun(self, project_id: str, base_branch: Optional[str] = None) -> PatchExtractionResponse:
        return await self.get_patches(project_id, base_branch)

    def run(self, project_id: str, base_branch: Optional[str] = None) -> PatchExtractionResponse:
        return asyncio.run(self.get_patches(project_id, base_branch))


def get_patch_extraction_tool(user_id: str) -> StructuredTool:
    """
    Get a LangChain Tool for extracting patches.
    """
    patch_extraction_tool = PatchExtractionTool(next(get_db()), user_id)
    return StructuredTool.from_function(
        coroutine=patch_extraction_tool.arun,
        func=patch_extraction_tool.run,
        name="Get patches",
        description="""
            Extract only the patches dictionary from a Git repository.
            This tool analyzes the differences between branches in a Git repository and retrieves only the patch information.
            Inputs:
            - project_id (str): The ID of the project being evaluated, this is a UUID.
            - base_branch (str): (optional) The base branch to compare against, this is a branch name. If none is provided, the default branch will be used.
            The output includes only a dictionary of file patches with filenames as keys and patch content as values.
            """,
        args_schema=PatchExtractionInput,
    ) 