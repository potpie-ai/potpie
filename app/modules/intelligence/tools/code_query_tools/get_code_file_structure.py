import asyncio
import logging
from typing import Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.modules.code_provider.code_provider_service import CodeProviderService
from app.modules.intelligence.tools.tool_utils import truncate_response

logger = logging.getLogger(__name__)

# Character limit for tool responses to prevent sending insanely large content to LLM
MAX_RESPONSE_LENGTH = 80000  # 80k characters

DEFAULT_STRUCTURE_DEPTH = 4
MAX_STRUCTURE_DEPTH = 6
MIN_STRUCTURE_DEPTH = 1


class RepoStructureRequest(BaseModel):
    project_id: str
    path: Optional[str] = None
    max_depth: Optional[int] = Field(
        default=None,
        ge=MIN_STRUCTURE_DEPTH,
        le=MAX_STRUCTURE_DEPTH,
        description=f"Optional traversal depth; defaults to {DEFAULT_STRUCTURE_DEPTH} and is capped at {MAX_STRUCTURE_DEPTH}.",
    )


class GetCodeFileStructureTool:
    name = "get_code_file_structure"
    description = """Retrieve the hierarchical file structure of a specified repository.
        :param project_id: string, the repository ID (UUID) to get the file structure for.
        :param max_depth: optional integer to control directory traversal depth (defaults to provider setting, typically 4).

            example:
            {
                "project_id": "550e8400-e29b-41d4-a716-446655440000"
            }

        Returns string containing the hierarchical file structure.

        ⚠️ IMPORTANT: Large repositories may result in truncated responses (max 80,000 characters).
        If the response is truncated, a notice will be included indicating the truncation occurred.
        """

    def __init__(self, db: Session):
        self.cp_service = CodeProviderService(db)

    async def fetch_repo_structure(
        self, project_id: str, path: Optional[str] = None, max_depth: Optional[int] = None
    ) -> str:
        effective_depth = self._sanitize_depth(max_depth)
        return await self.cp_service.get_project_structure_async(
            project_id, path, effective_depth
        )

    async def arun(
        self, project_id: str, path: Optional[str] = None, max_depth: Optional[int] = None
    ) -> str:
        try:
            result = await self.fetch_repo_structure(project_id, path, max_depth)
            truncated_result = truncate_response(result)
            if len(result) > 80000:
                logger.warning(
                    f"get_code_file_structure output truncated from {len(result)} to 80000 characters "
                    f"for project {project_id}, path={path}"
                )
            return truncated_result
        except Exception as e:
            logger.error(f"Error fetching file structure: {e}")
            return "error fetching data"

    def run(self, project_id: str, path: Optional[str] = None, max_depth: Optional[int] = None) -> str:
        try:
            result = asyncio.run(self.fetch_repo_structure(project_id, path, max_depth))
            truncated_result = truncate_response(result)
            if len(result) > 80000:
                logger.warning(
                    f"get_code_file_structure output truncated from {len(result)} to 80000 characters "
                    f"for project {project_id}, path={path}"
                )
            return truncated_result
        except Exception as e:
            logger.error(f"Error fetching file structure: {e}")
            return "error fetching data"

    def _sanitize_depth(self, max_depth: Optional[int]) -> int:
        depth = max_depth if max_depth is not None else DEFAULT_STRUCTURE_DEPTH
        try:
            depth = int(depth)
        except Exception:
            depth = DEFAULT_STRUCTURE_DEPTH
        return max(MIN_STRUCTURE_DEPTH, min(depth, MAX_STRUCTURE_DEPTH))


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
        the path for the subdir_name should be dir_name/subdir_name. Optionally pass max_depth to limit traversal (default is 4, capped at 6).

        ⚠️ IMPORTANT: Large repositories may result in truncated responses (max 80,000 characters).
        If the response is truncated, a notice will be included indicating the truncation occurred.""",
        coroutine=GetCodeFileStructureTool(db).arun,
        func=GetCodeFileStructureTool(db).run,
        args_schema=RepoStructureRequest,
    )
