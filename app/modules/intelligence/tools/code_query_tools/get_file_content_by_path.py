import logging
from typing import Optional, Type, Dict, Any
from pydantic import BaseModel, Field
from redis import Redis
from sqlalchemy.orm import Session
from langchain_core.tools import StructuredTool
from app.modules.code_provider.code_provider_service import CodeProviderService
from app.modules.projects.projects_service import ProjectService
from app.core.config_provider import config_provider


class FetchFileToolInput(BaseModel):
    project_id: str = Field(
        ..., description="Project ID that references the repository"
    )
    file_path: str = Field(..., description="Path to the file within the repo")
    start_line: Optional[int] = Field(
        None, description="First line to fetch (1-based, inclusive)"
    )
    end_line: Optional[int] = Field(None, description="Last line to fetch (inclusive)")


class FetchFileTool:
    name: str = "fetch_file"
    description: str = (
        """Fetch file content from a repository using the project_id and file path.
        Returns the content between optional start_line and end_line.
        Make sure the file exists before querying for it, confirm it by checking the file structure.
        File content is hashed for caching purposes. Cache won't be used if start_line or end_line are different.
        Use with_line_numbers to include line numbers in the response to better understand the context and location of the code.

        IMPORTANT LIMITS:
        - Maximum 1200 lines can be fetched at once
        - If the entire file is requested and it has more than 1200 lines, an error will be returned
        - If start_line and end_line span more than 1200 lines, an error will be returned
        - Always use start_line and end_line to fetch specific sections of large files

        param project_id: string, the repository ID (UUID) to get the file content for.
        param file_path: string, the path to the file in the repository.
        param with_line_numbers: bool, whether to include line numbers in the response.
        param start_line: int, the first line to fetch (1-based, inclusive).
        param end_line: int, the last line to fetch (inclusive).

        IMPORTANT: Use line numbers as much as possible, some files are large. Use other tools to access small parts of the file.
        You can use knowledge graph tools and node_ids to fetch code snippets

        example:
        {
            "project_id": "550e8400-e29b-41d4-a716-446655440000",
            "file_path": "src/main.py",
            "start_line": 1,
            "end_line": 10
        }
        Returns string containing the content of the file.

        If with_line_numbers is true, the content will be formatted with line numbers, starting from 1.

        format:
        line_number:line

        no extra spaces or tabs in between.

        Example:
        1:def hello_world():
        2:    print("Hello, world!")
        3:
        4:hello_world()

        """
    )
    args_schema: Type[BaseModel] = FetchFileToolInput

    def __init__(self, sql_db: Session, user_id: str, internal_call: bool = False):
        self.sql_db = sql_db
        self.user_id = user_id
        self.cp_service = CodeProviderService(self.sql_db)
        self.project_service = ProjectService(self.sql_db)
        self.redis = Redis.from_url(config_provider.get_redis_url())
        self.internal_call = internal_call

    def _get_project_details(self, project_id: str) -> Dict[str, str]:
        details = self.project_service.get_project_from_db_by_id_sync(project_id)
        if not details or "project_name" not in details:
            raise ValueError(f"Cannot find repo details for project_id: {project_id}")
        if details["user_id"] != self.user_id:
            raise ValueError(
                f"Cannot find repo details for project_id: {project_id} for current user"
            )
        return details

    def with_line_numbers(
        self, content: str, include_line_number: bool, starting_line: int
    ) -> str:
        if include_line_number:
            lines = content.splitlines()
            numbered_lines = [
                f"{starting_line+ i}:{line}" for i, line in enumerate(lines)
            ]
            return "\n".join(numbered_lines)
        return content

    def _check_line_limits(
        self, content: str, start_line: Optional[int], end_line: Optional[int]
    ) -> Dict[str, Any]:
        """Check if the content or requested range exceeds 1200 lines"""
        lines = content.splitlines()
        total_lines = len(lines)

        # If specific lines are requested, check the range
        if start_line is not None and end_line is not None:
            requested_lines = end_line - start_line + 1
            if requested_lines > 1200 and not self.internal_call:
                return {
                    "success": False,
                    "error": f"Too much content requested. You asked for {requested_lines} lines (lines {start_line}-{end_line}), but maximum allowed is 1200 lines. Please make your request more targeted by specifying a smaller line range.",
                    "content": None,
                    "total_lines": total_lines,
                }
        # If no specific lines requested and total file exceeds 1200 lines
        elif (
            start_line is None
            and end_line is None
            and total_lines > 1200
            and not self.internal_call
        ):
            return {
                "success": False,
                "error": f"Too much content requested. The file has {total_lines} lines, but maximum allowed is 1200 lines. Please specify start_line and end_line parameters to fetch a smaller section of the file.",
                "content": None,
                "total_lines": total_lines,
            }

        return {"success": True}

    def _run(
        self,
        project_id: str,
        file_path: str,
        with_line_numbers: bool = False,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
    ) -> Dict[str, Any]:
        try:
            details = self._get_project_details(project_id)
            # Modify cache key to reflect that we're only caching the specific path
            cache_key = f"file_content:{project_id}:exact_path_{file_path}:start_line_{start_line}:end_line_{end_line}"
            result = self.redis.get(cache_key)

            if result:
                cached_content = result.decode("utf-8")
                # Check line limits for cached content
                limit_check = self._check_line_limits(
                    cached_content, start_line, end_line
                )
                if not limit_check["success"]:
                    return limit_check

                content = self.with_line_numbers(
                    cached_content,
                    with_line_numbers,
                    starting_line=start_line or 1,
                )
                return {
                    "success": True,
                    "content": content,
                }

            content = self.cp_service.get_file_content(
                repo_name=details["project_name"],
                file_path=file_path,
                branch_name=details["branch_name"],
                start_line=start_line if start_line is not None else None,
                end_line=end_line if end_line is not None else None,
                project_id=project_id,
                commit_id=details["commit_id"],
            )

            # Check line limits for new content
            limit_check = self._check_line_limits(content, start_line, end_line)
            if not limit_check["success"]:
                return limit_check

            self.redis.setex(cache_key, 600, content)  # Cache for 10 minutes
            content = self.with_line_numbers(
                content, with_line_numbers, starting_line=start_line or 1
            )
            return {
                "success": True,
                "content": content,
            }
        except Exception as e:
            logging.exception(f"Failed to fetch file content for {file_path}: {str(e)}")
            return {"success": False, "error": str(e), "content": None}

    async def _arun(
        self,
        project_id: str,
        file_path: str,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
    ) -> Dict[str, Any]:
        # Synchronously for compatibility, like GithubTool
        return self._run(project_id, file_path, start_line, end_line)


def fetch_file_tool(
    sql_db: Session, user_id: str, internal_call: bool = False
) -> StructuredTool:
    tool_instance = FetchFileTool(sql_db, user_id, internal_call=internal_call)
    return StructuredTool.from_function(
        coroutine=tool_instance._arun,
        func=tool_instance._run,
        name="fetch_file",
        description=tool_instance.description,
        args_schema=FetchFileToolInput,
    )
