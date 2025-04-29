import logging
from typing import Optional, Type, Dict, Any
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from langchain_core.tools import StructuredTool
from app.modules.code_provider.code_provider_service import CodeProviderService
from app.modules.projects.projects_service import ProjectService


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
        "Fetch file content from a repository using the project_id and file path. "
        "Returns the content between optional start_line and end_line."
    )
    args_schema: Type[BaseModel] = FetchFileToolInput

    def __init__(self, sql_db: Session, user_id: str):
        self.sql_db = sql_db
        self.user_id = user_id
        self.cp_service = CodeProviderService(self.sql_db)
        self.project_service = ProjectService(self.sql_db)

    def _get_project_details(self, project_id: str) -> Dict[str, str]:
        details = self.project_service.get_project_from_db_by_id_sync(project_id)
        if not details or "project_name" not in details:
            raise ValueError(f"Cannot find repo details for project_id: {project_id}")
        if details["user_id"] != self.user_id:
            raise ValueError(
                f"Cannot find repo details for project_id: {project_id} for current user"
            )
        return details

    def _run(
        self,
        project_id: str,
        file_path: str,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
    ) -> Dict[str, Any]:
        try:
            details = self._get_project_details(project_id)
            content = self.cp_service.get_file_content(
                repo_name=details["project_name"],
                file_path=file_path,
                branch_name=details["branch_name"],
                start_line=start_line,
                end_line=end_line,
                project_id=project_id,
            )
            return {"success": True, "content": content}
        except Exception as e:
            logging.exception(f"Failed to fetch file content: {str(e)}")
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


def fetch_file_tool(sql_db: Session, user_id: str):
    tool_instance = FetchFileTool(sql_db, user_id)
    return StructuredTool.from_function(
        coroutine=tool_instance._arun,
        func=tool_instance._run,
        name="fetch_file",
        description=tool_instance.description,
        args_schema=FetchFileToolInput,
    )
