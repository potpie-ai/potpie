from typing import Optional

from pydantic import BaseModel, Field


class ParsingRequest(BaseModel):
    repo_name: Optional[str] = Field(default=None)
    repo_path: Optional[str] = Field(default=None)
    branch_name: Optional[str] = Field(default=None)
    commit_id: Optional[str] = Field(default=None)

    def __init__(self, **data):
        super().__init__(**data)
        if not self.repo_name and not self.repo_path:
            raise ValueError("Either repo_name or repo_path must be provided.")


class ParsingResponse(BaseModel):
    message: str
    status: str
    project_id: str


class RepoDetails(BaseModel):
    repo_name: str
    branch_name: str
    repo_path: Optional[str] = None
    commit_id: Optional[str] = None


class ParsingStatusRequest(BaseModel):
    repo_name: str = Field(..., description="Repository name to check status for")
    commit_id: Optional[str] = Field(
        default=None, description="Commit ID to check status for"
    )
    branch_name: Optional[str] = Field(
        default=None, description="Branch name (used if commit_id is not provided)"
    )
