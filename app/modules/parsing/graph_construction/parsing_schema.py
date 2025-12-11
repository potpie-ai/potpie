from typing import Optional

from pydantic import BaseModel, Field


class ParsingRequest(BaseModel):
    repository_identifier: str
    branch_name: Optional[str] = Field(default=None)
    commit_id: Optional[str] = Field(default=None)


class ParsingResponse(BaseModel):
    message: str
    status: str
    project_id: str


class RepoDetails(BaseModel):
    repository_identifier: str
    repo_name: str
    branch_name: Optional[str] = None
    repo_path: Optional[str] = None
    commit_id: Optional[str] = None
    is_local: bool = False
