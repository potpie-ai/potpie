from enum import Enum
from typing import Optional
from pydantic import BaseModel


class ProjectStatusEnum(str, Enum):
    SUBMITTED = "submitted"
    CLONED = "cloned"
    PARSED = "parsed"
    PROCESSING = "processing"
    READY = "ready"
    ERROR = "error"


class RepoDetails(BaseModel):
    repo_name: str


class CreateProjectRequest(BaseModel):
    idea: str


class CreateProjectResponse(BaseModel):
    id: str
    idea: str
    status: str
    created_at: str


class SelectRepositoryRequest(BaseModel):
    project_id: str
    repo_id: str


class SelectRepositoryResponse(BaseModel):
    status: str
    analysis_id: Optional[str] = None