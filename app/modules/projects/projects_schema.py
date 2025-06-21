from enum import Enum
from typing import Optional

from pydantic import BaseModel


class ProjectStatusEnum(str, Enum):
    SUBMITTED = "submitted"
    CLONED = "cloned"
    PARSED = "parsed"
    READY = "ready"
    ERROR = "error"


class RepoDetails(BaseModel):
    repo_name: str


class RepositoryResponse(BaseModel):
    id: str
    name: str
    full_name: str
    private: bool
    url: str
    owner: str


class UserRepositoriesResponse(BaseModel):
    repositories: list[RepositoryResponse]
