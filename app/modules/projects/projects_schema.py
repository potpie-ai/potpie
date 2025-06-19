from enum import Enum

from pydantic import BaseModel


class ProjectStatusEnum(str, Enum):
    SUBMITTED = "submitted"
    CLONED = "cloned"
    PARSED = "parsed"
    READY = "ready"
    ERROR = "error"


class RepoDetails(BaseModel):
    repo_name: str
