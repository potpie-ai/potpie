from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ProjectStatusEnum(str, Enum):
    CREATED = "created"
    SUBMITTED = "submitted"
    CLONED = "cloned"
    PARSED = "parsed"
    PROCESSING = "processing"
    INFERRING = "inferring"
    READY = "ready"
    ERROR = "error"


class RepoDetails(BaseModel):
    repo_name: str


class WorkspaceCreateBody(BaseModel):
    """Optional label for a DB project row used for integrations (e.g. Linear) before any repo is parsed."""

    display_name: Optional[str] = Field(
        None,
        max_length=200,
        description="Shown as project name; defaults to 'Workspace'",
    )
