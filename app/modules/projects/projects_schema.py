from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field

class ProjectStatusEnum(str, Enum):
    SUBMITTED = "submitted"
    CLONED = "cloned"
    PARSED = "parsed"
    PROCESSING = "processing"
    READY = "ready"
    ERROR = "error"


class RepoDetails(BaseModel):
    repo_name: str
