from enum import Enum

from pydantic import BaseModel


class ProjectStatusEnum(str, Enum):
    SUBMITTED = "submitted"
    CLONED = "cloned"
    PARSED = "parsed"
    PROCESSING = "processing"
    READY = "ready"
    PARTIALLY_READY = "partially_ready"  # Inference completed with 75-95% success rate
    ERROR = "error"


class RepoDetails(BaseModel):
    repo_name: str
