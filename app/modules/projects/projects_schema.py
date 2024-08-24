from enum import Enum


class ProjectStatusEnum(str, Enum):
    SUBMITTED = "submitted"
    CLONED = "cloned"
    PARSED = "parsed"
    PROCESSING = "processing"
    READY = "ready"
    ERROR = "error"
