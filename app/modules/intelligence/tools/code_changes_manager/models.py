"""Core data models for code changes."""

from datetime import datetime
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ChangeType(str, Enum):
    """Type of code change"""

    ADD = "add"  # New file
    UPDATE = "update"  # Modified file
    DELETE = "delete"  # Deleted file


@dataclass
class FileChange:
    """Represents a change to a single file"""

    file_path: str
    change_type: ChangeType
    content: Optional[str] = None  # None for DELETE
    previous_content: Optional[str] = None  # For UPDATE/DELETE
    created_at: str = ""
    updated_at: str = ""
    description: Optional[str] = None  # Optional change description

    def __post_init__(self):
        now = datetime.now().isoformat()
        if not self.created_at:
            self.created_at = now
        if not self.updated_at:
            self.updated_at = now
