"""Project-related type definitions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional


class ProjectStatus(str, Enum):
    """Status of a project in the system."""

    SUBMITTED = "submitted"
    CLONED = "cloned"
    PARSED = "parsed"
    PROCESSING = "processing"
    READY = "ready"
    ERROR = "error"

    @classmethod
    def from_string(cls, value: str) -> ProjectStatus:
        """Create ProjectStatus from string value."""
        try:
            return cls(value.lower())
        except ValueError:
            return cls.ERROR


@dataclass
class ProjectInfo:
    """Information about a project."""

    id: str
    repo_name: str
    branch_name: str
    status: ProjectStatus
    commit_id: Optional[str] = None
    repo_path: Optional[str] = None
    user_id: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    @classmethod
    def from_dict(cls, data: dict) -> ProjectInfo:
        """Create ProjectInfo from dictionary (as returned by ProjectService)."""
        status_value = data.get("status", "error")
        if isinstance(status_value, str):
            status = ProjectStatus.from_string(status_value)
        else:
            status = status_value

        return cls(
            id=data.get("id") or data.get("project_id", ""),
            repo_name=data.get("repo_name") or data.get("project_name", ""),
            branch_name=data.get("branch_name", ""),
            status=status,
            commit_id=data.get("commit_id"),
            repo_path=data.get("repo_path"),
            user_id=data.get("user_id"),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
        )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "repo_name": self.repo_name,
            "branch_name": self.branch_name,
            "status": self.status.value,
            "commit_id": self.commit_id,
            "repo_path": self.repo_path,
            "user_id": self.user_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
