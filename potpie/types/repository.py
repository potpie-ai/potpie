"""Type definitions for repository management."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional


class RepositoryStatus(str, Enum):
    """Status of a repository in the RepoManager.

    Repositories can be available for use, have been evicted to free space,
    or are missing (never registered or manually deleted).
    """

    AVAILABLE = "available"
    """Repository is available locally and ready for use."""

    EVICTED = "evicted"
    """Repository was evicted to free disk space."""

    MISSING = "missing"
    """Repository does not exist (never registered or manually deleted)."""


@dataclass
class RepositoryInfo:
    """Information about a repository managed by RepoManager.

    Contains metadata about a repository's registration, status, and storage.
    """

    repo_key: str
    """Unique identifier for this repository (repo:branch:commit:user)."""

    repo_name: str
    """Full repository name (e.g., 'owner/repo')."""

    local_path: str
    """Local filesystem path to the repository or worktree."""

    branch: Optional[str] = None
    """Branch name if tracked by branch."""

    commit_id: Optional[str] = None
    """Commit SHA if tracked by specific commit."""

    user_id: Optional[str] = None
    """User ID who owns this repository (for multi-tenant support)."""

    registered_at: Optional[datetime] = None
    """When the repository was first registered."""

    last_accessed: Optional[datetime] = None
    """When the repository was last accessed."""

    volume_bytes: Optional[int] = None
    """Size of the repository in bytes (calculated via du -s)."""

    metadata: Optional[Dict[str, Any]] = None
    """Additional metadata about the repository.

    Includes 'type' (bare_repo/worktree), 'repo_url', etc.
    """

    status: RepositoryStatus = RepositoryStatus.AVAILABLE
    """Current status of the repository."""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RepositoryInfo":
        """Create RepositoryInfo from a dictionary (typically from RepoManager).

        Args:
            data: Dictionary with repository information

        Returns:
            RepositoryInfo instance
        """
        # Handle datetime strings
        registered_at = data.get("registered_at")
        if isinstance(registered_at, str):
            registered_at = datetime.fromisoformat(registered_at)

        last_accessed = data.get("last_accessed")
        if isinstance(last_accessed, str):
            last_accessed = datetime.fromisoformat(last_accessed)

        status_input = data.get("status")
        if isinstance(status_input, RepositoryStatus):
            status = status_input
        elif isinstance(status_input, str):
            try:
                status = RepositoryStatus(status_input.lower())
            except ValueError:
                status = RepositoryStatus.AVAILABLE
        else:
            status = RepositoryStatus.AVAILABLE

        return cls(
            repo_key=data.get("repo_key", ""),
            repo_name=data.get("repo_name", ""),
            local_path=data.get("local_path", ""),
            branch=data.get("branch"),
            commit_id=data.get("commit_id"),
            user_id=data.get("user_id"),
            registered_at=registered_at,
            last_accessed=last_accessed,
            volume_bytes=data.get("volume_bytes"),
            metadata=data.get("metadata", {}),
            status=status,
        )


@dataclass
class VolumeInfo:
    """Information about disk volume usage for repositories."""

    total_volume_bytes: int
    """Total volume used by all repositories in bytes."""

    volume_limit_bytes: int
    """Configured volume limit in bytes."""

    volume_percentage: float
    """Percentage of volume limit used (0.0 to 100.0)."""

    repo_count: int
    """Number of repositories currently tracked."""

    @property
    def volume_used_gb(self) -> float:
        """Volume used in gigabytes."""
        return self.total_volume_bytes / (1024**3)

    @property
    def volume_limit_gb(self) -> float:
        """Volume limit in gigabytes."""
        return self.volume_limit_bytes / (1024**3)

    @property
    def available_gb(self) -> float:
        """Available volume in gigabytes."""
        return max(0, self.volume_limit_gb - self.volume_used_gb)
