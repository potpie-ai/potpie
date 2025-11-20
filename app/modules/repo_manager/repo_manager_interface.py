"""
Repo Manager Interface

Abstract interface for repository manager implementations.
Used to track and manage local copies of repositories that have been parsed.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List


class IRepoManager(ABC):
    """
    Abstract interface for repository manager implementations.

    This interface defines methods for managing local copies of repositories
    that have been parsed. Repositories can be evicted if they aren't used
    for a while to free up storage space.
    """

    @abstractmethod
    def is_repo_available(
        self,
        repo_name: str,
        branch: Optional[str] = None,
        commit_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> bool:
        """
        Check if a repository is available locally.

        Args:
            repo_name: Full repository name (e.g., 'owner/repo')
            branch: Branch name (optional, for branch-specific checks)
            commit_id: Commit SHA (optional, for commit-specific checks)
            user_id: User ID (optional, for user-specific checks)

        Returns:
            True if the repository is available locally, False otherwise
        """
        pass

    @abstractmethod
    def register_repo(
        self,
        repo_name: str,
        local_path: str,
        branch: Optional[str] = None,
        commit_id: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Register a repository that has been downloaded/parsed.

        Args:
            repo_name: Full repository name (e.g., 'owner/repo')
            local_path: Local filesystem path where the repo is stored
            branch: Branch name (optional)
            commit_id: Commit SHA (optional)
            user_id: User ID who requested the parse (optional)
            metadata: Additional metadata about the repository (optional)

        Returns:
            Repository identifier/tracking ID
        """
        pass

    @abstractmethod
    def get_repo_path(
        self,
        repo_name: str,
        branch: Optional[str] = None,
        commit_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Optional[str]:
        """
        Get the local filesystem path for a repository.

        Args:
            repo_name: Full repository name (e.g., 'owner/repo')
            branch: Branch name (optional)
            commit_id: Commit SHA (optional)
            user_id: User ID (optional)

        Returns:
            Local filesystem path if available, None otherwise
        """
        pass

    @abstractmethod
    def update_last_accessed(
        self,
        repo_name: str,
        branch: Optional[str] = None,
        commit_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> None:
        """
        Update the last accessed timestamp for a repository.

        This is used to track repository usage for eviction purposes.

        Args:
            repo_name: Full repository name (e.g., 'owner/repo')
            branch: Branch name (optional)
            commit_id: Commit SHA (optional)
            user_id: User ID (optional)
        """
        pass

    @abstractmethod
    def get_repo_info(
        self,
        repo_name: str,
        branch: Optional[str] = None,
        commit_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Get information about a registered repository.

        Args:
            repo_name: Full repository name (e.g., 'owner/repo')
            branch: Branch name (optional)
            commit_id: Commit SHA (optional)
            user_id: User ID (optional)

        Returns:
            Dictionary with repository information including:
            - local_path: Local filesystem path
            - registered_at: When the repo was registered
            - last_accessed: Last access timestamp
            - metadata: Additional metadata
            None if repository is not registered
        """
        pass

    @abstractmethod
    def list_available_repos(
        self,
        user_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        List all available repositories.

        Args:
            user_id: Filter by user ID (optional)
            limit: Maximum number of repos to return (optional)

        Returns:
            List of dictionaries containing repository information
        """
        pass

    @abstractmethod
    def evict_repo(
        self,
        repo_name: str,
        branch: Optional[str] = None,
        commit_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> bool:
        """
        Evict a repository from local storage.

        This removes the repository from tracking and optionally deletes
        the local filesystem copy.

        Args:
            repo_name: Full repository name (e.g., 'owner/repo')
            branch: Branch name (optional)
            commit_id: Commit SHA (optional)
            user_id: User ID (optional)

        Returns:
            True if the repository was evicted, False if it wasn't found
        """
        pass

    @abstractmethod
    def evict_stale_repos(
        self,
        max_age_days: int,
        user_id: Optional[str] = None,
    ) -> List[str]:
        """
        Evict repositories that haven't been accessed in a while.

        Args:
            max_age_days: Maximum age in days since last access
            user_id: Filter by user ID (optional)

        Returns:
            List of repository identifiers that were evicted
        """
        pass

    @abstractmethod
    def get_repo_size(
        self,
        repo_name: str,
        branch: Optional[str] = None,
        commit_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Optional[int]:
        """
        Get the size of a repository in bytes.

        Args:
            repo_name: Full repository name (e.g., 'owner/repo')
            branch: Branch name (optional)
            commit_id: Commit SHA (optional)
            user_id: User ID (optional)

        Returns:
            Size in bytes if available, None otherwise
        """
        pass
