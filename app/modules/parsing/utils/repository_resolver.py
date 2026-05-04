"""
Repository identifier resolution utilities.

Centralizes the logic for classifying and resolving a single repository
identifier into its component parts (repo_name, repo_path, is_local).

This replaces the scattered auto-detection logic that was previously
distributed across parsing_controller.py and other modules.
"""

import os
from enum import Enum
from typing import NamedTuple

from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)


class RepositoryType(Enum):
    """Classification of a repository identifier."""

    REMOTE = "remote"
    LOCAL = "local"


class ResolvedRepository(NamedTuple):
    """Result of resolving a repository identifier.

    Attributes:
        repository_identifier: The original identifier string.
        repo_name: Normalized repository name (e.g. 'owner/repo' or directory basename).
        repo_path: Resolved local filesystem path, or None for remote repos.
        is_local: True if the identifier refers to a local filesystem path.
    """

    repository_identifier: str
    repo_name: str
    repo_path: str | None
    is_local: bool


class RepositoryResolver:
    """Centralized repository identifier classification and resolution.

    A repository identifier is a single string that can represent either:
    - A remote GitHub repository (e.g. ``owner/repo``)
    - A local filesystem path (e.g. ``/path/to/repo``, ``~/repos/myproject``,
      ``./myproject``)
    """

    @staticmethod
    def looks_like_path(identifier: str) -> bool:
        """Check if an identifier looks like a filesystem path.

        Args:
            identifier: The repository identifier string.

        Returns:
            True if the identifier appears to be a filesystem path.
        """
        if not identifier:
            return False

        # Absolute or home-relative paths
        if os.path.isabs(identifier) or identifier.startswith(("~", "./", "../")):
            return True

        # Check if the identifier is an existing directory on the filesystem
        expanded = os.path.expanduser(identifier)
        if os.path.isdir(expanded):
            return True

        return False

    @staticmethod
    def classify(identifier: str) -> ResolvedRepository:
        """Classify and resolve a repository identifier.

        Determines whether the identifier refers to a local path or a remote
        repository and returns a :class:`ResolvedRepository` with the resolved
        fields populated.

        Args:
            identifier: The repository identifier string.

        Returns:
            A :class:`ResolvedRepository` with all fields populated.

        Raises:
            ValueError: If the identifier is empty or blank.
        """
        if not identifier or not identifier.strip():
            raise ValueError("Repository identifier cannot be empty.")

        identifier = identifier.strip()

        if RepositoryResolver.looks_like_path(identifier):
            repo_path = os.path.expanduser(identifier)
            repo_name = RepositoryResolver.extract_repo_name_from_path(repo_path)
            logger.info(
                f"Resolved identifier as local path: "
                f"repo_path={repo_path}, repo_name={repo_name}"
            )
            return ResolvedRepository(
                repository_identifier=identifier,
                repo_name=repo_name,
                repo_path=repo_path,
                is_local=True,
            )

        # Treat as remote repository identifier (e.g. 'owner/repo')
        logger.info(f"Resolved identifier as remote repo: repo_name={identifier}")
        return ResolvedRepository(
            repository_identifier=identifier,
            repo_name=identifier,
            repo_path=None,
            is_local=False,
        )

    @staticmethod
    def extract_repo_name_from_path(path: str) -> str:
        """Extract a repository name from a filesystem path.

        Uses the last component of the path as the repository name.

        Args:
            path: A filesystem path string.

        Returns:
            The basename of the path.
        """
        # Normalize trailing slashes before extracting basename
        normalized = path.rstrip("/").rstrip("\\")
        return os.path.basename(normalized) or path
