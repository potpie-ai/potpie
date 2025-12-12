"""
Repository Resolver Service

This module provides centralized repository resolution logic, converting a single
repository identifier (either 'owner/repo' or '/path/to/repo') into a fully
resolved RepoDetails object with normalized paths and metadata.

This eliminates the need for scattered resolution logic across multiple layers.
"""

import logging
import os
from pathlib import Path
from typing import Optional, Tuple

from fastapi import HTTPException

from app.modules.parsing.graph_construction.parsing_schema import RepoDetails
from app.modules.parsing.utils.repo_name_normalizer import normalize_repo_name

logger = logging.getLogger(__name__)


class RepositoryResolver:
    """
    Centralized service for resolving repository identifiers into normalized RepoDetails.
    
    This class handles:
    - Detection of local vs remote repositories
    - Path validation and normalization
    - Repository name extraction
    - Early validation of repository access
    """

    @staticmethod
    def is_local_path(identifier: str) -> bool:
        """
        Determine if the identifier represents a local filesystem path.
        
        Args:
            identifier: Repository identifier (e.g., 'owner/repo' or '/path/to/repo')
            
        Returns:
            True if identifier is a local path, False if it's a remote repo
        """
        # First, reject URL formats
        if identifier.startswith(("https://", "http://", "git@", "ssh://", "ftp://")):
            return False
        
        # Check for absolute paths (Unix: /, Windows: C:\)
        if os.path.isabs(identifier):
            return True
        
        # Check for relative path indicators
        if identifier.startswith(("~", "./", "../", ".\\")):
            return True
        
        # Check if it's actually a directory that exists
        expanded_path = os.path.expanduser(identifier)
        if os.path.isdir(expanded_path):
            return True
        
        # Windows-style paths (e.g., C:\path)
        if len(identifier) > 1 and identifier[1] == ':':
            return True
        
        return False

    @staticmethod
    def extract_repo_name_from_path(repo_path: str) -> str:
        """
        Extract repository name from a filesystem path.
        
        Args:
            repo_path: Full path to repository
            
        Returns:
            Repository name (last directory component)
        """
        # Normalize path separators
        normalized_path = repo_path.replace("\\", "/")
        
        # Remove trailing slashes
        normalized_path = normalized_path.rstrip("/")
        
        # Get last component
        repo_name = normalized_path.split("/")[-1]
        
        return repo_name

    @staticmethod
    def validate_local_repository(repo_path: str) -> str:
        """
        Validate and normalize a local repository path.
        
        Args:
            repo_path: Path to local repository
            
        Returns:
            Normalized absolute path
            
        Raises:
            HTTPException: If path doesn't exist or isn't accessible
        """
        # Expand user home directory if present
        expanded_path = os.path.expanduser(repo_path)
        
        # Convert to absolute path
        absolute_path = os.path.abspath(expanded_path)
        
        # Validate existence
        if not os.path.exists(absolute_path):
            raise HTTPException(
                status_code=400,
                detail=f"Local repository does not exist at path: {repo_path}"
            )
        
        # Validate it's a directory
        if not os.path.isdir(absolute_path):
            raise HTTPException(
                status_code=400,
                detail=f"Path is not a directory: {repo_path}"
            )
        
        # Check if it's a git repository (optional but recommended)
        git_dir = os.path.join(absolute_path, ".git")
        if not os.path.exists(git_dir):
            logger.warning(
                f"Path {absolute_path} does not appear to be a git repository "
                f"(no .git directory found). Proceeding anyway."
            )
        
        logger.info(f"Validated local repository at: {absolute_path}")
        return absolute_path

    @staticmethod
    def parse_remote_repository(identifier: str) -> Tuple[str, str]:
        """
        Parse a remote repository identifier into owner and repo name.
        
        Args:
            identifier: Repository identifier (e.g., 'owner/repo')
            
        Returns:
            Tuple of (owner, repo_name)
            
        Raises:
            HTTPException: If identifier format is invalid
        """
        # Remove any leading/trailing whitespace
        identifier = identifier.strip()
        
        # Reject URL formats
        if identifier.startswith(("https://", "http://", "git@", "ssh://", "ftp://")):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Please provide repository in 'owner/repo' format, not URL: {identifier}. "
                    f"Example: 'octocat/Hello-World'"
                )
            )
        
        # Split by slash
        parts = identifier.split("/")
        
        # Must be exactly 2 parts (owner/repo)
        if len(parts) != 2:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Invalid repository format: {identifier}. "
                    f"Expected format: 'owner/repo'. Example: 'octocat/Hello-World'"
                )
            )
        
        owner, repo_name = parts
        
        # Validate non-empty
        if not owner or not repo_name:
            raise HTTPException(
                status_code=400,
                detail=f"Repository owner and name cannot be empty: {identifier}"
            )
        
        # Additional validation: check for .git suffix or other invalid characters
        if repo_name.endswith('.git'):
            repo_name = repo_name[:-4]  # Remove .git suffix
            logger.info(f"Removed .git suffix from repo name: {repo_name}")
        
        return owner, repo_name

    @classmethod
    def resolve(
        cls,
        repository: str,
        branch_name: Optional[str] = None,
        commit_id: Optional[str] = None,
    ) -> RepoDetails:
        """
        Resolve a repository identifier into a complete RepoDetails object.
        
        This is the main entry point for repository resolution. It detects whether
        the identifier is a local path or remote repository, validates it, and
        returns a fully populated RepoDetails object.
        
        Args:
            repository: Repository identifier ('owner/repo' or '/path/to/repo')
            branch_name: Optional branch name
            commit_id: Optional commit ID
            
        Returns:
            RepoDetails object with all fields properly populated
            
        Raises:
            HTTPException: If repository cannot be resolved or validated
        """
        logger.info(f"Resolving repository identifier: {repository}")
        
        # Determine if this is a local or remote repository
        is_local = cls.is_local_path(repository)
        
        if is_local:
            # Handle local repository
            logger.info(f"Detected local repository: {repository}")
            
            # Validate and normalize the path
            normalized_path = cls.validate_local_repository(repository)
            
            # Extract repository name from path
            repo_name = cls.extract_repo_name_from_path(normalized_path)
            
            # Normalize the repo name for consistency
            normalized_repo_name = normalize_repo_name(repo_name)
            
            logger.info(
                f"Resolved local repository: path={normalized_path}, "
                f"name={normalized_repo_name}, is_local=True"
            )
            
            return RepoDetails(
                repository=repository,  # Original identifier
                repo_name=normalized_repo_name,
                repo_path=normalized_path,
                branch_name=branch_name,
                commit_id=commit_id,
                is_local=True
            )
        else:
            # Handle remote repository
            logger.info(f"Detected remote repository: {repository}")
            
            # Parse and validate the remote identifier
            owner, repo_name = cls.parse_remote_repository(repository)
            
            # Normalize the full repository name
            full_name = f"{owner}/{repo_name}"
            normalized_repo_name = normalize_repo_name(full_name)
            
            logger.info(
                f"Resolved remote repository: name={normalized_repo_name}, "
                f"is_local=False"
            )
            
            return RepoDetails(
                repository=repository,  # Original identifier
                repo_name=normalized_repo_name,
                repo_path=None,
                branch_name=branch_name,
                commit_id=commit_id,
                is_local=False
            )
