"""
Repository name normalization utilities for different code providers.
"""

import os
import logging

logger = logging.getLogger(__name__)


def normalize_repo_name(repo_name: str, provider_type: str = None) -> str:
    """
    Normalize repository name based on the code provider.

    This function handles provider-specific naming conventions:
    - GitBucket: Uses 'root' as owner name, normalize to actual username
    - GitHub: No normalization needed
    - GitLab: No normalization needed
    - Bitbucket: No normalization needed

    Args:
        repo_name: Repository name in format 'owner/repo'
        provider_type: Code provider type (gitbucket, github, etc.)

    Returns:
        Normalized repository name
    """
    if not repo_name or "/" not in repo_name:
        return repo_name

    # Get provider type from environment if not provided
    if not provider_type:
        provider_type = os.getenv("CODE_PROVIDER", "github").lower()

    # GitBucket specific normalization
    if provider_type == "gitbucket":
        # GitBucket uses 'root' as owner name, but we want to normalize to actual username
        # for consistency with database lookups
        if repo_name.startswith("root/"):
            # Extract the actual username from environment or use a default
            actual_username = os.getenv("GITBUCKET_USERNAME", "dhirenmathur")
            normalized_name = repo_name.replace("root/", f"{actual_username}/", 1)
            logger.info(f"GitBucket: Normalized '{repo_name}' to '{normalized_name}'")
            return normalized_name

    # For other providers, return as-is
    return repo_name


def get_actual_repo_name_for_lookup(repo_name: str, provider_type: str = None) -> str:
    """
    Get the actual repository name that should be used for database lookups.

    This is the reverse of normalize_repo_name - it converts the normalized name
    back to the format that the provider actually uses.

    Args:
        repo_name: Normalized repository name
        provider_type: Code provider type

    Returns:
        Actual repository name for provider API calls
    """
    if not repo_name or "/" not in repo_name:
        return repo_name

    # Get provider type from environment if not provided
    if not provider_type:
        provider_type = os.getenv("CODE_PROVIDER", "github").lower()

    # GitBucket specific handling
    if provider_type == "gitbucket":
        # If the repo name doesn't start with 'root/', it might be normalized
        # We need to convert it back to 'root/' for GitBucket API calls
        if not repo_name.startswith("root/"):
            # Check if it's a normalized name (username/repo)
            parts = repo_name.split("/")
            if len(parts) == 2:
                # Convert back to root/repo format for GitBucket
                actual_name = f"root/{parts[1]}"
                logger.info(
                    f"GitBucket: Converting '{repo_name}' to '{actual_name}' for API calls"
                )
                return actual_name

    return repo_name
