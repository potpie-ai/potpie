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
            actual_username = os.getenv("GITBUCKET_USERNAME")
            if not actual_username:
                logger.debug(
                    "GitBucket: Skipping normalization for '%s' because GITBUCKET_USERNAME is not set",
                    repo_name,
                )
                return repo_name

            normalized_name = repo_name.replace("root/", f"{actual_username}/", 1)
            logger.info(
                "GitBucket: Normalized '%s' to '%s'", repo_name, normalized_name
            )
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
        # Only reverse-map when we previously normalized from 'root/<repo>' to '<username>/<repo>'
        actual_username = os.getenv("GITBUCKET_USERNAME")
        if actual_username and repo_name.startswith(f"{actual_username}/"):
            parts = repo_name.split("/")
            if len(parts) == 2:
                actual_name = f"root/{parts[1]}"
                logger.debug(
                    "GitBucket: Converting '%s' to '%s' for API calls",
                    repo_name,
                    actual_name,
                )
                return actual_name

    return repo_name
