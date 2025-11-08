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
    Get the actual repository name that should be used for API calls.

    For GitBucket: The API documentation (https://github.com/gitbucket/gitbucket/wiki/API-WebHook)
    does not explicitly require 'root/repo' format. However, GitBucket API responses return
    'root' as the owner name. We try the username format first, and if that fails, fall back
    to 'root/repo' format.

    Args:
        repo_name: Normalized repository name (e.g., 'dhiren/repo')
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
        # According to GitBucket API docs, the format is owner/repo
        # The docs don't explicitly require 'root' format, but API responses return 'root' as owner
        # Try using the username format first (as per GitHub API v3 compatibility)
        # If API calls fail, the caller should retry with 'root/repo' format
        actual_username = os.getenv("GITBUCKET_USERNAME")
        
        if actual_username and repo_name.startswith(f"{actual_username}/"):
            logger.debug(
                "GitBucket: Using username format '%s' for API calls (per GitHub API v3 compatibility)",
                repo_name,
            )
            return repo_name
        
        # If repo_name is already 'root/repo', use it as-is
        if repo_name.startswith("root/"):
            logger.debug(
                "GitBucket: Using 'root' format '%s' for API calls",
                repo_name,
            )
            return repo_name
        
        # If we have a username but repo_name doesn't match, try converting to root format
        # This handles cases where normalization happened but we need the API format
        if actual_username:
            repo_part = repo_name.split("/", 1)[1] if "/" in repo_name else repo_name
            root_repo_name = f"root/{repo_part}"
            logger.debug(
                "GitBucket: Converting '%s' to '%s' for API calls (fallback to root format)",
                repo_name,
                root_repo_name,
            )
            return root_repo_name

    return repo_name
