from fastapi import HTTPException
from sqlalchemy.orm import Session
from typing import Dict, Any

from app.modules.code_provider.code_provider_service import CodeProviderService
from app.modules.code_provider.provider_factory import CodeProviderFactory


class CodeProviderController:
    """
    Generic controller that uses the provider factory to support multiple code providers
    (GitHub, GitBucket, GitLab, Bitbucket) based on environment configuration.
    """

    def __init__(self, db: Session):
        self.db = db
        self.code_provider_service = CodeProviderService(db)

    async def get_branch_list(self, repo_name: str) -> Dict[str, Any]:
        """
        Get branch list for a repository using the configured provider.

        Args:
            repo_name: Repository name (e.g., "owner/repo")

        Returns:
            Dictionary containing branch information
        """
        try:
            # Get the configured provider (this will auto-authenticate if credentials are available)
            provider = CodeProviderFactory.create_provider()

            # Use the provider's list_branches method
            branches = provider.list_branches(repo_name)

            # Format the response to match the expected API format
            return {"branches": branches}

        except Exception as e:
            raise HTTPException(
                status_code=404,
                detail=f"Repository {repo_name} not found or error fetching branches: {str(e)}",
            )

    async def get_user_repos(self, user: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get user repositories using the configured provider.

        Args:
            user: User information dictionary

        Returns:
            Dictionary containing repository information
        """
        try:
            # Get the configured provider (this will auto-authenticate if credentials are available)
            provider = CodeProviderFactory.create_provider()

            # Don't pass user_id to avoid Firebase user ID vs GitBucket username mismatch
            # The provider will use the authenticated user's repositories instead
            repositories = provider.list_user_repositories()

            # Format the response to match the expected API format
            return {"repositories": repositories}

        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error fetching user repositories: {str(e)}"
            )

    async def check_public_repo(self, repo_name: str) -> bool:
        """
        Check if a repository is public using the configured provider.

        Args:
            repo_name: Repository name (e.g., "owner/repo")

        Returns:
            Boolean indicating if repository is public
        """
        try:
            # Get the configured provider (this will auto-authenticate if credentials are available)
            provider = CodeProviderFactory.create_provider()

            # Try to access the repository - if successful, it's accessible
            # This is a simple check; more sophisticated logic could be added
            provider.get_repository(repo_name)
            return True

        except Exception:
            # If we can't access it, assume it's private or doesn't exist
            return False
