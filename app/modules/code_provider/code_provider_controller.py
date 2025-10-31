from fastapi import HTTPException
from sqlalchemy.orm import Session
from typing import Dict, Any
import os

from app.modules.code_provider.code_provider_service import CodeProviderService
from app.modules.code_provider.provider_factory import CodeProviderFactory
from app.core.config_provider import config_provider
from app.modules.code_provider.github.github_service import GithubService


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
        Uses fallback authentication (PAT-first, then GitHub App) for private repos.

        Args:
            repo_name: Repository name (e.g., "owner/repo")

        Returns:
            Dictionary containing branch information
        """
        import logging
        
        logger = logging.getLogger(__name__)
        
        try:
            # Use fallback provider that tries PAT first, then GitHub App for private repos
            provider = CodeProviderFactory.create_provider_with_fallback(repo_name)

            # Use the provider's list_branches method
            branches = provider.list_branches(repo_name)

            # Format the response to match the expected API format
            return {"branches": branches}

        except Exception as e:
            # Check if this is a 404 (not found) or 403 (forbidden) - likely PAT doesn't have access
            is_access_error = (
                "404" in str(e) 
                or "403" in str(e) 
                or "Not Found" in str(e)
                or "UnknownObjectException" in str(type(e))
            )
            
            if is_access_error:
                logger.info(
                    f"PAT authentication failed for {repo_name} (likely no access to private repo): {str(e)}"
                )
            else:
                logger.error(
                    f"Error fetching branches for {repo_name}: {str(e)}", exc_info=True
                )
            
            # If this is a GitHub repo and PAT failed, try GitHub App directly
            provider_type = os.getenv("CODE_PROVIDER", "github").lower()
            if provider_type == "github":
                app_id = os.getenv("GITHUB_APP_ID")
                private_key = config_provider.get_github_key()
                if app_id and private_key:
                    try:
                        logger.info(f"Retrying branch fetch for {repo_name} with GitHub App auth")
                        provider = CodeProviderFactory.create_github_app_provider(repo_name)
                        branches = provider.list_branches(repo_name)
                        logger.info(f"Successfully fetched {len(branches)} branches for {repo_name} using GitHub App auth")
                        return {"branches": branches}
                    except Exception as app_error:
                        logger.warning(
                            f"GitHub App auth also failed for {repo_name}: {str(app_error)}"
                        )
                else:
                    logger.debug("GitHub App credentials not configured, skipping App auth retry")
            
            raise HTTPException(
                status_code=404,
                detail=f"Repository {repo_name} not found or error fetching branches: {str(e)}",
            )

    async def get_user_repos(self, user: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get user repositories using the configured provider.

        When the provider is GitHub and GitHub App credentials are configured,
        use the GitHub App pathway to include installations/repos linked via the app.
        Otherwise, fall back to the generic provider listing (e.g., GitBucket).
        """
        try:
            provider_type = os.getenv("CODE_PROVIDER", "github").lower()

            if (
                provider_type == "github"
                and os.getenv("GITHUB_APP_ID")
                and config_provider.get_github_key()
            ):
                github_service = GithubService(self.db)
                return await github_service.get_combined_user_repos(user["user_id"])

            provider = CodeProviderFactory.create_provider()
            repositories = provider.list_user_repositories()
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
