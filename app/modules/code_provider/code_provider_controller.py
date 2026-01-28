from fastapi import HTTPException
from sqlalchemy.orm import Session
from typing import Dict, Any
import os

from app.modules.code_provider.code_provider_service import CodeProviderService
from app.modules.code_provider.provider_factory import CodeProviderFactory
from app.core.config_provider import config_provider
from app.modules.code_provider.github.github_service import GithubService
from app.modules.code_provider.branch_cache import BranchCache

try:
    from github.GithubException import GithubException, BadCredentialsException
except ImportError:
    GithubException = None
    BadCredentialsException = None


class CodeProviderController:
    """
    Generic controller that uses the provider factory to support multiple code providers
    (GitHub, GitBucket, GitLab, Bitbucket) based on environment configuration.
    """

    def __init__(self, db: Session):
        self.db = db
        self.code_provider_service = CodeProviderService(db)
        self.branch_cache = BranchCache()

    def _filter_branches(self, branches: list, search_query: str = None) -> list:
        """
        Filter branches by search query (case-insensitive).
        
        Args:
            branches: List of branch names
            search_query: Optional search query to filter branches
            
        Returns:
            Filtered list of branches
        """
        if not search_query:
            return branches
        return [branch for branch in branches if search_query in branch.lower()]

    def _paginate_branches(
        self, 
        branches: list, 
        limit: int = None, 
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Apply pagination to branches and return formatted response.
        
        Args:
            branches: List of branch names
            limit: Optional limit on number of branches to return
            offset: Number of branches to skip
            
        Returns:
            Dictionary with branches, has_next_page, and total_count
        """
        total_branches = len(branches)
        
        if limit is not None:
            paginated_branches = branches[offset:offset + limit]
            has_next_page = (offset + limit) < total_branches
            return {
                "branches": paginated_branches,
                "has_next_page": has_next_page,
                "total_count": total_branches
            }
        else:
            return {"branches": branches}

    async def get_branch_list(
        self, 
        repo_name: str, 
        limit: int = None, 
        offset: int = 0,
        search: str = None
    ) -> Dict[str, Any]:
        """
        Get branch list for a repository using the configured provider.
        Uses fallback authentication (PAT-first, then GitHub App) for private repos.
        Caches all fetched branches in Redis for future requests.
        Returns paginated results if limit is specified.

        Args:
            repo_name: Repository name (e.g., "owner/repo")
            limit: Optional limit on number of branches to return
            offset: Number of branches to skip (default: 0)
            search: Optional search query to filter branches by name

        Returns:
            Dictionary containing branch information with pagination metadata
        """
        from app.modules.utils.logger import setup_logger

        logger = setup_logger(__name__)

        # Normalize search query
        search_query = search.strip().lower() if search and search.strip() else None

        # Check cache first for all branches (without search filter)
        cached_branches = self.branch_cache.get_branches(repo_name, search_query=None)
        if cached_branches is not None:
            logger.info(f"Found {len(cached_branches)} cached branches for {repo_name}")
            
            filtered_branches = self._filter_branches(cached_branches, search_query)
            if search_query:
                logger.info(f"Filtered to {len(filtered_branches)} branches matching '{search}'")
            
            return self._paginate_branches(filtered_branches, limit, offset)

        try:
            # Use fallback provider that tries PAT first, then GitHub App for private repos
            provider = CodeProviderFactory.create_provider_with_fallback(repo_name)

            # Use the provider's list_branches method - this fetches ALL branches
            all_branches = provider.list_branches(repo_name)

            # Cache all branches in Redis
            self.branch_cache.cache_all_branches(repo_name, all_branches, ttl=3600)
            logger.info(f"Cached {len(all_branches)} branches for {repo_name}")

            filtered_branches = self._filter_branches(all_branches, search_query)
            if search_query:
                logger.info(f"Filtered to {len(filtered_branches)} branches matching '{search}'")

            return self._paginate_branches(filtered_branches, limit, offset)

        except Exception as e:
            # Check if this is a 404 (not found), 401 (bad credentials), or 403 (forbidden)
            is_404_error = (
                (GithubException and isinstance(e, GithubException) and e.status == 404)
                or "404" in str(e)
                or "Not Found" in str(e)
                or (hasattr(e, "status") and e.status == 404)
            )
            is_401_error = (
                (BadCredentialsException and isinstance(e, BadCredentialsException))
                or (
                    GithubException
                    and isinstance(e, GithubException)
                    and e.status == 401
                )
                or "401" in str(e)
                or "Bad credentials" in str(e)
                or (hasattr(e, "status") and e.status == 401)
            )
            is_403_error = (
                (GithubException and isinstance(e, GithubException) and e.status == 403)
                or "403" in str(e)
                or (hasattr(e, "status") and e.status == 403)
            )

            provider_type = os.getenv("CODE_PROVIDER", "github").lower()

            # If this is a GitHub repo and PAT failed with 404 or 401, try unauthenticated access for public repos
            # 401 can happen when token is invalid/expired, but repo might still be public
            if provider_type == "github" and (is_404_error or is_401_error):
                error_type = "401 (Bad credentials)" if is_401_error else "404"
                logger.info(
                    f"PAT authentication failed with {error_type} for {repo_name}, "
                    "trying unauthenticated access for public repo"
                )
                try:
                    from app.modules.code_provider.github.github_provider import (
                        GitHubProvider,
                    )

                    provider = GitHubProvider()
                    provider.set_unauthenticated_client()
                    all_branches = provider.list_branches(repo_name)
                    logger.info(
                        f"Successfully accessed {repo_name} without authentication"
                    )
                    # Cache all branches in Redis
                    self.branch_cache.cache_all_branches(repo_name, all_branches, ttl=3600)
                    logger.info(f"Cached {len(all_branches)} branches for {repo_name} (unauthenticated)")
                    
                    filtered_branches = self._filter_branches(all_branches, search_query)
                    if search_query:
                        logger.info(f"Filtered to {len(filtered_branches)} branches matching '{search}'")
                    
                    return self._paginate_branches(filtered_branches, limit, offset)
                except Exception as unauth_error:
                    logger.warning(
                        f"Unauthenticated access also failed for {repo_name}: {unauth_error}"
                    )
                    # Continue to try GitHub App below

            # If GitHub App is configured, try it as fallback
            if provider_type == "github":
                app_id = os.getenv("GITHUB_APP_ID")
                private_key = config_provider.get_github_key()
                if app_id and private_key:
                    try:
                        logger.info(
                            f"Retrying branch fetch for {repo_name} with GitHub App auth"
                        )
                        provider = CodeProviderFactory.create_github_app_provider(
                            repo_name
                        )
                        all_branches = provider.list_branches(repo_name)
                        logger.info(
                            f"Successfully fetched {len(all_branches)} branches for {repo_name} using GitHub App auth"
                        )
                        # Cache all branches in Redis
                        self.branch_cache.cache_all_branches(repo_name, all_branches, ttl=3600)
                        logger.info(f"Cached {len(all_branches)} branches for {repo_name} (GitHub App)")
                        
                        filtered_branches = self._filter_branches(all_branches, search_query)
                        if search_query:
                            logger.info(f"Filtered to {len(filtered_branches)} branches matching '{search}'")
                        
                        return self._paginate_branches(filtered_branches, limit, offset)
                    except Exception as app_error:
                        logger.warning(
                            f"GitHub App auth also failed for {repo_name}: {str(app_error)}"
                        )
                else:
                    logger.debug(
                        "GitHub App credentials not configured, skipping App auth retry"
                    )

            # Log the error appropriately
            if is_404_error or is_401_error or is_403_error:
                logger.info(f"Authentication failed for {repo_name}: {str(e)}")
            else:
                logger.error(
                    f"Error fetching branches for {repo_name}: {str(e)}", exc_info=True
                )

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
