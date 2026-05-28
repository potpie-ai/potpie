import asyncio
import os
from typing import Any, Dict, List, Optional

from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

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

# Maximum search query length to prevent DoS attacks
MAX_SEARCH_QUERY_LENGTH = 200


class CodeProviderController:
    """
    Generic controller that uses the provider factory to support multiple code providers
    (GitHub, GitBucket, GitLab, Bitbucket) based on environment configuration.
    """

    def __init__(self, db: Session):
        self.db = db
        self.code_provider_service = CodeProviderService(db)
        self.branch_cache = BranchCache()

    @staticmethod
    def _normalize_search_query(search: Optional[str]) -> Optional[str]:
        """
        Normalize and validate search query.
        
        Args:
            search: Raw search query string (can be None, empty string, or whitespace-only)
            
        Returns:
            Normalized lowercase search query, or None if invalid/empty
            
        Raises:
            HTTPException: If search query is too long (DoS protection)
        """
        if not search:
            return None
        
        # Strip whitespace
        search = search.strip()
        
        # Return None if empty after stripping
        if not search:
            return None
        
        # Validate length to prevent DoS attacks
        if len(search) > MAX_SEARCH_QUERY_LENGTH:
            raise HTTPException(
                status_code=400,
                detail=f"Search query too long. Maximum length is {MAX_SEARCH_QUERY_LENGTH} characters."
            )
        
        # Normalize to lowercase for case-insensitive matching
        return search.lower()

    def _filter_branches(self, branches: list, search_query: Optional[str] = None) -> list:
        """
        Filter branches by search query (case-insensitive).

        Args:
            branches: List of branch names (strings or dicts with 'name' key)
            search_query: Optional normalized search query to filter branches

        Returns:
            Filtered list of branches
        """
        if not search_query or not branches:
            return branches or []
        
        filtered = []
        for branch in branches:
            # Handle both string branches and dict branches
            if isinstance(branch, str):
                branch_name = branch.lower()
            elif isinstance(branch, dict):
                branch_name = branch.get("name", "").lower()
            else:
                continue
            
            if search_query in branch_name:
                filtered.append(branch)
        
        return filtered

    def _filter_repositories(self, repositories: list, search_query: Optional[str] = None) -> list:
        """
        Filter repositories by search query (case-insensitive).
        Searches in both full_name and name fields.

        Args:
            repositories: List of repository dictionaries with 'full_name' and/or 'name' keys
            search_query: Optional normalized search query to filter repositories

        Returns:
            Filtered list of repositories
        """
        if not search_query or not repositories:
            return repositories or []
        
        filtered = []
        for repo in repositories:
            if not isinstance(repo, dict):
                continue
            
            # Check both full_name and name fields
            full_name = repo.get("full_name", "").lower()
            name = repo.get("name", "").lower()
            
            # Match if search query appears in either field
            if search_query in full_name or search_query in name:
                filtered.append(repo)
        
        return filtered

    def _paginate_branches(
        self, branches: list, limit: int = None, offset: int = 0
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
            paginated_branches = branches[offset : offset + limit]
            has_next_page = (offset + limit) < total_branches
            return {
                "branches": paginated_branches,
                "has_next_page": has_next_page,
                "total_count": total_branches,
            }
        else:
            return {"branches": branches}

    def _fetch_branches_from_provider_sync(self, repo_name: str) -> List[Any]:
        """
        Sync helper: fetch all branches from provider with fallbacks (PAT, unauthenticated, GitHub App).
        Runs in thread pool to avoid blocking the event loop on slow GitHub API.
        """
        from app.modules.utils.logger import setup_logger

        logger = setup_logger(__name__)
        provider = CodeProviderFactory.create_provider_with_fallback(repo_name)
        all_branches = provider.list_branches(repo_name)
        self.branch_cache.cache_all_branches(repo_name, all_branches, ttl=3600)
        logger.info(f"Cached {len(all_branches)} branches for {repo_name}")
        return all_branches

    def _fetch_branches_with_fallbacks_sync(self, repo_name: str) -> List[Any]:
        """
        Sync helper: fetch branches with full fallback chain (PAT -> unauthenticated -> GitHub App).
        Used from asyncio.to_thread to avoid blocking the event loop.
        """
        from app.modules.utils.logger import setup_logger

        logger = setup_logger(__name__)
        try:
            return self._fetch_branches_from_provider_sync(repo_name)
        except Exception as e:
            is_404_error = (
                (GithubException and isinstance(e, GithubException) and e.status == 404)
                or "404" in str(e)
                or "Not Found" in str(e)
                or (hasattr(e, "status") and e.status == 404)
            )
            is_401_error = (
                (BadCredentialsException and isinstance(e, BadCredentialsException))
                or (GithubException and isinstance(e, GithubException) and e.status == 401)
                or "401" in str(e)
                or "Bad credentials" in str(e)
                or (hasattr(e, "status") and e.status == 401)
            )
            provider_type = os.getenv("CODE_PROVIDER", "github").lower()

            if provider_type == "github" and (is_404_error or is_401_error):
                error_type = "401 (Bad credentials)" if is_401_error else "404"
                logger.info(
                    f"PAT authentication failed with {error_type} for {repo_name}, "
                    "trying unauthenticated access for public repo"
                )
                try:
                    from app.modules.code_provider.github.github_provider import GitHubProvider

                    provider = GitHubProvider()
                    provider.set_unauthenticated_client()
                    all_branches = provider.list_branches(repo_name)
                    self.branch_cache.cache_all_branches(repo_name, all_branches, ttl=3600)
                    logger.info(f"Cached {len(all_branches)} branches for {repo_name} (unauthenticated)")
                    return all_branches
                except Exception as unauth_error:
                    logger.warning(f"Unauthenticated access also failed for {repo_name}: {unauth_error}")

            if provider_type == "github" and os.getenv("GITHUB_APP_ID") and config_provider.get_github_key():
                try:
                    logger.info(f"Retrying branch fetch for {repo_name} with GitHub App auth")
                    provider = CodeProviderFactory.create_github_app_provider(repo_name)
                    all_branches = provider.list_branches(repo_name)
                    self.branch_cache.cache_all_branches(repo_name, all_branches, ttl=3600)
                    logger.info(f"Cached {len(all_branches)} branches for {repo_name} (GitHub App)")
                    return all_branches
                except Exception as app_error:
                    logger.warning(f"GitHub App auth also failed for {repo_name}: {str(app_error)}")

            if is_404_error or is_401_error or (hasattr(e, "status") and getattr(e, "status", None) == 403):
                logger.info(f"Authentication failed for {repo_name}: {str(e)}")
            else:
                logger.error(f"Error fetching branches for {repo_name}: {str(e)}", exc_info=True)
            raise

    async def get_branch_list(
        self, repo_name: str, limit: Optional[int] = None, offset: int = 0, search: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get branch list for a repository using the configured provider.
        Uses fallback authentication (PAT-first, then GitHub App) for private repos.
        Caches all fetched branches in Redis for future requests.
        Returns paginated results if limit is specified.
        GitHub API calls run in a thread pool to avoid blocking the event loop.
        """
        from app.modules.utils.logger import setup_logger

        logger = setup_logger(__name__)
        search_query = self._normalize_search_query(search)

        # Fast path: check cache (async Redis when available)
        cached_branches = await self.branch_cache.get_branches_async(
            repo_name, search_query=None
        )
        if cached_branches is not None:
            logger.info(f"Found {len(cached_branches)} cached branches for {repo_name}")
            filtered_branches = self._filter_branches(cached_branches, search_query)
            if search_query:
                logger.info(f"Filtered to {len(filtered_branches)} branches matching '{search_query}'")
            return self._paginate_branches(filtered_branches, limit, offset)

        try:
            all_branches = await asyncio.to_thread(
                self._fetch_branches_with_fallbacks_sync, repo_name
            )
            filtered_branches = self._filter_branches(all_branches, search_query)
            if search_query:
                logger.info(f"Filtered to {len(filtered_branches)} branches matching '{search_query}'")
            return self._paginate_branches(filtered_branches, limit, offset)
        except Exception as e:
            raise HTTPException(
                status_code=404,
                detail=f"Repository {repo_name} not found or error fetching branches: {str(e)}",
            )

    async def get_user_repos(
        self,
        user: Dict[str, Any],
        search: Optional[str] = None,
        async_db: Optional[AsyncSession] = None,
    ) -> Dict[str, Any]:
        """
        Get user repositories using the configured provider.

        When the provider is GitHub and GitHub App credentials are configured,
        use the GitHub App pathway to include installations/repos linked via the app.
        Otherwise, fall back to the generic provider listing (e.g., GitBucket).

        Args:
            user: User dictionary containing user_id
            search: Optional search query to filter repositories by name

        Returns:
            Dictionary containing filtered repositories
        """
        try:
            from app.modules.utils.logger import setup_logger
            logger = setup_logger(__name__)

            # Normalize and validate search query
            search_query = self._normalize_search_query(search)

            provider_type = os.getenv("CODE_PROVIDER", "github").lower()

            if (
                provider_type == "github"
                and os.getenv("GITHUB_APP_ID")
                and config_provider.get_github_key()
            ):
                github_service = GithubService(self.db)
                result = await github_service.get_combined_user_repos(
                    user["user_id"], async_session=async_db
                )
            else:
                def _list_repos_sync():
                    provider = CodeProviderFactory.create_provider()
                    return provider.list_user_repositories()

                repositories = await asyncio.to_thread(_list_repos_sync)
                result = {"repositories": repositories}

            # Apply search filter if provided
            if search_query:
                original_count = len(result.get("repositories", []))
                result["repositories"] = self._filter_repositories(result.get("repositories", []), search_query)
                logger.info(
                    f"Filtered to {len(result['repositories'])} repositories matching '{search_query}' "
                    f"(from {original_count} total)"
                )

            return result

        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error fetching user repositories: {str(e)}"
            )

    def _check_public_repo_sync(self, repo_name: str) -> bool:
        """Sync helper: try to access repo via provider. Runs in thread to avoid blocking."""
        provider = CodeProviderFactory.create_provider()
        provider.get_repository(repo_name)
        return True

    async def check_public_repo(self, repo_name: str) -> bool:
        """
        Check if a repository is public using the configured provider.
        GitHub API call runs in a thread pool to avoid blocking the event loop.
        """
        try:
            return await asyncio.to_thread(self._check_public_repo_sync, repo_name)
        except Exception:
            return False
