import os
from enum import Enum
from typing import Any, Dict, Optional

from app.core.config_provider import config_provider
from app.modules.code_provider.base.code_provider_interface import (
    AuthMethod,
    ICodeProvider,
)
from app.modules.code_provider.github.github_provider import GitHubProvider
from app.modules.utils.logger import setup_logger

try:
    from github.GithubException import GithubException
except ImportError:
    GithubException = None

logger = setup_logger(__name__)


class ProviderType(str, Enum):
    GITHUB = "github"
    GITBUCKET = "gitbucket"
    GITLAB = "gitlab"
    BITBUCKET = "bitbucket"
    LOCAL = "local"


class CodeProviderFactory:
    """
    Factory for creating code provider instances.

    Configuration via environment variables:
    - CODE_PROVIDER: Provider type (github, gitbucket, gitlab, bitbucket)
    - CODE_PROVIDER_BASE_URL: Base URL for provider (for self-hosted instances)
    - CODE_PROVIDER_TOKEN: Personal access token (recommended)
    - CODE_PROVIDER_TOKEN_POOL: Comma-separated multiple PATs
    - GITHUB_APP_ID, GITHUB_PRIVATE_KEY: For GitHub App auth (legacy)
    - GH_TOKEN_LIST: Legacy PAT pool (deprecated)
    """

    @staticmethod
    def create_provider(
        provider_type: Optional[str] = None,
        base_url: Optional[str] = None,
        credentials: Optional[Dict[str, Any]] = None,
        auth_method: Optional[AuthMethod] = None,
        repo_name: Optional[str] = None,
    ) -> ICodeProvider:
        """
        Create and configure a code provider instance.

        Args:
            provider_type: Override default provider type
            base_url: Override default base URL
            credentials: Authentication credentials
            auth_method: Authentication method to use
            repo_name: Optional repository identifier used for local path detection

        Returns:
            Configured ICodeProvider instance
        """
        # Detect local repositories first
        local_repo_path = CodeProviderFactory._resolve_local_repo_path(repo_name)
        if local_repo_path:
            from app.modules.code_provider.local_repo.local_provider import (
                LocalProvider,
            )

            logger.debug(f"Using LocalProvider for repository path: {local_repo_path}")
            return LocalProvider(default_repo_path=local_repo_path)

        # Determine provider type
        if not provider_type:
            provider_type = os.getenv("CODE_PROVIDER", "github").lower()

        # Determine base URL
        if not base_url:
            base_url = os.getenv("CODE_PROVIDER_BASE_URL")

        # Create provider instance
        if provider_type == ProviderType.GITHUB:
            base_url = base_url or "https://api.github.com"
            provider = GitHubProvider(base_url=base_url)

        elif provider_type == ProviderType.GITBUCKET:
            if not base_url:
                raise ValueError(
                    "GitBucket requires CODE_PROVIDER_BASE_URL environment variable. "
                    "Example: CODE_PROVIDER_BASE_URL=http://localhost:8080/api/v3"
                )
            from app.modules.code_provider.gitbucket.gitbucket_provider import (
                GitBucketProvider,
            )

            provider = GitBucketProvider(base_url=base_url)

        elif provider_type == ProviderType.GITLAB:
            base_url = base_url or "https://gitlab.com"
            # provider = GitLabProvider(base_url=base_url)
            raise NotImplementedError("GitLab provider not yet implemented")

        elif provider_type == ProviderType.BITBUCKET:
            base_url = base_url or "https://api.bitbucket.org/2.0"
            # provider = BitbucketProvider(base_url=base_url)
            raise NotImplementedError("Bitbucket provider not yet implemented")

        elif provider_type == ProviderType.LOCAL:
            from app.modules.code_provider.local_repo.local_provider import (
                LocalProvider,
            )

            # For local provider, base_url is the repository path
            if not base_url:
                raise ValueError(
                    "Local provider requires CODE_PROVIDER_BASE_URL to be set to repository path. "
                    "Example: CODE_PROVIDER_BASE_URL=/path/to/repo"
                )

            provider = LocalProvider(default_repo_path=base_url)
            logger.info(f"Created LocalProvider for path: {base_url}")
            return provider  # Return early, no authentication needed

        else:
            raise ValueError(f"Unknown provider type: {provider_type}")

        # Authenticate if credentials provided
        if credentials and auth_method:
            provider.authenticate(credentials, auth_method)
        elif credentials:
            # Auto-detect auth method
            if "token" in credentials:
                provider.authenticate(credentials, AuthMethod.PERSONAL_ACCESS_TOKEN)
            elif "access_token" in credentials:
                provider.authenticate(credentials, AuthMethod.OAUTH_TOKEN)
            elif "username" in credentials and "password" in credentials:
                provider.authenticate(credentials, AuthMethod.BASIC_AUTH)
        else:
            # Try to authenticate with environment variables (PAT-first)
            token = os.getenv("CODE_PROVIDER_TOKEN")
            if token:
                logger.info("Authenticating with CODE_PROVIDER_TOKEN (PAT)")
                provider.authenticate(
                    {"token": token}, AuthMethod.PERSONAL_ACCESS_TOKEN
                )
            else:
                # Try Basic Auth from environment
                username = os.getenv("CODE_PROVIDER_USERNAME")
                password = os.getenv("CODE_PROVIDER_PASSWORD")
                if username and password:
                    logger.info(
                        "Authenticating with CODE_PROVIDER_USERNAME/PASSWORD (Basic Auth)"
                    )
                    provider.authenticate(
                        {"username": username, "password": password},
                        AuthMethod.BASIC_AUTH,
                    )
                else:
                    # Fallback to legacy GH_TOKEN_LIST
                    token_list_str = os.getenv("GH_TOKEN_LIST", "")
                    if token_list_str:
                        import random

                        tokens = [
                            t.strip() for t in token_list_str.split(",") if t.strip()
                        ]
                        if tokens:
                            token = random.choice(tokens)
                            logger.info(
                                "Authenticating with GH_TOKEN_LIST (legacy PAT pool)"
                            )
                            provider.authenticate(
                                {"token": token}, AuthMethod.PERSONAL_ACCESS_TOKEN
                            )

        return provider

    @staticmethod
    def create_github_app_provider(repo_name: str) -> ICodeProvider:
        """
        Create GitHub provider with App authentication for specific repo.
        Legacy method for backward compatibility.

        Args:
            repo_name: Repository name to get installation ID for

        Returns:
            GitHubProvider authenticated with App installation token
        """
        provider = GitHubProvider()

        app_id = os.getenv("GITHUB_APP_ID")
        private_key = config_provider.get_github_key()

        if not app_id or not private_key:
            raise ValueError("GitHub App credentials not configured")

        # Get installation ID for repo
        import requests
        from github.Auth import AppAuth

        if not private_key.startswith("-----BEGIN"):
            private_key = f"-----BEGIN RSA PRIVATE KEY-----\n{private_key}\n-----END RSA PRIVATE KEY-----\n"

        auth = AppAuth(app_id=app_id, private_key=private_key)
        jwt = auth.create_jwt()

        url = f"https://api.github.com/repos/{repo_name}/installation"
        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {jwt}",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        response = requests.get(url, headers=headers, timeout=60)

        if response.status_code == 404:
            # App not installed on this repository (likely public repo or no access)
            raise ValueError(
                f"GitHub App not installed on repository {repo_name}. "
                f"This is expected for public repos or repos where the app isn't installed."
            )
        elif response.status_code != 200:
            raise Exception(
                f"Failed to get installation ID for {repo_name}: "
                f"HTTP {response.status_code} - {response.text}"
            )

        installation_id = response.json()["id"]

        provider.authenticate(
            {
                "app_id": app_id,
                "private_key": private_key,
                "installation_id": installation_id,
            },
            AuthMethod.APP_INSTALLATION,
        )

        return provider

    @staticmethod
    def get_default_provider() -> ICodeProvider:
        """Get default provider configured via environment variables."""
        return CodeProviderFactory.create_provider()

    @staticmethod
    def create_provider_with_fallback(repo_name: str) -> ICodeProvider:
        """
        Create provider with comprehensive authentication fallback strategy.

        This method implements the ONLY authentication fallback chain in the codebase.
        All callers should rely on this method's fallback behavior and NOT implement
        their own retry logic.

        Authentication priority order:
        1. Local repository (if repo_name is a local path)
        2. GitHub App (if GITHUB_APP_ID configured and provider is GitHub)
        3. PAT from GH_TOKEN_LIST (GitHub only, random selection for load distribution)
        4. PAT from CODE_PROVIDER_TOKEN (universal fallback for all providers)
        5. Unauthenticated access (GitHub only, for public repos)

        Args:
            repo_name: Repository name or local path

        Returns:
            Authenticated ICodeProvider instance

        Raises:
            ValueError: If no authentication method is available
        """
        # Handle local repositories without authentication
        local_repo_path = CodeProviderFactory._resolve_local_repo_path(repo_name)
        if local_repo_path:
            from app.modules.code_provider.local_repo.local_provider import (
                LocalProvider,
            )

            logger.debug(
                f"Using LocalProvider (fallback) for repository path: {local_repo_path}"
            )
            return LocalProvider(default_repo_path=local_repo_path)

        provider_type = os.getenv("CODE_PROVIDER", "github").lower()

        # For GitHub, validate that repo_name looks like a valid GitHub repo name
        # GitHub repos must be in "owner/repo" format
        if provider_type == "github" and "/" not in repo_name:
            logger.error(
                f"Invalid GitHub repository name: '{repo_name}'. "
                f"GitHub repos must be in 'owner/repo' format. "
                f"If this is a local repository, ensure repo_path is set in the database."
            )
            raise ValueError(
                f"Invalid repository name '{repo_name}'. "
                f"GitHub repos must be in 'owner/repo' format (e.g., 'facebook/react'). "
                f"For local repositories, use the full filesystem path."
            )

        # Check if GitHub App is configured (only relevant for GitHub provider)
        app_id = os.getenv("GITHUB_APP_ID")
        private_key = (
            config_provider.get_github_key()
            if provider_type == ProviderType.GITHUB
            else None
        )
        is_github_app_configured = bool(app_id and private_key)

        # For GitHub with App configured: Try GitHub App first, then PAT
        if provider_type == ProviderType.GITHUB and is_github_app_configured:
            logger.info(
                "GitHub App is configured, trying App auth first", repo_name=repo_name
            )
            try:
                return CodeProviderFactory.create_github_app_provider(repo_name)
            except Exception as e:
                # Check if this is an expected failure (app not installed on repo)
                error_msg = str(e)
                is_expected_failure = (
                    "GitHub App not installed" in error_msg
                    or "404" in error_msg
                )
                
                if is_expected_failure:
                    # This is expected for public repos or repos where app isn't installed
                    logger.debug(
                        f"GitHub App not installed on repository {repo_name} (expected for public repos or repos where app isn't installed), falling back to PAT"
                    )
                else:
                    # Unexpected error - log as warning
                    logger.warning(
                        f"GitHub App authentication failed for {repo_name}: {e}, falling back to PAT"
                    )
                # Continue to PAT fallback below

        # For GitHub: Try GH_TOKEN_LIST first (where GitHub PATs are stored)
        # For other providers: Try CODE_PROVIDER_TOKEN first
        if provider_type == ProviderType.GITHUB:
            # For GitHub, prioritize GH_TOKEN_LIST over CODE_PROVIDER_TOKEN
            token_list_str = os.getenv("GH_TOKEN_LIST", "")

            # Debug: Log the raw token list string
            if token_list_str:
                token_repr = repr(token_list_str)
                logger.debug("Raw GH_TOKEN_LIST from environment:")
                logger.debug(f"  - Length: {len(token_list_str)}")
                logger.debug(f"  - Has newlines: {chr(10) in token_list_str}")
                logger.debug(f"  - Has carriage returns: {chr(13) in token_list_str}")
                logger.debug(f"  - Repr: {token_repr[:50]}...")

            if token_list_str:
                import random

                tokens = [t.strip() for t in token_list_str.split(",") if t.strip()]
                logger.debug(f"Parsed {len(tokens)} token(s) from GH_TOKEN_LIST")
                if tokens:
                    logger.info(
                        f"Using GH_TOKEN_LIST for authentication ({len(tokens)} token(s) available)"
                    )
                    # GH_TOKEN_LIST is specifically for GitHub.com, not GitBucket or other providers
                    # Always use GitHub's API endpoint when using GH_TOKEN_LIST
                    base_url = "https://api.github.com"
                    provider = GitHubProvider(base_url=base_url)
                    token = random.choice(tokens)

                    provider.authenticate(
                        {"token": token}, AuthMethod.PERSONAL_ACCESS_TOKEN
                    )
                    return provider

        # Try CODE_PROVIDER_TOKEN (for non-GitHub providers, or as fallback for GitHub)
        token = os.getenv("CODE_PROVIDER_TOKEN")
        if token:
            logger.info("Using CODE_PROVIDER_TOKEN for authentication")
            provider = CodeProviderFactory.create_provider()
            provider.authenticate({"token": token}, AuthMethod.PERSONAL_ACCESS_TOKEN)
            return provider

        # If we get here and it's GitHub without App configured, try unauthenticated access
        if provider_type == ProviderType.GITHUB:
            logger.info(
                f"No PAT configured, trying unauthenticated access for {repo_name}"
            )
            try:
                # Use GitHub.com API for GitHub provider (not GitBucket or other configured base URLs)
                base_url = "https://api.github.com"
                provider = GitHubProvider(base_url=base_url)
                provider.set_unauthenticated_client()
                return provider
            except Exception as e:
                logger.warning(f"Failed to create unauthenticated provider: {e}")

        # If we get here, we have no auth method
        raise ValueError(
            "No authentication method available. "
            "Please configure CODE_PROVIDER_TOKEN, GH_TOKEN_LIST, or GitHub App credentials."
        )

    @staticmethod
    def _resolve_local_repo_path(repo_name: Optional[str]) -> Optional[str]:
        """
        Resolve repo_name to a local repository path if it points to a git directory.

        Returns:
            Absolute path to the repository or None if not local.
        """
        if not repo_name:
            return None

        expanded_path = os.path.abspath(os.path.expanduser(repo_name))

        if not os.path.isdir(expanded_path):
            return None

        git_dir = os.path.join(expanded_path, ".git")
        if os.path.isdir(git_dir) or os.path.isfile(git_dir):
            return expanded_path

        # Handle bare repositories where .git is the repository itself
        try:
            from git import Repo

            Repo(expanded_path)
            return expanded_path
        except Exception:
            return None


def has_code_provider_credentials() -> bool:
    """
    Check if any valid code provider credentials are configured.

    This function checks for credentials in the same order as
    create_provider_with_fallback() to ensure consistency.

    Checks for:
    1. CODE_PROVIDER_TOKEN (works for all providers)
    2. GH_TOKEN_LIST (legacy, works for GitHub/GitBucket)
    3. CODE_PROVIDER_USERNAME + CODE_PROVIDER_PASSWORD (GitBucket Basic Auth)
    4. GITHUB_APP_ID + private key (GitHub only)

    Returns:
        bool: True if any valid credentials exist, False otherwise

    Example:
        >>> os.environ['CODE_PROVIDER_TOKEN'] = 'ghp_xxx'
        >>> has_code_provider_credentials()
        True
    """
    # Check for primary PAT (works for all providers)
    if os.getenv("CODE_PROVIDER_TOKEN"):
        return True

    # Check for legacy PAT pool (works for GitHub/GitBucket)
    token_list_str = os.getenv("GH_TOKEN_LIST", "")
    if token_list_str:
        tokens = [t.strip() for t in token_list_str.split(",") if t.strip()]
        if tokens:
            return True

    # Check for Basic Auth credentials (works for GitBucket)
    if os.getenv("CODE_PROVIDER_USERNAME") and os.getenv("CODE_PROVIDER_PASSWORD"):
        return True

    # Check for GitHub App credentials (GitHub only)
    if os.getenv("GITHUB_APP_ID") and config_provider.get_github_key():
        return True

    return False
