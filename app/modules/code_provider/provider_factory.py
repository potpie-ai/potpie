import os
import logging
from typing import Optional, Dict, Any
from enum import Enum

from app.modules.code_provider.base.code_provider_interface import (
    ICodeProvider,
    AuthMethod,
)
from app.modules.code_provider.github.github_provider import GitHubProvider
from app.core.config_provider import config_provider

logger = logging.getLogger(__name__)


class ProviderType(str, Enum):
    GITHUB = "github"
    GITBUCKET = "gitbucket"
    GITLAB = "gitlab"
    BITBUCKET = "bitbucket"


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
    ) -> ICodeProvider:
        """
        Create and configure a code provider instance.

        Args:
            provider_type: Override default provider type
            base_url: Override default base URL
            credentials: Authentication credentials
            auth_method: Authentication method to use

        Returns:
            Configured ICodeProvider instance
        """
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
        from github.Auth import AppAuth
        import requests

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
        response = requests.get(url, headers=headers)

        if response.status_code != 200:
            raise Exception(f"Failed to get installation ID for {repo_name}")

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
        Create provider with authentication fallback (PAT-first, then App auth).

        This method implements the PAT-first strategy:
        1. Try CODE_PROVIDER_TOKEN (new PAT config)
        2. Try GH_TOKEN_LIST (legacy PAT pool)
        3. Try GitHub App authentication (if configured)
        4. Raise error if all methods fail

        Args:
            repo_name: Repository name (needed for App auth)

        Returns:
            Authenticated ICodeProvider instance
        """
        # Try PAT authentication first (new config)
        token = os.getenv("CODE_PROVIDER_TOKEN")
        if token:
            logger.info("Using CODE_PROVIDER_TOKEN for authentication")
            # Use the configured provider type instead of hardcoded GitHubProvider
            provider = CodeProviderFactory.create_provider()
            provider.authenticate({"token": token}, AuthMethod.PERSONAL_ACCESS_TOKEN)
            return provider

        # Try legacy PAT pool
        token_list_str = os.getenv("GH_TOKEN_LIST", "")
        if token_list_str:
            import random

            tokens = [t.strip() for t in token_list_str.split(",") if t.strip()]
            if tokens:
                logger.info("Using GH_TOKEN_LIST for authentication")
                # Use the configured provider type instead of hardcoded GitHubProvider
                provider = CodeProviderFactory.create_provider()
                token = random.choice(tokens)
                provider.authenticate(
                    {"token": token}, AuthMethod.PERSONAL_ACCESS_TOKEN
                )
                return provider

        # Try GitHub App authentication as fallback
        app_id = os.getenv("GITHUB_APP_ID")
        private_key = config_provider.get_github_key()
        if app_id and private_key:
            logger.info("Using GitHub App authentication as fallback")
            try:
                return CodeProviderFactory.create_github_app_provider(repo_name)
            except Exception as e:
                logger.warning(f"GitHub App authentication failed: {e}")

        raise ValueError(
            "No authentication method available. "
            "Please configure CODE_PROVIDER_TOKEN, GH_TOKEN_LIST, or GitHub App credentials."
        )
