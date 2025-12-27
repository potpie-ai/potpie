"""
Base class for Atlassian OAuth integrations (Jira, Confluence, etc.)

All Atlassian products use the same OAuth 2.0 (3LO) infrastructure:
- Same authorization endpoint: https://auth.atlassian.com/authorize
- Same token endpoint: https://auth.atlassian.com/oauth/token
- Same resource endpoint: https://api.atlassian.com/oauth/token/accessible-resources

This base class implements the common OAuth flow, with product-specific
customization (scopes, API URLs) handled by subclasses.
"""

from typing import Dict, Optional, Any
from abc import ABC, abstractmethod
from starlette.config import Config
import httpx
import urllib.parse
import time
from app.modules.utils.logger import setup_logger
from app.modules.integrations import hash_user_id

logger = setup_logger(__name__)


class AtlassianOAuthStore:
    """In-memory store for Atlassian OAuth tokens (shared across all Atlassian products)"""

    def __init__(self) -> None:
        self._tokens: Dict[str, Dict[str, Any]] = {}

    def store_tokens(self, user_id: str, tokens: Dict[str, Any]) -> None:
        """Store OAuth tokens for a user"""
        self._tokens[user_id] = {**tokens, "stored_at": time.time()}

    def get_tokens(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve OAuth tokens for a user"""
        return self._tokens.get(user_id)

    def remove_tokens(self, user_id: str) -> None:
        """Remove OAuth tokens for a user"""
        self._tokens.pop(user_id, None)

    def is_token_valid(self, user_id: str) -> bool:
        """Check if stored token is still valid"""
        tokens = self.get_tokens(user_id)
        if not tokens:
            return False

        expires_at = tokens.get("expires_at")
        if expires_at and time.time() > expires_at:
            self.remove_tokens(user_id)
            return False
        return True


class AtlassianOAuthBase(ABC):
    """
    Base class for Atlassian product OAuth (Jira, Confluence, etc.)

    Implements OAuth 2.0 (3LO) flow common to all Atlassian products.
    Subclasses must define product_name and default_scope.
    """

    # Shared Atlassian OAuth endpoints
    AUTH_BASE_URL = "https://auth.atlassian.com"
    API_BASE_URL = "https://api.atlassian.com"

    def __init__(self, config: Config) -> None:
        self.config = config
        self.token_store = AtlassianOAuthStore()

        # Get OAuth credentials from config (shared across all Atlassian products)
        self.client_id = config(f"{self.product_name.upper()}_CLIENT_ID", default="")
        self.client_secret = config(
            f"{self.product_name.upper()}_CLIENT_SECRET", default=""
        )

        if not self.client_id or not self.client_secret:
            logger.warning(
                f"{self.product_name.capitalize()} OAuth credentials not configured"
            )

    @property
    @abstractmethod
    def product_name(self) -> str:
        """
        Product name identifier (e.g., 'jira', 'confluence')
        Used for config keys and API endpoints
        """
        pass

    @property
    @abstractmethod
    def default_scope(self) -> str:
        """
        Default OAuth scopes for this product
        Example for Jira: 'read:jira-work write:jira-work offline_access'
        Example for Confluence: 'read:confluence-content.all write:confluence-content offline_access'
        """
        pass

    def get_api_base_url(self, cloud_id: str) -> str:
        """
        Get API base URL for this product

        Args:
            cloud_id: Atlassian cloud ID (site ID)

        Returns:
            API base URL (e.g., 'https://api.atlassian.com/ex/jira/{cloud_id}')
        """
        return f"{self.API_BASE_URL}/ex/{self.product_name}/{cloud_id}"

    def get_authorization_url(
        self,
        redirect_uri: str,
        state: Optional[str] = None,
        scope: Optional[str] = None,
        prompt: str = "consent",
    ) -> str:
        """
        Generate the Atlassian authorization URL

        Args:
            redirect_uri: OAuth callback URL
            state: Optional state parameter for security
            scope: Optional custom scopes (uses default_scope if not provided)
            prompt: OAuth prompt type (default: 'consent')

        Returns:
            Authorization URL to redirect user to
        """
        if not redirect_uri:
            raise ValueError(
                f"redirect_uri is required for {self.product_name.capitalize()} OAuth"
            )

        scope_str = scope or self.default_scope

        params = {
            "audience": "api.atlassian.com",
            "client_id": self.client_id,
            "scope": scope_str,
            "redirect_uri": redirect_uri,
            "state": state or "SECURE_RANDOM",
            "response_type": "code",
            "prompt": prompt,
        }

        query_string = urllib.parse.urlencode(params, quote_via=urllib.parse.quote)
        auth_url = f"{self.AUTH_BASE_URL}/authorize?{query_string}"

        logger.info(
            f"Generated {self.product_name.capitalize()} authorization URL: %s",
            auth_url,
        )
        return auth_url

    async def exchange_code_for_tokens(
        self, authorization_code: str, redirect_uri: str
    ) -> Dict[str, Any]:
        """
        Exchange authorization code for access and refresh tokens

        Args:
            authorization_code: Code from OAuth callback
            redirect_uri: Same redirect URI used in authorization

        Returns:
            Dictionary containing access_token, refresh_token, etc.
        """
        if not self.client_id or not self.client_secret:
            raise Exception(
                f"{self.product_name.capitalize()} OAuth credentials not configured"
            )

        token_url = f"{self.AUTH_BASE_URL}/oauth/token"
        payload = {
            "grant_type": "authorization_code",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "code": authorization_code,
            "redirect_uri": redirect_uri,
        }

        logger.info(
            f"Calling {self.product_name.capitalize()} token endpoint at %s", token_url
        )

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                token_url,
                json=payload,
                headers={"Content-Type": "application/json"},
            )

        if response.status_code != 200:
            logger.error(
                f"{self.product_name.capitalize()} token exchange failed: %s - %s",
                response.status_code,
                response.text,
            )
            raise Exception(
                f"Token exchange failed ({response.status_code}): {response.text}"
            )

        token_response = response.json()
        expires_at = time.time() + token_response.get("expires_in", 3600)

        tokens = {
            "access_token": token_response.get("access_token"),
            "refresh_token": token_response.get("refresh_token"),
            "token_type": token_response.get("token_type", "Bearer"),
            "scope": token_response.get("scope"),
            "expires_in": token_response.get("expires_in"),
            "expires_at": expires_at,
        }

        logger.info(
            f"Received {self.product_name.capitalize()} tokens: %s", list(tokens.keys())
        )
        return tokens

    async def refresh_access_token(self, refresh_token: str) -> Dict[str, Any]:
        """
        Refresh an expired access token using the refresh token

        Args:
            refresh_token: Refresh token from previous authentication

        Returns:
            New tokens dictionary
        """
        if not self.client_id or not self.client_secret:
            raise Exception(
                f"{self.product_name.capitalize()} OAuth credentials not configured"
            )

        token_url = f"{self.AUTH_BASE_URL}/oauth/token"
        payload = {
            "grant_type": "refresh_token",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "refresh_token": refresh_token,
        }

        logger.info(f"Refreshing {self.product_name.capitalize()} access token")

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                token_url,
                json=payload,
                headers={"Content-Type": "application/json"},
            )

        if response.status_code != 200:
            logger.error(
                f"{self.product_name.capitalize()} token refresh failed: %s - %s",
                response.status_code,
                response.text,
            )
            raise Exception(
                f"Token refresh failed ({response.status_code}): {response.text}"
            )

        token_response = response.json()
        expires_at = time.time() + token_response.get("expires_in", 3600)

        tokens = {
            "access_token": token_response.get("access_token"),
            "refresh_token": token_response.get("refresh_token"),
            "token_type": token_response.get("token_type", "Bearer"),
            "scope": token_response.get("scope"),
            "expires_in": token_response.get("expires_in"),
            "expires_at": expires_at,
        }

        logger.info(
            f"Successfully refreshed {self.product_name.capitalize()} access token"
        )
        return tokens

    async def get_accessible_resources(self, access_token: str) -> Dict[str, Any]:
        """
        Get the list of Atlassian cloud resources the token can access

        Args:
            access_token: Valid OAuth access token

        Returns:
            List of accessible resources with cloud_id, name, url, scopes, etc.
        """
        url = f"{self.API_BASE_URL}/oauth/token/accessible-resources"
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/json",
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url, headers=headers)

        if response.status_code != 200:
            logger.error(
                "Failed to fetch accessible resources: %s - %s",
                response.status_code,
                response.text,
            )
            raise Exception(
                f"Failed to fetch accessible resources: {response.status_code}"
            )

        return response.json()

    def handle_callback(self, request, user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Parse OAuth callback query parameters

        Args:
            request: FastAPI Request object
            user_id: Optional user ID

        Returns:
            Dictionary with status, code, state, etc.
        """
        code = request.query_params.get("code")
        state = request.query_params.get("state")
        error = request.query_params.get("error")
        error_description = request.query_params.get("error_description")

        if error:
            logger.error(
                f"{self.product_name.capitalize()} OAuth returned error: %s - %s",
                error,
                error_description,
            )
            return {
                "status": "error",
                "message": f"OAuth error: {error}",
                "user_id": user_id or "unknown",
                "error_description": error_description,
            }

        if not code:
            logger.error(
                f"{self.product_name.capitalize()} OAuth callback missing authorization code"
            )
            return {
                "status": "error",
                "message": "No authorization code received",
                "user_id": user_id or "unknown",
            }

        logger.info(
            f"{self.product_name.capitalize()} OAuth callback received code for state %s",
            state,
        )
        return {
            "status": "success",
            "message": f"{self.product_name.capitalize()} OAuth callback received",
            "user_id": user_id or "unknown",
            "code": code,
            "state": state,
        }

    def get_user_info(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve token metadata for a cached user"""
        return self.token_store.get_tokens(user_id)

    def revoke_access(self, user_id: str) -> bool:
        """Remove cached tokens for a user"""
        try:
            self.token_store.remove_tokens(user_id)
            logger.info(
                f"Revoked {self.product_name.capitalize()} OAuth tokens for user %s",
                hash_user_id(user_id),
            )
            return True
        except Exception:
            logger.exception(
                f"Failed to revoke {self.product_name.capitalize()} OAuth tokens"
            )
            return False
