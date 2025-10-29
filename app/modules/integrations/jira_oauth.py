"""Jira OAuth integration following Atlassian 3LO documentation."""

from typing import Dict, Optional, Any
from starlette.config import Config
import httpx
import urllib.parse
import logging
import time


class JiraOAuthStore:
    """In-memory store for Jira OAuth tokens."""

    def __init__(self) -> None:
        self._tokens: Dict[str, Dict[str, Any]] = {}

    def store_tokens(self, user_id: str, tokens: Dict[str, Any]) -> None:
        """Store OAuth tokens for a user."""
        self._tokens[user_id] = {**tokens, "stored_at": time.time()}

    def get_tokens(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve OAuth tokens for a user."""
        return self._tokens.get(user_id)

    def remove_tokens(self, user_id: str) -> None:
        """Remove OAuth tokens for a user."""
        self._tokens.pop(user_id, None)

    def is_token_valid(self, user_id: str) -> bool:
        """Return True when the cached token has not expired."""
        tokens = self.get_tokens(user_id)
        if not tokens:
            return False

        expires_at = tokens.get("expires_at")
        if expires_at and time.time() > expires_at:
            self.remove_tokens(user_id)
            return False
        return True


class JiraOAuth:
    """Helper class for Jira OAuth exchanges."""

    AUTH_BASE_URL = "https://auth.atlassian.com"
    API_BASE_URL = "https://api.atlassian.com"

    def __init__(self, config: Config) -> None:
        self.config = config
        self.token_store = JiraOAuthStore()

        self.client_id = config("JIRA_CLIENT_ID", default="")
        self.client_secret = config("JIRA_CLIENT_SECRET", default="")
        self.default_scope = config(
            "JIRA_OAUTH_SCOPE", default="read:jira-user read:jira-work write:jira-work manage:jira-webhooks offline_access"
        )

        if not self.client_id or not self.client_secret:
            logging.warning("Jira OAuth credentials not configured")

    def get_authorization_url(
        self,
        redirect_uri: str,
        state: Optional[str] = None,
        scope: Optional[str] = None,
        prompt: str = "consent",
    ) -> str:
        """Generate the Atlassian authorization URL."""
        if not redirect_uri:
            raise ValueError("redirect_uri is required for Jira OAuth")

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

        logging.info("Generated Jira authorization URL: %s", auth_url)
        return auth_url

    async def exchange_code_for_tokens(
        self, authorization_code: str, redirect_uri: str
    ) -> Dict[str, Any]:
        """Exchange authorization code for access and refresh tokens."""
        if not self.client_id or not self.client_secret:
            raise Exception("Jira OAuth credentials not configured")

        token_url = f"{self.AUTH_BASE_URL}/oauth/token"
        payload = {
            "grant_type": "authorization_code",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "code": authorization_code,
            "redirect_uri": redirect_uri,
        }

        logging.info("Calling Jira token endpoint at %s", token_url)

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                token_url,
                json=payload,
                headers={"Content-Type": "application/json"},
            )

        if response.status_code != 200:
            logging.error(
                "Jira token exchange failed: %s - %s",
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

        logging.info("Received Jira tokens: %s", list(tokens.keys()))
        return tokens

    async def get_accessible_resources(self, access_token: str) -> Dict[str, Any]:
        """Return the list of Atlassian cloud resources the token can access."""
        url = f"{self.API_BASE_URL}/oauth/token/accessible-resources"
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/json",
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url, headers=headers)

        if response.status_code != 200:
            logging.error(
                "Failed to fetch accessible resources: %s - %s",
                response.status_code,
                response.text,
            )
            raise Exception(
                f"Failed to fetch accessible resources: {response.status_code}"
            )

        return response.json()

    def handle_callback(
        self, request, user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Parse Jira OAuth callback query parameters."""
        code = request.query_params.get("code")
        state = request.query_params.get("state")
        error = request.query_params.get("error")
        error_description = request.query_params.get("error_description")

        if error:
            logging.error("Jira OAuth returned error: %s - %s", error, error_description)
            return {
                "status": "error",
                "message": f"OAuth error: {error}",
                "user_id": user_id or "unknown",
                "error_description": error_description,
            }

        if not code:
            logging.error("Jira OAuth callback missing authorization code")
            return {
                "status": "error",
                "message": "No authorization code received",
                "user_id": user_id or "unknown",
            }

        logging.info("Jira OAuth callback received code for state %s", state)
        return {
            "status": "success",
            "message": "Jira OAuth callback received",
            "user_id": user_id or "unknown",
            "code": code,
            "state": state,
        }

    def get_user_info(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve token metadata for a cached user."""
        return self.token_store.get_tokens(user_id)

    def revoke_access(self, user_id: str) -> bool:
        """Remove cached tokens for a user."""
        try:
            self.token_store.remove_tokens(user_id)
            logging.info("Revoked Jira OAuth tokens for user %s", user_id)
            return True
        except Exception as exc:
            logging.error("Failed to revoke Jira OAuth tokens: %s", exc)
            return False
