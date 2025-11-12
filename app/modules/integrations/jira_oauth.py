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
            "JIRA_OAUTH_SCOPE",
            default="read:jira-user read:jira-work write:jira-work manage:jira-webhook offline_access manage:jira-configuration",
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

    async def refresh_access_token(self, refresh_token: str) -> Dict[str, Any]:
        """Refresh an expired access token using the refresh token."""
        if not self.client_id or not self.client_secret:
            raise Exception("Jira OAuth credentials not configured")

        token_url = f"{self.AUTH_BASE_URL}/oauth/token"
        payload = {
            "grant_type": "refresh_token",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "refresh_token": refresh_token,
        }

        logging.info("Refreshing Jira access token")

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                token_url,
                json=payload,
                headers={"Content-Type": "application/json"},
            )

        if response.status_code != 200:
            logging.error(
                "Jira token refresh failed: %s - %s",
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

        logging.info("Successfully refreshed Jira access token")
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

    def handle_callback(self, request, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Parse Jira OAuth callback query parameters."""
        code = request.query_params.get("code")
        state = request.query_params.get("state")
        error = request.query_params.get("error")
        error_description = request.query_params.get("error_description")

        if error:
            logging.error(
                "Jira OAuth returned error: %s - %s", error, error_description
            )
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

    async def create_webhook(
        self,
        cloud_id: str,
        access_token: str,
        webhook_url: str,
        events: Optional[list] = None,
        jql: Optional[str] = None,
        name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a Jira webhook for a Jira Cloud site (cloud_id).

        Returns the created webhook object (including id) on success.
        """
        if events is None:
            events = [
                "jira:issue_created",
                "jira:issue_updated",
                "jira:issue_deleted",
                "comment_created",
            ]

        if not cloud_id:
            raise Exception("cloud_id (site id) is required to create webhook")

        url = f"{self.API_BASE_URL}/ex/jira/{cloud_id}/rest/api/3/webhook"

        # Use the dynamic registration format required for OAuth2/Connect apps:
        # { "url": "https://app/webhook", "webhooks": [ { "events": [...], "jqlFilter": "..." } ] }
        # Note: OAuth webhooks require a non-empty JQL filter with supported operators only
        # Supported fields: project, issueKey, issuetype, status, priority, assignee, reporter
        webhook_entry: Dict[str, Any] = {"events": events}

        # Set JQL filter - default to matching all issues if not specified
        if jql is not None:
            webhook_entry["jqlFilter"] = jql
        else:
            # Match all issues across all projects using a condition that's always true
            # Using "priority != NonExistentPriority" as a workaround to match all issues
            webhook_entry["jqlFilter"] = (
                "priority != NonExistentPriority OR priority = NonExistentPriority"
            )

        payload: Dict[str, Any] = {"url": webhook_url, "webhooks": [webhook_entry]}

        headers = {
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

        # Log the payload we are about to send for diagnostics
        try:
            logging.info(
                "Creating Jira webhook for site %s -> %s", cloud_id, webhook_url
            )
            logging.info("Jira create_webhook payload: %s", payload)
        except Exception:
            # best-effort logging; don't fail the request because logging failed
            pass

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json=payload, headers=headers)

        # Log the response for debugging
        logging.info(
            "Webhook creation response status: %s, body: %s",
            response.status_code,
            response.text,
        )

        if response.status_code not in (200, 201):
            logging.error(
                "Failed to create Jira webhook (%s): %s",
                response.status_code,
                response.text,
            )
            raise Exception(
                f"Failed to create Jira webhook: {response.status_code} {response.text}"
            )

        # Successful response; return parsed JSON
        try:
            result = response.json()
            logging.info("Webhook creation result: %s", result)
            return result
        except Exception as e:
            logging.warning("Failed to parse webhook response as JSON: %s", e)
            return {"status_code": response.status_code, "text": response.text}

    async def delete_webhook(
        self, cloud_id: str, access_token: str, webhook_id: str
    ) -> bool:
        """Delete a Jira webhook by id for a given cloud_id (site)."""
        if not cloud_id or not webhook_id:
            raise Exception("cloud_id and webhook_id are required to delete webhook")

        # OAuth apps must use bulk delete endpoint
        url = f"{self.API_BASE_URL}/ex/jira/{cloud_id}/rest/api/3/webhook"

        headers = {
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

        # Webhook IDs must be sent as an array in the request body
        payload = {"webhookIds": [int(webhook_id)]}

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.request(
                "DELETE", url, headers=headers, json=payload
            )

        if response.status_code not in (200, 202, 204):
            logging.error(
                "Failed to delete Jira webhook (%s): %s",
                response.status_code,
                response.text,
            )
            return False

        logging.info(f"Successfully deleted webhook {webhook_id} for site {cloud_id}")
        return True
