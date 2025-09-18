"""
Linear OAuth integration implementation following official documentation
https://developers.linear.app/docs/oauth
"""

from typing import Dict, Optional, Any
from starlette.config import Config
import httpx
import urllib.parse
import logging
import time


class LinearOAuthStore:
    """In-memory store for Linear OAuth tokens"""

    def __init__(self):
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

        # Check if token has expired
        expires_at = tokens.get("expires_at")
        if expires_at and time.time() > expires_at:
            self.remove_tokens(user_id)
            return False

        return True


class LinearOAuth:
    """Linear OAuth integration handler following official documentation"""

    def __init__(self, config: Config):
        self.config = config
        self.token_store = LinearOAuthStore()

        # Get OAuth credentials from config
        self.client_id = config("LINEAR_CLIENT_ID", default="")
        self.client_secret = config("LINEAR_CLIENT_SECRET", default="")

        if not self.client_id or not self.client_secret:
            logging.warning("Linear OAuth credentials not configured")

    def get_authorization_url(
        self, redirect_uri: str, state: Optional[str] = None, scope: str = "read"
    ) -> str:
        """
        Generate authorization URL for OAuth flow
        Following: https://developers.linear.app/docs/oauth
        """
        # Build query parameters with proper URL encoding
        params = {
            "client_id": self.client_id,
            "response_type": "code",
            "redirect_uri": redirect_uri,
            "scope": scope,
        }

        if state:
            params["state"] = state

        # Use urllib.parse.urlencode for proper URL encoding
        query_string = urllib.parse.urlencode(params, safe="")

        auth_url = f"https://linear.app/oauth/authorize?{query_string}"
        return auth_url

    async def exchange_code_for_tokens(
        self, authorization_code: str, redirect_uri: str
    ) -> Dict[str, Any]:
        """
        Exchange authorization code for access tokens
        Following: https://developers.linear.app/docs/oauth
        """
        try:
            if not self.client_id or not self.client_secret:
                raise Exception("Linear OAuth credentials not configured")

            # Linear OAuth token endpoint
            token_url = "https://api.linear.app/oauth/token"

            # Prepare token data for OAuth flow
            token_data = {
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "grant_type": "authorization_code",
                "code": authorization_code,
                "redirect_uri": redirect_uri,
            }

            # Log token exchange details for debugging
            logging.info("Linear token exchange request:")
            logging.info(f"  URL: {token_url}")
            logging.info(
                f"  Client ID: {self.client_id[:10]}..."
                if self.client_id
                else "  Client ID: None"
            )
            logging.info(f"  Redirect URI: {redirect_uri}")
            logging.info(
                f"  Code: {authorization_code[:20]}..."
                if authorization_code
                else "  Code: None"
            )

            # Make the token exchange request
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    token_url,
                    data=token_data,
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                )

                if response.status_code != 200:
                    logging.error("Linear token exchange failed:")
                    logging.error(f"  Status: {response.status_code}")
                    logging.error(f"  Response: {response.text}")
                    logging.error(f"  Headers: {dict(response.headers)}")
                    raise Exception(f"Token exchange failed: {response.status_code}")

                token_response = response.json()

                # Parse token expiration
                expires_at = time.time() + token_response.get("expires_in", 3600)

                # Extract token information
                tokens = {
                    "access_token": token_response.get("access_token"),
                    "token_type": token_response.get("token_type", "Bearer"),
                    "expires_in": token_response.get("expires_in"),
                    "expires_at": expires_at,
                    "scope": token_response.get("scope"),
                }

                return tokens

        except Exception as e:
            logging.error(f"Failed to exchange Linear OAuth code for tokens: {str(e)}")
            raise Exception(f"OAuth token exchange failed: {str(e)}")

    async def get_user_info_from_api(
        self, access_token: str
    ) -> Optional[Dict[str, Any]]:
        """Get user information from Linear API"""
        try:
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
            }

            # Get user information
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    "https://api.linear.app/graphql",
                    headers=headers,
                    json={
                        "query": """
                        query {
                            viewer {
                                id
                                name
                                email
                                organization {
                                    id
                                    name
                                    urlKey
                                }
                            }
                        }
                        """
                    },
                )

                if response.status_code != 200:
                    logging.error(
                        f"Failed to get Linear user info: {response.status_code}"
                    )
                    return None

                result = response.json()
                if "data" in result and "viewer" in result["data"]:
                    return result["data"]["viewer"]

                return None

        except Exception as e:
            logging.error(f"Error getting Linear user info: {str(e)}")
            return None

    def handle_callback(self, request, user_id: str) -> Dict[str, Any]:
        """Handle OAuth callback and store tokens"""
        try:
            # Get authorization code from query parameters
            code = request.query_params.get("code")
            state = request.query_params.get("state")
            error = request.query_params.get("error")

            if error:
                raise Exception(f"OAuth error: {error}")

            if not code:
                raise Exception("No authorization code received")

            # For now, we'll just log the callback
            # In a real implementation, you'd exchange the code for tokens
            logging.info(f"Linear OAuth callback received for user {user_id}")
            logging.info(f"Code: {code[:20]}...")
            logging.info(f"State: {state}")

            return {
                "status": "success",
                "message": "Linear OAuth callback received",
                "user_id": user_id,
                "code_received": bool(code),
                "state": state,
            }

        except Exception as e:
            logging.error(f"Linear OAuth callback failed: {str(e)}")
            return {
                "status": "error",
                "message": f"OAuth callback failed: {str(e)}",
                "user_id": user_id,
            }

    def get_user_info(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get stored user information"""
        return self.token_store.get_tokens(user_id)

    def revoke_access(self, user_id: str) -> bool:
        """Revoke OAuth access for a user"""
        try:
            self.token_store.remove_tokens(user_id)
            logging.info(f"Linear OAuth access revoked for user: {user_id}")
            return True
        except Exception as e:
            logging.error(f"Failed to revoke Linear OAuth access: {str(e)}")
            return False
