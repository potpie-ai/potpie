"""
Sentry OAuth integration implementation following official documentation
https://docs.sentry.io/product/partnership-platform/oauth-integration/
"""

from typing import Dict, Optional, Any
from starlette.config import Config
from fastapi import HTTPException
import httpx
import urllib.parse
import logging
import time


class SentryOAuthStore:
    """In-memory store for Sentry OAuth tokens (compatibility with old implementation)"""

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


class SentryOAuthV2:
    """Sentry OAuth integration handler following official documentation"""

    def __init__(self, config: Config):
        self.config = config
        self.token_store = SentryOAuthStore()

        # Get OAuth credentials from config
        self.client_id = config("SENTRY_CLIENT_ID", default="")
        self.client_secret = config("SENTRY_CLIENT_SECRET", default="")

        if not self.client_id or not self.client_secret:
            raise HTTPException(
                status_code=500, detail="Sentry OAuth credentials not configured"
            )

    def get_authorization_url(
        self, redirect_uri: str, state: Optional[str] = None
    ) -> str:
        """
        Generate authorization URL for OAuth flow
        Following: https://docs.sentry.io/product/partnership-platform/oauth-integration/
        """
        # Build query parameters with proper scope encoding
        params = {
            "client_id": self.client_id,
            "response_type": "code",
            "redirect_uri": redirect_uri,
        }

        if state:
            params["state"] = state

        # Construct URL manually to avoid + in scope parameter
        safe_chars = ":/?#[]@!$&'()*+,;="
        auth_url = (
            f"https://sentry.io/oauth/authorize/"
            f"?client_id={self.client_id}"
            f"&response_type=code"
            f"&redirect_uri={urllib.parse.quote(redirect_uri, safe=safe_chars)}"
        )

        if state:
            auth_url += f"&state={urllib.parse.quote(state, safe=safe_chars)}"

        logging.info(f"Generated Sentry OAuth URL: {auth_url}")
        return auth_url

    async def exchange_code_for_tokens(
        self, authorization_code: str, redirect_uri: str
    ) -> Dict[str, Any]:
        """
        Exchange authorization code for access tokens
        Following: https://docs.sentry.io/product/partnership-platform/oauth-integration/
        """
        try:
            # Token exchange endpoint
            token_url = "https://sentry.io/oauth/token/"

            # Prepare token data exactly as shown in Sentry documentation
            token_data = {
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "grant_type": "authorization_code",
                "code": authorization_code,
                "redirect_uri": redirect_uri,
            }

            logging.info("=== Sentry OAuth Token Exchange ===")
            logging.info(f"Token URL: {token_url}")
            logging.info(f"Client ID: {self.client_id}")
            logging.info(f"Code: {authorization_code}")
            logging.info(f"Code length: {len(authorization_code)}")
            logging.info(f"Code first 10 chars: {authorization_code[:10]}")
            logging.info(f"Code last 10 chars: {authorization_code[-10:]}")
            logging.info(f"Redirect URI: {redirect_uri}")
            logging.info(f"Request data keys: {list(token_data.keys())}")

            # Log the exact request payload (without secrets)
            debug_payload = {
                k: v for k, v in token_data.items() if k != "client_secret"
            }
            debug_payload["client_secret"] = "***REDACTED***"
            logging.info(f"Request payload: {debug_payload}")

            # Make the token exchange request using httpx (as shown in Sentry docs)
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    token_url,
                    data=token_data,
                    headers={
                        "Content-Type": "application/x-www-form-urlencoded",
                        "Accept": "application/json",
                    },
                )

                logging.info(f"Response status: {response.status_code}")
                logging.info(f"Response headers: {dict(response.headers)}")
                logging.info(f"Response content: {response.text}")

                if response.status_code != 200:
                    logging.error(f"Token exchange failed: {response.status_code}")
                    logging.error(f"Response: {response.text}")
                    raise HTTPException(
                        status_code=response.status_code,
                        detail=f"Token exchange failed: {response.text}",
                    )

                # Parse the token response
                tokens = response.json()
                logging.info(f"Received tokens: {list(tokens.keys())}")

                return tokens

        except httpx.HTTPError as e:
            logging.error(f"HTTP error during token exchange: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"HTTP error during token exchange: {str(e)}"
            )
        except Exception as e:
            logging.error(f"Unexpected error during token exchange: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Token exchange failed: {str(e)}"
            )

    async def get_user_organizations(self, access_token: str) -> Dict[str, Any]:
        """
        Get user organizations using access token
        Following: https://docs.sentry.io/product/partnership-platform/oauth-integration/
        """
        try:
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
            }

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    "https://sentry.io/api/0/organizations/", headers=headers
                )

                if response.status_code != 200:
                    logging.error(
                        f"Failed to get organizations: {response.status_code}"
                    )
                    logging.error(f"Response: {response.text}")
                    raise HTTPException(
                        status_code=response.status_code,
                        detail=f"Failed to get organizations: {response.text}",
                    )

                organizations = response.json()
                logging.info(f"Retrieved {len(organizations)} organizations")

                return organizations

        except httpx.HTTPError as e:
            logging.error(f"HTTP error getting organizations: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Failed to get organizations: {str(e)}"
            )
        except Exception as e:
            logging.error(f"Unexpected error getting organizations: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Failed to get organizations: {str(e)}"
            )

    def get_user_info(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user information from Sentry using stored tokens (compatibility method)"""
        if not self.token_store.is_token_valid(user_id):
            return None

        tokens = self.token_store.get_tokens(user_id)
        if not tokens:
            return None

        # Return basic token info for compatibility
        return {
            "user_id": user_id,
            "has_valid_token": True,
            "token_type": tokens.get("token_type"),
            "scope": tokens.get("scope"),
            "expires_at": tokens.get("expires_at"),
        }

    def revoke_access(self, user_id: str) -> bool:
        """Revoke OAuth access for a user (compatibility method)"""
        try:
            self.token_store.remove_tokens(user_id)
            return True
        except Exception:
            return False

    def handle_callback(self, request, user_id: str) -> Dict[str, Any]:
        """Handle OAuth callback and store tokens (compatibility method)"""
        # This method is deprecated in favor of the new flow
        raise HTTPException(
            status_code=501,
            detail="This method is deprecated. Use integrations_service.save_sentry_integration instead.",
        )
