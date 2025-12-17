"""
Google SSO Provider Implementation

Handles Google Workspace / Gmail authentication.
"""

import os
import logging
from typing import Dict, Any, Optional
from urllib.parse import urlencode

from google.auth.transport import requests
from google.oauth2 import id_token

from app.modules.auth.sso_providers.base_provider import (
    BaseSSOProvider,
    SSOUserInfo,
)

logger = logging.getLogger(__name__)


class GoogleSSOProvider(BaseSSOProvider):
    """
    Google SSO Provider.

    Supports:
    - Google Workspace accounts
    - Personal Gmail accounts
    - ID token verification
    - OAuth 2.0 flow
    """

    GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
    GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
    GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v3/userinfo"

    @property
    def provider_name(self) -> str:
        return "google"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Google SSO provider.

        Configuration:
            - client_id: Google OAuth 2.0 Client ID
            - client_secret: Google OAuth 2.0 Client Secret (for server-side flow)
            - hd: Hosted domain for Google Workspace (optional)
        """
        super().__init__(config)

        # Load from config or environment
        self.client_id = self.config.get("client_id") or os.getenv(
            "GOOGLE_SSO_CLIENT_ID"
        )
        self.client_secret = self.config.get("client_secret") or os.getenv(
            "GOOGLE_SSO_CLIENT_SECRET"
        )
        self.hosted_domain = self.config.get("hd") or os.getenv(
            "GOOGLE_SSO_HOSTED_DOMAIN"
        )

    async def verify_token(self, id_token_str: str) -> SSOUserInfo:
        """
        Verify a Google ID token.

        Uses Google's official library to verify:
        - Token signature
        - Token expiration
        - Token audience (client_id)
        - Issuer (accounts.google.com)
        """
        try:
            # Verify the token
            idinfo = id_token.verify_oauth2_token(
                id_token_str,
                requests.Request(),
                self.client_id,
            )

            # Verify issuer
            if idinfo["iss"] not in [
                "accounts.google.com",
                "https://accounts.google.com",
            ]:
                raise ValueError("Invalid token issuer")

            # Verify hosted domain if configured
            if self.hosted_domain:
                token_hd = idinfo.get("hd")
                if token_hd != self.hosted_domain:
                    raise ValueError(
                        f"Token from wrong domain: expected {self.hosted_domain}, "
                        f"got {token_hd}"
                    )

            # Extract user info
            user_info = SSOUserInfo(
                email=idinfo["email"],
                email_verified=idinfo.get("email_verified", False),
                provider_uid=idinfo["sub"],
                display_name=idinfo.get("name"),
                given_name=idinfo.get("given_name"),
                family_name=idinfo.get("family_name"),
                picture=idinfo.get("picture"),
                raw_data=idinfo,
            )

            logger.info(f"Successfully verified Google token for {user_info.email}")
            return user_info

        except ValueError as e:
            logger.error(f"Google token verification failed: {str(e)}")
            raise ValueError(f"Invalid Google ID token: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error verifying Google token: {str(e)}")
            raise ValueError(f"Failed to verify Google ID token: {str(e)}")

    def get_authorization_url(
        self,
        redirect_uri: str,
        state: Optional[str] = None,
        scopes: Optional[list] = None,
    ) -> str:
        """
        Generate Google OAuth authorization URL.

        Default scopes:
        - openid
        - email
        - profile
        """
        if not scopes:
            scopes = ["openid", "email", "profile"]

        params = {
            "client_id": self.client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": " ".join(scopes),
            "access_type": "offline",
            "prompt": "consent",
        }

        if state:
            params["state"] = state

        if self.hosted_domain:
            params["hd"] = self.hosted_domain

        return f"{self.GOOGLE_AUTH_URL}?{urlencode(params)}"

    def get_required_config_keys(self) -> list:
        """Google requires client_id at minimum"""
        return ["client_id"]

    def validate_config(self) -> bool:
        """Validate Google configuration"""
        if not self.client_id:
            raise ValueError("Google SSO requires client_id")

        return True
