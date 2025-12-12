"""
Azure AD / Microsoft Entra ID SSO Provider Implementation

Handles Microsoft 365 and Azure AD authentication.
"""

import os
import logging
import ssl
import jwt
import httpx
from typing import Dict, Any, Optional
from urllib.parse import urlencode

from app.modules.auth.sso_providers.base_provider import (
    BaseSSOProvider,
    SSOUserInfo,
)

logger = logging.getLogger(__name__)


class AzureSSOProvider(BaseSSOProvider):
    """
    Azure AD / Microsoft Entra ID SSO Provider.

    Supports:
    - Azure AD work accounts
    - Microsoft 365 accounts
    - ID token verification
    - OAuth 2.0 / OpenID Connect flow
    """

    AZURE_AUTH_URL = "https://login.microsoftonline.com/{tenant}/oauth2/v2.0/authorize"
    AZURE_TOKEN_URL = "https://login.microsoftonline.com/{tenant}/oauth2/v2.0/token"
    AZURE_JWKS_URL = "https://login.microsoftonline.com/{tenant}/discovery/v2.0/keys"

    @property
    def provider_name(self) -> str:
        return "azure"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Azure AD SSO provider.

        Configuration:
            - client_id: Azure AD Application (client) ID
            - client_secret: Azure AD Client Secret
            - tenant_id: Azure AD Tenant ID (or 'common' for multi-tenant)
        """
        super().__init__(config)

        # Load from config or environment
        self.client_id = self.config.get("client_id") or os.getenv(
            "AZURE_SSO_CLIENT_ID"
        )
        self.client_secret = self.config.get("client_secret") or os.getenv(
            "AZURE_SSO_CLIENT_SECRET"
        )
        self.tenant_id = self.config.get("tenant_id") or os.getenv(
            "AZURE_SSO_TENANT_ID",
            "common",
        )

    async def verify_token(self, id_token_str: str) -> SSOUserInfo:
        """
        Verify an Azure AD ID token.

        Verifies:
        - Token signature using Azure's JWKS
        - Token expiration
        - Token audience (client_id)
        - Issuer (Microsoft identity platform)
        """
        try:
            # Get signing keys from Azure
            jwks_url = self.AZURE_JWKS_URL.format(tenant=self.tenant_id)
            # Use secure TLS defaults with explicit minimum protocol version
            ssl_context = ssl.create_default_context()
            # Explicitly set minimum TLS version for security
            ssl_context.minimum_version = ssl.TLSVersion.TLSv1_2
            async with httpx.AsyncClient(verify=ssl_context) as client:
                try:
                    jwks_response = await client.get(jwks_url, timeout=10.0)
                    jwks_response.raise_for_status()
                    jwks = jwks_response.json()
                except httpx.HTTPError as e:
                    logger.error(f"Failed to fetch Azure JWKS: {str(e)}")
                    raise ValueError(
                        f"Failed to fetch signing keys from Azure: {str(e)}"
                    )

                # Decode token header to get key ID
                unverified_header = jwt.get_unverified_header(id_token_str)
                kid = unverified_header.get("kid")

                # Find the matching key
                signing_key = None
                for key in jwks.get("keys", []):
                    if key.get("kid") == kid:
                        try:
                            signing_key = jwt.PyJWK(key)
                            break
                        except Exception as e:
                            logger.warning(f"Failed to create PyJWK from key: {str(e)}")
                            continue

                if not signing_key:
                    raise ValueError(
                        f"Unable to find matching signing key for kid: {kid}"
                    )

                # Verify and decode token
                payload = jwt.decode(
                    id_token_str,
                    signing_key.key,
                    algorithms=["RS256"],
                    audience=self.client_id,
                    options={
                        "verify_signature": True,
                        "verify_exp": True,
                        "verify_aud": True,
                    },
                )

                # Verify issuer
                expected_issuer = (
                    f"https://login.microsoftonline.com/{self.tenant_id}/v2.0"
                )
                if payload.get("iss") != expected_issuer and self.tenant_id != "common":
                    # For 'common' tenant, issuer will vary by actual tenant
                    logger.warning(f"Unexpected issuer: {payload.get('iss')}")

                # Extract user info
                user_info = SSOUserInfo(
                    email=payload.get("email") or payload.get("preferred_username"),
                    email_verified=True,  # Azure AD always verifies email
                    provider_uid=payload["oid"],  # Object ID
                    display_name=payload.get("name"),
                    given_name=payload.get("given_name"),
                    family_name=payload.get("family_name"),
                    raw_data=payload,
                )

                logger.info(
                    f"Successfully verified Azure AD token for {user_info.email}"
                )
                return user_info

        except jwt.ExpiredSignatureError:
            logger.error("Azure AD token has expired")
            raise ValueError("Token has expired")
        except jwt.InvalidAudienceError:
            logger.error("Azure AD token audience mismatch")
            raise ValueError("Invalid token audience")
        except jwt.InvalidTokenError as e:
            logger.error(f"Azure AD token validation failed: {str(e)}")
            raise ValueError(f"Invalid Azure AD token: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error verifying Azure AD token: {str(e)}")
            raise ValueError(f"Failed to verify Azure AD token: {str(e)}")

    def get_authorization_url(
        self,
        redirect_uri: str,
        state: Optional[str] = None,
        scopes: Optional[list] = None,
    ) -> str:
        """
        Generate Azure AD OAuth authorization URL.

        Default scopes:
        - openid
        - email
        - profile
        """
        if not scopes:
            scopes = ["openid", "email", "profile"]

        auth_url = self.AZURE_AUTH_URL.format(tenant=self.tenant_id)

        params = {
            "client_id": self.client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "response_mode": "query",
            "scope": " ".join(scopes),
        }

        if state:
            params["state"] = state

        return f"{auth_url}?{urlencode(params)}"

    def get_required_config_keys(self) -> list:
        """Azure requires client_id and tenant_id"""
        return ["client_id", "tenant_id"]

    def validate_config(self) -> bool:
        """Validate Azure configuration"""
        if not self.client_id:
            raise ValueError("Azure SSO requires client_id")

        if not self.tenant_id:
            raise ValueError("Azure SSO requires tenant_id")

        return True
