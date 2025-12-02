"""
SSO Provider Integrations

This package contains implementations for various SSO providers.
Each provider handles token verification and user info extraction.
"""

from app.modules.auth.sso_providers.base_provider import BaseSSOProvider
from app.modules.auth.sso_providers.google_provider import GoogleSSOProvider
from app.modules.auth.sso_providers.azure_provider import AzureSSOProvider

__all__ = [
    "BaseSSOProvider",
    "GoogleSSOProvider",
    "AzureSSOProvider",
]

