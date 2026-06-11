"""OAuth and Atlassian API configuration for CLI integration auth."""

from __future__ import annotations

import os
from typing import Literal
from urllib.parse import urlparse

Provider = Literal["linear", "github", "atlassian", "jira", "confluence"]
OAuthProvider = Literal["linear"]
AtlassianProduct = Literal["jira", "confluence"]

DEFAULT_CALLBACK_PORT = 8080
DEFAULT_CALLBACK_PATH = "/callback"
DEFAULT_REDIRECT_URI = (
    f"http://localhost:{DEFAULT_CALLBACK_PORT}{DEFAULT_CALLBACK_PATH}"
)
DEFAULT_CALLBACK_HOST = "localhost"

LINEAR_AUTH_URL = "https://linear.app/oauth/authorize"
LINEAR_TOKEN_URL = "https://api.linear.app/oauth/token"
LINEAR_DEFAULT_SCOPE = "read"

ATLASSIAN_API_TOKEN_PAGE = "https://id.atlassian.com/manage-profile/security/api-tokens"
ATLASSIAN_API_GATEWAY = "https://api.atlassian.com"
ATLASSIAN_ACCESSIBLE_RESOURCES_URL = (
    f"{ATLASSIAN_API_GATEWAY}/oauth/token/accessible-resources"
)


def get_redirect_uri() -> str:
    return os.getenv("POTPIE_CLI_OAUTH_REDIRECT_URI", DEFAULT_REDIRECT_URI).strip()


def _parsed_redirect_uri():
    parsed = urlparse(get_redirect_uri())
    if parsed.scheme != "http" or not parsed.hostname or not parsed.port:
        raise ValueError(
            "POTPIE_CLI_OAUTH_REDIRECT_URI must be an http localhost URL with a port, "
            "for example http://localhost:8001/api/v1/integrations/linear/callback."
        )
    if parsed.hostname not in {"localhost", "127.0.0.1"}:
        raise ValueError(
            "POTPIE_CLI_OAUTH_REDIRECT_URI must use localhost or 127.0.0.1."
        )
    return parsed


def get_callback_port() -> int:
    raw = os.getenv("POTPIE_CLI_OAUTH_CALLBACK_PORT", "").strip()
    if raw:
        try:
            port = int(raw)
            if 1 <= port <= 65535:
                return port
        except ValueError:
            pass
    return int(_parsed_redirect_uri().port or DEFAULT_CALLBACK_PORT)


def get_client_id(provider: OAuthProvider) -> str:
    if provider != "linear":
        return ""
    val = os.getenv("LINEAR_CLIENT_ID", "").strip()
    if val:
        return val
    try:
        from adapters.outbound.cli_auth._build_config import LINEAR_CLIENT_ID  # noqa: PLC0415

        return LINEAR_CLIENT_ID
    except (ImportError, AttributeError):
        return ""


def get_client_secret(provider: OAuthProvider) -> str:
    return ""


def get_callback_host() -> str:
    return _parsed_redirect_uri().hostname or DEFAULT_CALLBACK_HOST


def get_callback_path() -> str:
    path = _parsed_redirect_uri().path or DEFAULT_CALLBACK_PATH
    return path if path.startswith("/") else f"/{path}"


def get_scopes(provider: OAuthProvider) -> str:
    if provider != "linear":
        return ""
    return os.getenv("LINEAR_OAUTH_SCOPE", LINEAR_DEFAULT_SCOPE).strip()


def authorization_url(provider: OAuthProvider) -> str:
    if provider != "linear":
        raise ValueError(f"Unsupported OAuth provider: {provider!r}")
    return LINEAR_AUTH_URL


def token_url(provider: OAuthProvider) -> str:
    if provider != "linear":
        raise ValueError(f"Unsupported OAuth provider: {provider!r}")
    return LINEAR_TOKEN_URL


def atlassian_jira_gateway_url(cloud_id: str) -> str:
    return f"{ATLASSIAN_API_GATEWAY}/ex/jira/{cloud_id.strip()}"


def atlassian_confluence_gateway_url(cloud_id: str) -> str:
    return f"{ATLASSIAN_API_GATEWAY}/ex/confluence/{cloud_id.strip()}"
