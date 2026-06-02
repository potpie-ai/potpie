"""Atlassian API configuration for CLI integration auth."""

from __future__ import annotations

from typing import Literal

Provider = Literal["github", "atlassian", "jira", "confluence"]
AtlassianProduct = Literal["jira", "confluence"]

ATLASSIAN_API_TOKEN_PAGE = "https://id.atlassian.com/manage-profile/security/api-tokens"
ATLASSIAN_API_GATEWAY = "https://api.atlassian.com"
ATLASSIAN_ACCESSIBLE_RESOURCES_URL = (
    f"{ATLASSIAN_API_GATEWAY}/oauth/token/accessible-resources"
)


def atlassian_jira_gateway_url(cloud_id: str) -> str:
    return f"{ATLASSIAN_API_GATEWAY}/ex/jira/{cloud_id.strip()}"


def atlassian_confluence_gateway_url(cloud_id: str) -> str:
    return f"{ATLASSIAN_API_GATEWAY}/ex/confluence/{cloud_id.strip()}"
