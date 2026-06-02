"""Tests for Atlassian CLI provider configuration."""

from __future__ import annotations

from adapters.inbound.cli.provider_config import (
    ATLASSIAN_API_GATEWAY,
    atlassian_confluence_gateway_url,
    atlassian_jira_gateway_url,
)


def test_atlassian_gateway_urls() -> None:
    assert atlassian_jira_gateway_url("abc") == (
        f"{ATLASSIAN_API_GATEWAY}/ex/jira/abc"
    )
    assert atlassian_confluence_gateway_url("abc") == (
        f"{ATLASSIAN_API_GATEWAY}/ex/confluence/abc"
    )
