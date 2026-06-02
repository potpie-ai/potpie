"""Tests for Atlassian site discovery helpers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from adapters.inbound.cli.atlassian_auth import (
    _fetch_accessible_resources,
    _parse_accessible_resources,
    discover_sites_with_api_token,
    fetch_accessible_resources,
)


def test_parse_accessible_resources() -> None:
    data = [
        {"id": "cloud-1", "url": "https://team.atlassian.net", "name": "Team"},
        {"invalid": True},
    ]
    sites = _parse_accessible_resources(data)
    assert len(sites) == 1
    assert sites[0]["cloud_id"] == "cloud-1"
    assert sites[0]["site_url"] == "https://team.atlassian.net"


def test_fetch_accessible_resources_bearer_fallback() -> None:
    basic = MagicMock()
    basic.status_code = 401
    bearer = MagicMock()
    bearer.status_code = 200
    bearer.json.return_value = [
        {"id": "c2", "url": "https://other.atlassian.net", "name": "Other"},
    ]
    with patch("adapters.inbound.cli.atlassian_auth.httpx.Client") as mock_cls:
        client = MagicMock()
        mock_cls.return_value.__enter__.return_value = client
        client.get.side_effect = [basic, bearer]
        sites = _fetch_accessible_resources("user@example.com", "token")
    assert len(sites) == 1
    assert sites[0]["cloud_id"] == "c2"


def test_fetch_accessible_resources_basic_success() -> None:
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = [
        {"id": "c1", "url": "https://team.atlassian.net", "name": "Team"},
    ]
    with patch("adapters.inbound.cli.atlassian_auth.httpx.Client") as mock_cls:
        client = MagicMock()
        mock_cls.return_value.__enter__.return_value = client
        client.get.return_value = response
        sites = _fetch_accessible_resources("user@example.com", "token")
    assert len(sites) == 1


def test_discover_sites_with_api_token_filters_by_gateway() -> None:
    from adapters.inbound.cli.atlassian_auth import AtlassianVerifyResult

    candidates = [
        {
            "cloud_id": "c1",
            "site_url": "https://team.atlassian.net",
            "site_name": "Team",
        }
    ]
    with (
        patch(
            "adapters.inbound.cli.atlassian_auth.collect_site_candidates",
            return_value=candidates,
        ),
        patch(
            "adapters.inbound.cli.atlassian_auth.verify_gateway_product",
            return_value=AtlassianVerifyResult(ok=True, display_name="Ada"),
        ),
    ):
        sites = discover_sites_with_api_token("u@example.com", "tok", "jira")
    assert len(sites) == 1
    assert sites[0]["cloud_id"] == "c1"


def test_collect_login_site_candidates_merges_resources_and_email_hints() -> None:
    from adapters.inbound.cli.atlassian_auth import collect_login_site_candidates

    with (
        patch(
            "adapters.inbound.cli.atlassian_auth._fetch_accessible_resources",
            return_value=[
                {
                    "cloud_id": "c1",
                    "site_url": "https://team.atlassian.net",
                    "site_name": "Team",
                }
            ],
        ),
        patch(
            "adapters.inbound.cli.atlassian_auth.fetch_cloud_id_for_site",
            return_value="c1",
        ),
    ):
        sites = collect_login_site_candidates(
            "user@team.com",
            "token",
            include_email_hints=True,
        )
    assert sites
    assert any(s["source"] == "accessible-resources" for s in sites)


def test_fetch_accessible_resources_alias() -> None:
    with patch(
        "adapters.inbound.cli.atlassian_auth.discover_sites_with_api_token",
        return_value=[],
    ) as discover:
        fetch_accessible_resources("u@example.com", "tok")
    discover.assert_called_once_with("u@example.com", "tok", "jira")
