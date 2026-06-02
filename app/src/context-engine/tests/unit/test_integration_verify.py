"""Tests for Atlassian integration verify probes."""

from __future__ import annotations

from unittest.mock import patch

from adapters.inbound.cli.atlassian_auth import AtlassianAuthErrorKind, AtlassianVerifyResult
from adapters.inbound.cli.integration_verify import (
    _verify_atlassian_product,
    _verify_message_for_kind,
    verify_integration_access,
)


def test_verify_message_for_kind() -> None:
    assert "invalid" in _verify_message_for_kind(
        "jira", AtlassianAuthErrorKind.INVALID_CREDENTIALS
    )
    assert "scope" in _verify_message_for_kind(
        "confluence", AtlassianAuthErrorKind.INSUFFICIENT_SCOPES
    )


def test_verify_integration_access_unknown_provider() -> None:
    ok, message = verify_integration_access("github", {})  # type: ignore[arg-type]
    assert ok is False
    assert "unknown provider" in message


def test_verify_atlassian_product_success(monkeypatch) -> None:
    monkeypatch.setattr(
        "adapters.inbound.cli.integration_verify.fetch_cloud_id_for_site",
        lambda _url: "cloud-1",
    )
    monkeypatch.setattr(
        "adapters.inbound.cli.integration_verify.verify_gateway_product",
        lambda *args, **kwargs: AtlassianVerifyResult(
            ok=True,
            display_name="Ada",
            error_kind=None,
        ),
    )
    ok, message = _verify_atlassian_product(
        "jira",
        {
            "email": "ada@example.com",
            "api_token": "tok",
            "site_url": "https://team.atlassian.net",
            "site_name": "team",
        },
    )
    assert ok is True
    assert "Ada" in message


def test_verify_integration_access_jira(monkeypatch) -> None:
    monkeypatch.setattr(
        "adapters.inbound.cli.integration_verify._verify_atlassian_product",
        lambda _p, _c: (True, "ok (Ada @ team)"),
    )
    ok, message = verify_integration_access(
        "jira",
        {"email": "a@example.com", "api_token": "t", "site_url": "https://x.net"},
    )
    assert ok is True
    assert "Ada" in message
