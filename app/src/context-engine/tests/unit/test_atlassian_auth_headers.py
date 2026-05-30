"""Auth header selection for Atlassian tenant vs gateway URLs."""

from __future__ import annotations

from adapters.inbound.cli.atlassian_read import _auth_header_variants


def test_tenant_base_uses_basic_only() -> None:
    headers = _auth_header_variants(
        "user@example.com",
        "secret-token",
        base="https://potpie-team.atlassian.net",
    )
    assert len(headers) == 1
    assert headers[0]["Authorization"].startswith("Basic ")


def test_gateway_base_allows_bearer_fallback() -> None:
    headers = _auth_header_variants(
        "user@example.com",
        "secret-token",
        base="https://api.atlassian.com/ex/jira/cloud-id",
    )
    assert len(headers) == 2
    assert headers[0]["Authorization"].startswith("Basic ")
    assert headers[1]["Authorization"].startswith("Bearer ")
