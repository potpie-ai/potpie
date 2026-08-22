from __future__ import annotations

# ruff: noqa: S101 - pytest unit tests use assertions intentionally.

import pytest

from potpie.runtime import RuntimeEndpoint, generate_bearer_token


def test_tcp_endpoint_rejects_non_loopback_binding() -> None:
    with pytest.raises(ValueError, match="loopback-only"):
        RuntimeEndpoint(kind="tcp", address="0.0.0.0", port=7777)  # noqa: S104


def test_uds_endpoint_requires_absolute_path() -> None:
    with pytest.raises(ValueError, match="must be absolute"):
        RuntimeEndpoint(kind="uds", address="relative.sock")


def test_ipv6_loopback_endpoint_formats_valid_http_authority() -> None:
    endpoint = RuntimeEndpoint(kind="tcp", address="::1", port=7777)

    assert endpoint.display == "[::1]:7777"


def test_generated_bearer_tokens_are_unique_256_bit_secrets() -> None:
    first = generate_bearer_token()
    second = generate_bearer_token()

    assert first != second
    assert len(first.encode()) >= 32
    assert len(second.encode()) >= 32
