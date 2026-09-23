"""Linear webhook signature fail-closed behavior."""

from __future__ import annotations

import hashlib
import hmac

import pytest

from integrations.adapters.inbound.http.sources_router import (
    _linear_webhook_auth_failure,
)

pytestmark = pytest.mark.unit


def test_linear_webhook_rejects_missing_secret_without_opt_in() -> None:
    err = _linear_webhook_auth_failure(
        secret="",
        signature="abc",
        body=b"{}",
        allow_unsigned=False,
    )
    assert err is not None
    assert "LINEAR_WEBHOOK_SECRET" in err


def test_linear_webhook_opt_in_allows_unsigned_local_dev() -> None:
    err = _linear_webhook_auth_failure(
        secret="",
        signature=None,
        body=b"{}",
        allow_unsigned=True,
    )
    assert err is None


def test_linear_webhook_rejects_bad_signature() -> None:
    secret = "s3cret"
    body = b'{"action":"create"}'
    err = _linear_webhook_auth_failure(
        secret=secret,
        signature="deadbeef",
        body=body,
        allow_unsigned=False,
    )
    assert err == "invalid_signature"
    expected = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    ok = _linear_webhook_auth_failure(
        secret=secret,
        signature=expected,
        body=body,
        allow_unsigned=False,
    )
    assert ok is None
