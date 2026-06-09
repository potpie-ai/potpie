# ruff: noqa: S101
from __future__ import annotations

from observability.sinks.sentry_privacy import SentryRateLimiter


def test_rate_limiter_allows_new_bucket_values() -> None:
    limiter = SentryRateLimiter(max_events_per_minute=1)
    base = {
        "tags": {"release": "potpie-cli@1", "error.code": "unexpected_cli_error"},
        "contexts": {"telemetry": {"anonymous_install_id": "install-1"}},
    }

    assert limiter.allow(base) is True
    assert limiter.allow(base) is False
    assert (
        limiter.allow({**base, "tags": {**base["tags"], "error.code": "other"}}) is True
    )
    assert (
        limiter.allow(
            {
                **base,
                "contexts": {"telemetry": {"anonymous_install_id": "install-2"}},
            }
        )
        is True
    )


def test_rate_limiter_resets_after_minute() -> None:
    now = 10.0

    def clock() -> float:
        return now

    limiter = SentryRateLimiter(max_events_per_minute=1, clock=clock)
    event = {
        "tags": {"release": "potpie-cli@1", "error.code": "unexpected_cli_error"},
        "contexts": {"telemetry": {"anonymous_install_id": "install-1"}},
    }

    assert limiter.allow(event) is True
    assert limiter.allow(event) is False
    now = 71.0
    assert limiter.allow(event) is True
