from __future__ import annotations

import pytest

from observability.sinks.sentry_privacy import require_staging_smoke_environment


def test_staging_smoke_skips_by_default(monkeypatch) -> None:
    monkeypatch.delenv("RUN_SENTRY_STAGING_SMOKE", raising=False)

    with pytest.raises(pytest.skip.Exception):
        require_staging_smoke_environment()


def test_staging_smoke_refuses_non_staging(monkeypatch) -> None:
    monkeypatch.setenv("RUN_SENTRY_STAGING_SMOKE", "1")
    monkeypatch.setenv("SENTRY_DSN", "https://public@example.com/1")
    monkeypatch.setenv("SENTRY_ENVIRONMENT", "prod")
    monkeypatch.setenv("SENTRY_RELEASE", "potpie-cli@test")

    with pytest.raises(RuntimeError, match="SENTRY_ENVIRONMENT=staging"):
        require_staging_smoke_environment()
