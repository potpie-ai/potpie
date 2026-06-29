from __future__ import annotations

import pytest

from adapters.inbound.cli.telemetry.settings import (
    load_sentry_settings as load_cli_sentry_settings,
    telemetry_environment as cli_telemetry_environment,
)
from bootstrap import runtime_settings
from bootstrap import sentry_settings as shared_sentry_settings

_SENTRY_ENV_NAMES = (
    "POTPIE_ENVIRONMENT",
    "POTPIE_TELEMETRY_DISABLED",
    "POTPIE_SENTRY_ENABLED",
    "POTPIE_SENTRY_DSN",
    "POTPIE_SENTRY_RELEASE",
    "POTPIE_SENTRY_DIST",
)


@pytest.fixture(autouse=True)
def _clear_sentry_config(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _SENTRY_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("POTPIE_ENVIRONMENT", "test")
    monkeypatch.setattr(runtime_settings, "load_distribution_defaults", lambda: {})
    monkeypatch.setattr(shared_sentry_settings, "build_git_sha", lambda: None)


def test_sentry_settings_disabled_without_dsn() -> None:
    settings = load_cli_sentry_settings()

    assert settings.enabled is False
    assert settings.dsn is None
    assert settings.environment == "test"


def test_potpie_sentry_dsn_enables_sentry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("POTPIE_SENTRY_DSN", "https://public@example.invalid/1")

    settings = load_cli_sentry_settings()

    assert settings.enabled is True
    assert settings.dsn == "https://public@example.invalid/1"


def test_potpie_environment_is_used_for_sentry_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("POTPIE_ENVIRONMENT", "staging")
    monkeypatch.setenv("POTPIE_SENTRY_DSN", "https://public@example.invalid/1")

    settings = load_cli_sentry_settings()

    assert settings.environment == "staging"
    assert cli_telemetry_environment() == "staging"


def test_sentry_release_comes_from_env_or_package_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("POTPIE_SENTRY_DSN", "https://public@example.invalid/1")
    monkeypatch.setenv("POTPIE_SENTRY_RELEASE", "potpie-cli@test")

    settings = load_cli_sentry_settings()

    assert settings.release == "potpie-cli@test"

    monkeypatch.delenv("POTPIE_SENTRY_RELEASE")
    assert load_cli_sentry_settings().release.startswith("potpie-cli@")


def test_sentry_dist_comes_from_env_or_build_info(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("POTPIE_SENTRY_DSN", "https://public@example.invalid/1")
    monkeypatch.setattr(shared_sentry_settings, "build_git_sha", lambda: "abc123")

    settings = load_cli_sentry_settings()

    assert settings.dist == "abc123"

    monkeypatch.setenv("POTPIE_SENTRY_DIST", "explicit-dist")
    assert load_cli_sentry_settings().dist == "explicit-dist"


def test_sentry_uses_distribution_defaults_without_runtime_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("POTPIE_ENVIRONMENT", raising=False)
    monkeypatch.setattr(
        runtime_settings,
        "load_distribution_defaults",
        lambda: {
            "environment": "prod_oss",
            "sentry_dsn": "https://dist@example.invalid/1",
        },
    )

    settings = shared_sentry_settings.load_sentry_settings()

    assert settings.enabled is True
    assert settings.dsn == "https://dist@example.invalid/1"
    assert settings.environment == "prod_oss"


def test_sentry_opt_outs_disable_canonical_dsn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("POTPIE_SENTRY_DSN", "https://public@example.invalid/1")
    monkeypatch.setenv("POTPIE_SENTRY_ENABLED", "0")

    assert load_cli_sentry_settings().enabled is False

    monkeypatch.setenv("POTPIE_SENTRY_ENABLED", "1")
    monkeypatch.setenv("POTPIE_TELEMETRY_DISABLED", "1")

    assert load_cli_sentry_settings().enabled is False
