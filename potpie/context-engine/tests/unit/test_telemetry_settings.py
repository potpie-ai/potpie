from __future__ import annotations

import pytest

from adapters.inbound.cli.telemetry import _build_defaults as build_defaults
from bootstrap import sentry_settings as shared_sentry_settings
from adapters.inbound.cli.telemetry.settings import (
    load_sentry_settings as load_cli_sentry_settings,
    telemetry_environment as cli_telemetry_environment,
)

_SENTRY_ENV_NAMES = (
    "POTPIE_TELEMETRY_DISABLED",
    "POTPIE_SENTRY_ENABLED",
    "POTPIE_SENTRY_DSN",
    "SENTRY_DSN",
    "POTPIE_SENTRY_ENVIRONMENT",
    "SENTRY_ENVIRONMENT",
    "POTPIE_SENTRY_RELEASE",
    "SENTRY_RELEASE",
    "POTPIE_SENTRY_DIST",
    "SENTRY_DIST",
)

_BUILD_DEFAULT_NAMES = (
    "POTPIE_TELEMETRY_DISABLED",
    "POTPIE_SENTRY_ENABLED",
    "POTPIE_SENTRY_DSN",
    "POTPIE_SENTRY_ENVIRONMENT",
    "POTPIE_SENTRY_RELEASE",
    "POTPIE_SENTRY_DIST",
)


@pytest.fixture(autouse=True)
def _clear_sentry_config(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _SENTRY_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    for name in _BUILD_DEFAULT_NAMES:
        monkeypatch.setattr(build_defaults, name, "")


def test_sentry_settings_disabled_without_dsn(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = load_cli_sentry_settings()

    assert settings.enabled is False
    assert settings.dsn is None


def test_sentry_settings_respects_global_kill_switch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("POTPIE_SENTRY_DSN", "https://public@example.invalid/1")
    monkeypatch.setenv("POTPIE_TELEMETRY_DISABLED", "1")

    settings = load_cli_sentry_settings()

    assert settings.enabled is False
    assert settings.dsn == "https://public@example.invalid/1"


def test_sentry_settings_ignores_blank_global_kill_switch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("POTPIE_SENTRY_DSN", "https://public@example.invalid/1")
    monkeypatch.setenv("POTPIE_TELEMETRY_DISABLED", "   ")
    monkeypatch.setenv("POTPIE_SENTRY_ENABLED", "")
    monkeypatch.setenv("SENTRY_ENABLED", "")

    settings = load_cli_sentry_settings()

    assert settings.enabled is True
    assert settings.dsn == "https://public@example.invalid/1"


def test_sentry_settings_respects_sentry_kill_switch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("POTPIE_SENTRY_DSN", "https://public@example.invalid/1")
    monkeypatch.setenv("POTPIE_SENTRY_ENABLED", "0")

    settings = load_cli_sentry_settings()

    assert settings.enabled is False


def test_potpie_sentry_env_wins_over_generic_sentry_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("POTPIE_SENTRY_DSN", "https://potpie@example.invalid/1")
    monkeypatch.setenv("SENTRY_DSN", "https://generic@example.invalid/1")
    monkeypatch.setenv("POTPIE_SENTRY_ENVIRONMENT", "staging")
    monkeypatch.setenv("SENTRY_ENVIRONMENT", "prod")
    monkeypatch.setenv("POTPIE_SENTRY_RELEASE", "potpie-cli@test")
    monkeypatch.setenv("SENTRY_RELEASE", "generic@test")
    monkeypatch.setenv("POTPIE_SENTRY_DIST", "cli-dist")
    monkeypatch.setenv("SENTRY_DIST", "generic-dist")

    settings = load_cli_sentry_settings()

    assert settings.enabled is True
    assert settings.dsn == "https://potpie@example.invalid/1"
    assert settings.environment == "staging"
    assert cli_telemetry_environment() == "staging"
    assert settings.release == "potpie-cli@test"
    assert settings.dist == "cli-dist"


def test_sentry_settings_default_release_is_potpie_cli(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SENTRY_DSN", "https://public@example.invalid/1")

    settings = load_cli_sentry_settings()

    assert settings.release.startswith("potpie-cli@")


def test_sentry_settings_use_baked_config_without_runtime_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        build_defaults, "POTPIE_SENTRY_DSN", "https://baked@example.invalid/1"
    )
    monkeypatch.setattr(build_defaults, "POTPIE_SENTRY_ENABLED", "1")
    monkeypatch.setattr(build_defaults, "POTPIE_TELEMETRY_DISABLED", "0")
    monkeypatch.setattr(build_defaults, "POTPIE_SENTRY_ENVIRONMENT", "production")
    monkeypatch.setattr(build_defaults, "POTPIE_SENTRY_RELEASE", "potpie-cli@baked")
    monkeypatch.setattr(build_defaults, "POTPIE_SENTRY_DIST", "sha-baked")

    settings = load_cli_sentry_settings()

    assert settings.enabled is True
    assert settings.dsn == "https://baked@example.invalid/1"
    assert settings.environment == "production"
    assert cli_telemetry_environment() == "production"
    assert settings.release == "potpie-cli@baked"
    assert settings.dist == "sha-baked"


def test_sentry_runtime_env_overrides_baked_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        build_defaults, "POTPIE_SENTRY_DSN", "https://baked@example.invalid/1"
    )
    monkeypatch.setattr(build_defaults, "POTPIE_SENTRY_ENVIRONMENT", "production")
    monkeypatch.setattr(build_defaults, "POTPIE_SENTRY_RELEASE", "potpie-cli@baked")
    monkeypatch.setattr(build_defaults, "POTPIE_SENTRY_DIST", "sha-baked")
    monkeypatch.setenv("POTPIE_SENTRY_DSN", "https://runtime@example.invalid/1")
    monkeypatch.setenv("POTPIE_SENTRY_ENVIRONMENT", "staging")
    monkeypatch.setenv("POTPIE_SENTRY_RELEASE", "potpie-cli@runtime")
    monkeypatch.setenv("POTPIE_SENTRY_DIST", "sha-runtime")

    settings = load_cli_sentry_settings()

    assert settings.enabled is True
    assert settings.dsn == "https://runtime@example.invalid/1"
    assert settings.environment == "staging"
    assert settings.release == "potpie-cli@runtime"
    assert settings.dist == "sha-runtime"


def test_sentry_runtime_opt_out_overrides_baked_enablement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        build_defaults, "POTPIE_SENTRY_DSN", "https://baked@example.invalid/1"
    )
    monkeypatch.setattr(build_defaults, "POTPIE_SENTRY_ENABLED", "1")
    monkeypatch.setenv("POTPIE_SENTRY_ENABLED", "0")

    settings = load_cli_sentry_settings()

    assert settings.enabled is False


def test_sentry_runtime_global_opt_out_overrides_baked_enablement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        build_defaults, "POTPIE_SENTRY_DSN", "https://baked@example.invalid/1"
    )
    monkeypatch.setattr(build_defaults, "POTPIE_TELEMETRY_DISABLED", "0")
    monkeypatch.setenv("POTPIE_TELEMETRY_DISABLED", "1")

    settings = load_cli_sentry_settings()

    assert settings.enabled is False


def test_shared_sentry_settings_ignore_cli_baked_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        build_defaults, "POTPIE_SENTRY_DSN", "https://baked@example.invalid/1"
    )
    monkeypatch.setattr(build_defaults, "POTPIE_SENTRY_ENVIRONMENT", "production")

    settings = shared_sentry_settings.load_sentry_settings()

    assert settings.enabled is False
    assert settings.dsn is None
    assert settings.environment == "dev"
