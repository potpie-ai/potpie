from __future__ import annotations

from adapters.inbound.cli.telemetry.settings import load_sentry_settings


def test_sentry_settings_disabled_without_dsn(monkeypatch) -> None:
    monkeypatch.delenv("POTPIE_SENTRY_DSN", raising=False)
    monkeypatch.delenv("SENTRY_DSN", raising=False)

    settings = load_sentry_settings()

    assert settings.enabled is False
    assert settings.dsn is None


def test_sentry_settings_respects_global_kill_switch(monkeypatch) -> None:
    monkeypatch.setenv("POTPIE_SENTRY_DSN", "https://public@example.invalid/1")
    monkeypatch.setenv("POTPIE_TELEMETRY_DISABLED", "1")

    settings = load_sentry_settings()

    assert settings.enabled is False
    assert settings.dsn == "https://public@example.invalid/1"


def test_sentry_settings_ignores_blank_global_kill_switch(monkeypatch) -> None:
    monkeypatch.setenv("POTPIE_SENTRY_DSN", "https://public@example.invalid/1")
    monkeypatch.setenv("POTPIE_TELEMETRY_DISABLED", "   ")

    settings = load_sentry_settings()

    assert settings.enabled is True
    assert settings.dsn == "https://public@example.invalid/1"


def test_sentry_settings_respects_sentry_kill_switch(monkeypatch) -> None:
    monkeypatch.setenv("POTPIE_SENTRY_DSN", "https://public@example.invalid/1")
    monkeypatch.setenv("POTPIE_SENTRY_ENABLED", "0")

    settings = load_sentry_settings()

    assert settings.enabled is False


def test_potpie_sentry_env_wins_over_generic_sentry_env(monkeypatch) -> None:
    monkeypatch.setenv("POTPIE_SENTRY_DSN", "https://potpie@example.invalid/1")
    monkeypatch.setenv("SENTRY_DSN", "https://generic@example.invalid/1")
    monkeypatch.setenv("POTPIE_SENTRY_ENVIRONMENT", "staging")
    monkeypatch.setenv("SENTRY_ENVIRONMENT", "prod")
    monkeypatch.setenv("POTPIE_SENTRY_RELEASE", "potpie-cli@test")
    monkeypatch.setenv("SENTRY_RELEASE", "generic@test")
    monkeypatch.setenv("POTPIE_SENTRY_DIST", "cli-dist")
    monkeypatch.setenv("SENTRY_DIST", "generic-dist")

    settings = load_sentry_settings()

    assert settings.enabled is True
    assert settings.dsn == "https://potpie@example.invalid/1"
    assert settings.environment == "staging"
    assert settings.release == "potpie-cli@test"
    assert settings.dist == "cli-dist"


def test_sentry_settings_default_release_is_potpie_cli(monkeypatch) -> None:
    monkeypatch.setenv("SENTRY_DSN", "https://public@example.invalid/1")
    monkeypatch.delenv("POTPIE_SENTRY_RELEASE", raising=False)
    monkeypatch.delenv("SENTRY_RELEASE", raising=False)

    settings = load_sentry_settings()

    assert settings.release.startswith("potpie-cli@")
