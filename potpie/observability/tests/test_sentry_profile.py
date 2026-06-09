# ruff: noqa: S101
from __future__ import annotations

from observability import profiles


def test_cli_profile_is_disabled_without_dsn(monkeypatch) -> None:
    monkeypatch.delenv("SENTRY_DSN", raising=False)
    monkeypatch.delenv("POTPIE_SENTRY_ENABLED", raising=False)

    cfg = profiles.cli()

    assert cfg.sentry.enabled is False
    assert "sentry" not in cfg.sinks


def test_cli_profile_kill_switch_disables_sentry(monkeypatch) -> None:
    monkeypatch.setenv("SENTRY_DSN", "https://public@example.com/1")
    monkeypatch.setenv("POTPIE_SENTRY_ENABLED", "0")

    cfg = profiles.cli()

    assert cfg.sentry.enabled is False
    assert "sentry" not in cfg.sinks


def test_cli_profile_uses_release_override(monkeypatch) -> None:
    monkeypatch.setenv("SENTRY_DSN", "https://public@example.com/1")
    monkeypatch.setenv("SENTRY_RELEASE", "potpie-cli@test")
    monkeypatch.setenv("SENTRY_ENVIRONMENT", "staging")
    monkeypatch.delenv("POTPIE_SENTRY_ENABLED", raising=False)

    cfg = profiles.cli()

    assert cfg.sentry.enabled is True
    assert cfg.sentry.release == "potpie-cli@test"
    assert cfg.sentry.environment == "staging"
    assert "sentry" in cfg.sinks


def test_cli_profile_defaults_local_environment_to_dev(monkeypatch) -> None:
    monkeypatch.setenv("SENTRY_DSN", "https://public@example.com/1")
    monkeypatch.delenv("SENTRY_ENVIRONMENT", raising=False)
    monkeypatch.delenv("ENV", raising=False)
    monkeypatch.delenv("POTPIE_SENTRY_ENABLED", raising=False)

    cfg = profiles.cli()

    assert cfg.env == "dev"
    assert cfg.sentry.environment == "dev"
    assert cfg.sentry.release == "potpie-cli@0.1.0"
