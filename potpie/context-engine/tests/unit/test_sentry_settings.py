from __future__ import annotations

import pytest

from potpie_context_engine.bootstrap import standalone_env
from potpie_context_engine.bootstrap.sentry_settings import load_sentry_settings


def test_standalone_sentry_settings_preserve_environment_and_opt_outs() -> None:
    defaults = {
        "environment": "prod_oss",
        "sentry_dsn": "https://dist.example.invalid/1",
    }

    settings = load_sentry_settings({}, distribution_defaults=defaults)

    assert settings.enabled is True
    assert settings.dsn == "https://dist.example.invalid/1"
    assert settings.environment == "prod_oss"
    assert settings.release.startswith("potpie-cli@")

    disabled = load_sentry_settings(
        {"POTPIE_TELEMETRY_DISABLED": "1"},
        distribution_defaults=defaults,
    )

    assert disabled.enabled is False
    assert disabled.dsn == "https://dist.example.invalid/1"


def test_standalone_sentry_runtime_environment_overrides_packaged_defaults() -> None:
    settings = load_sentry_settings(
        {
            "POTPIE_ENVIRONMENT": "staging",
            "POTPIE_SENTRY_DSN": "https://runtime.example.invalid/1",
            "POTPIE_SENTRY_RELEASE": "potpie-cli@test",
            "POTPIE_SENTRY_DIST": "abc123",
        },
        distribution_defaults={
            "environment": "prod_oss",
            "sentry_dsn": "https://dist.example.invalid/1",
        },
    )

    assert settings.enabled is True
    assert settings.dsn == "https://runtime.example.invalid/1"
    assert settings.environment == "staging"
    assert settings.release == "potpie-cli@test"
    assert settings.dist == "abc123"


def test_dev_standalone_settings_preserve_trusted_dotenv_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'tmp'\n")
    (tmp_path / ".env").write_text(
        "POTPIE_ENVIRONMENT=prod\nPOTPIE_SENTRY_DSN=https://dotenv.example.invalid/1\n"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("POTPIE_ENVIRONMENT", raising=False)
    monkeypatch.delenv("POTPIE_SENTRY_DSN", raising=False)
    monkeypatch.setattr(standalone_env, "_loaded", False)

    settings = load_sentry_settings(distribution_defaults={})

    assert settings.environment == "dev"
    assert settings.dsn == "https://dotenv.example.invalid/1"
    assert settings.enabled is True
