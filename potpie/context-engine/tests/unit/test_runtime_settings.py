from __future__ import annotations

import os

import pytest

from bootstrap import env_bootstrap
from bootstrap.runtime_settings import (
    RuntimeSettings,
    ensure_runtime_environment_loaded,
    load_runtime_settings,
    project_child_environment,
    resolve_bootstrap_environment,
)

_CONFIG_ENV_NAMES = (
    "POTPIE_ENVIRONMENT",
    "POTPIE_API_URL",
    "POTPIE_UI_URL",
    "POTPIE_API_KEY",
    "POTPIE_TELEMETRY_DISABLED",
    "POTPIE_SENTRY_ENABLED",
    "POTPIE_SENTRY_DSN",
    "POTPIE_PRODUCT_ANALYTICS_ENABLED",
    "POTPIE_POSTHOG_API_KEY",
    "POTPIE_POSTHOG_HOST",
    "LINEAR_CLIENT_ID",
    "POTPIE_GITHUB_CLIENT_ID",
    "CONTEXT_ENGINE_GITHUB_TOKEN",
    "GITHUB_WEBHOOK_SECRET",
)


@pytest.fixture(autouse=True)
def _clear_runtime_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _CONFIG_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(env_bootstrap, "_loaded", False)
    monkeypatch.setattr(
        "bootstrap.runtime_settings.load_distribution_defaults",
        lambda: {},
    )


def test_process_env_wins_over_distribution_defaults() -> None:
    settings = load_runtime_settings(
        {
            "POTPIE_ENVIRONMENT": "staging",
            "POTPIE_SENTRY_DSN": "https://runtime.example.invalid/1",
            "POTPIE_POSTHOG_API_KEY": "phc_runtime",
        },
        distribution_defaults={
            "environment": "prod_oss",
            "sentry_dsn": "https://dist.example.invalid/1",
            "posthog_api_key": "phc_dist",
        },
    )

    assert settings.environment == "staging"
    assert settings.sentry_dsn == "https://runtime.example.invalid/1"
    assert settings.posthog_api_key == "phc_runtime"


def test_distribution_defaults_win_over_code_defaults() -> None:
    settings = load_runtime_settings(
        {},
        distribution_defaults={
            "environment": "prod_oss",
            "posthog_host": "https://posthog.dist.invalid",
            "linear_client_id": "linear-client",
            "github_client_id": "github-client",
        },
    )

    assert settings.environment == "prod_oss"
    assert settings.posthog_host == "https://posthog.dist.invalid"
    assert settings.linear_client_id == "linear-client"
    assert settings.github_client_id == "github-client"


def test_source_checkout_default_environment_is_dev() -> None:
    settings = load_runtime_settings({}, distribution_defaults={})

    assert settings.environment == "dev"


def test_installed_distribution_default_environment_is_prod_oss() -> None:
    assert resolve_bootstrap_environment({}, {"environment": "prod_oss"}) == "prod_oss"


def test_dev_environment_reads_dotenv_for_missing_keys(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'tmp'\n")
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "POTPIE_ENVIRONMENT=prod",
                "POTPIE_SENTRY_DSN=https://dotenv.example.invalid/1",
                "POTPIE_API_URL=https://api.dotenv.invalid",
            ]
        )
    )
    monkeypatch.chdir(tmp_path)

    settings = load_runtime_settings()

    assert settings.environment == "dev"
    assert settings.sentry_dsn == "https://dotenv.example.invalid/1"
    assert settings.potpie_api_url == "https://api.dotenv.invalid"


def test_non_dev_environment_does_not_read_dotenv(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'tmp'\n")
    (tmp_path / ".env").write_text("POTPIE_SENTRY_DSN=https://dotenv.invalid/1\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("POTPIE_ENVIRONMENT", "staging")

    settings = load_runtime_settings()

    assert settings.environment == "staging"
    assert settings.sentry_dsn is None


def test_dotenv_does_not_override_existing_process_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'tmp'\n")
    (tmp_path / ".env").write_text(
        "POTPIE_SENTRY_DSN=https://dotenv.example.invalid/1\n"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("POTPIE_SENTRY_DSN", "https://runtime.example.invalid/1")

    settings = load_runtime_settings()

    assert settings.sentry_dsn == "https://runtime.example.invalid/1"


def test_dotenv_cannot_set_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'tmp'\n")
    (tmp_path / ".env").write_text("POTPIE_ENVIRONMENT=prod\n")
    monkeypatch.chdir(tmp_path)

    ensure_runtime_environment_loaded({})

    assert os.getenv("POTPIE_ENVIRONMENT") is None


def test_direct_dotenv_loader_cannot_set_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'tmp'\n")
    (tmp_path / ".env").write_text("POTPIE_ENVIRONMENT=prod\n")
    monkeypatch.chdir(tmp_path)

    env_bootstrap.load_cli_env()

    assert os.getenv("POTPIE_ENVIRONMENT") is None


def test_blank_env_values_are_treated_as_missing() -> None:
    settings = load_runtime_settings(
        {"POTPIE_SENTRY_DSN": "  "},
        distribution_defaults={"sentry_dsn": "https://dist.example.invalid/1"},
    )

    assert settings.sentry_dsn == "https://dist.example.invalid/1"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1", True),
        ("true", True),
        ("yes", True),
        ("on", True),
        ("0", False),
        ("false", False),
        ("no", False),
        ("off", False),
    ],
)
def test_boolean_flags_parse_canonical_values(raw: str, expected: bool) -> None:
    settings = load_runtime_settings(
        {"POTPIE_PRODUCT_ANALYTICS_ENABLED": raw},
        distribution_defaults={},
    )

    assert settings.product_analytics_enabled is expected


def test_api_and_ui_urls_are_separate() -> None:
    settings = load_runtime_settings(
        {
            "POTPIE_API_URL": "https://api.example.invalid/",
            "POTPIE_UI_URL": "https://ui.example.invalid/",
        },
        distribution_defaults={},
    )

    assert settings.potpie_api_url == "https://api.example.invalid"
    assert settings.potpie_ui_url == "https://ui.example.invalid"


def test_project_child_environment_emits_canonical_values_and_drops_aliases() -> None:
    settings = RuntimeSettings(
        environment="staging",
        potpie_api_url="https://api.example.invalid",
        potpie_ui_url="https://ui.example.invalid",
        potpie_api_key="api-key",
        telemetry_disabled=False,
        sentry_enabled=True,
        sentry_dsn="https://sentry.example.invalid/1",
        product_analytics_enabled=True,
        posthog_api_key="phc_public",
        posthog_host="https://posthog.example.invalid",
        linear_client_id="linear-client",
        github_client_id="github-client",
        context_engine_github_token="github-token",
        github_webhook_secret="webhook-secret",
    )

    child = project_child_environment(
        settings,
        {
            "PATH": "/bin",
            "SENTRY_DSN": "legacy",
            "POTPIE_BASE_URL": "legacy",
            "GITHUB_TOKEN": "legacy",
        },
        overrides={
            "CONTEXT_ENGINE_HOME": "/tmp/potpie",
            "POTPIE_API_URL": "override",
            "SENTRY_DSN": "override-legacy",
        },
    )

    assert child["PATH"] == "/bin"
    assert child["POTPIE_ENVIRONMENT"] == "staging"
    assert child["POTPIE_API_URL"] == "override"
    assert child["POTPIE_UI_URL"] == "https://ui.example.invalid"
    assert child["POTPIE_SENTRY_DSN"] == "https://sentry.example.invalid/1"
    assert child["POTPIE_POSTHOG_API_KEY"] == "phc_public"
    assert child["LINEAR_CLIENT_ID"] == "linear-client"
    assert child["POTPIE_GITHUB_CLIENT_ID"] == "github-client"
    assert child["CONTEXT_ENGINE_GITHUB_TOKEN"] == "github-token"
    assert child["GITHUB_WEBHOOK_SECRET"] == "webhook-secret"
    assert child["CONTEXT_ENGINE_HOME"] == "/tmp/potpie"
    assert "SENTRY_DSN" not in child
    assert "POTPIE_BASE_URL" not in child
    assert "GITHUB_TOKEN" not in child
