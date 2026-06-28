from __future__ import annotations

import pytest

import build_config_values


def test_distribution_defaults_use_internal_field_names() -> None:
    values = build_config_values.distribution_default_values(
        {
            "POTPIE_ENVIRONMENT": "prod_oss",
            "POTPIE_SENTRY_DSN": "https://sentry.example.invalid/1",
            "POTPIE_POSTHOG_API_KEY": "phc_public",
            "POTPIE_POSTHOG_HOST": "https://posthog.invalid",
            "LINEAR_CLIENT_ID": "linear-client",
            "POTPIE_GITHUB_CLIENT_ID": "github-client",
        }
    )

    assert values == {
        "environment": "prod_oss",
        "sentry_dsn": "https://sentry.example.invalid/1",
        "posthog_api_key": "phc_public",
        "posthog_host": "https://posthog.invalid",
        "linear_client_id": "linear-client",
        "github_client_id": "github-client",
    }


def test_distribution_defaults_have_safe_public_defaults() -> None:
    values = build_config_values.distribution_default_values({})

    assert values["environment"] == "prod_oss"
    assert values["posthog_host"] == "https://us.i.posthog.com"
    assert values["sentry_dsn"] == ""
    assert values["posthog_api_key"] == ""


def test_build_info_defaults_git_sha_to_github_sha() -> None:
    values = build_config_values.build_info_values(
        {"GITHUB_SHA": "abc123", "POTPIE_BUILD_TIME": "2026-06-28T00:00:00Z"}
    )

    assert values == {"GIT_SHA": "abc123", "BUILD_TIME": "2026-06-28T00:00:00Z"}


def test_write_distribution_mapping_does_not_emit_env_var_constants(tmp_path) -> None:
    out = tmp_path / "_distribution_defaults.py"

    build_config_values.write_python_mapping(
        out,
        "DISTRIBUTION_DEFAULTS",
        {"environment": "prod_oss", "sentry_dsn": ""},
    )

    text = out.read_text(encoding="utf-8")
    assert "DISTRIBUTION_DEFAULTS = {" in text
    assert "'environment': 'prod_oss'" in text
    assert "POTPIE_SENTRY_DSN =" not in text
    assert "LINEAR_CLIENT_ID =" not in text


def test_release_validation_fails_when_required_defaults_are_missing() -> None:
    with pytest.raises(RuntimeError, match="sentry_dsn"):
        build_config_values.validate_distribution_defaults(
            {
                "environment": "prod_oss",
                "sentry_dsn": "",
                "posthog_api_key": "phc_public",
                "posthog_host": "https://us.i.posthog.com",
                "linear_client_id": "linear-client",
                "github_client_id": "github-client",
            }
        )


def test_release_validation_passes_when_required_defaults_are_present() -> None:
    build_config_values.validate_distribution_defaults(
        {
            "environment": "prod_oss",
            "sentry_dsn": "https://sentry.example.invalid/1",
            "posthog_api_key": "phc_public",
            "posthog_host": "https://us.i.posthog.com",
            "linear_client_id": "linear-client",
            "github_client_id": "github-client",
        }
    )


def test_local_build_without_validation_does_not_fail() -> None:
    values = build_config_values.distribution_default_values({})

    assert build_config_values.should_validate_distribution_defaults({}) is False
    assert values["sentry_dsn"] == ""


def test_explicit_validation_flag_is_required() -> None:
    assert (
        build_config_values.should_validate_distribution_defaults(
            {"POTPIE_VALIDATE_DISTRIBUTION_DEFAULTS": "1"}
        )
        is True
    )
