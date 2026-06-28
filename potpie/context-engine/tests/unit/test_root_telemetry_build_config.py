from __future__ import annotations

import telemetry_build_config_values as build_config


def test_sentry_build_config_defaults() -> None:
    values = build_config.sentry_config_values({})

    assert values == {
        "POTPIE_TELEMETRY_DISABLED": "0",
        "POTPIE_SENTRY_ENABLED": "1",
        "POTPIE_SENTRY_DSN": "",
        "POTPIE_SENTRY_ENVIRONMENT": "prod_oss",
        "POTPIE_SENTRY_RELEASE": "",
        "POTPIE_SENTRY_DIST": "",
    }


def test_sentry_build_config_reads_runtime_defaults() -> None:
    values = build_config.sentry_config_values(
        {
            "POTPIE_TELEMETRY_DISABLED": "1",
            "POTPIE_SENTRY_ENABLED": "0",
            "POTPIE_SENTRY_DSN": "https://sentry.example/1",
            "POTPIE_SENTRY_ENVIRONMENT": "staging",
            "POTPIE_SENTRY_RELEASE": "potpie@test",
            "GITHUB_SHA": "abc123",
        }
    )

    assert values == {
        "POTPIE_TELEMETRY_DISABLED": "1",
        "POTPIE_SENTRY_ENABLED": "0",
        "POTPIE_SENTRY_DSN": "https://sentry.example/1",
        "POTPIE_SENTRY_ENVIRONMENT": "staging",
        "POTPIE_SENTRY_RELEASE": "potpie@test",
        "POTPIE_SENTRY_DIST": "abc123",
    }


def test_posthog_build_config_defaults() -> None:
    values = build_config.posthog_config_values({})

    assert values == {
        "POTPIE_TELEMETRY_DISABLED": "0",
        "POTPIE_POSTHOG_ENABLED": "1",
        "POTPIE_PRODUCT_ANALYTICS_ENABLED": "1",
        "POTPIE_POSTHOG_API_KEY": "",
        "POTPIE_POSTHOG_HOST": "https://us.i.posthog.com",
    }


def test_posthog_build_config_reads_cli_defaults() -> None:
    values = build_config.posthog_config_values(
        {
            "POTPIE_TELEMETRY_DISABLED": "1",
            "POTPIE_POSTHOG_ENABLED": "0",
            "POTPIE_PRODUCT_ANALYTICS_ENABLED": "0",
            "POTPIE_POSTHOG_API_KEY": "ph_test",
            "POTPIE_POSTHOG_HOST": "https://eu.i.posthog.com",
        }
    )

    assert values == {
        "POTPIE_TELEMETRY_DISABLED": "1",
        "POTPIE_POSTHOG_ENABLED": "0",
        "POTPIE_PRODUCT_ANALYTICS_ENABLED": "0",
        "POTPIE_POSTHOG_API_KEY": "ph_test",
        "POTPIE_POSTHOG_HOST": "https://eu.i.posthog.com",
    }


def test_write_python_constants(tmp_path) -> None:
    out = tmp_path / "runtime" / "telemetry" / "_build_config.py"

    build_config.write_python_constants(out, {"A": "x", "B": ""})

    assert out.read_text(encoding="utf-8").splitlines() == [
        "# Auto-generated at wheel build time - do not edit manually.",
        "# Override any value at runtime by setting the corresponding environment variable.",
        "A = 'x'",
        "B = ''",
    ]
