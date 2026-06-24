from __future__ import annotations

import build_config_values


def test_telemetry_build_config_defaults_dist_to_github_sha() -> None:
    values = build_config_values.telemetry_config_values({"GITHUB_SHA": "abc123"})

    assert values == {
        "POTPIE_TELEMETRY_DISABLED": "0",
        "POTPIE_SENTRY_ENABLED": "1",
        "POTPIE_SENTRY_DSN": "",
        "POTPIE_SENTRY_ENVIRONMENT": "production",
        "POTPIE_SENTRY_RELEASE": "",
        "POTPIE_SENTRY_DIST": "abc123",
        "POTPIE_POSTHOG_ENABLED": "1",
        "POTPIE_PRODUCT_ANALYTICS_ENABLED": "1",
        "POTPIE_POSTHOG_API_KEY": "",
        "POTPIE_POSTHOG_HOST": "https://us.i.posthog.com",
    }


def test_telemetry_build_config_explicit_dist_wins_over_github_sha(
) -> None:
    values = build_config_values.telemetry_config_values(
        {
            "GITHUB_SHA": "abc123",
            "POTPIE_SENTRY_DIST": "explicit-dist",
        }
    )

    assert values["POTPIE_SENTRY_DIST"] == "explicit-dist"


def test_write_python_constants(tmp_path) -> None:
    out = tmp_path / "_build_config.py"

    build_config_values.write_python_constants(out, {"A": "x", "B": ""})

    assert out.read_text(encoding="utf-8").splitlines() == [
        "# Auto-generated at wheel build time - do not edit manually.",
        "# Override any value at runtime by setting the corresponding environment variable.",
        "A = 'x'",
        "B = ''",
    ]
