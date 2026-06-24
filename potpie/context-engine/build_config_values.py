from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

OAUTH_OUT = Path("adapters/outbound/cli_auth/_build_config.py")
TELEMETRY_OUT = Path("adapters/inbound/cli/telemetry/_build_config.py")

HEADER = """\
# Auto-generated at wheel build time - do not edit manually.
# Override any value at runtime by setting the corresponding environment variable.
"""

TELEMETRY_NAMES = (
    "POTPIE_TELEMETRY_DISABLED",
    "POTPIE_SENTRY_ENABLED",
    "POTPIE_SENTRY_DSN",
    "POTPIE_SENTRY_ENVIRONMENT",
    "POTPIE_SENTRY_RELEASE",
    "POTPIE_SENTRY_DIST",
    "POTPIE_POSTHOG_ENABLED",
    "POTPIE_PRODUCT_ANALYTICS_ENABLED",
    "POTPIE_POSTHOG_API_KEY",
    "POTPIE_POSTHOG_HOST",
)


def oauth_config_values(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    return {
        "LINEAR_CLIENT_ID": _env("LINEAR_CLIENT_ID", environ),
        "POTPIE_GITHUB_CLIENT_ID": _env("POTPIE_GITHUB_CLIENT_ID", environ),
    }


def telemetry_config_values(
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    values = {
        "POTPIE_TELEMETRY_DISABLED": _env_or_default(
            "POTPIE_TELEMETRY_DISABLED", "0", environ
        ),
        "POTPIE_SENTRY_ENABLED": _env_or_default(
            "POTPIE_SENTRY_ENABLED", "1", environ
        ),
        "POTPIE_SENTRY_DSN": _env("POTPIE_SENTRY_DSN", environ),
        "POTPIE_SENTRY_ENVIRONMENT": _env_or_default(
            "POTPIE_SENTRY_ENVIRONMENT", "production", environ
        ),
        "POTPIE_SENTRY_RELEASE": _env("POTPIE_SENTRY_RELEASE", environ),
        "POTPIE_SENTRY_DIST": _env("POTPIE_SENTRY_DIST", environ)
        or _env("GITHUB_SHA", environ),
        "POTPIE_POSTHOG_ENABLED": _env_or_default(
            "POTPIE_POSTHOG_ENABLED", "1", environ
        ),
        "POTPIE_PRODUCT_ANALYTICS_ENABLED": _env_or_default(
            "POTPIE_PRODUCT_ANALYTICS_ENABLED", "1", environ
        ),
        "POTPIE_POSTHOG_API_KEY": _env("POTPIE_POSTHOG_API_KEY", environ),
        "POTPIE_POSTHOG_HOST": _env_or_default(
            "POTPIE_POSTHOG_HOST", "https://us.i.posthog.com", environ
        ),
    }
    return {name: values[name] for name in TELEMETRY_NAMES}


def write_python_constants(path: Path, values: Mapping[str, str]) -> None:
    path.write_text(
        HEADER + "".join(f"{name} = {value!r}\n" for name, value in values.items()),
        encoding="utf-8",
    )


def _env(name: str, environ: Mapping[str, str] | None = None) -> str:
    source = os.environ if environ is None else environ
    value = source.get(name)
    if value is None:
        return ""
    return value.strip()


def _env_or_default(
    name: str,
    default: str,
    environ: Mapping[str, str] | None = None,
) -> str:
    return _env(name, environ) or default
