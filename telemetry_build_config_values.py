from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

POSTHOG_TELEMETRY_OUT = Path("potpie/cli/telemetry/_build_config.py")
SENTRY_TELEMETRY_OUT = Path("potpie/runtime/telemetry/_build_config.py")

HEADER = """\
# Auto-generated at wheel build time - do not edit manually.
# Override any value at runtime by setting the corresponding environment variable.
"""

SENTRY_TELEMETRY_NAMES = (
    "POTPIE_TELEMETRY_DISABLED",
    "POTPIE_SENTRY_ENABLED",
    "POTPIE_SENTRY_DSN",
    "POTPIE_SENTRY_ENVIRONMENT",
    "POTPIE_SENTRY_RELEASE",
    "POTPIE_SENTRY_DIST",
)

POSTHOG_TELEMETRY_NAMES = (
    "POTPIE_TELEMETRY_DISABLED",
    "POTPIE_POSTHOG_ENABLED",
    "POTPIE_PRODUCT_ANALYTICS_ENABLED",
    "POTPIE_POSTHOG_API_KEY",
    "POTPIE_POSTHOG_HOST",
)

DEFAULT_POTPIE_SENTRY_ENVIRONMENT = "prod_oss"


def telemetry_config_values(
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    return {
        "POTPIE_TELEMETRY_DISABLED": _env_or_default(
            "POTPIE_TELEMETRY_DISABLED", "0", environ
        ),
        **sentry_config_values(environ),
        **posthog_config_values(environ),
    }


def sentry_config_values(
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
            "POTPIE_SENTRY_ENVIRONMENT",
            DEFAULT_POTPIE_SENTRY_ENVIRONMENT,
            environ,
        ),
        "POTPIE_SENTRY_RELEASE": _env("POTPIE_SENTRY_RELEASE", environ),
        "POTPIE_SENTRY_DIST": _env("POTPIE_SENTRY_DIST", environ)
        or _env("GITHUB_SHA", environ),
    }
    return {name: values[name] for name in SENTRY_TELEMETRY_NAMES}


def posthog_config_values(
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    values = {
        "POTPIE_TELEMETRY_DISABLED": _env_or_default(
            "POTPIE_TELEMETRY_DISABLED", "0", environ
        ),
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
    return {name: values[name] for name in POSTHOG_TELEMETRY_NAMES}


def write_python_constants(path: Path, values: Mapping[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
