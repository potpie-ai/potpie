from __future__ import annotations

import os
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

DISTRIBUTION_DEFAULTS_OUT = Path("bootstrap/_distribution_defaults.py")
BUILD_INFO_OUT = Path("bootstrap/_build_info.py")

HEADER = """\
# Auto-generated at wheel build time - do not edit manually.
# Runtime environment variables override these packaged public defaults.
"""

DEFAULT_DISTRIBUTION_ENVIRONMENT = "prod_oss"
DEFAULT_POSTHOG_HOST = "https://us.i.posthog.com"

REQUIRED_DISTRIBUTION_DEFAULTS = (
    "environment",
    "sentry_dsn",
    "posthog_api_key",
    "posthog_host",
    "linear_client_id",
    "github_client_id",
)


def distribution_default_values(
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    return {
        "environment": _env_or_default(
            "POTPIE_ENVIRONMENT", DEFAULT_DISTRIBUTION_ENVIRONMENT, environ
        ),
        "sentry_dsn": _env("POTPIE_SENTRY_DSN", environ),
        "posthog_api_key": _env("POTPIE_POSTHOG_API_KEY", environ),
        "posthog_host": _env_or_default(
            "POTPIE_POSTHOG_HOST", DEFAULT_POSTHOG_HOST, environ
        ),
        "linear_client_id": _env("LINEAR_CLIENT_ID", environ),
        "github_client_id": _env("POTPIE_GITHUB_CLIENT_ID", environ),
    }


def build_info_values(
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    return {
        "GIT_SHA": _env("POTPIE_BUILD_GIT_SHA", environ)
        or _env("GITHUB_SHA", environ),
        "BUILD_TIME": _env("POTPIE_BUILD_TIME", environ) or _utc_now(),
    }


def should_validate_distribution_defaults(
    environ: Mapping[str, str] | None = None,
) -> bool:
    return _env("POTPIE_VALIDATE_DISTRIBUTION_DEFAULTS", environ).lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def validate_distribution_defaults(values: Mapping[str, str]) -> None:
    missing = [
        name for name in REQUIRED_DISTRIBUTION_DEFAULTS if not _clean(values.get(name))
    ]
    if missing:
        raise RuntimeError(
            "Missing required distribution defaults: " + ", ".join(missing)
        )


def write_python_mapping(path: Path, name: str, values: Mapping[str, str]) -> None:
    lines = [HEADER, f"{name} = {{\n"]
    lines.extend(f"    {key!r}: {value!r},\n" for key, value in values.items())
    lines.append("}\n")
    path.write_text("".join(lines), encoding="utf-8")


def write_python_constants(path: Path, values: Mapping[str, str]) -> None:
    path.write_text(
        HEADER + "".join(f"{name} = {value!r}\n" for name, value in values.items()),
        encoding="utf-8",
    )


def _env(name: str, environ: Mapping[str, str] | None = None) -> str:
    source = os.environ if environ is None else environ
    return _clean(source.get(name))


def _env_or_default(
    name: str,
    default: str,
    environ: Mapping[str, str] | None = None,
) -> str:
    return _env(name, environ) or default


def _clean(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _utc_now() -> str:
    return datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )
