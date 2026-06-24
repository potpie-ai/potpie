from __future__ import annotations

import ast
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Final

OAUTH_OUT = Path("adapters/outbound/cli_auth/_build_config.py")
TELEMETRY_OUT = Path("adapters/inbound/cli/telemetry/_build_config.py")
OAUTH_NAMES = ("LINEAR_CLIENT_ID", "POTPIE_GITHUB_CLIENT_ID")

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
TELEMETRY_INPUT_NAMES = (*TELEMETRY_NAMES, "GITHUB_SHA")

DEFAULT_POTPIE_SENTRY_ENVIRONMENT = "prod_oss"
_DOTENV_SEARCH_START: Final[Path] = Path(__file__).resolve().parent


def oauth_config_values(
    environ: Mapping[str, str] | None = None,
    *,
    dotenv_start: Path | None = None,
) -> dict[str, str]:
    source = _merged_build_environ(environ, dotenv_start=dotenv_start)
    return {
        "LINEAR_CLIENT_ID": _env("LINEAR_CLIENT_ID", source),
        "POTPIE_GITHUB_CLIENT_ID": _env("POTPIE_GITHUB_CLIENT_ID", source),
    }


def telemetry_config_values(
    environ: Mapping[str, str] | None = None,
    *,
    dotenv_start: Path | None = None,
) -> dict[str, str]:
    source = _merged_build_environ(environ, dotenv_start=dotenv_start)
    values = {
        "POTPIE_TELEMETRY_DISABLED": _env_or_default(
            "POTPIE_TELEMETRY_DISABLED", "0", source
        ),
        "POTPIE_SENTRY_ENABLED": _env_or_default("POTPIE_SENTRY_ENABLED", "1", source),
        "POTPIE_SENTRY_DSN": _env("POTPIE_SENTRY_DSN", source),
        "POTPIE_SENTRY_ENVIRONMENT": _env_or_default(
            "POTPIE_SENTRY_ENVIRONMENT",
            DEFAULT_POTPIE_SENTRY_ENVIRONMENT,
            source,
        ),
        "POTPIE_SENTRY_RELEASE": _env("POTPIE_SENTRY_RELEASE", source),
        "POTPIE_SENTRY_DIST": _env("POTPIE_SENTRY_DIST", source)
        or _env("GITHUB_SHA", source),
        "POTPIE_POSTHOG_ENABLED": _env_or_default(
            "POTPIE_POSTHOG_ENABLED", "1", source
        ),
        "POTPIE_PRODUCT_ANALYTICS_ENABLED": _env_or_default(
            "POTPIE_PRODUCT_ANALYTICS_ENABLED", "1", source
        ),
        "POTPIE_POSTHOG_API_KEY": _env("POTPIE_POSTHOG_API_KEY", source),
        "POTPIE_POSTHOG_HOST": _env_or_default(
            "POTPIE_POSTHOG_HOST", "https://us.i.posthog.com", source
        ),
    }
    return {name: values[name] for name in TELEMETRY_NAMES}


def write_python_constants(path: Path, values: Mapping[str, str]) -> None:
    path.write_text(
        HEADER + "".join(f"{name} = {value!r}\n" for name, value in values.items()),
        encoding="utf-8",
    )


def prefer_existing_config_values(
    path: Path,
    values: Mapping[str, str],
) -> dict[str, str]:
    """Keep generated sdist values when a wheel build lacks build env inputs."""
    existing = _read_python_constants(path)
    if not existing:
        return dict(values)
    merged = dict(values)
    for name in values:
        if name in existing:
            merged[name] = existing[name]
    return merged


def has_build_config_inputs(
    names: tuple[str, ...],
    environ: Mapping[str, str] | None = None,
    *,
    dotenv_start: Path | None = None,
) -> bool:
    source = os.environ if environ is None else environ
    if any(name in source for name in names):
        return True
    return _find_nearest_dotenv(dotenv_start or _DOTENV_SEARCH_START) is not None


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


def _merged_build_environ(
    environ: Mapping[str, str] | None = None,
    *,
    dotenv_start: Path | None = None,
) -> Mapping[str, str]:
    source = os.environ if environ is None else environ
    if dotenv_start is None and environ is not None:
        return source

    dotenv = _read_nearest_dotenv(dotenv_start or _DOTENV_SEARCH_START)
    if not dotenv:
        return source

    merged = dict(dotenv)
    merged.update(source)
    return merged


def _read_nearest_dotenv(start: Path) -> dict[str, str]:
    path = _find_nearest_dotenv(start)
    if path is None:
        return {}

    values: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}
    for line in lines:
        parsed = _parse_dotenv_line(line)
        if parsed is None:
            continue
        key, value = parsed
        values[key] = value
    return values


def _find_nearest_dotenv(start: Path) -> Path | None:
    cur = start.resolve()
    if cur.is_file():
        cur = cur.parent
    for ancestor in (cur, *cur.parents):
        candidate = ancestor / ".env"
        if candidate.is_file():
            return candidate
    return None


def _parse_dotenv_line(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    if stripped.lower().startswith("export "):
        stripped = stripped[7:].strip()
    if "=" not in stripped:
        return None
    key, value = stripped.split("=", 1)
    key = key.strip()
    if not key or not key.replace("_", "").isalnum() or key[0].isdigit():
        return None
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    return key, value


def _read_python_constants(path: Path) -> dict[str, str]:
    try:
        module = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return {}

    values: dict[str, str] = {}
    for node in module.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            values[target.id] = node.value.value
    return values
