from __future__ import annotations

import ast
import os
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

DISTRIBUTION_DEFAULTS_OUT = Path("bootstrap/_distribution_defaults.py")
BUILD_INFO_OUT = Path("bootstrap/_build_info.py")
DISTRIBUTION_DEFAULT_INPUT_NAMES = (
    "POTPIE_ENVIRONMENT",
    "POTPIE_SENTRY_DSN",
    "POTPIE_POSTHOG_API_KEY",
    "POTPIE_POSTHOG_HOST",
    "LINEAR_CLIENT_ID",
    "POTPIE_GITHUB_CLIENT_ID",
)
BUILD_INFO_INPUT_NAMES = ("POTPIE_BUILD_GIT_SHA", "GITHUB_SHA", "POTPIE_BUILD_TIME")

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
_DOTENV_SEARCH_START: Final[Path] = Path(__file__).resolve().parent


def distribution_default_values(
    environ: Mapping[str, str] | None = None,
    *,
    dotenv_start: Path | None = None,
) -> dict[str, str]:
    source = _merged_build_environ(environ, dotenv_start=dotenv_start)
    return {
        "environment": _env_or_default(
            "POTPIE_ENVIRONMENT", DEFAULT_DISTRIBUTION_ENVIRONMENT, source
        ),
        "sentry_dsn": _env("POTPIE_SENTRY_DSN", source),
        "posthog_api_key": _env("POTPIE_POSTHOG_API_KEY", source),
        "posthog_host": _env_or_default(
            "POTPIE_POSTHOG_HOST", DEFAULT_POSTHOG_HOST, source
        ),
        "linear_client_id": _env("LINEAR_CLIENT_ID", source),
        "github_client_id": _env("POTPIE_GITHUB_CLIENT_ID", source),
    }


def build_info_values(
    environ: Mapping[str, str] | None = None,
    *,
    dotenv_start: Path | None = None,
) -> dict[str, str]:
    source = _merged_build_environ(environ, dotenv_start=dotenv_start)
    return {
        "GIT_SHA": _env("POTPIE_BUILD_GIT_SHA", source)
        or _env("GITHUB_SHA", source),
        "BUILD_TIME": _env("POTPIE_BUILD_TIME", source) or _utc_now(),
    }


def should_validate_distribution_defaults(
    environ: Mapping[str, str] | None = None,
    *,
    dotenv_start: Path | None = None,
) -> bool:
    source = _merged_build_environ(environ, dotenv_start=dotenv_start)
    return _env("POTPIE_VALIDATE_DISTRIBUTION_DEFAULTS", source).lower() in {
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
    path.write_text(
        HEADER
        + f"{name} = {{\n"
        + "".join(f"    {key!r}: {value!r},\n" for key, value in values.items())
        + "}\n",
        encoding="utf-8",
    )


def write_python_constants(path: Path, values: Mapping[str, str]) -> None:
    path.write_text(
        HEADER + "".join(f"{name} = {value!r}\n" for name, value in values.items()),
        encoding="utf-8",
    )


def prefer_existing_mapping_values(
    path: Path,
    mapping_name: str,
    values: Mapping[str, str],
) -> dict[str, str]:
    """Keep generated sdist mapping values when a wheel build lacks build inputs."""
    existing = _read_python_mapping(path, mapping_name)
    if not existing:
        return dict(values)
    merged = dict(values)
    for name in values:
        if name in existing:
            merged[name] = existing[name]
    return merged


def prefer_existing_config_values(
    path: Path,
    values: Mapping[str, str],
) -> dict[str, str]:
    """Keep generated sdist constant values when a wheel build lacks build inputs."""
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


def _read_python_mapping(path: Path, mapping_name: str) -> dict[str, str]:
    try:
        module = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return {}

    for node in module.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or target.id != mapping_name:
            continue
        try:
            raw = ast.literal_eval(node.value)
        except (ValueError, SyntaxError):
            return {}
        if not isinstance(raw, dict):
            return {}
        return {str(key): _clean(value) for key, value in raw.items()}
    return {}


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
