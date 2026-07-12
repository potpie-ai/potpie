from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from potpie_context_engine.bootstrap import env_bootstrap

_CODE_DEFAULT_ENVIRONMENT: Final[str] = "dev"
_CODE_DEFAULT_API_URL: Final[str] = "http://localhost:8001"
_CODE_DEFAULT_UI_URL: Final[str] = "http://localhost:3000"
_DEPRECATED_CHILD_ENV_KEYS: Final[frozenset[str]] = frozenset(
    {
        "POTPIE_BASE_URL",
        "POTPIE_CLI_API_BASE_URL",
        "POTPIE_CLI_BASE_URL",
        "POTPIE_PORT",
        "POTPIE_API_PORT",
        "POTPIE_CLI_UI_BASE_URL",
        "POTPIE_CLI_APP_BASE_URL",
        "GITHUB_TOKEN",
    }
)


@dataclass(frozen=True)
class RuntimeSettings:
    environment: str
    potpie_api_url: str
    potpie_ui_url: str
    potpie_api_key: str | None
    linear_client_id: str | None
    github_client_id: str | None
    context_engine_github_token: str | None
    github_webhook_secret: str | None


def load_runtime_settings(
    environ: Mapping[str, str] | None = None,
    *,
    distribution_defaults: Mapping[str, object] | None = None,
) -> RuntimeSettings:
    """Resolve runtime settings from env, distribution defaults, then code defaults."""
    defaults = _normalize_distribution_defaults(
        distribution_defaults
        if distribution_defaults is not None
        else load_distribution_defaults()
    )
    if environ is None:
        ensure_runtime_environment_loaded(defaults)
        environ = os.environ

    environment = (
        _env(environ, "POTPIE_ENVIRONMENT")
        or _default(defaults, "environment")
        or _CODE_DEFAULT_ENVIRONMENT
    )
    return RuntimeSettings(
        environment=environment,
        potpie_api_url=(
            _env(environ, "POTPIE_API_URL") or _CODE_DEFAULT_API_URL
        ).rstrip("/"),
        potpie_ui_url=(_env(environ, "POTPIE_UI_URL") or _CODE_DEFAULT_UI_URL).rstrip(
            "/"
        ),
        potpie_api_key=_env(environ, "POTPIE_API_KEY"),
        linear_client_id=_env(environ, "LINEAR_CLIENT_ID")
        or _default(defaults, "linear_client_id"),
        github_client_id=_env(environ, "POTPIE_GITHUB_CLIENT_ID")
        or _default(defaults, "github_client_id"),
        context_engine_github_token=_env(environ, "CONTEXT_ENGINE_GITHUB_TOKEN"),
        github_webhook_secret=_env(environ, "GITHUB_WEBHOOK_SECRET"),
    )


def ensure_runtime_environment_loaded(
    distribution_defaults: Mapping[str, str] | None = None,
) -> None:
    """Load local .env only for dev bootstrap environments."""
    defaults = (
        _normalize_distribution_defaults(distribution_defaults)
        if distribution_defaults is not None
        else load_distribution_defaults()
    )
    if resolve_bootstrap_environment(os.environ, defaults) == "dev":
        env_bootstrap.load_cli_env()


def resolve_bootstrap_environment(
    environ: Mapping[str, str],
    distribution_defaults: Mapping[str, str],
) -> str:
    return (
        _env(environ, "POTPIE_ENVIRONMENT")
        or _default(distribution_defaults, "environment")
        or _CODE_DEFAULT_ENVIRONMENT
    )


def project_child_environment(
    settings: RuntimeSettings,
    base: Mapping[str, str],
    overrides: Mapping[str, str] | None = None,
) -> dict[str, str]:
    child = {
        key: value
        for key, value in base.items()
        if key not in _DEPRECATED_CHILD_ENV_KEYS
    }
    child.update(
        {
            "POTPIE_ENVIRONMENT": settings.environment,
            "POTPIE_API_URL": settings.potpie_api_url,
            "POTPIE_UI_URL": settings.potpie_ui_url,
        }
    )
    _set_if_present(child, "POTPIE_API_KEY", settings.potpie_api_key)
    _set_if_present(child, "LINEAR_CLIENT_ID", settings.linear_client_id)
    _set_if_present(child, "POTPIE_GITHUB_CLIENT_ID", settings.github_client_id)
    _set_if_present(
        child, "CONTEXT_ENGINE_GITHUB_TOKEN", settings.context_engine_github_token
    )
    _set_if_present(child, "GITHUB_WEBHOOK_SECRET", settings.github_webhook_secret)
    if overrides:
        child.update(
            {
                key: value
                for key, value in overrides.items()
                if key not in _DEPRECATED_CHILD_ENV_KEYS
            }
        )
    return child


def load_distribution_defaults() -> Mapping[str, str]:
    try:
        from potpie_context_engine.bootstrap._distribution_defaults import (
            DISTRIBUTION_DEFAULTS,
        )
    except ImportError:
        return MappingProxyType({})
    return MappingProxyType(_normalize_distribution_defaults(DISTRIBUTION_DEFAULTS))


def build_git_sha() -> str | None:
    try:
        from potpie_context_engine.bootstrap._build_info import GIT_SHA
    except ImportError:
        return None
    return _clean(GIT_SHA)


def _normalize_distribution_defaults(
    values: Mapping[str, object],
) -> dict[str, str]:
    return {
        str(key): cleaned
        for key, value in values.items()
        if (cleaned := _clean(value)) is not None
    }


def _env(environ: Mapping[str, str], name: str) -> str | None:
    return _clean(environ.get(name))


def _default(defaults: Mapping[str, str], name: str) -> str | None:
    return _clean(defaults.get(name))


def _clean(value: object) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned or None


def _set_if_present(target: dict[str, str], name: str, value: str | None) -> None:
    if value is None:
        target.pop(name, None)
    else:
        target[name] = value


__all__ = [
    "RuntimeSettings",
    "build_git_sha",
    "ensure_runtime_environment_loaded",
    "load_distribution_defaults",
    "load_runtime_settings",
    "project_child_environment",
    "resolve_bootstrap_environment",
]
