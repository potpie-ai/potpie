"""Hatch build hook: inject public defaults into the wheel at build time.

During ``hatch build`` / ``uv build``, reads ``LINEAR_CLIENT_ID`` and
``POTPIE_GITHUB_CLIENT_ID`` from the build environment and writes
``adapters/outbound/cli_auth/_build_config.py`` so installed users do not need
to configure those variables locally.

It also writes ``adapters/inbound/cli/telemetry/_build_config.py`` with public
CLI telemetry ingest defaults (Sentry DSN and PostHog project API key). Runtime
env vars still take precedence for local development overrides and opt-outs.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

_CONFIG_VALUES = "build_config_values.py"


class OAuthClientIdInjectionHook(BuildHookInterface):
    """Inject public OAuth and telemetry defaults into the wheel."""

    def initialize(self, version: str, build_data: dict) -> None:
        config_values = _load_config_values_module()
        config_values.write_python_constants(
            config_values.OAUTH_OUT,
            config_values.oauth_config_values(),
        )
        config_values.write_python_constants(
            config_values.TELEMETRY_OUT,
            config_values.telemetry_config_values(),
        )
        build_data.setdefault("artifacts", []).extend(
            [str(config_values.OAUTH_OUT), str(config_values.TELEMETRY_OUT)]
        )


def _load_config_values_module() -> ModuleType:
    path = Path(__file__).with_name(_CONFIG_VALUES)
    spec = importlib.util.spec_from_file_location("_potpie_build_config_values", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load build config helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
