"""Hatch build hook: inject public CLI telemetry defaults into the root wheel."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

_CONFIG_VALUES = "telemetry_build_config_values.py"


class TelemetryBuildHook(BuildHookInterface):
    """Inject telemetry defaults owned by the root CLI/runtime package."""

    def initialize(self, version: str, build_data: dict) -> None:
        config_values = _load_config_values_module()
        config_values.write_python_constants(
            config_values.SENTRY_TELEMETRY_OUT,
            config_values.sentry_config_values(),
        )
        config_values.write_python_constants(
            config_values.POSTHOG_TELEMETRY_OUT,
            config_values.posthog_config_values(),
        )
        artifacts = build_data.setdefault("artifacts", [])
        artifacts.append(str(config_values.SENTRY_TELEMETRY_OUT))
        artifacts.append(str(config_values.POSTHOG_TELEMETRY_OUT))


def _load_config_values_module() -> ModuleType:
    path = Path(__file__).with_name(_CONFIG_VALUES)
    spec = importlib.util.spec_from_file_location(
        "_potpie_telemetry_config_values",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load build config helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
