"""Hatch build hook: generate packaged distribution defaults.

During ``hatch build`` / ``uv build``, this writes public packaged defaults for
installed wheels to ``bootstrap/_distribution_defaults.py`` and runtime build
metadata to ``bootstrap/_build_info.py``. Process environment variables still
win at runtime.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

_CONFIG_VALUES = "build_config_values.py"


class DistributionDefaultsHook(BuildHookInterface):
    """Generate public distribution defaults and build metadata."""

    def initialize(self, version: str, build_data: dict) -> None:
        del version
        config_values = _load_config_values_module()
        distribution_defaults = config_values.distribution_default_values()
        build_info = config_values.build_info_values()
        if not config_values.has_build_config_inputs(
            config_values.DISTRIBUTION_DEFAULT_INPUT_NAMES
        ):
            distribution_defaults = config_values.prefer_existing_mapping_values(
                config_values.DISTRIBUTION_DEFAULTS_OUT,
                "DISTRIBUTION_DEFAULTS",
                distribution_defaults,
            )
        if not config_values.has_build_config_inputs(
            config_values.BUILD_INFO_INPUT_NAMES
        ):
            build_info = config_values.prefer_existing_config_values(
                config_values.BUILD_INFO_OUT,
                build_info,
            )
        if config_values.should_validate_distribution_defaults():
            config_values.validate_distribution_defaults(distribution_defaults)
        config_values.write_python_mapping(
            config_values.DISTRIBUTION_DEFAULTS_OUT,
            "DISTRIBUTION_DEFAULTS",
            distribution_defaults,
        )
        config_values.write_python_constants(
            config_values.BUILD_INFO_OUT,
            build_info,
        )
        build_data.setdefault("artifacts", []).extend(
            [
                str(config_values.DISTRIBUTION_DEFAULTS_OUT),
                str(config_values.BUILD_INFO_OUT),
            ]
        )


def _load_config_values_module() -> ModuleType:
    path = Path(__file__).with_name(_CONFIG_VALUES)
    spec = importlib.util.spec_from_file_location("_potpie_build_config_values", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load build config helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
