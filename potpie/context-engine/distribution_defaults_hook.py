"""Hatch build hook: generate packaged distribution defaults.

During ``hatch build`` / ``uv build``, this writes public packaged defaults for
installed wheels to ``bootstrap/_distribution_defaults.py`` and runtime build
metadata to ``bootstrap/_build_info.py``. Process environment variables still
win at runtime.
"""

from __future__ import annotations

import importlib.util
import shutil
import tempfile
from pathlib import Path
from types import ModuleType

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

_CONFIG_VALUES = "build_config_values.py"
_GENERATED_BUILD_DIRS_KEY = "_potpie_generated_build_dirs"
_GENERATED_DIR_PREFIX = "potpie-context-engine-build-"


class DistributionDefaultsHook(BuildHookInterface):
    """Generate public distribution defaults and build metadata."""

    def initialize(self, version: str, build_data: dict) -> None:
        del version
        config_values = _load_config_values_module()
        distribution_defaults = config_values.distribution_default_values()
        build_info = config_values.build_info_values()
        distribution_defaults = (
            config_values.prefer_existing_distribution_default_values(
                config_values.DISTRIBUTION_DEFAULTS_OUT,
                distribution_defaults,
            )
        )
        build_info = config_values.prefer_existing_build_info_values(
            config_values.BUILD_INFO_OUT,
            build_info,
        )
        if config_values.should_validate_distribution_defaults():
            config_values.validate_distribution_defaults(distribution_defaults)
        generated_dir = Path(tempfile.mkdtemp(prefix=_GENERATED_DIR_PREFIX))
        distribution_defaults_out = (
            generated_dir / config_values.DISTRIBUTION_DEFAULTS_OUT.name
        )
        build_info_out = generated_dir / config_values.BUILD_INFO_OUT.name
        config_values.write_python_mapping(
            distribution_defaults_out,
            "DISTRIBUTION_DEFAULTS",
            distribution_defaults,
        )
        config_values.write_python_constants(
            build_info_out,
            build_info,
        )
        build_data.setdefault("force_include", {}).update(
            {
                str(
                    distribution_defaults_out
                ): config_values.DISTRIBUTION_DEFAULTS_OUT.as_posix(),
                str(build_info_out): config_values.BUILD_INFO_OUT.as_posix(),
            }
        )
        build_data.setdefault(_GENERATED_BUILD_DIRS_KEY, []).append(str(generated_dir))

    def finalize(self, version: str, build_data: dict, artifact_path: str) -> None:
        del version, artifact_path
        _remove_generated_artifacts(build_data)


def _load_config_values_module() -> ModuleType:
    path = Path(__file__).with_name(_CONFIG_VALUES)
    spec = importlib.util.spec_from_file_location("_potpie_build_config_values", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load build config helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _remove_generated_artifacts(build_data: dict) -> None:
    temp_root = Path(tempfile.gettempdir()).resolve()
    for raw_path in build_data.get(_GENERATED_BUILD_DIRS_KEY, []):
        path = Path(raw_path)
        resolved = path.resolve(strict=False)
        if (
            not path.is_absolute()
            or resolved.parent != temp_root
            or not resolved.name.startswith(_GENERATED_DIR_PREFIX)
        ):
            raise RuntimeError(f"Refusing to remove unexpected build directory: {path}")
        if resolved.exists():
            shutil.rmtree(resolved)
