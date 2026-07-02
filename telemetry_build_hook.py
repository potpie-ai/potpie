"""Hatch build hook: inject public CLI telemetry defaults into the root wheel."""

from __future__ import annotations

import importlib.util
import shutil
import tempfile
from pathlib import Path
from types import ModuleType

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

_CONFIG_VALUES = "telemetry_build_config_values.py"
_GENERATED_BUILD_DIRS_KEY = "_potpie_runtime_generated_build_dirs"
_GENERATED_DIR_PREFIX = "potpie-runtime-build-"


class TelemetryBuildHook(BuildHookInterface):
    """Inject telemetry defaults owned by the root CLI/runtime package."""

    def initialize(self, version: str, build_data: dict) -> None:
        del version
        config_values = _load_config_values_module()
        generated_dir = Path(tempfile.mkdtemp(prefix=_GENERATED_DIR_PREFIX))
        sentry_out = generated_dir / config_values.SENTRY_TELEMETRY_OUT
        posthog_out = generated_dir / config_values.POSTHOG_TELEMETRY_OUT
        try:
            config_values.write_python_constants(
                sentry_out,
                config_values.sentry_config_values(),
            )
            config_values.write_python_constants(
                posthog_out,
                config_values.posthog_config_values(),
            )
            build_data.setdefault("force_include", {}).update(
                {
                    str(sentry_out): config_values.SENTRY_TELEMETRY_OUT.as_posix(),
                    str(posthog_out): config_values.POSTHOG_TELEMETRY_OUT.as_posix(),
                }
            )
            build_data.setdefault(_GENERATED_BUILD_DIRS_KEY, []).append(
                str(generated_dir)
            )
        except Exception:
            shutil.rmtree(generated_dir, ignore_errors=True)
            raise

    def finalize(self, version: str, build_data: dict, artifact_path: str) -> None:
        del version, artifact_path
        _remove_generated_artifacts(build_data)


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
