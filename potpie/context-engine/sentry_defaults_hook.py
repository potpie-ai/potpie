"""Hatch hook for standalone Context Engine Sentry metrics defaults."""

from __future__ import annotations

import ast
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

_DEFAULTS_OUT = Path("src/potpie_context_engine/bootstrap/_distribution_defaults.py")
_BUILD_INFO_OUT = Path("src/potpie_context_engine/bootstrap/_build_info.py")
_GENERATED_BUILD_DIRS_KEY = "_context_engine_generated_build_dirs"
_GENERATED_DIR_PREFIX = "potpie-context-engine-build-"
_DOTENV_VALUES = None


class SentryDefaultsHook(BuildHookInterface):
    """Generate only the engine-owned defaults used by standalone metrics."""

    def initialize(self, version: str, build_data: dict) -> None:
        del version
        defaults = _prefer_existing_defaults(_DEFAULTS_OUT, _runtime_defaults())
        build_info = _prefer_existing_build_info(_BUILD_INFO_OUT, _build_info())
        if _flag(os.getenv("POTPIE_VALIDATE_DISTRIBUTION_DEFAULTS", "0")):
            missing = [name for name, value in defaults.items() if not value]
            if missing:
                raise RuntimeError(
                    "Missing required Context Engine Sentry defaults: "
                    + ", ".join(missing)
                )

        generated_dir = Path(tempfile.mkdtemp(prefix=_GENERATED_DIR_PREFIX))
        defaults_out = generated_dir / _DEFAULTS_OUT.name
        build_info_out = generated_dir / _BUILD_INFO_OUT.name
        try:
            defaults_out.write_text(
                "# Auto-generated at wheel build time - do not edit manually.\n"
                f"DISTRIBUTION_DEFAULTS = {defaults!r}\n",
                encoding="utf-8",
            )
            build_info_out.write_text(
                "# Auto-generated at wheel build time - do not edit manually.\n"
                + "".join(
                    f"{name} = {value!r}\n" for name, value in build_info.items()
                ),
                encoding="utf-8",
            )
            build_data.setdefault("force_include", {}).update(
                {
                    str(defaults_out): self._artifact_path(_DEFAULTS_OUT),
                    str(build_info_out): self._artifact_path(_BUILD_INFO_OUT),
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
        temp_root = Path(tempfile.gettempdir()).resolve()
        for raw_path in build_data.get(_GENERATED_BUILD_DIRS_KEY, []):
            path = Path(raw_path).resolve(strict=False)
            if path.parent != temp_root or not path.name.startswith(
                _GENERATED_DIR_PREFIX
            ):
                raise RuntimeError(
                    f"Refusing to remove unexpected build directory: {path}"
                )
            if path.exists():
                shutil.rmtree(path)

    def _artifact_path(self, source_tree_path: Path) -> str:
        rel = source_tree_path.as_posix()
        if getattr(self, "target_name", "wheel") == "wheel" and rel.startswith("src/"):
            return rel[len("src/") :]
        return rel


def _runtime_defaults() -> dict[str, str]:
    return {
        "environment": _env("POTPIE_ENVIRONMENT") or "prod_oss",
        "sentry_dsn": _env("POTPIE_SENTRY_DSN"),
    }


def _build_info() -> dict[str, str]:
    return {
        "GIT_SHA": _env("POTPIE_BUILD_GIT_SHA") or _env("GITHUB_SHA"),
        "BUILD_TIME": _env("POTPIE_BUILD_TIME") or _utc_now(),
    }


def _prefer_existing_defaults(path: Path, values: dict[str, str]) -> dict[str, str]:
    existing = _read_mapping(path, "DISTRIBUTION_DEFAULTS")
    merged = dict(values)
    for field, env_name in {
        "environment": "POTPIE_ENVIRONMENT",
        "sentry_dsn": "POTPIE_SENTRY_DSN",
    }.items():
        if field in existing and not _env(env_name):
            merged[field] = existing[field]
    return merged


def _prefer_existing_build_info(path: Path, values: dict[str, str]) -> dict[str, str]:
    existing = _read_constants(path)
    merged = dict(values)
    if "GIT_SHA" in existing and not (
        _env("POTPIE_BUILD_GIT_SHA") or _env("GITHUB_SHA")
    ):
        merged["GIT_SHA"] = existing["GIT_SHA"]
    if "BUILD_TIME" in existing and not _env("POTPIE_BUILD_TIME"):
        merged["BUILD_TIME"] = existing["BUILD_TIME"]
    return merged


def _read_mapping(path: Path, name: str) -> dict[str, str]:
    try:
        module = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return {}
    for node in module.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or target.id != name:
            continue
        try:
            value = ast.literal_eval(node.value)
        except (ValueError, SyntaxError):
            return {}
        if isinstance(value, dict):
            return {str(key): str(item).strip() for key, item in value.items()}
    return {}


def _read_constants(path: Path) -> dict[str, str]:
    try:
        module = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return {}
    return {
        node.targets[0].id: node.value.value
        for node in module.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    }


def _env(name: str) -> str:
    if name in os.environ:
        return os.environ[name].strip()
    return _dotenv_values().get(name, "")


def _dotenv_values() -> dict[str, str]:
    global _DOTENV_VALUES
    if _DOTENV_VALUES is not None:
        return _DOTENV_VALUES
    for ancestor in (
        Path(__file__).resolve().parent,
        *Path(__file__).resolve().parents,
    ):
        candidate = ancestor / ".env"
        if not candidate.is_file():
            continue
        values: dict[str, str] = {}
        try:
            lines = candidate.read_text(encoding="utf-8").splitlines()
        except OSError:
            break
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            if stripped.lower().startswith("export "):
                stripped = stripped[7:].strip()
            key, value = stripped.split("=", 1)
            key = key.strip()
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
                value = value[1:-1]
            if key:
                values[key] = value
        _DOTENV_VALUES = values
        return values
    _DOTENV_VALUES = {}
    return _DOTENV_VALUES


def _flag(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _utc_now() -> str:
    return (
        datetime.now(tz=timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
