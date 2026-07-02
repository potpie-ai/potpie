from __future__ import annotations

import json
import stat
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from potpie.runtime.paths import config_dir
from potpie.runtime.settings import RuntimeSettings, load_runtime_settings

TelemetryState = Literal["blocked", "disabled", "enabled"]


@dataclass(frozen=True, slots=True)
class TelemetryPreferences:
    enabled: bool = True


@dataclass(frozen=True, slots=True)
class TelemetryRuntimeResolution:
    runtime: RuntimeSettings
    telemetry: TelemetryState


class TelemetryPreferenceWriteError(RuntimeError):
    """Raised when the local telemetry preference file cannot be written."""

    def __init__(self, path: Path, cause: OSError) -> None:
        self.path = path
        self.cause = cause
        super().__init__(f"Unable to write telemetry preferences at {path}: {cause}")


def preferences_path() -> Path:
    return config_dir() / "telemetry" / "settings.json"


def load_preferences() -> TelemetryPreferences:
    try:
        payload = _read_payload(preferences_path())
    except FileNotFoundError:
        return TelemetryPreferences()
    if payload is None:
        return TelemetryPreferences(enabled=False)
    enabled = payload.get("enabled")
    if isinstance(enabled, bool):
        return TelemetryPreferences(enabled=enabled)
    return TelemetryPreferences(enabled=False)


def save_preferences(preferences: TelemetryPreferences) -> None:
    path = preferences_path()
    tmp: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w",
            dir=path.parent,
            encoding="utf-8",
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp = Path(handle.name)
            _ = handle.write(
                json.dumps(
                    {
                        "schema_version": 1,
                        "enabled": preferences.enabled,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
        tmp.chmod(stat.S_IRUSR | stat.S_IWUSR)
        _ = tmp.replace(path)
    except OSError as exc:
        if tmp is not None:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
        raise TelemetryPreferenceWriteError(path, exc) from exc


def telemetry_enabled_by_preference() -> bool:
    return load_preferences().enabled


def load_runtime_settings_with_telemetry_preference() -> TelemetryRuntimeResolution:
    return resolve_runtime_telemetry(load_runtime_settings())


def resolve_runtime_telemetry(settings: RuntimeSettings) -> TelemetryRuntimeResolution:
    if settings.telemetry_disabled:
        return TelemetryRuntimeResolution(runtime=settings, telemetry="blocked")
    if telemetry_enabled_by_preference():
        return TelemetryRuntimeResolution(runtime=settings, telemetry="enabled")
    return TelemetryRuntimeResolution(
        runtime=replace(settings, telemetry_disabled=True),
        telemetry="disabled",
    )


def _read_payload(path: Path) -> dict[str, object] | None:
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise
    except OSError:
        return None
    try:
        data: object = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    return {str(key): value for key, value in data.items()}


__all__ = [
    "TelemetryPreferenceWriteError",
    "TelemetryPreferences",
    "TelemetryRuntimeResolution",
    "TelemetryState",
    "load_preferences",
    "load_runtime_settings_with_telemetry_preference",
    "preferences_path",
    "resolve_runtime_telemetry",
    "save_preferences",
    "telemetry_enabled_by_preference",
]
