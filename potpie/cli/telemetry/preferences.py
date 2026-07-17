from __future__ import annotations

import json
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path

from potpie_context_engine.adapters.outbound.cli_auth.credentials_store import config_dir


@dataclass(frozen=True, slots=True)
class TelemetryPreferences:
    enabled: bool = True


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
    "load_preferences",
    "preferences_path",
    "save_preferences",
    "telemetry_enabled_by_preference",
]
