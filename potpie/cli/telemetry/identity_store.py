from __future__ import annotations

import json
import stat
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import ClassVar

from adapters.outbound.cli_auth.credentials_store import config_dir


@dataclass(frozen=True)
class TelemetryIdentity:
    __slots__: ClassVar[tuple[str, ...]] = (
        "anonymous_install_id",
        "created_at",
        "last_seen_at",
    )

    anonymous_install_id: str
    created_at: str
    last_seen_at: str


def identity_path() -> Path:
    return config_dir() / "telemetry" / "identity.json"


def load_or_create_identity() -> TelemetryIdentity:
    path = identity_path()
    now = datetime.now(timezone.utc).isoformat()
    payload = _read_payload(path)
    install_id = _string_value(payload, "anonymous_install_id")
    created_at = _string_value(payload, "created_at")
    identity = TelemetryIdentity(
        anonymous_install_id=install_id or f"install_{uuid.uuid4().hex}",
        created_at=created_at or now,
        last_seen_at=now,
    )
    _write_payload(path, identity)
    return identity


def _read_payload(path: Path) -> dict[str, object]:
    try:
        raw = path.read_text(encoding="utf-8")
        data: object = json.loads(raw)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(key): value for key, value in data.items()}


def _string_value(payload: dict[str, object], key: str) -> str | None:
    value = payload.get(key)
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _write_payload(path: Path, identity: TelemetryIdentity) -> None:
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
                        "anonymous_install_id": identity.anonymous_install_id,
                        "created_at": identity.created_at,
                        "last_seen_at": identity.last_seen_at,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
        tmp.chmod(stat.S_IRUSR | stat.S_IWUSR)
        _ = tmp.replace(path)
    except OSError:
        if tmp is not None:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
        return
