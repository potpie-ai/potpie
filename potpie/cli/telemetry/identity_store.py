from __future__ import annotations

import json
import stat
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import ClassVar

from potpie_context_engine.adapters.outbound.cli_auth.credentials_store import (
    config_dir,
)


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


_ACTIVATION_SENT_KEY = "activation_sent"


def identity_path() -> Path:
    return config_dir() / "telemetry" / "identity.json"


def load_or_create_identity() -> TelemetryIdentity:
    path = identity_path()
    payload = _read_payload(path)
    identity = _identity_from_payload(payload)
    _write_payload(path, identity, activation_sent=_activation_sent(payload))
    return identity


def activation_sent(name: str) -> bool:
    """Whether the once-only activation event ``name`` already went out."""
    return name in _activation_sent(_read_payload(identity_path()))


def mark_activation_sent(name: str) -> bool:
    """Record that once-only event ``name`` is being sent; ``True`` the first time.

    The "first use" onboarding events were firing on every ``status``,
    ``search`` and ``resolve`` because nothing remembered that they had already
    gone out — one or two extra analytics POSTs on every command, forever. The
    marker lives next to the install id because it describes the same install.
    A write that fails leaves the event unmarked, so the worst case is the old
    behaviour (sent again next time), never a lost first-use signal.
    """
    path = identity_path()
    payload = _read_payload(path)
    sent = _activation_sent(payload)
    if name in sent:
        return False
    _write_payload(path, _identity_from_payload(payload), activation_sent=(*sent, name))
    return True


def _identity_from_payload(payload: dict[str, object]) -> TelemetryIdentity:
    now = datetime.now(timezone.utc).isoformat()
    install_id = _string_value(payload, "anonymous_install_id")
    created_at = _string_value(payload, "created_at")
    return TelemetryIdentity(
        anonymous_install_id=install_id or f"install_{uuid.uuid4().hex}",
        created_at=created_at or now,
        last_seen_at=now,
    )


def _activation_sent(payload: dict[str, object]) -> tuple[str, ...]:
    value = payload.get(_ACTIVATION_SENT_KEY)
    if not isinstance(value, list):
        return ()
    return tuple(
        dict.fromkeys(
            item.strip() for item in value if isinstance(item, str) and item.strip()
        )
    )


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


def _write_payload(
    path: Path,
    identity: TelemetryIdentity,
    *,
    activation_sent: tuple[str, ...] = (),
) -> None:
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
                        _ACTIVATION_SENT_KEY: list(activation_sent),
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
