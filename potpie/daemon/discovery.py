"""Canonical owner-only daemon discovery and local credential records."""

from __future__ import annotations

import hashlib
import json
import os
import socket
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, cast

from potpie.runtime.protocol import PROTOCOL_MAX_VERSION, PROTOCOL_MIN_VERSION
from potpie.runtime.transport import RuntimeEndpoint


DISCOVERY_SCHEMA_VERSION = 1
DISCOVERY_FILENAME = "discovery.json"
CREDENTIAL_FILENAME = "daemon.credential"
PID_FILENAME = "daemon.pid"
AUTHENTICATION_SCHEME = "bearer"
_PRIVATE_FILE_MODE = 0o600
_PRIVATE_DIRECTORY_MODE = 0o700
_CONSERVATIVE_UDS_PATH_LIMIT = 90


class DaemonDiscoveryError(RuntimeError):
    """A canonical discovery or credential record is missing or invalid.

    ``code`` classifies *why* so callers can tell an absent record (first run)
    from one this build cannot read (an upgrade left an older daemon's record
    behind). The wedge in the 2.0.1 report came from collapsing both into one
    opaque "unavailable", which routed users to a recovery that could not run.
    """

    def __init__(
        self,
        message: str,
        *,
        code: str = "daemon_discovery_unavailable",
        recommended_next_action: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.recommended_next_action = recommended_next_action

    @property
    def message(self) -> str:
        return str(self)

    @property
    def recoverable_by_replacement(self) -> bool:
        """True when replacing the daemon process is the documented recovery."""

        return self.code in _REPLACEABLE_DISCOVERY_CODES


DISCOVERY_ABSENT = "daemon_discovery_absent"
DISCOVERY_UNREADABLE = "daemon_discovery_unreadable"
DISCOVERY_SCHEMA_UNSUPPORTED = "daemon_discovery_schema_unsupported"
DISCOVERY_CREDENTIAL_UNAVAILABLE = "daemon_credential_unavailable"

_REPLACEABLE_DISCOVERY_CODES = frozenset(
    {
        DISCOVERY_UNREADABLE,
        DISCOVERY_SCHEMA_UNSUPPORTED,
        DISCOVERY_CREDENTIAL_UNAVAILABLE,
    }
)

# Keys written by pre-2.0.1 daemons. Recognising them lets the CLI say
# "written by an older potpie" instead of "invalid".
_LEGACY_DISCOVERY_KEYS = frozenset({"base_url", "token", "transport"})


@dataclass(frozen=True, slots=True)
class DaemonDiscovery:
    schema_version: int
    instance_id: str
    pid: int
    endpoint: RuntimeEndpoint
    protocol_min: int
    protocol_max: int
    authentication_scheme: str
    credential_file: Path

    def __post_init__(self) -> None:
        if self.schema_version != DISCOVERY_SCHEMA_VERSION:
            raise ValueError("unsupported daemon discovery schema")
        if not self.instance_id.strip():
            raise ValueError("daemon discovery instance ID must not be empty")
        if self.pid <= 0:
            raise ValueError("daemon discovery PID must be positive")
        if self.protocol_min <= 0 or self.protocol_max < self.protocol_min:
            raise ValueError("daemon discovery protocol range is invalid")
        if self.authentication_scheme != AUTHENTICATION_SCHEME:
            raise ValueError("daemon discovery authentication scheme is invalid")
        if not self.credential_file.is_absolute():
            raise ValueError("daemon credential-file reference must be absolute")

    def to_document(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "instance_id": self.instance_id,
            "pid": self.pid,
            "transport": {
                "kind": self.endpoint.kind,
                "address": self.endpoint.address,
                "port": self.endpoint.port,
            },
            "protocol": {
                "min": self.protocol_min,
                "max": self.protocol_max,
            },
            "authentication": {
                "scheme": self.authentication_scheme,
                "credential_file": str(self.credential_file),
            },
        }


@dataclass(frozen=True, slots=True)
class DaemonConnection:
    discovery: DaemonDiscovery
    bearer_token: str


def discovery_path(home: Path) -> Path:
    return Path(home) / DISCOVERY_FILENAME


def credential_path(home: Path) -> Path:
    return Path(home) / CREDENTIAL_FILENAME


def pid_path(home: Path) -> Path:
    return Path(home) / PID_FILENAME


def select_runtime_endpoint(home: Path, *, instance_id: str) -> RuntimeEndpoint:
    """Prefer a private UDS and use authenticated loopback TCP as fallback."""

    home = Path(home).resolve()
    if os.name == "posix" and hasattr(socket, "AF_UNIX"):
        try:
            socket_path = _uds_path(home, instance_id=instance_id)
            _prepare_private_directory(socket_path.parent)
            return RuntimeEndpoint(kind="uds", address=str(socket_path))
        except (OSError, RuntimeError, ValueError):
            pass
    return RuntimeEndpoint(
        kind="tcp",
        address="127.0.0.1",
        port=_available_loopback_port(),
    )


def write_daemon_credential(home: Path, bearer_token: str) -> Path:
    if len(bearer_token.encode()) < 32:
        raise ValueError("daemon bearer token must contain at least 256 bits")
    path = credential_path(Path(home).resolve())
    _prepare_private_directory(path.parent)
    _atomic_private_write(path, f"{bearer_token}\n")
    return path


def write_daemon_discovery(home: Path, discovery: DaemonDiscovery) -> Path:
    home = Path(home).resolve()
    expected_credential = credential_path(home)
    if discovery.credential_file != expected_credential:
        raise ValueError(
            "daemon discovery must reference the canonical credential file"
        )
    path = discovery_path(home)
    _prepare_private_directory(path.parent)
    _atomic_private_write(
        path,
        json.dumps(discovery.to_document(), sort_keys=True, separators=(",", ":")),
    )
    return path


def write_daemon_pid(home: Path, pid: int) -> Path:
    if pid <= 0:
        raise ValueError("daemon PID must be positive")
    path = pid_path(Path(home).resolve())
    _prepare_private_directory(path.parent)
    _atomic_private_write(path, f"{pid}\n")
    return path


def read_daemon_pid(home: Path) -> int | None:
    """Read the canonical owner-only PID record, treating invalid state as stale."""

    path = pid_path(Path(home).resolve())
    if not path.exists():
        return None
    try:
        _require_private_regular_file(path)
        pid = int(path.read_text(encoding="utf-8").strip())
    except (DaemonDiscoveryError, OSError, ValueError):
        return None
    return pid if pid > 0 else None


def read_daemon_discovery(home: Path) -> DaemonDiscovery | None:
    home = Path(home).resolve()
    path = discovery_path(home)
    if not path.exists():
        return None
    _require_private_regular_file(path)
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DaemonDiscoveryError(
            "the daemon runtime record could not be read",
            code=DISCOVERY_UNREADABLE,
            recommended_next_action="run 'potpie daemon restart'",
        ) from exc
    if _looks_like_legacy_document(document):
        raise DaemonDiscoveryError(
            "the daemon runtime record was written by an older potpie and this "
            "build cannot authenticate that daemon",
            code=DISCOVERY_SCHEMA_UNSUPPORTED,
            recommended_next_action=(
                "run 'potpie daemon restart' to replace the pre-upgrade daemon"
            ),
        )
    try:
        return _decode_discovery(document, home=home)
    except (TypeError, ValueError) as exc:
        raise DaemonDiscoveryError(
            f"the daemon runtime record is not valid for this build: {exc}",
            code=DISCOVERY_SCHEMA_UNSUPPORTED,
            recommended_next_action="run 'potpie daemon restart'",
        ) from exc


def load_daemon_connection(home: Path) -> DaemonConnection:
    home = Path(home).resolve()
    discovery = read_daemon_discovery(home)
    if discovery is None:
        raise DaemonDiscoveryError(
            "no daemon runtime record was found",
            code=DISCOVERY_ABSENT,
            recommended_next_action="run 'potpie daemon start'",
        )
    token = read_daemon_credential(home)
    return DaemonConnection(discovery=discovery, bearer_token=token)


def read_daemon_credential(home: Path) -> str:
    home = Path(home).resolve()
    path = credential_path(home)
    _require_private_regular_file(path)
    try:
        token = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise DaemonDiscoveryError(
            "the daemon credential could not be read",
            code=DISCOVERY_CREDENTIAL_UNAVAILABLE,
            recommended_next_action="run 'potpie daemon restart'",
        ) from exc
    if len(token.encode()) < 32:
        raise DaemonDiscoveryError(
            "the daemon credential is not valid for this build",
            code=DISCOVERY_CREDENTIAL_UNAVAILABLE,
            recommended_next_action="run 'potpie daemon restart'",
        )
    return token


def _looks_like_legacy_document(document: object) -> bool:
    """Recognise a pre-2.0.1 discovery record by its distinctive flat shape."""

    if not isinstance(document, Mapping):
        return False
    if "schema_version" in document:
        return False
    return bool(_LEGACY_DISCOVERY_KEYS & set(document))


def remove_daemon_runtime_records(
    home: Path,
    *,
    expected_instance_id: str | None = None,
    expected_pid: int | None = None,
) -> None:
    """Remove only the canonical records and endpoint owned by this runtime."""

    home = Path(home).resolve()
    endpoint_path: Path | None = None
    try:
        discovery = read_daemon_discovery(home)
    except DaemonDiscoveryError:
        discovery = None
    if expected_instance_id is not None and (
        discovery is None or discovery.instance_id != expected_instance_id
    ):
        return
    if expected_pid is not None and (
        discovery is None or discovery.pid != expected_pid
    ):
        return
    if discovery is not None and discovery.endpoint.kind == "uds":
        endpoint_path = Path(discovery.endpoint.address)
    for path in (discovery_path(home), credential_path(home), pid_path(home)):
        try:
            path.unlink()
        except FileNotFoundError:
            pass
    if endpoint_path is not None:
        try:
            endpoint_path.unlink()
        except FileNotFoundError:
            pass


def _decode_discovery(document: object, *, home: Path) -> DaemonDiscovery:
    if not isinstance(document, Mapping) or set(document) != {
        "schema_version",
        "instance_id",
        "pid",
        "transport",
        "protocol",
        "authentication",
    }:
        raise ValueError("daemon discovery fields are invalid")
    transport = _required_mapping(document, "transport")
    protocol = _required_mapping(document, "protocol")
    authentication = _required_mapping(document, "authentication")
    if set(transport) != {"kind", "address", "port"}:
        raise ValueError("daemon transport discovery fields are invalid")
    if set(protocol) != {"min", "max"}:
        raise ValueError("daemon protocol discovery fields are invalid")
    if set(authentication) != {"scheme", "credential_file"}:
        raise ValueError("daemon authentication discovery fields are invalid")
    kind = _required_string(transport, "kind")
    if kind not in {"uds", "tcp"}:
        raise ValueError("daemon transport kind is invalid")
    endpoint = RuntimeEndpoint(
        kind=cast(Literal["uds", "tcp"], kind),
        address=_required_string(transport, "address"),
        port=_optional_int(transport, "port"),
    )
    credential_file = Path(_required_string(authentication, "credential_file"))
    expected_credential = credential_path(home)
    if credential_file != expected_credential:
        raise ValueError("daemon discovery references a noncanonical credential file")
    return DaemonDiscovery(
        schema_version=_required_int(document, "schema_version"),
        instance_id=_required_string(document, "instance_id"),
        pid=_required_int(document, "pid"),
        endpoint=endpoint,
        protocol_min=_required_int(protocol, "min"),
        protocol_max=_required_int(protocol, "max"),
        authentication_scheme=_required_string(authentication, "scheme"),
        credential_file=credential_file,
    )


def _uds_path(home: Path, *, instance_id: str) -> Path:
    candidate = home / "daemon.sock"
    if len(os.fsencode(candidate)) <= _CONSERVATIVE_UDS_PATH_LIMIT:
        return candidate
    uid = os.getuid() if hasattr(os, "getuid") else 0
    private_tmp = Path(tempfile.gettempdir()) / f"potpie-runtime-{uid}"
    digest = hashlib.sha256(str(home).encode()).hexdigest()[:16]
    return private_tmp / f"{digest}-{instance_id[:8]}.sock"


def _available_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _prepare_private_directory(path: Path) -> None:
    path.mkdir(parents=True, mode=_PRIVATE_DIRECTORY_MODE, exist_ok=True)
    if path.is_symlink() or not path.is_dir():
        raise RuntimeError("daemon runtime directory must be a real directory")
    if os.name == "posix":
        details = path.stat()
        if hasattr(os, "getuid") and details.st_uid != os.getuid():
            raise RuntimeError("daemon runtime directory must be owner-controlled")
        path.chmod(_PRIVATE_DIRECTORY_MODE)


def _atomic_private_write(path: Path, data: str) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        if hasattr(os, "fchmod"):
            os.fchmod(descriptor, _PRIVATE_FILE_MODE)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        path.chmod(_PRIVATE_FILE_MODE)
    except Exception:
        try:
            os.close(descriptor)
        except OSError:
            pass
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _require_private_regular_file(path: Path) -> None:
    try:
        details = path.stat()
    except OSError as exc:
        raise DaemonDiscoveryError(
            f"daemon runtime file is unavailable: {path.name}"
        ) from exc
    if path.is_symlink() or not stat.S_ISREG(details.st_mode):
        raise DaemonDiscoveryError(f"daemon runtime file is invalid: {path.name}")
    if os.name == "posix":
        if hasattr(os, "getuid") and details.st_uid != os.getuid():
            raise DaemonDiscoveryError(
                f"daemon runtime file has the wrong owner: {path.name}"
            )
        if stat.S_IMODE(details.st_mode) & 0o077:
            raise DaemonDiscoveryError(
                f"daemon runtime file is not owner-only: {path.name}"
            )


def _required_mapping(document: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = document.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be an object")
    return value


def _required_string(document: Mapping[str, Any], key: str) -> str:
    value = document.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _required_int(document: Mapping[str, Any], key: str) -> int:
    value = document.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key} must be an integer")
    return value


def _optional_int(document: Mapping[str, Any], key: str) -> int | None:
    value = document.get(key)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key} must be an integer or null")
    return value


def canonical_discovery(
    *,
    home: Path,
    instance_id: str,
    pid: int,
    endpoint: RuntimeEndpoint,
) -> DaemonDiscovery:
    return DaemonDiscovery(
        schema_version=DISCOVERY_SCHEMA_VERSION,
        instance_id=instance_id,
        pid=pid,
        endpoint=endpoint,
        protocol_min=PROTOCOL_MIN_VERSION,
        protocol_max=PROTOCOL_MAX_VERSION,
        authentication_scheme=AUTHENTICATION_SCHEME,
        credential_file=credential_path(Path(home).resolve()),
    )


__all__ = [
    "AUTHENTICATION_SCHEME",
    "CREDENTIAL_FILENAME",
    "DISCOVERY_ABSENT",
    "DISCOVERY_CREDENTIAL_UNAVAILABLE",
    "DISCOVERY_FILENAME",
    "DISCOVERY_SCHEMA_UNSUPPORTED",
    "DISCOVERY_SCHEMA_VERSION",
    "DISCOVERY_UNREADABLE",
    "PID_FILENAME",
    "DaemonConnection",
    "DaemonDiscovery",
    "DaemonDiscoveryError",
    "canonical_discovery",
    "credential_path",
    "discovery_path",
    "load_daemon_connection",
    "pid_path",
    "read_daemon_discovery",
    "read_daemon_credential",
    "remove_daemon_runtime_records",
    "select_runtime_endpoint",
    "write_daemon_credential",
    "write_daemon_discovery",
    "write_daemon_pid",
]
