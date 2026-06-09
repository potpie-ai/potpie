from __future__ import annotations

import json
import platform
import uuid
from contextvars import ContextVar
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

import typer

from adapters.outbound.pots.local_pot_store import default_home

_DAEMON_SESSION_ID = f"daemon_{uuid.uuid4().hex}"
_CURRENT: ContextVar["TelemetryContext | None"] = ContextVar(
    "potpie_cli_telemetry_context",
    default=None,
)
_obs_token: object | None = None


@dataclass(frozen=True, slots=True)
class TelemetryContext:
    anonymous_install_id: str
    invocation_id: str
    daemon_session_id: str
    command: str | None
    subcommand: str | None
    output_mode: str
    cli_version: str
    python_version: str
    os: str
    arch: str

    def fields(self) -> dict[str, str]:
        return {
            key: value
            for key, value in {
                "anonymous_install_id": self.anonymous_install_id,
                "invocation_id": self.invocation_id,
                "daemon_session_id": self.daemon_session_id,
                "command": self.command,
                "subcommand": self.subcommand,
                "output_mode": self.output_mode,
                "cli_version": self.cli_version,
                "python_version": self.python_version,
                "os": self.os,
                "arch": self.arch,
            }.items()
            if value is not None
        }


def load_anonymous_install_id(home: Path) -> str:
    identity_path = _identity_path(home)
    try:
        payload = json.loads(identity_path.read_text(encoding="utf-8"))
        install_id = payload.get("anonymous_install_id")
        if isinstance(install_id, str) and install_id:
            return install_id
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        pass
    install_id = f"install_{uuid.uuid4().hex}"
    try:
        identity_path.parent.mkdir(parents=True, exist_ok=True)
        identity_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "anonymous_install_id": install_id,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    except OSError:
        return install_id
    return install_id


def bind_cli_telemetry_context(
    ctx: typer.Context, *, json_output: bool
) -> TelemetryContext:
    command, subcommand = _command_parts(ctx)
    telemetry = TelemetryContext(
        anonymous_install_id=load_anonymous_install_id(default_home()),
        invocation_id=f"invoke_{uuid.uuid4().hex}",
        daemon_session_id=_DAEMON_SESSION_ID,
        command=command,
        subcommand=subcommand,
        output_mode="json" if json_output else "human",
        cli_version=_cli_version(),
        python_version=platform.python_version(),
        os=platform.system().lower(),
        arch=platform.machine(),
    )
    _CURRENT.set(telemetry)
    _bind_observability(telemetry)
    return telemetry


def current_telemetry_context() -> TelemetryContext | None:
    return _CURRENT.get()


def _bind_observability(telemetry: TelemetryContext) -> None:
    global _obs_token
    try:
        from observability import bind_context, reset_context

        if _obs_token is not None:
            reset_context(_obs_token)
        _obs_token = bind_context(**telemetry.fields())
    except Exception:  # noqa: BLE001
        _obs_token = None


def _command_parts(ctx: typer.Context) -> tuple[str | None, str | None]:
    command = ctx.invoked_subcommand
    subcommand = None
    protected_args = [str(arg) for arg in getattr(ctx, "protected_args", [])]
    args = [str(arg) for arg in getattr(ctx, "args", [])]
    parts = protected_args + args
    if command is None and parts:
        command = parts[0]
    if len(parts) > 1:
        subcommand = parts[1]
    return command, subcommand


def _identity_path(home: Path) -> Path:
    return home / "telemetry" / "identity.json"


def _cli_version() -> str:
    try:
        return metadata.version("potpie-context-engine")
    except metadata.PackageNotFoundError:
        return "0.1.0"
