from __future__ import annotations

import platform
import uuid
from contextvars import ContextVar
from dataclasses import dataclass
from typing import ClassVar

import typer

from .identity_store import load_or_create_identity
from .settings import default_cli_release, telemetry_environment

_DAEMON_SESSION_ID = f"daemon_{uuid.uuid4().hex}"
_CURRENT: ContextVar["TelemetryContext | None"] = ContextVar(
    "potpie_cli_telemetry_context",
    default=None,
)


@dataclass(frozen=True)
class TelemetryContext:
    __slots__: ClassVar[tuple[str, ...]] = (
        "anonymous_install_id",
        "arch",
        "cli_version",
        "command",
        "daemon_session_id",
        "environment",
        "invocation_id",
        "os",
        "output_mode",
        "python_version",
        "subcommand",
    )

    anonymous_install_id: str
    invocation_id: str
    daemon_session_id: str
    environment: str
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
                "environment": self.environment,
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

    def analytics_properties(self) -> dict[str, str]:
        return {
            "anonymous_install_id": self.anonymous_install_id,
            "invocation_id": self.invocation_id,
            "daemon_session_id": self.daemon_session_id,
            "environment": self.environment,
            "output_mode": self.output_mode,
            "cli_version": self.cli_version,
            "python_version": self.python_version,
            "platform": self.os,
            "arch": self.arch,
            **_optional_command_properties(self),
        }


def bind_telemetry_context(
    ctx: typer.Context, *, json_output: bool
) -> TelemetryContext:
    identity = load_or_create_identity()
    command, subcommand = _command_parts(ctx)
    telemetry = TelemetryContext(
        anonymous_install_id=identity.anonymous_install_id,
        invocation_id=f"invoke_{uuid.uuid4().hex}",
        daemon_session_id=_DAEMON_SESSION_ID,
        environment=telemetry_environment(),
        command=command,
        subcommand=subcommand,
        output_mode="json" if json_output else "human",
        cli_version=default_cli_release().removeprefix("potpie-cli@"),
        python_version=platform.python_version(),
        os=platform.system().lower(),
        arch=platform.machine(),
    )
    _ = _CURRENT.set(telemetry)
    return telemetry


def current_telemetry_context() -> TelemetryContext | None:
    return _CURRENT.get()


def _command_parts(ctx: typer.Context) -> tuple[str | None, str | None]:
    protected_args = _string_parts(getattr(ctx, "protected_args", ()))
    args = _string_parts(getattr(ctx, "args", ()))
    parts = protected_args + args
    command = ctx.invoked_subcommand
    if command is None and parts:
        command = parts[0]
    subcommand = parts[1] if len(parts) > 1 else None
    return command, subcommand


def _string_parts(value: object) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return [str(part) for part in value]


def _optional_command_properties(telemetry: TelemetryContext) -> dict[str, str]:
    properties: dict[str, str] = {}
    if telemetry.command is not None:
        properties["command"] = telemetry.command
    if telemetry.subcommand is not None:
        properties["subcommand"] = telemetry.subcommand
    return properties
