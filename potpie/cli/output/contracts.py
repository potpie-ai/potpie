"""Uniform success, error, metadata, and list envelopes for CLI JSON output."""

from __future__ import annotations

from collections.abc import Mapping
from contextvars import ContextVar
from functools import wraps
from typing import Any

import click

CLI_SCHEMA_VERSION = "1"
_CURRENT_COMMAND: ContextVar[str | None] = ContextVar(
    "potpie_cli_output_command",
    default=None,
)

_LIST_KEYS = (
    "items",
    "pots",
    "sources",
    "skills",
    "integrations",
    "workspaces",
    "projects",
    "spaces",
    "repos",
)
_LIST_COMMANDS = {
    "integration.list",
    "ledger.sources",
    "pot.list",
    "skills.list",
    "source.list",
}


def current_command() -> str:
    """Return the current dotted Click command path without the program name."""

    bound = _CURRENT_COMMAND.get()
    if bound:
        return bound

    context = click.get_current_context(silent=True)
    parts: list[str] = []
    while context is not None:
        name = str(context.info_name or "").strip()
        if name and name not in {"potpie", "root"}:
            parts.append(name)
        context = context.parent
    if parts:
        return ".".join(reversed(parts))

    try:
        from potpie.cli.telemetry.context import current_telemetry_context

        telemetry = current_telemetry_context()
    except Exception:  # noqa: BLE001 - output formatting must not mask a result
        telemetry = None
    if telemetry is None or telemetry.command is None:
        return "unknown"
    values = [telemetry.command]
    if telemetry.subcommand:
        values.append(telemetry.subcommand)
    return ".".join(values)


def bind_command_paths(typer_app: Any) -> None:
    """Bind every registered Typer callback to its stable dotted command path."""

    def bind(app: Any, path: tuple[str, ...] = ()) -> None:
        for command in app.registered_commands:
            callback = command.callback
            if callback is None:
                continue
            name = str(command.name or callback.__name__.replace("_", "-"))
            command_path = ".".join((*path, name))
            if getattr(callback, "__potpie_command_path__", None) == command_path:
                continue

            @wraps(callback)
            def wrapped(
                *args: Any,
                __callback: Any = callback,
                __path: str = command_path,
                **kwargs: Any,
            ) -> Any:
                token = _CURRENT_COMMAND.set(__path)
                try:
                    return __callback(*args, **kwargs)
                finally:
                    _CURRENT_COMMAND.reset(token)

            wrapped.__potpie_command_path__ = command_path
            command.callback = wrapped
        for group in app.registered_groups:
            bind(group.typer_instance, (*path, str(group.name)))

    bind(typer_app)


def current_runtime_mode() -> str:
    """Report the selected product runtime mode without constructing an engine."""

    try:
        from potpie.cli.commands import _common

        runtime = _common._state.get("runtime")
        if runtime is not None:
            return str(runtime.settings.runtime_mode)
        host = _common._state.get("host")
        host_runtime = getattr(host, "runtime", None)
        if host_runtime is not None:
            return str(host_runtime.settings.runtime_mode)
        from potpie.runtime.settings import ProductSettings

        return ProductSettings.load().runtime_mode
    except Exception:  # noqa: BLE001 - metadata must not make output fail
        return "daemon"


def success_envelope(
    data: Mapping[str, Any] | None,
    *,
    command: str | None = None,
    request_id: str | None = None,
) -> dict[str, Any]:
    """Wrap one successful command result in schema version 1."""

    command_name = command or current_command()
    payload = dict(data or {})
    payload = _normalize_list_data(command_name, payload)
    return {
        "ok": True,
        "data": payload,
        "meta": _meta(command_name, request_id=request_id),
    }


def error_envelope(
    *,
    code: str,
    message: str,
    details: Any = None,
    retryable: bool = False,
    recommended_next_action: Mapping[str, Any] | str | None = None,
    command: str | None = None,
    request_id: str | None = None,
) -> dict[str, Any]:
    """Wrap one failed command result in schema version 1."""

    command_name = command or current_command()
    action = _recommended_action(recommended_next_action, reason=message)
    return {
        "ok": False,
        "error": {
            "code": code,
            "message": message,
            "details": details if details is not None else {},
            "retryable": bool(retryable),
            "recommended_next_action": action,
        },
        "meta": _meta(command_name, request_id=request_id),
    }


def _meta(command: str, *, request_id: str | None) -> dict[str, Any]:
    return {
        "schema_version": CLI_SCHEMA_VERSION,
        "command": command,
        "runtime_mode": current_runtime_mode(),
        "request_id": request_id,
    }


def _recommended_action(
    value: Mapping[str, Any] | str | None,
    *,
    reason: str,
) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        command = value.get("command")
        if not command:
            return None
        return {
            "command": str(command),
            "reason": str(value.get("reason") or reason),
        }
    command = str(value).strip()
    if not command:
        return None
    return {"command": command, "reason": reason}


def _normalize_list_data(command: str, data: dict[str, Any]) -> dict[str, Any]:
    is_list = command in _LIST_COMMANDS or command.endswith(".list")
    if not is_list:
        return data
    for key in _LIST_KEYS:
        value = data.get(key)
        if not isinstance(value, (list, tuple)):
            continue
        normalized = {name: item for name, item in data.items() if name != key}
        normalized["items"] = list(value)
        normalized["count"] = int(normalized.get("count", len(value)))
        normalized["next_cursor"] = normalized.get("next_cursor")
        return normalized
    return data


__all__ = [
    "CLI_SCHEMA_VERSION",
    "bind_command_paths",
    "current_command",
    "current_runtime_mode",
    "error_envelope",
    "success_envelope",
]
