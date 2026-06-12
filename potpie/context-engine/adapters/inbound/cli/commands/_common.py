"""Shared CLI plumbing for the host-routed command surface.

Every command in this package routes ``CLI -> HostShell -> service(s) -> ports``.
This module owns the cross-cutting concerns so the command bodies stay thin:

- one cached ``HostShell`` per process (``get_host``);
- ``--json`` output state + ``emit`` / ``fail`` helpers;
- the ``contract()`` error boundary that maps domain errors to the documented
  exit codes (0 ok / 1 validation / 2 unavailable / 3 degraded / 4 auth) and the
  structured JSON error shape (``code``/``message``/``detail``/
  ``recommended_next_action``);
- active-pot resolution.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from typing import Any, Iterator

import typer

from domain.errors import (
    CapabilityNotImplemented,
    ContextEngineDisabled,
    PotNotFound,
)
from domain.ports.cli_auth.credentials import CredentialStore

# --- exit codes (cli-flow.md output contract) -------------------------------
EXIT_OK = 0
EXIT_VALIDATION = 1
EXIT_UNAVAILABLE = 2
EXIT_DEGRADED = 3
EXIT_AUTH = 4

_state: dict[str, Any] = {"json": False, "verbose": False, "host": None, "store": None}


def set_json(value: bool) -> None:
    _state["json"] = bool(value)


def is_json() -> bool:
    return bool(_state["json"])


def set_verbose(value: bool) -> None:
    _state["verbose"] = bool(value)


def is_verbose() -> bool:
    return bool(_state["verbose"])


def get_host():
    """Return the process-wide ``HostShell`` (built lazily)."""
    if _state["host"] is None:
        from bootstrap.host_wiring import build_host_shell

        _state["host"] = build_host_shell()
    return _state["host"]


def set_host(host: Any) -> None:
    """Inject a host (tests / alternate wiring)."""
    _state["host"] = host


def get_store() -> CredentialStore:
    """Return the process-wide ``CredentialStore`` (built lazily).

    The auth/credential subsystem persists through this domain port; the concrete
    is chosen at the composition root (``bootstrap.cli_auth_wiring``), so this
    inbound module never imports an adapter. The default is the real
    keychain-backed store; tests inject an in-memory fake via ``set_store``.
    """
    if _state["store"] is None:
        from bootstrap.cli_auth_wiring import build_credential_store

        _state["store"] = build_credential_store()
    return _state["store"]


def set_store(store: CredentialStore) -> None:
    """Inject a credential store (tests / alternate wiring)."""
    _state["store"] = store


def emit(payload: dict[str, Any], *, human: str) -> None:
    """Emit a success result: JSON when ``--json``, else a human line."""
    if is_json():
        typer.echo(json.dumps(payload, default=str))
    else:
        from adapters.inbound.cli.ui.format import print_human_block

        print_human_block(human)


def fail(
    *,
    code: str,
    message: str,
    detail: str | None = None,
    next_action: str | None = None,
    exit_code: int = EXIT_VALIDATION,
) -> None:
    """Emit the structured error contract and exit with the given code."""
    if is_json():
        typer.echo(
            json.dumps(
                {
                    "code": code,
                    "message": message,
                    "detail": detail,
                    "recommended_next_action": next_action,
                },
                default=str,
            )
        )
    else:
        from adapters.inbound.cli.ui.format import print_structured_error

        print_structured_error(
            title=message,
            message=message,
            hint=detail,
            next_action=next_action,
        )
    raise typer.Exit(code=exit_code)


@contextmanager
def contract() -> Iterator[None]:
    """Error boundary: map domain errors to the documented exit codes.

    No command should leak a traceback; an unbuilt capability returns the
    structured not-implemented contract (exit 2) rather than crashing.
    """
    try:
        yield
    except CapabilityNotImplemented as exc:
        fail(
            code="not_implemented",
            message=str(exc),
            detail=exc.detail,
            next_action=exc.recommended_next_action,
            exit_code=EXIT_UNAVAILABLE,
        )
    except ContextEngineDisabled as exc:
        fail(
            code="unavailable",
            message=str(exc),
            next_action="check backend/daemon readiness with 'potpie doctor'",
            exit_code=EXIT_UNAVAILABLE,
        )
    except PotNotFound as exc:
        fail(
            code="pot_not_found",
            message=str(exc),
            next_action="list pots with 'potpie pot list' or create one with 'potpie setup'",
            exit_code=EXIT_VALIDATION,
        )
    except ValueError as exc:
        fail(code="validation_error", message=str(exc), exit_code=EXIT_VALIDATION)
    except typer.Exit:
        raise
    except Exception as exc:  # noqa: BLE001
        from adapters.inbound.cli.sentry_runtime import capture_unexpected_cli_error

        capture_unexpected_cli_error(
            exc,
            error_code="unexpected_cli_error",
            error_kind="unexpected",
        )
        fail(
            code="unexpected_cli_error",
            message="Unexpected internal error.",
            exit_code=EXIT_VALIDATION,
        )


def resolve_pot_id(host: Any, explicit: str | None = None) -> str:
    """Resolve ``--pot`` ref → id, else the active pot. Fails (exit 1) if none."""
    pots = host.pots
    if explicit:
        for pot in pots.list_pots():
            if explicit in (pot.pot_id, pot.name):
                return pot.pot_id
        fail(
            code="pot_not_found",
            message=f"No pot matching '{explicit}'.",
            next_action="run 'potpie pot list'",
        )
    active = pots.active_pot()
    if active is None:
        fail(
            code="no_active_pot",
            message="No active pot.",
            next_action="run 'potpie setup' to create and activate a pot",
        )
    return active.pot_id


__all__ = [
    "EXIT_AUTH",
    "EXIT_DEGRADED",
    "EXIT_OK",
    "EXIT_UNAVAILABLE",
    "EXIT_VALIDATION",
    "contract",
    "emit",
    "fail",
    "get_host",
    "get_store",
    "is_json",
    "is_verbose",
    "resolve_pot_id",
    "set_host",
    "set_store",
    "set_json",
    "set_verbose",
]
