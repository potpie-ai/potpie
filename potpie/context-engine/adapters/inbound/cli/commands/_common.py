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
import os
import subprocess
from contextlib import contextmanager
from pathlib import Path
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
        if os.getenv("CONTEXT_ENGINE_HOST_MODE", "").strip().lower() == "in_process":
            from bootstrap.host_wiring import build_host_shell

            _state["host"] = build_host_shell()
        else:
            from host.daemon_client import RemoteHostShell

            _state["host"] = RemoteHostShell()
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
        typer.echo(human)


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
        lines = [f"error: {message}"]
        if detail:
            lines.append(f"  detail: {detail}")
        if next_action:
            lines.append(f"  next: {next_action}")
        typer.echo("\n".join(lines), err=True)
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


def resolve_pot_id(
    host: Any, explicit: str | None = None, *, infer_from_repo: bool = True
) -> str:
    """Resolve ``--pot`` ref → id, else current-repo pot, else active pot.

    ``infer_from_repo=False`` skips current-repo inference and goes straight to
    the active pot. Source registration uses this: ``source add repo .`` is the
    command that *establishes* the repo→pot mapping, so inferring its target
    from existing registrations would route the new source to the wrong pot
    (or fail as ambiguous when other pots already track the same repo).
    """
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
    matches = _pots_matching_current_repo(host) if infer_from_repo else []
    active = pots.active_pot()
    if len(matches) == 1:
        return matches[0][0]
    if len(matches) > 1:
        if active is not None and any(active.pot_id == pid for pid, _ in matches):
            return active.pot_id
        names = ", ".join(f"{name} ({pid})" for pid, name in matches)
        fail(
            code="ambiguous_pot",
            message=f"Current repo is registered in multiple pots: {names}.",
            next_action="pick one with '--pot <id-or-name>' or set it active with 'potpie pot use <id-or-name>'",
        )
    if active is None:
        fail(
            code="no_active_pot",
            message="No active pot, and the current repo is not registered as a source in any pot.",
            next_action="run 'potpie setup', or create a pot with 'potpie pot create <name> --use' and register this repo with 'potpie source add repo .'",
        )
    return active.pot_id


def _pots_matching_current_repo(host: Any) -> list[tuple[str, str]]:
    """Return ``(pot_id, name)`` for every pot whose repo source matches cwd.

    A pot is the project boundary, not a single repository. This helper only
    chooses the pot from the current working tree; it does not inject a repo
    scope into reads. Timeline queries therefore default to the whole project
    across all repositories attached to the pot. The caller decides how to
    disambiguate multiple matches (active pot wins; otherwise a structured
    ``ambiguous_pot`` error).
    """

    try:
        cwd = Path.cwd().resolve()
    except OSError:
        return []
    remote = _current_git_remote(cwd)
    matches: list[tuple[str, str]] = []
    try:
        pots = list(host.pots.list_pots())
    except Exception:  # noqa: BLE001 - pot resolution should not mask commands
        return []
    for pot in pots:
        try:
            sources = host.pots.list_sources(pot_id=pot.pot_id)
        except Exception:  # noqa: BLE001
            continue
        for source in sources:
            if getattr(source, "kind", None) != "repo":
                continue
            refs = {
                str(getattr(source, "name", "") or "").strip(),
                str(getattr(source, "location", "") or "").strip(),
            }
            if any(
                _repo_source_matches_cwd(ref, cwd=cwd, remote=remote)
                for ref in refs
                if ref
            ):
                matches.append((pot.pot_id, pot.name))
                break
    return matches


def _repo_source_matches_cwd(source_name: str, *, cwd: Path, remote: str | None) -> bool:
    if not source_name:
        return False
    source_path = Path(source_name).expanduser()
    if source_path.is_absolute() or source_name.startswith((".", "~")):
        try:
            resolved = source_path.resolve(strict=False)
        except OSError:
            resolved = source_path.absolute()
        if cwd == resolved or cwd.is_relative_to(resolved) or resolved.is_relative_to(cwd):
            return True

    normalized_source = _normalize_repo_ref(source_name)
    return bool(remote and normalized_source and normalized_source == remote)


def _current_git_remote(cwd: Path) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "config", "--get", "remote.origin.url"],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=1,
            check=False,
        )
    except Exception:  # noqa: BLE001
        return None
    if proc.returncode != 0:
        return None
    return _normalize_repo_ref(proc.stdout.strip())


def _normalize_repo_ref(value: str) -> str | None:
    text = value.strip()
    if not text:
        return None
    if text.endswith(".git"):
        text = text[:-4]
    if text.startswith("git@"):
        # git@github.com:owner/repo
        text = text[4:].replace(":", "/", 1)
    elif "://" in text:
        from urllib.parse import urlparse

        parsed = urlparse(text)
        path = parsed.path.strip("/")
        if parsed.netloc and path:
            text = f"{parsed.netloc}/{path}"
    return text.strip("/").lower().replace(" ", "-") or None


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
