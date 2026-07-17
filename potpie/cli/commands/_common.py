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
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Final, Iterator, NoReturn, Sequence

import click
import typer

from potpie.cli.repo_location import (
    current_git_remote as shared_current_git_remote,
    normalize_repo_ref as shared_normalize_repo_ref,
    repo_identity_key,
)
from potpie_context_core.domain.errors import (
    CapabilityNotImplemented,
    ContextEngineDisabled,
    PotNotFound,
)
from potpie_context_engine.domain.ports.cli_auth.credentials import CredentialStore

# --- exit codes (cli-flow.md output contract) -------------------------------
EXIT_OK = 0
EXIT_VALIDATION = 1
EXIT_UNAVAILABLE = 2
EXIT_DEGRADED = 3
EXIT_AUTH = 4

_state: dict[str, Any] = {
    "json": False,
    "verbose": False,
    "host": None,
    "store": None,
    "json_error_formatter": None,
}
_CLI_METRIC_ATTRIBUTE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "arch",
        "cli_version",
        "command",
        "error_code",
        "os",
        "output_mode",
        "result",
        "subcommand",
    }
)


def set_json(value: bool) -> None:
    _state["json"] = bool(value)


def is_json() -> bool:
    return bool(_state["json"])


def bootstrap_output_flags_from_argv(argv: Sequence[str] | None = None) -> None:
    """Apply ``--json`` / ``--verbose`` before Typer finishes parsing.

    The root callback only runs after a command line parses successfully. Early
    bootstrap keeps parse-time failures on the documented JSON error contract.
    """
    args = tuple(argv or ())
    scan_args = args
    if "--" in args:
        scan_args = args[: args.index("--")]
    set_json("--json" in scan_args)
    set_verbose("--verbose" in scan_args or "-v" in scan_args)


def set_verbose(value: bool) -> None:
    _state["verbose"] = bool(value)


def is_verbose() -> bool:
    return bool(_state["verbose"])


def get_host():
    """Return the process-wide ``HostShell`` (built lazily)."""
    if _state["host"] is None:
        mode = os.getenv("CONTEXT_ENGINE_HOST_MODE", "").strip().lower()
        if mode != "in_process":
            try:
                from potpie.daemon.daemon_client import RemoteHostShell
            except ModuleNotFoundError as exc:
                if exc.name != "potpie.daemon.daemon_client":
                    raise
            else:
                _state["host"] = RemoteHostShell()
                return _state["host"]

        from potpie.services.host_wiring import build_host_shell

        _state["host"] = build_host_shell()
    return _state["host"]


def set_host(host: Any) -> None:
    """Inject a host (tests / alternate wiring)."""
    _state["host"] = host


def get_store() -> CredentialStore:
    """Return the process-wide ``CredentialStore`` (built lazily).

    The auth/credential subsystem persists through this domain port; the concrete
    is chosen at the composition root (``potpie_context_engine.bootstrap.cli_auth_wiring``), so this
    inbound module never imports an adapter. The default is the real
    file-backed store; tests inject an in-memory fake via ``set_store``.
    """
    if _state["store"] is None:
        from potpie_context_engine.bootstrap.cli_auth_wiring import build_credential_store

        _state["store"] = build_credential_store()
    return _state["store"]


def set_store(store: CredentialStore) -> None:
    """Inject a credential store (tests / alternate wiring)."""
    _state["store"] = store


@contextmanager
def json_error_formatter(
    formatter: Callable[[dict[str, Any]], dict[str, Any]] | None,
) -> Iterator[None]:
    """Temporarily wrap JSON errors emitted by ``fail``.

    This lets command groups with stricter envelopes, such as ``potpie graph``,
    reuse the shared error boundary without changing every other CLI command's
    documented error contract.
    """
    old = _state.get("json_error_formatter")
    _state["json_error_formatter"] = formatter
    try:
        yield
    finally:
        _state["json_error_formatter"] = old


def emit(payload: dict[str, Any], *, human: str) -> None:
    """Emit a success result: JSON when ``--json``, else a human line."""
    if is_json():
        typer.echo(json.dumps(payload, default=str))
    else:
        from potpie.cli.ui.format import print_human_block

        print_human_block(human)


def fail(
    *,
    code: str,
    message: str,
    detail: Any = None,
    next_action: str | None = None,
    exit_code: int = EXIT_VALIDATION,
) -> NoReturn:
    """Emit the structured error contract and exit with the given code."""
    if is_json():
        payload = {
            "code": code,
            "message": message,
            "detail": detail,
            "recommended_next_action": next_action,
        }
        formatter = _state.get("json_error_formatter")
        if callable(formatter):
            payload = formatter(payload)
        typer.echo(
            json.dumps(
                payload,
                default=str,
            )
        )
    else:
        from potpie.cli.ui.format import print_structured_error

        print_structured_error(
            title=message,
            message=message,
            # Structured (mapping) detail is for the JSON envelope; the human
            # message already carries the guidance in prose.
            hint=detail if isinstance(detail, str) else None,
            next_action=next_action,
        )
    raise typer.Exit(code=exit_code)


@contextmanager
def contract() -> Iterator[None]:
    """Error boundary: map domain errors to the documented exit codes.

    No command should leak a traceback; an unbuilt capability returns the
    structured not-implemented contract (exit 2) rather than crashing.
    """
    start = time.perf_counter()
    result = "ok"
    error_code = "none"
    try:
        yield
    except CapabilityNotImplemented as exc:
        result = "not_implemented"
        error_code = "not_implemented"
        fail(
            code="not_implemented",
            message=str(exc),
            detail=exc.detail,
            next_action=exc.recommended_next_action,
            exit_code=EXIT_UNAVAILABLE,
        )
    except ContextEngineDisabled as exc:
        result = "unavailable"
        error_code = "unavailable"
        fail(
            code="unavailable",
            message=str(exc),
            next_action="check backend/daemon readiness with 'potpie doctor'",
            exit_code=EXIT_UNAVAILABLE,
        )
    except PotNotFound as exc:
        result = "pot_not_found"
        error_code = "pot_not_found"
        fail(
            code="pot_not_found",
            message=str(exc),
            next_action="list pots with 'potpie pot list' or create one with 'potpie setup'",
            exit_code=EXIT_VALIDATION,
        )
    except ValueError as exc:
        result = "validation_error"
        error_code = "validation_error"
        # Domain errors may carry structured guidance (e.g. UnknownGraphViewError's
        # detail.did_you_mean) for the JSON envelope.
        fail(
            code="validation_error",
            message=str(exc),
            detail=getattr(exc, "detail", None),
            next_action=getattr(exc, "recommended_next_action", None),
            exit_code=EXIT_VALIDATION,
        )
    except typer.Exit:
        result = "exit"
        error_code = "exit"
        raise
    except (KeyboardInterrupt, EOFError):
        raise
    except Exception as exc:  # noqa: BLE001
        if isinstance(exc, click.Abort) or type(exc).__name__ == "Abort":
            raise
        result = "unexpected"
        error_code = "unexpected_cli_error"
        from potpie.cli.telemetry.sentry_runtime import (
            capture_unexpected_cli_error,
        )

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
    finally:
        _record_cli_contract_metrics(
            started_at=start,
            result=result,
            error_code=error_code,
        )


def _record_cli_contract_metrics(
    *,
    started_at: float,
    result: str,
    error_code: str,
) -> None:
    from potpie_context_engine.bootstrap import sentry_metrics_runtime

    attributes = _cli_metric_attributes(result=result, error_code=error_code)
    duration_ms = max((time.perf_counter() - started_at) * 1000.0, 0.0)
    try:
        sentry_metrics_runtime.count(
            "ce.cli.invocations_total",
            attributes=attributes,
        )
        sentry_metrics_runtime.distribution(
            "ce.cli.duration_ms",
            duration_ms,
            unit="millisecond",
            attributes=attributes,
        )
    except Exception:  # noqa: BLE001
        pass
    finally:
        try:
            sentry_metrics_runtime.flush(timeout=2.0)
        except Exception:  # noqa: BLE001
            pass


def _cli_metric_attributes(
    *,
    result: str,
    error_code: str,
) -> dict[str, str | int | float | bool]:
    from potpie.cli.telemetry.context import current_telemetry_context

    telemetry = current_telemetry_context()
    attributes: dict[str, str | int | float | bool] = {
        "error_code": error_code,
        "result": result,
    }
    if telemetry is None:
        return attributes
    for key, value in telemetry.fields().items():
        if key in _CLI_METRIC_ATTRIBUTE_KEYS:
            attributes[key] = value
    return attributes


def resolve_pot_scope(
    host: Any, explicit: str | None = None, *, infer_from_repo: bool = True
) -> tuple[str, str]:
    """Resolve pot id and how it was chosen for CLI scope hints.

    ``resolved_via`` is one of ``explicit``, ``repo_default``, ``linked_repo``, or
    ``active_pot``. See :func:`resolve_pot_id` for resolution order.

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
                return pot.pot_id, "explicit"
        fail(
            code="pot_not_found",
            message=f"No pot matching '{explicit}'.",
            next_action="run 'potpie pot list'",
        )
    repo_identity = _current_repo_identity() if infer_from_repo else None
    default_pot = _repo_default_pot_id(host, repo_identity)
    if default_pot:
        return default_pot, "repo_default"
    matches = _pots_matching_current_repo(host) if infer_from_repo else []
    active = pots.active_pot()
    if len(matches) == 1:
        return matches[0][0], "linked_repo"
    if len(matches) > 1:
        if active is not None and any(active.pot_id == pid for pid, _ in matches):
            return active.pot_id, "active_pot"
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
    return active.pot_id, "active_pot"


def resolve_pot_id(
    host: Any, explicit: str | None = None, *, infer_from_repo: bool = True
) -> str:
    """Resolve ``--pot`` ref → id, else current-repo pot, else active pot."""
    pot_id, _ = resolve_pot_scope(host, explicit, infer_from_repo=infer_from_repo)
    return pot_id


def current_repo_identity_for_cli() -> str | None:
    return _current_repo_identity()


def repo_pot_candidates(
    host: Any, repo: str | None = None, *, include_counts: bool = True
) -> dict[str, Any]:
    repo_identity = _repo_identity_from_option(repo)
    matches = (
        _pots_matching_current_repo(host)
        if repo in (None, "", ".", "current")
        else _pots_matching_repo_identity(host, repo_identity)
    )
    default_pot_id = _repo_default_pot_id(host, repo_identity)
    active = _safe_call(lambda: host.pots.active_pot(), None)
    rows: list[dict[str, Any]] = []
    for pot_id, name in matches:
        row = {
            "pot_id": pot_id,
            "name": name,
            "active": bool(
                active is not None and getattr(active, "pot_id", None) == pot_id
            ),
            "default": bool(default_pot_id == pot_id),
            "source_count": pot_source_count(host, pot_id),
        }
        if include_counts:
            row["counts"] = pot_graph_counts(host, pot_id)
        rows.append(row)
    return {
        "repo": repo_identity,
        "default_pot_id": default_pot_id,
        "candidates": rows,
    }


def repo_effective_pot_info(host: Any, repo: str | None = None) -> dict[str, Any]:
    repo_identity = _repo_identity_from_option(repo)
    matches = (
        _pots_matching_current_repo(host)
        if repo in (None, "", ".", "current")
        else _pots_matching_repo_identity(host, repo_identity)
    )
    default_pot_id = _repo_default_pot_id(host, repo_identity)
    active = _safe_call(lambda: host.pots.active_pot(), None)
    active_id = getattr(active, "pot_id", None) if active is not None else None
    match_ids = {pot_id for pot_id, _ in matches}

    effective_id: str | None = None
    reason = "unresolved"
    status = "resolved"
    if default_pot_id:
        effective_id = default_pot_id
        reason = "repo_default"
    elif len(matches) == 1:
        effective_id = matches[0][0]
        reason = "single_linked_repo_pot"
    elif len(matches) > 1:
        if active_id and active_id in match_ids:
            effective_id = str(active_id)
            reason = "active_linked_repo_pot"
        else:
            status = "ambiguous"
            reason = "multiple_linked_repo_pots"
    elif active_id:
        effective_id = str(active_id)
        reason = "active_pot"
    else:
        status = "unresolved"
        reason = "no_active_pot"

    return {
        "repo": repo_identity,
        "active_pot_id": active_id,
        "default_pot_id": default_pot_id,
        "effective_pot": _pot_summary(host, effective_id) if effective_id else None,
        "reason": reason,
        "status": status,
        "candidates": [
            {"id": pot_id, "name": name, "active": bool(active_id == pot_id)}
            for pot_id, name in matches
        ],
    }


def repo_effective_pot_human(routing: dict[str, Any]) -> str | None:
    repo = routing.get("repo")
    if not repo:
        return None
    effective = routing.get("effective_pot")
    if effective:
        reason = _ROUTING_REASON_LABELS.get(
            str(routing.get("reason") or ""), str(routing.get("reason") or "")
        )
        return (
            f"current repo effective pot: {effective.get('name')} "
            f"({effective.get('id')}) via {reason}"
        )
    if routing.get("status") == "ambiguous":
        names = ", ".join(
            f"{row.get('name')} ({row.get('id')})"
            for row in routing.get("candidates", ())
        )
        return f"current repo effective pot: ambiguous ({names})"
    return "current repo effective pot: (unresolved)"


def repo_default_mismatch_warning(
    host: Any, routing: dict[str, Any], *, selected_pot_id: str
) -> str | None:
    default_id = routing.get("default_pot_id")
    repo = routing.get("repo")
    if not default_id or default_id == selected_pot_id or not repo:
        return None
    default = _pot_summary(host, str(default_id))
    return (
        f"repo {repo} default remains {default.get('name')} ({default_id}); "
        "repo-scoped commands here use that pot. Run "
        f"`potpie pot default set --repo current {selected_pot_id}` "
        "or retry with `--also-default-for-current-repo` to switch both."
    )


def pot_scope_info(host: Any, pot_id: str) -> dict[str, Any]:
    pot = _pot_for_id(host, pot_id)
    return {
        "id": pot_id,
        "name": getattr(pot, "name", pot_id) if pot is not None else pot_id,
        "active": bool(getattr(pot, "active", False)) if pot is not None else False,
        "source_count": pot_source_count(host, pot_id),
        "counts": pot_graph_counts(host, pot_id),
    }


def pot_scope_resolution_human(resolved_via: str, *, repo: str | None = None) -> str:
    if resolved_via == "explicit":
        return "via --pot"
    if resolved_via == "repo_default":
        return f"via repo default for {repo}" if repo else "via repo default"
    if resolved_via == "linked_repo":
        return f"via linked repo {repo}" if repo else "via linked repo"
    return "via active pot"


def pot_scope_human(
    host: Any,
    pot_id: str,
    *,
    resolved_via: str | None = None,
    repo: str | None = None,
) -> str:
    info = pot_scope_info(host, pot_id)
    counts = info.get("counts") or {}
    claims = counts.get("claims", 0)
    entities = counts.get("entities", 0)
    scope = (
        f"pot={info.get('name')} ({pot_id}) "
        f"sources={info.get('source_count', 0)} claims={claims} entities={entities}"
    )
    if resolved_via:
        scope += f" {pot_scope_resolution_human(resolved_via, repo=repo)}"
    return scope


def _known_claims_count(counts: dict[str, int]) -> int | None:
    if "claims" not in counts:
        return None
    return int(counts["claims"])


def _pot_claims_count(host: Any, pot_id: str) -> int | None:
    return _known_claims_count(pot_graph_counts(host, pot_id))


def empty_pot_warnings(
    host: Any, pot_id: str, repo: str | None = None
) -> tuple[str, ...]:
    claims = _pot_claims_count(host, pot_id)
    if claims is None or claims != 0:
        return ()
    linked = repo_pot_candidates(host, repo)
    alternatives = [
        row
        for row in linked.get("candidates", ())
        if row.get("pot_id") != pot_id
        and (alt_claims := _known_claims_count(row.get("counts") or {})) is not None
        and alt_claims > 0
    ]
    if not alternatives:
        return ()
    best = sorted(
        alternatives,
        key=lambda row: _known_claims_count(row.get("counts") or {}) or 0,
        reverse=True,
    )[0]
    claims = _known_claims_count(best.get("counts") or {}) or 0
    return (
        (
            f"current pot has 0 claims; repo {linked.get('repo')} also links to "
            f"{best.get('name')} ({best.get('pot_id')}) with {claims} claims. "
            f"Retry with --pot {best.get('pot_id')} or run "
            f"`potpie pot default set --repo current {best.get('pot_id')}`."
        ),
    )


def empty_pot_guidance(
    host: Any, pot_id: str, repo: str | None = None
) -> tuple[str, ...]:
    """Recovery hints when a pot has no graph claims yet."""
    warnings = list(empty_pot_warnings(host, pot_id, repo))
    claims = _pot_claims_count(host, pot_id)
    if claims is None or claims != 0:
        return tuple(warnings)
    if pot_source_count(host, pot_id) > 0:
        warnings.append(
            "pot has registered sources but 0 claims; next: run harness-led ingestion "
            "(agent skills + `potpie graph propose/commit`), switch with "
            "`potpie pot use <id>` or inspect `potpie pot linked --repo current`, "
            "or keep this empty pot intentionally."
        )
    elif not warnings:
        warnings.append(
            "pot has 0 claims and no sources; next: `potpie source add repo .` "
            "then harness-led ingestion, or keep this empty pot intentionally."
        )
    return tuple(warnings)


def enrich_with_pot_guidance(
    host: Any,
    pot_id: str,
    payload: dict[str, Any],
    *,
    human: str,
    repo: str | None = None,
) -> tuple[dict[str, Any], str]:
    warnings = empty_pot_guidance(host, pot_id, repo)
    existing = payload.get("warnings") or []
    existing_warnings = [existing] if isinstance(existing, str) else list(existing)
    combined_warnings = [*existing_warnings, *warnings]
    if not combined_warnings:
        return payload, human
    human_lines = [human, *(f"! {warning}" for warning in warnings)]
    return (
        {
            **payload,
            "warnings": combined_warnings,
            "recommended_next_action": payload.get("recommended_next_action")
            or combined_warnings[0],
        },
        "\n".join(human_lines),
    )


def use_pot_selection(
    host: Any,
    ref: str,
    *,
    also_default_for_current_repo: bool = False,
    origin: str | None = None,
) -> tuple[dict[str, Any], str]:
    repo_key = None
    if also_default_for_current_repo:
        repo_key = current_repo_identity_for_cli()
        if not repo_key:
            raise ValueError("--also-default-for-current-repo requires a repo")

    pot = host.pots.use_pot(ref=ref)
    repo_default_set = False
    if repo_key:
        host.pots.set_repo_default(repo=repo_key, pot_id=pot.pot_id)
        repo_default_set = True

    routing = repo_effective_pot_info(host)
    warnings = []
    warning = repo_default_mismatch_warning(host, routing, selected_pot_id=pot.pot_id)
    if warning:
        warnings.append(warning)

    active_line = f"active pot → {pot.name}"
    if origin:
        active_line = f"{active_line} ({origin})"
    lines = [active_line]
    if repo_default_set:
        lines.append(f"repo {repo_key} default → {pot.name} ({pot.pot_id})")
    lines.extend(f"warning: {item}" for item in warnings)

    payload: dict[str, Any] = {
        "id": pot.pot_id,
        "name": pot.name,
        "repo_default_set": repo_default_set,
        "current_repo": routing,
        "warnings": warnings,
    }
    if origin:
        payload["origin"] = origin

    return enrich_with_pot_guidance(
        host,
        pot.pot_id,
        payload,
        human="\n".join(lines),
        repo=repo_key,
    )


def pot_graph_counts(host: Any, pot_id: str) -> dict[str, int]:
    graph = getattr(host, "graph", None)
    if graph is None:
        return {}
    status = _safe_call(lambda: graph.data_plane_status(pot_id), None)
    if status is None:
        return {}
    counts = getattr(status, "counts", {}) or {}
    out: dict[str, int] = {}
    for key, value in dict(counts).items():
        try:
            out[str(key)] = int(value)
        except (TypeError, ValueError):
            continue
    return out


def pot_source_count(host: Any, pot_id: str) -> int:
    return len(_safe_call(lambda: host.pots.list_sources(pot_id=pot_id), []) or [])


def _repo_identity_from_option(repo: str | None) -> str | None:
    return (
        _current_repo_identity()
        if repo in (None, "", ".", "current")
        else repo_identity_key(repo or "")
    )


def _pot_summary(host: Any, pot_id: str | None) -> dict[str, Any] | None:
    if not pot_id:
        return None
    pot = _pot_for_id(host, pot_id)
    return {
        "id": pot_id,
        "name": getattr(pot, "name", pot_id) if pot is not None else pot_id,
        "active": bool(getattr(pot, "active", False)) if pot is not None else False,
    }


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


def _pots_matching_repo_identity(
    host: Any, repo_identity: str | None
) -> list[tuple[str, str]]:
    if not repo_identity:
        return []
    matches: list[tuple[str, str]] = []
    try:
        pots = list(host.pots.list_pots())
    except Exception:  # noqa: BLE001
        return []
    for pot in pots:
        try:
            sources = host.pots.list_sources(pot_id=pot.pot_id)
        except Exception:  # noqa: BLE001
            continue
        for source in sources:
            if getattr(source, "kind", None) != "repo":
                continue
            refs = (
                str(getattr(source, "name", "") or "").strip(),
                str(getattr(source, "location", "") or "").strip(),
            )
            if any(repo_identity_key(ref) == repo_identity for ref in refs if ref):
                matches.append((pot.pot_id, pot.name))
                break
    return matches


def _current_repo_identity() -> str | None:
    try:
        cwd = Path.cwd().resolve()
    except OSError:
        return None
    return _current_git_remote(cwd) or str(cwd)


def repo_default_pot_id(host: Any, repo_identity: str | None) -> str | None:
    """Return the locally persisted default pot id for a repo identity, if valid."""
    if not repo_identity:
        return None
    getter = getattr(host.pots, "repo_default", None)
    if not callable(getter):
        return None
    pot_id = _safe_call(lambda: getter(repo=repo_identity), None)
    if not pot_id:
        return None
    pot_id = str(pot_id)
    return pot_id if _pot_for_id(host, pot_id) is not None else None


def repo_default_matches(host: Any, repo_key: str | None, pot_id: str) -> bool:
    """True when ``repo_key``'s repo default is set to ``pot_id``."""
    default_pot = repo_default_pot_id(host, repo_key)
    return bool(default_pot and default_pot == pot_id)


def _repo_default_pot_id(host: Any, repo_identity: str | None) -> str | None:
    return repo_default_pot_id(host, repo_identity)


def _pot_for_id(host: Any, pot_id: str):
    for pot in _safe_call(lambda: host.pots.list_pots(), []) or []:
        if getattr(pot, "pot_id", None) == pot_id:
            return pot
    return None


def _safe_call(fn, default):
    try:
        return fn()
    except Exception:  # noqa: BLE001
        return default


def _repo_source_matches_cwd(
    source_name: str, *, cwd: Path, remote: str | None
) -> bool:
    if not source_name:
        return False
    source_path = Path(source_name).expanduser()
    if source_path.is_absolute() or source_name.startswith((".", "~")):
        try:
            resolved = source_path.resolve(strict=False)
        except OSError:
            resolved = source_path.absolute()
        if (
            cwd == resolved
            or cwd.is_relative_to(resolved)
            or resolved.is_relative_to(cwd)
        ):
            return True

    normalized_source = _normalize_repo_ref(source_name)
    return bool(remote and normalized_source and normalized_source == remote)


def _current_git_remote(cwd: Path) -> str | None:
    return shared_current_git_remote(cwd)


def _normalize_repo_ref(value: str) -> str | None:
    return shared_normalize_repo_ref(value)


_ROUTING_REASON_LABELS: Final[dict[str, str]] = {
    "repo_default": "repo default",
    "single_linked_repo_pot": "single linked repo pot",
    "active_linked_repo_pot": "active linked repo pot",
    "active_pot": "active pot",
}


__all__ = [
    "EXIT_AUTH",
    "EXIT_DEGRADED",
    "EXIT_OK",
    "EXIT_UNAVAILABLE",
    "EXIT_VALIDATION",
    "bootstrap_output_flags_from_argv",
    "contract",
    "emit",
    "fail",
    "get_host",
    "get_store",
    "current_repo_identity_for_cli",
    "empty_pot_guidance",
    "empty_pot_warnings",
    "enrich_with_pot_guidance",
    "is_json",
    "is_verbose",
    "pot_graph_counts",
    "pot_scope_human",
    "pot_scope_info",
    "pot_scope_resolution_human",
    "pot_source_count",
    "repo_default_matches",
    "repo_default_mismatch_warning",
    "repo_default_pot_id",
    "repo_effective_pot_human",
    "repo_effective_pot_info",
    "repo_pot_candidates",
    "resolve_pot_id",
    "resolve_pot_scope",
    "set_host",
    "set_store",
    "set_json",
    "set_verbose",
    "use_pot_selection",
]
