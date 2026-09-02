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
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Final, Iterator, NoReturn, Sequence

import click
import typer

from potpie.cli.host_snapshot import invalidate_host_snapshot, memoized
from potpie.cli.repo_location import (
    REPO_MATCH_CONTAINED,
    classify_repo_source_match,
    current_git_remote as shared_current_git_remote,
    normalize_repo_ref as shared_normalize_repo_ref,
    repo_identity_key,
)
from potpie_context_core.errors import (
    CapabilityNotImplemented,
    ContextEngineDisabled,
    PotArchived,
    PotNameConflict,
    PotNotFound,
    SourceNotFound,
)
from potpie_context_engine.domain.ports.cli_auth.credentials import CredentialStore

# --- exit codes (cli-flow.md output contract) -------------------------------
EXIT_OK = 0
EXIT_VALIDATION = 1
EXIT_UNAVAILABLE = 2
EXIT_DEGRADED = 3
EXIT_AUTH = 4

#: Statuses that mean a host answered and refused the credential.
#:
#: Duplicated from ``potpie.daemon.client._CREDENTIAL_REFUSED`` on purpose: the
#: inbound CLI has to classify an exception it is handed, and importing the
#: transport that produced it to do so would tie the one error boundary every
#: command shares to one particular adapter. The adapter-level test in
#: tests/unit/test_cli_error_contract.py is what keeps the two in step.
CREDENTIAL_REFUSED_STATUSES: Final[frozenset[int]] = frozenset({401, 403})

#: The exit-code table. Documented once in docs/context-graph/cli-flow.md and
#: applied once, here; every other code means "command/validation failure".
_EXIT_BY_CODE: Final[dict[str, int]] = {
    "unavailable": EXIT_UNAVAILABLE,
    "not_implemented": EXIT_UNAVAILABLE,
    # The narrower unavailability codes belong in the table too, not just in the
    # `exit_code=` argument of whichever call site raises them. Left to the call
    # sites they drifted: `daemon_unavailable` exited 2 from `potpie status` and
    # 1 from `potpie ui` for the same dead daemon, because only one of the two
    # remembered to pass the number.
    "daemon_unavailable": EXIT_UNAVAILABLE,
    "daemon_start_failed": EXIT_UNAVAILABLE,
    "telemetry_preference_write_failed": EXIT_UNAVAILABLE,
    "degraded": EXIT_DEGRADED,
    "auth_error": EXIT_AUTH,
}


def exit_code_for(code: str) -> int:
    """The documented exit code for an error code — the table as a function.

    Every emitter resolves the number through here so it cannot depend on who
    is reading or on which call site happened to raise. Two drifts made that
    worth enforcing in code rather than in prose: the same usage error exited 2
    for a human and 1 under ``--json``, and ``unavailable`` exited 2 from the
    error boundary and 1 from a hand-written ``fail()``. A wrapper that retries
    on 2 because "a dependency is down" cannot tell either of those from a typo.

    An unrecognised code is a validation failure. A code nobody has classified
    is not grounds for claiming a dependency is unavailable.
    """
    return _EXIT_BY_CODE.get(code, EXIT_VALIDATION)


_state: dict[str, Any] = {
    "json": False,
    "verbose": False,
    "host": None,
    "store": None,
    "json_error_formatter": None,
    # What the argv scan below found, kept apart from the applied output mode;
    # see `bootstrap_output_flags_from_argv`.
    "argv_json": False,
    "argv_verbose": False,
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
    # An explicit setter is the caller's whole opinion about output mode, so it
    # also retires the argv scan: a fixture putting the CLI back into human
    # output must not be re-flipped by a `--json` some earlier invocation left
    # behind on process-wide state.
    _state["argv_json"] = False


def is_json() -> bool:
    return bool(_state["json"])


def bootstrap_output_flags_from_argv(argv: Sequence[str] | None = None) -> None:
    """Apply ``--json`` / ``--verbose`` before Typer finishes parsing.

    The root callback only runs after a command line parses successfully. Early
    bootstrap keeps parse-time failures on the documented JSON error contract.

    The scan is also *remembered*, because the root callback used to undo it.
    Click parses the group's arguments before the subcommand's, so
    ``potpie pot list --json`` reached the callback with ``--json`` unset, the
    callback wrote that back over the scan, and the ``NoSuchOption`` Click was
    about to raise a moment later came out as bare text on stderr at exit 2 —
    a rejection rendered in the one shape the caller had just said it could not
    parse.
    """
    args = tuple(argv or ())
    scan_args = args
    if "--" in args:
        scan_args = args[: args.index("--")]
    wants_json = "--json" in scan_args
    wants_verbose = "--verbose" in scan_args or "-v" in scan_args
    set_json(wants_json)
    set_verbose(wants_verbose)
    _state["argv_json"] = wants_json
    _state["argv_verbose"] = wants_verbose


def argv_requested_json() -> bool:
    """True when this invocation's argv asked for JSON, wherever it said so."""
    return bool(_state["argv_json"])


def argv_requested_verbose() -> bool:
    """True when this invocation's argv asked for verbose output."""
    return bool(_state["argv_verbose"])


def clear_argv_output_flags() -> None:
    """Forget the argv scan once the invocation that made it is over.

    The memory is scoped to one command line, not to the process. A run whose
    root callback never executes — an unknown command, an eager ``--version`` —
    would otherwise leave ``--json`` latched on for whatever drives the app
    next, which in practice is every in-process test after it.
    """
    _state["argv_json"] = False
    _state["argv_verbose"] = False


def set_verbose(value: bool) -> None:
    _state["verbose"] = bool(value)
    _state["argv_verbose"] = False


def is_verbose() -> bool:
    return bool(_state["verbose"])


class _ActiveHost:
    """Forwards to whichever origin is current when the attribute is read.

    Commands do ``host = get_host()`` once and then use that object for the
    rest of the body, so a plain host captured up front would keep pointing at
    the origin that was current at the *start* of the command. Resolving a
    qualified ref like ``--pot managed:api`` has to move the whole command
    across, not just the pot id — otherwise the id would be looked up on one
    host and used on the other, which is the one outcome worth engineering
    against. Binding late makes the 40-odd existing call sites correct without
    touching them.
    """

    __slots__ = ()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._resolve_current_host(), name)

    def _resolve_current_host(self) -> Any:
        """The host this proxy forwards to right now.

        A method rather than an attribute so the memo in
        :mod:`potpie.cli.host_snapshot` can key its answers on the real host:
        the proxy is a fresh object per ``get_host()`` call, while the host it
        stands for is built once per origin.
        """
        from potpie.cli import hosts

        return hosts.build_host(hosts.current_origin())

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        from potpie.cli import hosts

        return f"<ActiveHost origin={hosts.current_origin()}>"


def get_host():
    """Return the host for the current origin.

    An injected host (tests, alternate wiring) wins outright — it is a
    deliberate stand-in for the whole registry, not for one origin.
    """
    if _state["host"] is not None:
        return _state["host"]
    return _ActiveHost()


def get_host_for(origin: str):
    """Return the host for a specific origin, ignoring what is current.

    Used where a command already knows the origin — merged listings, and
    routing a resolved pot to the host that owns it.
    """
    if _state["host"] is not None:
        return _state["host"]
    from potpie.cli import hosts

    return hosts.build_host(origin)


def set_host(host: Any) -> None:
    """Inject a host (tests / alternate wiring).

    Also drops the registry's per-process origin override: an injected host
    defines the whole world for whatever runs next, so a leftover "current
    origin" from an earlier command would be a stale opinion about a registry
    that no longer applies.
    """
    _state["host"] = host
    invalidate_host_snapshot()
    from potpie.cli import hosts

    hosts.set_current_origin(None)


def get_store() -> CredentialStore:
    """Return the process-wide ``CredentialStore`` (built lazily).

    The auth/credential subsystem persists through this domain port; the concrete
    is chosen at the composition root (``potpie_context_engine.bootstrap.cli_auth_wiring``), so this
    inbound module never imports an adapter. The default is the real
    file-backed store; tests inject an in-memory fake via ``set_store``.
    """
    if _state["store"] is None:
        from potpie_context_engine.bootstrap.cli_auth_wiring import (
            build_credential_store,
        )

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
    exit_code: int | None = None,
) -> NoReturn:
    """Emit the structured error contract and exit with the documented code.

    ``exit_code`` defaults to :func:`exit_code_for` rather than to
    ``EXIT_VALIDATION``, so one call site cannot report ``unavailable`` at exit
    1 while the error boundary reports the same code at exit 2. Passing it
    explicitly stays available for the paths that mean something the code alone
    does not say.
    """
    if is_json():
        payload = {
            # The one key every error shape in this CLI shares, `graph`'s
            # workbench envelope included: a consumer can branch on failure
            # before it knows which command group answered.
            "ok": False,
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
    raise typer.Exit(code=exit_code_for(code) if exit_code is None else exit_code)


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
        )
    except ContextEngineDisabled as exc:
        # A 401/403 is the one member of this family where the host answered.
        # Collapsed into `unavailable` it told the operator to check the network
        # and restart a daemon that was never down, while the whole repair was
        # one wrong token — and `EXIT_AUTH` existed with no host path able to
        # reach it. The status code the transport attaches is the only thing
        # that ever knew the difference; the boundary just never looked.
        refused = getattr(exc, "status_code", None) in CREDENTIAL_REFUSED_STATUSES
        error_code = "auth_error" if refused else "unavailable"
        result = error_code
        fail(
            code=error_code,
            message=str(exc),
            next_action=getattr(exc, "recommended_next_action", None)
            or (
                "check which key that host expects with 'potpie host list'"
                if refused
                # Some unavailability has a specific, known repair (e.g. a graph
                # substrate that shut down uncleanly); prefer it over the generic
                # pointer at doctor.
                else "check backend/daemon readiness with 'potpie doctor'"
            ),
        )
    except PotArchived as exc:
        # Its own code, not `pot_not_found`: the ref resolved, and sending the
        # operator to `pot list` to look for a pot that is deliberately not in
        # that listing is the same misdirection this table exists to avoid.
        result = "pot_archived"
        error_code = "pot_archived"
        fail(
            code="pot_archived",
            message=str(exc),
            next_action=getattr(exc, "recommended_next_action", None),
        )
    except PotNameConflict as exc:
        result = "pot_name_conflict"
        error_code = "pot_name_conflict"
        fail(
            code="pot_name_conflict",
            message=str(exc),
            next_action=getattr(exc, "recommended_next_action", None),
        )
    except SourceNotFound as exc:
        # Above `PotNotFound`, which it subclasses. The subclassing exists so an
        # inbound boundary that has not learned this type still degrades to a
        # sensible refusal rather than "unexpected internal error"; this branch
        # is what stops the *pot* repair being offered for a missing source by
        # any command that has not mapped the error itself. Naming it here is
        # also what lets the subclassing be dropped later.
        result = "source_not_found"
        error_code = "source_not_found"
        fail(
            code="source_not_found",
            message=str(exc),
            next_action=getattr(exc, "recommended_next_action", None),
        )
    except PotNotFound as exc:
        result = "pot_not_found"
        error_code = "pot_not_found"
        fail(
            code="pot_not_found",
            message=str(exc),
            # Not every PotNotFound is *about* a pot: it is also how "this pot
            # does not hold that" is reported, e.g. `source remove` of an id the
            # pot never held. Hardcoding the pot repair there sends the operator
            # to `pot list` to hunt for a pot that was never missing, so a
            # raiser that knows the real next step gets to say so.
            next_action=getattr(exc, "recommended_next_action", None)
            or "list pots with 'potpie pot list' or create one with 'potpie setup'",
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
            # `--verbose` is documented as "Verbose tracebacks on errors" and
            # this is the one branch with a traceback worth seeing — it was
            # silently dropping it, so the only way to find out what an
            # "Unexpected internal error" actually was involved reproducing the
            # call by hand outside the CLI.
            detail=traceback.format_exc() if is_verbose() else None,
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
    # No flush here. The SDK's atexit hook sends everything this process
    # recorded in one envelope; a synchronous ``flush(2.0)`` on every command
    # waited on a full HTTPS round trip to Sentry inside the command's own
    # wall time, and graph commands paid it twice.
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


def _searchable_origins() -> list[str]:
    """Origins a bare ref may resolve in, current one first.

    An injected host stands in for the whole registry, so single-host mode is
    preserved: searching "both" would ask the same fake twice and report every
    pot as ambiguous with itself.

    :func:`hosts.targeting_origins` rather than ``configured_origins`` because
    this is the *targeting* candidate set: a configured host that cannot be
    enumerated at all must still be in it, so the refusal comes from
    :func:`_find_pot_in` instead of the candidate quietly going missing.
    """
    from potpie.cli import hosts

    if _state["host"] is not None:
        return [hosts.current_origin()]
    current = hosts.current_origin()
    return [current, *(o for o in hosts.targeting_origins() if o != current)]


def _qualify_hint(prefix: str, origin: str, ref: str) -> str:
    """Next action for a ref that has to route around an unreachable ``origin``.

    ``prefix`` is how the caller's own surface spells a pot ref (``--pot``,
    ``potpie use``), so the hint is a line the user can actually run.
    """
    from potpie.cli import hosts

    alternatives = " or ".join(
        f"'{prefix} {hosts.qualify(other, ref)}'"
        for other in hosts.ORIGINS
        if other != origin
    )
    return f"target one host explicitly, e.g. {alternatives}"


def _unreachable_host_hint(origin: str) -> str:
    """Next action when the ref *names* ``origin`` and ``origin`` is down.

    Distinct from :func:`_qualify_hint`, which routes a bare ref around the
    unreachable host: once the ref says ``managed:api`` there is nowhere to
    route to, and repeating "target one host explicitly" would be advice the
    user already followed. The only repair left is the host itself.
    """
    from potpie.cli import hosts

    if origin == hosts.MANAGED:
        return (
            "check the managed host with 'potpie host list', "
            "or re-point it with 'potpie host set <url>'"
        )
    return "check backend/daemon readiness with 'potpie doctor'"


def _refuse_origin_with_no_host_behind_it(
    origin: str | None, ref: str, *, drop: str
) -> None:
    """Refuse an origin an injected host cannot honestly answer for.

    An injected host (``potpie setup``, tests) stands in for the whole registry
    rather than for one origin of it: there is exactly one host and it is
    whichever origin is current. Naming any *other* origin — ``--managed``, or a
    qualified ``managed:api`` — targets a host that is not there, and the stand-in
    answers for it regardless: the pot id is read off the injected host while the
    origin the command moves to, reports and persists is one nothing was resolved
    against. Refusing beats relabelling, because the selection itself is the lie.

    Both resolution paths route through here — ``--pot`` via
    :func:`_resolve_explicit_pot` and ``potpie use`` via
    :func:`_select_origin_for_use` — so the two cannot drift into disagreeing
    about what one injected host means; that drift is exactly how ``--pot``
    kept resolving cross-host after ``potpie use`` was fixed. ``drop`` spells the
    target the way the calling surface does (``{origin}`` is substituted), so the
    hint names something the user actually typed.
    """
    if origin is None or _state["host"] is None:
        return
    from potpie.cli import hosts

    current = hosts.current_origin()
    if origin == current:
        return
    fail(
        code="validation_error",
        message=(
            f"Cannot target the {origin} host: this run is wired to "
            f"a single {current} host."
        ),
        next_action=f"drop {drop.format(origin=origin)} and pass '{ref}' on its own",
    )


@dataclass(frozen=True, slots=True)
class _PotMatch:
    """One host's answer for a pot ref, including whether it is archived.

    Archived pots are matched rather than skipped so the refusal can say
    ``pot_archived``. Filtering them out here instead produced ``pot_not_found``
    for a ref that resolved perfectly well, sending the operator to ``pot list``
    to hunt for a pot that listing deliberately hides.
    """

    origin: str
    pot_id: str
    name: str
    archived: bool


def _refuse_ambiguous_pot(
    live: Sequence[_PotMatch], *, ref: str, qualify_hint: str
) -> None:
    """Refuse a ref that resolved on more than one host, naming both candidates.

    One refusal shared by both resolution paths — ``--pot`` through
    :func:`_resolve_match` and ``potpie use`` through
    :func:`_select_origin_for_use` — because those two drifting apart is the
    documented history of this invariant rather than a hypothetical: ``--pot``
    kept picking a host silently for a round after ``potpie use`` stopped.

    Candidates are named the way ``pot list`` prints them, ``<origin>:<name>``,
    which is also the syntax the hint tells the caller to type. Qualifying the
    ref the caller *typed* instead answered a pot id with
    ``local:pot_37ab7f7fe4ef, managed:pot_37ab7f7fe4ef`` — two lines that both
    work and neither of which anyone can match against a pot they know by name.
    """
    if len(live) < 2:
        return
    from potpie.cli import hosts

    qualified = ", ".join(hosts.qualify(m.origin, m.name) for m in live)
    fail(
        code="ambiguous_pot",
        message=f"'{ref}' matches a pot on more than one host: {qualified}.",
        next_action=qualify_hint,
    )


def _resolve_match(
    matches: list[_PotMatch],
    *,
    ref: str,
    qualify_hint: str,
    list_hint: str,
    not_found_message: str | None = None,
) -> _PotMatch:
    """Pick the one pot a ref means, or refuse with the reason it cannot.

    A live pot always wins over an archived one, on any host: an archived pot
    still answering to a name is precisely how a dead pot shadows a live one, so
    it must not make a ref ambiguous either.
    """
    live = [match for match in matches if not match.archived]
    _refuse_ambiguous_pot(live, ref=ref, qualify_hint=qualify_hint)
    if live:
        return live[0]
    if matches:
        archived = matches[0]
        fail(
            code="pot_archived",
            message=(
                f"Pot '{archived.name}' ({archived.pot_id}) is archived, so it "
                f"cannot be used as a target. Archiving cleared its graph and "
                f"stored documents."
            ),
            next_action=(
                "see it with 'potpie pot list --archived', or start a new pot "
                "with 'potpie pot create <name> --use'"
            ),
        )
    fail(
        code="pot_not_found",
        message=not_found_message or f"No pot matching '{ref}'.",
        next_action=list_hint,
    )
    raise AssertionError("unreachable")  # pragma: no cover - fail() exits


def _find_pot_in(
    origin: str, ref: str, *, unreachable_hint: str | None = None
) -> _PotMatch | None:
    """The match for ``ref`` on ``origin``, or ``None``.

    Passing ``unreachable_hint`` turns a host that cannot be enumerated from
    "holds no match" into a refusal carrying that hint. Resolving a bare ref
    across two origins needs that: read as "no match", a momentarily
    unreachable host shrinks the ambiguity check to a single candidate and the
    command runs against the *other* graph without ever saying the first one
    was not asked. Callers that merely enumerate leave it unset and keep
    degrading, because a listing that dies when a remote is down is useless
    exactly when you need to see what is local.

    A host that cannot even be *built* is refused in the same breath but not in
    the same words: no socket was opened, so "cannot reach" would be the one
    fact that is not true of a managed address with a stray character in its
    port — the failure the caller has to fix is the address, not the network.
    """
    try:
        host = get_host_for(origin)
    except Exception as exc:  # noqa: BLE001 - see docstring: degrade or refuse
        if unreachable_hint is None:
            return None
        fail(
            code="unavailable",
            message=f"Cannot use the {origin} host to resolve '{ref}': {exc}",
            next_action=unreachable_hint,
        )
    try:
        pots = _list_pots(host)
    except Exception as exc:  # noqa: BLE001 - see docstring: degrade or refuse
        if unreachable_hint is None:
            return None
        # A host that answered 401 was reached, so saying "cannot reach" and
        # pointing at the network sends the operator to debug a connection that
        # worked while the whole repair is one wrong token. Same split, and the
        # same reason, as the `ContextEngineDisabled` branch in `contract()`.
        if getattr(exc, "status_code", None) in CREDENTIAL_REFUSED_STATUSES:
            fail(
                code="auth_error",
                message=f"The {origin} host refused the credential resolving '{ref}': {exc}",
                next_action="check which key that host expects with 'potpie host list'",
            )
        fail(
            code="unavailable",
            message=f"Cannot reach the {origin} host to resolve '{ref}': {exc}",
            next_action=unreachable_hint,
        )
    for pot in pots:
        if ref in (pot.pot_id, pot.name):
            return _PotMatch(
                origin=origin,
                pot_id=pot.pot_id,
                name=pot.name,
                archived=bool(getattr(pot, "archived", False)),
            )
    return None


def _resolve_explicit_pot(explicit: str) -> str:
    """Resolve ``--pot`` / a pot ref, moving the command to the owning origin.

    A qualified ``managed:api`` is unambiguous and searched only there. A bare
    ref prefers the current origin and only then looks elsewhere; matching on
    two origins is an error rather than a pick, because pot names are per-host
    labels and `default` very likely exists on both. Guessing would run the
    command against the wrong graph.

    Every branch here is a *targeting* path, so every one of them passes an
    ``unreachable_hint``: a host that cannot be enumerated must be reported as
    unreachable and named. Read as "holds no match" it becomes ``pot_not_found``
    instead, which sends the operator to ``pot list`` to look for a pot that was
    never missing while the host that owns it is the thing that is down.

    A qualifier naming an origin that has no host behind it is refused rather
    than resolved; see :func:`_refuse_origin_with_no_host_behind_it`. The bare
    branch needs no such guard because ``_searchable_origins`` already collapses
    to the one origin an injected host stands for.
    """
    from potpie.cli import hosts

    origin, ref = hosts.split_ref(explicit)
    if ref is None:  # pragma: no cover - `explicit` is non-empty here
        ref = explicit

    if origin is not None:
        _refuse_origin_with_no_host_behind_it(
            origin, ref, drop="the '{origin}:' qualifier"
        )
        hosts.set_current_origin(origin)
        found = _find_pot_in(
            origin, ref, unreachable_hint=_unreachable_host_hint(origin)
        )
        match = _resolve_match(
            [found] if found is not None else [],
            ref=ref,
            qualify_hint="qualify it, e.g. '--pot managed:<name>'",
            list_hint=f"run 'potpie pot list --{origin}'",
            not_found_message=f"No pot matching '{ref}' on the {origin} host.",
        )
        return match.pot_id

    candidates = _searchable_origins()
    matches: list[_PotMatch] = []
    for candidate in candidates:
        found = _find_pot_in(
            candidate,
            ref,
            unreachable_hint=(
                _qualify_hint("--pot", candidate, ref)
                if len(candidates) > 1
                else _unreachable_host_hint(candidate)
            ),
        )
        if found is not None:
            matches.append(found)

    match = _resolve_match(
        matches,
        ref=explicit,
        qualify_hint="qualify it, e.g. '--pot managed:<name>'",
        list_hint="run 'potpie pot list'",
    )
    hosts.set_current_origin(match.origin)
    return match.pot_id


def resolve_pot_scope(
    host: Any, explicit: str | None = None, *, infer_from_repo: bool = True
) -> tuple[str, str]:
    """Resolve pot id and how it was chosen for CLI scope hints.

    ``resolved_via`` is one of ``explicit``, ``repo_default``, ``linked_repo``,
    ``contained_repo``, or ``active_pot``. See :func:`resolve_pot_id` for
    resolution order.

    ``infer_from_repo=False`` skips current-repo inference and goes straight to
    the active pot. Source registration uses this: ``source add repo .`` is the
    command that *establishes* the repo→pot mapping, so inferring its target
    from existing registrations would route the new source to the wrong pot
    (or fail as ambiguous when other pots already track the same repo).
    """
    if explicit:
        # Selects the origin as well as the pot; see _resolve_explicit_pot.
        return _resolve_explicit_pot(explicit), "explicit"
    repo_identity = _current_repo_identity() if infer_from_repo else None
    default_pot = _repo_default_pot_id(host, repo_identity)
    if default_pot:
        return default_pot, "repo_default"
    selves, contained = _current_repo_matches(host) if infer_from_repo else ([], [])
    matches = selves + contained
    active = _active_pot(host)
    if len(matches) == 1:
        # Which relation matched is part of the answer. Scoping to a project
        # that happens to live *inside* the cwd used to be reported as
        # `linked_repo` — as though the caller were standing in the repo — so
        # every command silently ran against a child project's pot with nothing
        # in the output saying which project, or that a choice had been made.
        if contained:
            return contained[0][0], "contained_repo"
        return matches[0][0], "linked_repo"
    if len(matches) > 1:
        if active is not None and any(active.pot_id == pid for pid, _ in matches):
            return active.pot_id, "active_pot"
        # A workspace root is not an ambiguous repo, and saying so mattered:
        # "Current repo is registered in multiple pots" names something that is
        # not true of a directory registered in none of them, so the reader goes
        # looking for a duplicate registration that does not exist.
        if not selves and len(contained) > 1:
            names = ", ".join(f"{name} ({pid})" for pid, name in contained)
            fail(
                code="workspace_root_ambiguous",
                message=(
                    "This directory is not a registered project; it contains registered "
                    f"projects in more than one pot: {names}."
                ),
                next_action="cd into the project you mean, or pass '--pot <id-or-name>'",
            )
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
    active = _safe_call(lambda: _active_pot(host), None)
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
    active = _safe_call(lambda: _active_pot(host), None)
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
    if resolved_via == "contained_repo":
        return "via a registered project inside this directory"
    return "via active pot"


def pot_scope_human(
    host: Any,
    pot_id: str,
    *,
    resolved_via: str | None = None,
    repo: str | None = None,
) -> str:
    """The one-line pot header every read prints.

    Name and id only. The ``sources= claims= entities=`` triple cost two extra
    host calls per command — a source listing and the data-plane status — for
    a decoration nobody acted on, and on a managed host those two were a
    second of wall time on every read. They are still printed under
    ``--verbose``; ``potpie status`` and ``pot info`` keep them unconditionally
    because there the counts are the answer.
    """
    pot = _pot_for_id(host, pot_id)
    name = getattr(pot, "name", pot_id) if pot is not None else pot_id
    scope = f"pot={name} ({pot_id})"
    if is_verbose():
        info = pot_scope_info(host, pot_id)
        counts = info.get("counts") or {}
        scope += (
            f" sources={info.get('source_count', 0)} "
            f"claims={counts.get('claims', 0)} entities={counts.get('entities', 0)}"
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


def parse_scope_pairs(scope: str | None) -> dict[str, str]:
    """``key:value[,key:value]`` → dict, refusing anything that is not that.

    One parser for the whole surface, because the two that existed disagreed
    about the failure case and the lenient one was on the *write* path:
    ``potpie record --scope service`` dropped the malformed pair on the floor
    and wrote an unscoped claim at exit 0, so the caller's narrowing silently
    became no narrowing at all and the wrong data is now in the graph. A scope
    the CLI cannot read is a refusal, never a smaller filter.

    Raised as ``ValueError`` so the shared ``contract()`` renders it as
    ``validation_error`` in whichever envelope the caller asked for.
    """
    if not scope:
        return {}
    out: dict[str, str] = {}
    for pair in scope.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if ":" not in pair:
            raise ValueError(
                f"invalid --scope entry {pair!r}; expected key:value pairs"
            )
        key, value = pair.split(":", 1)
        key = key.strip()
        if not key:
            raise ValueError(
                f"invalid --scope entry {pair!r}; scope keys must not be empty"
            )
        value = value.strip()
        if not value:
            raise ValueError(
                f"invalid --scope entry {pair!r}; scope values must not be empty"
            )
        out[key] = value
    return out


def require_text(value: str | None, *, argument: str, example: str) -> str:
    """The trimmed ``value``, refusing one that says nothing.

    An empty or whitespace-only argument is not a narrower request, it is an
    absent one — and the commands that took it answered anyway: ``potpie search
    ''`` returned a ranked envelope with a confidence score attached to a query
    nobody made, and ``config get ''`` answered ``{"": null}`` as though the
    empty key were a setting that happens to be unset. Both read as results.
    """
    cleaned = (value or "").strip()
    if cleaned:
        return cleaned
    fail(
        code="validation_error",
        message=f"{argument} cannot be empty.",
        next_action=f"pass a value, e.g. {example}",
    )


def origin_from_use_flags(*, local: bool, managed: bool) -> str | None:
    """Map ``--local`` / ``--managed`` to a requested origin, or ``None``.

    Both at once is a refusal rather than a precedence rule: on the flag pair
    whose whole job is choosing which graph the CLI moves to, honouring one
    would mean silently ignoring the other.
    """
    from potpie.cli import hosts

    if local and managed:
        fail(
            code="validation_error",
            message="--local and --managed name different hosts; pass at most one.",
            next_action="pick one, or qualify the ref (e.g. 'potpie use managed:<name>')",
        )
    if local:
        return hosts.LOCAL
    if managed:
        return hosts.MANAGED
    return None


def _select_origin_for_use(
    ref: str, requested: str | None = None
) -> tuple[str, str, bool]:
    """Point the CLI at the host that owns ``ref``; return it and the bare ref.

    Returns ``(bare_ref, origin, move_pointer)``, where ``origin`` is the host
    the selection was actually resolved against — never a label chosen
    independently of it.

    ``requested`` (``--local`` / ``--managed``) and a qualified ``managed:api``
    say the same thing, so they resolve identically and contradicting each
    other is refused. Either way the origin is a *target*, not a starting
    point: the ref is used there and nowhere else, because falling through to
    the other host is precisely how "use this local pot" ends up selecting the
    managed pot of the same name.

    A bare ref is searched across every configured origin and a match on two is
    refused rather than picked — pot names are per-host labels, so guessing
    would eventually point the CLI at the wrong graph.

    An injected host (``potpie setup``, tests) is the whole registry rather than
    one origin of it, so there is exactly one host and it is whichever origin is
    current. Targeting any *other* origin is refused instead of run: the fake
    would happily answer for a host it is not, and the selection would be
    reported and persisted under an origin nothing was resolved against.
    """
    from potpie.cli import hosts

    qualifier, bare = hosts.split_ref(ref)
    if qualifier is not None and bare is not None:
        if requested is not None and requested != qualifier:
            fail(
                code="validation_error",
                message=(
                    f"--{requested} contradicts the ref '{ref}', "
                    f"which targets the {qualifier} host."
                ),
                next_action=(
                    f"drop --{requested}, or pass '{hosts.qualify(requested, bare)}'"
                ),
            )
        requested, ref = qualifier, bare

    if _state["host"] is not None:
        # One host, so nothing to search and no pointer to move — but the origin
        # reported is the one that is actually current, never a literal: a
        # hardcoded `local` prints "(local)" for a session sitting on `managed`,
        # which is the label/host disagreement this function returns an origin
        # to prevent. And an origin that is not the current one has no host
        # behind it at all, so it is refused before the fake can answer for it.
        _refuse_origin_with_no_host_behind_it(
            requested, ref, drop="the {origin} target"
        )
        return ref, hosts.current_origin(), False

    if requested is not None:
        # Move first and let the host itself report a miss, so a pot that is not
        # there fails on the host the user asked for instead of resolving on the
        # other one.
        hosts.set_current_origin(requested)
        return ref, requested, True

    candidates = _searchable_origins()
    matches = [
        found
        for candidate in candidates
        if (
            found := _find_pot_in(
                candidate,
                ref,
                unreachable_hint=(
                    _qualify_hint("potpie use", candidate, ref)
                    if len(candidates) > 1
                    else None
                ),
            )
        )
        is not None
    ]
    live = [match for match in matches if not match.archived]
    _refuse_ambiguous_pot(
        live,
        ref=ref,
        qualify_hint=(
            "qualify it, e.g. 'potpie use managed:<name>', or pass --local / --managed"
        ),
    )
    # Not found anywhere — or found only as an archived pot: hand it to the host
    # that holds it (or the current one) so the failure comes from the host,
    # with its own wording, exactly as it did before. The host's ``use_pot``
    # raises ``PotArchived`` for the archived case.
    origin = (live or matches or [None])[0]
    origin = origin.origin if origin is not None else hosts.current_origin()
    hosts.set_current_origin(origin)
    return ref, origin, True


def use_pot_selection(
    host: Any,
    ref: str,
    *,
    also_default_for_current_repo: bool = False,
    origin: str | None = None,
) -> tuple[dict[str, Any], str]:
    """Select the active pot and report the origin it was resolved against.

    ``origin`` is a *request* (``--local`` / ``--managed``), not a display
    label: what gets printed and persisted is whatever the resolution actually
    targeted, so the origin the user reads cannot disagree with the graph the
    CLI moved to.
    """
    repo_key = None
    if also_default_for_current_repo:
        repo_key = current_repo_identity_for_cli()
        if not repo_key:
            raise ValueError("--also-default-for-current-repo requires a repo")

    ref, origin, move_pointer = _select_origin_for_use(ref, origin)

    pot = host.pots.use_pot(ref=ref)
    invalidate_host_snapshot()
    # Persist only after the host accepted the ref, so a failed selection never
    # strands the CLI pointing at a host the user did not choose.
    if move_pointer:
        from potpie.cli import hosts

        hosts.set_persisted_origin(origin)
    repo_default_set = False
    if repo_key:
        host.pots.set_repo_default(repo=repo_key, pot_id=pot.pot_id)
        invalidate_host_snapshot()
        repo_default_set = True

    routing = repo_effective_pot_info(host)
    warnings = []
    warning = repo_default_mismatch_warning(host, routing, selected_pot_id=pot.pot_id)
    if warning:
        warnings.append(warning)

    lines = [f"active pot → {pot.name} ({origin})"]
    if repo_default_set:
        lines.append(f"repo {repo_key} default → {pot.name} ({pot.pot_id})")
    lines.extend(f"warning: {item}" for item in warnings)

    payload: dict[str, Any] = {
        "id": pot.pot_id,
        "name": pot.name,
        "origin": origin,
        "repo_default_set": repo_default_set,
        "current_repo": routing,
        "warnings": warnings,
    }

    return enrich_with_pot_guidance(
        host,
        pot.pot_id,
        payload,
        human="\n".join(lines),
        repo=repo_key,
    )


def pot_graph_counts(host: Any, pot_id: str) -> dict[str, int]:
    if getattr(host, "graph", None) is None:
        return {}
    status = _safe_call(lambda: pot_data_plane_status(host, pot_id), None)
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
    return len(_safe_call(lambda: _list_sources(host, pot_id), []) or [])


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


def _current_repo_matches(
    host: Any,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """``(self_matches, contained_matches)`` for the cwd.

    Two relations, kept apart. A pot is a *self* match when the cwd is (or sits
    inside) a repository that pot has registered — the pot that owns this tree.
    It is a *contained* match when the registered repository sits underneath the
    cwd, which is what standing in a workspace root looks like: the directory is
    registered nowhere and merely holds projects that are. Collapsed into one
    boolean, one registered child silently scoped every command to that child's
    pot and two of them failed with "Current repo is registered in multiple
    pots" — a sentence about a repo that is not registered at all.
    """
    try:
        cwd = Path.cwd().resolve()
    except OSError:
        return [], []
    remote = _current_git_remote(cwd)
    selves: list[tuple[str, str]] = []
    contained: list[tuple[str, str]] = []
    seen: set[str] = set()
    for pot_id, pot_name, refs in _repo_source_index(host):
        if pot_id in seen:
            continue
        kinds = {
            match
            for ref in refs
            if (match := classify_repo_source_match(ref, cwd=cwd, remote=remote))
        }
        if not kinds:
            continue
        seen.add(pot_id)
        (contained if kinds == {REPO_MATCH_CONTAINED} else selves).append(
            (pot_id, pot_name)
        )
    return selves, contained


def _pots_matching_current_repo(host: Any) -> list[tuple[str, str]]:
    """Return ``(pot_id, name)`` for every pot whose repo source matches cwd.

    A pot is the project boundary, not a single repository. This helper only
    chooses the pot from the current working tree; it does not inject a repo
    scope into reads. Timeline queries therefore default to the whole project
    across all repositories attached to the pot. The caller decides how to
    disambiguate multiple matches (active pot wins; otherwise a structured
    ``ambiguous_pot`` error).

    Both relations, in one list, because that is the candidate set every caller
    here has always had; :func:`resolve_pot_scope` is the one place that has to
    tell them apart.
    """
    selves, contained = _current_repo_matches(host)
    return selves + contained


def _pots_matching_repo_identity(
    host: Any, repo_identity: str | None
) -> list[tuple[str, str]]:
    if not repo_identity:
        return []
    return _pots_matching_repo_source(
        host, lambda ref: repo_identity_key(ref) == repo_identity
    )


def _pots_matching_repo_source(
    host: Any, matches_ref: Callable[[str], bool]
) -> list[tuple[str, str]]:
    """``(pot_id, name)`` for each pot with a repo source ``matches_ref`` accepts.

    One pot per match, in pot order, as the per-pot walk this replaced produced.
    """
    matches: list[tuple[str, str]] = []
    matched: set[str] = set()
    for pot_id, pot_name, refs in _repo_source_index(host):
        if pot_id in matched:
            continue
        if any(matches_ref(ref) for ref in refs):
            matches.append((pot_id, pot_name))
            matched.add(pot_id)
    return matches


def _repo_source_index(host: Any) -> list[tuple[str, str, tuple[str, ...]]]:
    """``(pot_id, pot_name, refs)`` for every repo source of every visible pot.

    ``refs`` are the registered strings a working tree is matched against. The
    control plane serves this as one call; the per-pot walk below is the
    fallback for a host that predates it. That walk is a dict read per pot
    in-process but one network + database round trip per pot against a hosted
    control plane, which made repo→pot resolution — and so ``status`` and every
    command that resolves a pot from the cwd — scale with the caller's pot
    count.
    """
    index = _safe_call(lambda: _list_repo_sources(host), None)
    if index is None:
        return _repo_source_index_per_pot(host)
    return [
        (
            str(getattr(row, "pot_id", "") or ""),
            str(getattr(row, "pot_name", "") or ""),
            _repo_source_refs(row),
        )
        for row in index
    ]


def _repo_source_index_per_pot(host: Any) -> list[tuple[str, str, tuple[str, ...]]]:
    try:
        pots = _list_pots(host)
    except Exception:  # noqa: BLE001 - pot resolution should not mask commands
        return []
    rows: list[tuple[str, str, tuple[str, ...]]] = []
    for pot in pots:
        # Matches the control-plane index above, which drops archived pots: an
        # archived pot is not a routing candidate for the current repo.
        if getattr(pot, "archived", False):
            continue
        try:
            sources = _list_sources(host, pot.pot_id)
        except Exception:  # noqa: BLE001
            continue
        rows.extend(
            (pot.pot_id, pot.name, _repo_source_refs(source))
            for source in sources
            if getattr(source, "kind", None) == "repo"
        )
    return rows


def _repo_source_refs(source: Any) -> tuple[str, ...]:
    refs = (
        str(getattr(source, "name", "") or "").strip(),
        str(getattr(source, "location", "") or "").strip(),
    )
    return tuple(dict.fromkeys(ref for ref in refs if ref))


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
    pot_id = _safe_call(
        lambda: memoized(
            _snapshot_host(host),
            "pots.repo_default",
            (repo_identity,),
            lambda: getter(repo=repo_identity),
        ),
        None,
    )
    if not pot_id:
        return None
    pot_id = str(pot_id)
    pot = _pot_for_id(host, pot_id)
    # A pointer at a pot that is gone, or archived, is stale rather than
    # authoritative — and honouring the archived case routed every repo-scoped
    # read and write into a pot whose graph had already been torn down, which
    # answers empty instead of failing.
    if pot is None or bool(getattr(pot, "archived", False)):
        return None
    return pot_id


def repo_default_matches(host: Any, repo_key: str | None, pot_id: str) -> bool:
    """True when ``repo_key``'s repo default is set to ``pot_id``."""
    default_pot = repo_default_pot_id(host, repo_key)
    return bool(default_pot and default_pot == pot_id)


def _repo_default_pot_id(host: Any, repo_identity: str | None) -> str | None:
    return repo_default_pot_id(host, repo_identity)


def _pot_for_id(host: Any, pot_id: str):
    for pot in _safe_call(lambda: _list_pots(host), []) or []:
        if getattr(pot, "pot_id", None) == pot_id:
            return pot
    return None


# --- the memoized host reads ---------------------------------------------------
#
# The six calls below are the ones pot resolution, the pot header, and the
# empty-pot guidance kept re-asking. Each goes through the process-wide memo in
# :mod:`potpie.cli.host_snapshot`; see that module for the invalidation rules.
# Listings are materialised so a lazy answer is consumed exactly once.


def _snapshot_host(host: Any) -> Any:
    """The object the memo keys on: the real host behind an ``_ActiveHost``."""
    if isinstance(host, _ActiveHost):
        return host._resolve_current_host()
    return host


def _list_pots(host: Any) -> list[Any]:
    return memoized(
        _snapshot_host(host), "pots.list_pots", (), lambda: list(host.pots.list_pots())
    )


def _active_pot(host: Any) -> Any:
    return memoized(_snapshot_host(host), "pots.active_pot", (), host.pots.active_pot)


def _list_repo_sources(host: Any) -> list[Any]:
    return memoized(
        _snapshot_host(host),
        "pots.list_repo_sources",
        (),
        lambda: list(host.pots.list_repo_sources()),
    )


def _list_sources(host: Any, pot_id: str) -> list[Any]:
    return memoized(
        _snapshot_host(host),
        "pots.list_sources",
        (pot_id,),
        lambda: list(host.pots.list_sources(pot_id=pot_id)),
    )


def pot_data_plane_status(host: Any, pot_id: str) -> Any:
    """``graph.data_plane_status(pot_id)``, asked once per process."""
    return memoized(
        _snapshot_host(host),
        "graph.data_plane_status",
        (pot_id,),
        lambda: host.graph.data_plane_status(pot_id),
    )


def _safe_call(fn, default):
    try:
        return fn()
    except Exception:  # noqa: BLE001
        return default


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
    "CREDENTIAL_REFUSED_STATUSES",
    "EXIT_AUTH",
    "EXIT_DEGRADED",
    "EXIT_OK",
    "EXIT_UNAVAILABLE",
    "EXIT_VALIDATION",
    "argv_requested_json",
    "argv_requested_verbose",
    "bootstrap_output_flags_from_argv",
    "clear_argv_output_flags",
    "contract",
    "emit",
    "exit_code_for",
    "fail",
    "get_host",
    "get_store",
    "current_repo_identity_for_cli",
    "empty_pot_guidance",
    "empty_pot_warnings",
    "enrich_with_pot_guidance",
    "is_json",
    "is_verbose",
    "origin_from_use_flags",
    "parse_scope_pairs",
    "pot_graph_counts",
    "pot_scope_human",
    "pot_scope_info",
    "pot_scope_resolution_human",
    "pot_source_count",
    "invalidate_host_snapshot",
    "pot_data_plane_status",
    "repo_default_matches",
    "repo_default_mismatch_warning",
    "repo_default_pot_id",
    "repo_effective_pot_human",
    "repo_effective_pot_info",
    "repo_pot_candidates",
    "require_text",
    "resolve_pot_id",
    "resolve_pot_scope",
    "set_host",
    "set_store",
    "set_json",
    "set_verbose",
    "use_pot_selection",
]
