"""Daemon-backed ``HostShell`` client for the CLI."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Final

import httpx

from potpie_context_engine.adapters.outbound.pots.local_pot_store import default_home
from potpie_context_core.errors import (
    CapabilityNotImplemented,
    ContextEngineDisabled,
    PotNotFound,
)
from potpie.daemon.lifecycle import Daemon
from potpie.daemon.rpc import decode, encode

#: Statuses that mean the endpoint answered *and* refused the credential.
#:
#: Kept apart from the rest of the 4xx family because the two failures have
#: nothing in common from where the user sits: an unreachable host is a service
#: to start and a network to check, and this is one wrong key. Both used to
#: arrive as the same sentence.
_CREDENTIAL_REFUSED: Final[frozenset[int]] = frozenset({401, 403})

#: Surfaces whose deadline is not the default one, keyed by surface name.
#:
#: ``None`` means "no client-side deadline". ``setup`` earns that because its
#: single RPC carries the entire first run — model download, backend provision,
#: migrations, skill install — and on a genuinely cold machine that exceeds any
#: number small enough to be useful elsewhere. Raising the number only moves the
#: cliff; what matters is that the client stops giving up on work the daemon is
#: still doing and reporting it as a failure.
#:
#: Surface names are spelled here *and* in the daemon's own ``_ALLOWED_RPC_SURFACES``.
#: A rename in one place and not the other reverts silently to the default
#: deadline rather than failing loudly, so keep this table to the few surfaces
#: that genuinely need it.
_SURFACE_DEADLINES: Final[dict[str, float | None]] = {"setup": None}

#: How a host that predates ``not_implemented`` for surfaces refuses one.
#:
#: The managed service runs its own build of this daemon with its own copy of
#: the RPC allowlist, and a surface missing from that copy comes back as
#: ``validation_error: invalid RPC surface: resources`` — the caller-was-wrong
#: code, at exit 1, with no next action, for a caller that asked a perfectly
#: reasonable question. We cannot deploy the fix into that repository, so the
#: translation happens on this side; see :func:`_raise_remote_error`.
_LEGACY_SURFACE_REFUSAL: Final = "invalid RPC surface: "


@dataclass(slots=True)
class DaemonRpcClient:
    """Small local HTTP client that calls operations inside the daemon."""

    daemon: Daemon = field(
        default_factory=lambda: Daemon(home=default_home(), in_process=False)
    )
    timeout_s: float = 30.0
    label: str = "Potpie daemon"
    """What to call this endpoint in errors.

    The same client now also drives a managed host, where "the Potpie daemon is
    unavailable" points at the wrong machine entirely — the local daemon may be
    perfectly healthy. Callers that are not the local daemon pass their own
    label so the message names what actually failed.
    """

    def call(self, surface: str, method: str, *args: Any, **kwargs: Any) -> Any:
        discovery = self._rpc_discovery()
        url = f"{discovery['base_url'].rstrip('/')}/rpc"
        payload = {
            "surface": surface,
            "method": method,
            "args": encode(args),
            "kwargs": encode(kwargs),
        }
        # The deadline rides on the surface rather than on a keyword argument
        # because ``**kwargs`` is forwarded verbatim to the remote method — any
        # control parameter added here would collide with a service parameter of
        # the same name the first time one is introduced.
        return self._result(
            self._post(
                url,
                payload,
                token=discovery["token"],
                timeout=self._deadline_for(surface),
            ),
            url,
        )

    def attr(self, surface: str, name: str) -> Any:
        discovery = self._rpc_discovery()
        url = f"{discovery['base_url'].rstrip('/')}/attr"
        payload = {"surface": surface, "name": name}
        return self._result(
            self._post(
                url,
                payload,
                token=discovery["token"],
                timeout=self._deadline_for(surface),
            ),
            url,
        )

    def _deadline_for(self, surface: str) -> float | None:
        """Seconds this surface may take before the client gives up, or ``None``."""
        if surface in _SURFACE_DEADLINES:
            return _SURFACE_DEADLINES[surface]
        return self.timeout_s

    def _post(
        self,
        url: str,
        payload: dict[str, Any],
        *,
        token: str,
        timeout: float | None,
    ) -> httpx.Response:
        # No default: ``None`` is a meaningful deadline here ("wait as long as
        # the daemon needs"), so a caller that forgets to pass one must fail
        # loudly rather than silently inherit either interpretation.
        deadline = timeout
        try:
            return httpx.post(
                url,
                json=payload,
                headers={"Authorization": f"Bearer {token}"},
                timeout=deadline,
            )
        except httpx.TimeoutException as exc:
            # Caught above RequestError, which it subclasses, because the two are
            # opposite facts about the remote side. A deadline says nothing about
            # whether the work failed — the daemon holds one lock for the whole
            # call and keeps going — yet this arrived as "is unavailable", which
            # the CLI renders as a failed command with a "check daemon readiness"
            # repair. Users then re-ran the operation that was still running.
            raise _client_deadline_exceeded(
                label=self.label, deadline=deadline, cause=exc
            ) from exc
        except httpx.RequestError as exc:
            # Nothing answered, so there is no status to carry: this is the one
            # failure that really is "unreachable", and it is what every other
            # failure here used to be reported as.
            raise ContextEngineDisabled(f"{self.label} is unavailable: {exc}") from exc

    def _result(self, response: httpx.Response, url: str) -> Any:
        """The decoded result, or the error the response carries.

        The credential check runs *before* the body is parsed. A managed host
        behind a proxy answers 401 with an HTML page, and the JSON decode would
        turn that into "returned a non-JSON response" — technically true, and it
        buries the only fact worth reporting: the endpoint is up and does not
        accept this key.
        """
        if response.status_code in _CREDENTIAL_REFUSED:
            raise _credential_refused(
                status_code=response.status_code,
                reason=_response_reason(response),
                label=self.label,
                endpoint=url,
            )
        data = _response_json(response, label=self.label)
        if response.status_code >= 400 or not data.get("ok", False):
            _raise_remote_error(
                data,
                status_code=response.status_code,
                endpoint=url,
                label=self.label,
            )
        return decode(data.get("result"))

    def _rpc_discovery(self) -> dict[str, str]:
        discovery = self.daemon.discovery()
        if discovery is None:
            raise ContextEngineDisabled(
                f"{self.label} is not running. Run 'potpie setup' to start it."
            )
        if not discovery.get("base_url") or not discovery.get("token"):
            raise ContextEngineDisabled(
                f"{self.label} is running but does not expose the CLI RPC surface. "
                "Run 'potpie daemon restart'."
            )
        return discovery


class RemoteSurface:
    """Dynamic proxy for a ``HostShell`` surface or nested capability port."""

    _NESTED = frozenset(
        {
            "mutation",
            "claim_query",
            "semantic",
            "inspection",
            "analytics",
            "snapshot",
        }
    )
    _REMOTE_ATTRS = frozenset({"profile"})

    def __init__(self, client: DaemonRpcClient, path: str) -> None:
        self._client = client
        self._path = path

    def __getattr__(self, name: str) -> Any:
        if name in self._NESTED:
            return RemoteSurface(self._client, f"{self._path}.{name}")
        if name in self._REMOTE_ATTRS:
            return self._client.attr(self._path, name)

        def _call(*args: Any, **kwargs: Any) -> Any:
            return self._client.call(self._path, name, *args, **kwargs)

        return _call


@dataclass
class RemoteHostShell:
    """CLI facade whose service calls are executed inside the daemon."""

    rpc: DaemonRpcClient = field(default_factory=DaemonRpcClient)
    profile: str = "local"

    def __post_init__(self) -> None:
        self.daemon = self.rpc.daemon
        self.agent_context = RemoteSurface(self.rpc, "agent_context")
        self.graph = RemoteSurface(self.rpc, "graph")
        self.graph_workbench = RemoteSurface(self.rpc, "graph_workbench")
        self.pots = RemoteSurface(self.rpc, "pots")
        self.skills = RemoteSurface(self.rpc, "skills")
        self.backend = RemoteSurface(self.rpc, "backend")
        self.ledger = RemoteSurface(self.rpc, "ledger")
        self.resources = RemoteSurface(self.rpc, "resources")
        self.nudge = RemoteSurface(self.rpc, "nudge")
        self.config = RemoteSurface(self.rpc, "config")
        self.installer = RemoteSurface(self.rpc, "installer")
        self.auth = RemoteSurface(self.rpc, "auth")
        self.setup = RemoteSurface(self.rpc, "setup")


def _response_json(
    response: httpx.Response, *, label: str = "Potpie daemon"
) -> dict[str, Any]:
    try:
        data = response.json()
    except ValueError as exc:
        raise ContextEngineDisabled(
            f"{label} returned a non-JSON response ({response.status_code})."
        ) from exc
    if not isinstance(data, dict):
        raise ContextEngineDisabled(f"{label} returned an invalid response.")
    return data


def _response_reason(response: httpx.Response) -> str:
    """Whatever the endpoint said, out of a body of unknown shape.

    A refusal arrives as this daemon's own error envelope, as FastAPI's
    ``{"detail": ...}``, or as a reverse proxy's HTML — and which one it is says
    nothing, while the sentence inside says everything.
    """
    try:
        data = response.json()
    except ValueError:
        return _readable(response.text)
    return _envelope_reason(data) if isinstance(data, dict) else ""


def _envelope_reason(data: dict[str, Any]) -> str:
    error = data.get("error")
    if isinstance(error, dict) and error.get("message"):
        return str(error["message"])
    return str(data.get("detail") or "")


def _readable(text: str) -> str:
    """A body worth quoting back, or nothing.

    Markup is dropped rather than clipped: a proxy's error page contributes no
    fact the status code has not already given, and pasting a screenful of HTML
    into a CLI error hides the sentence in front of it.
    """
    collapsed = " ".join((text or "").split())
    if not collapsed or collapsed.startswith("<"):
        return ""
    return collapsed if len(collapsed) <= 200 else f"{collapsed[:197]}..."


def _client_deadline_exceeded(
    *,
    label: str,
    deadline: float | None,
    cause: Exception,
) -> ContextEngineDisabled:
    """The error for a request this client stopped waiting for.

    Reported as ``unavailable`` like the rest of the family — the exit-code
    table is not this function's to change — but it must not say the work
    failed, because it does not know that and usually the opposite is true. The
    daemon runs the whole call under one lock and finishes it; the client simply
    left. Told "the daemon is unavailable", operators and CI re-ran the command,
    which is how a cold ``potpie --json setup`` produced duplicate work on top of
    a false failure. The only honest next step is to go and look.
    """
    waited = f" within {deadline:g}s" if deadline is not None else ""
    exc = ContextEngineDisabled(
        f"{label} did not answer{waited} ({cause.__class__.__name__}). That is a "
        "client-side deadline, not a failure: the request may still be running "
        "there."
    )
    exc.recommended_next_action = (
        "check whether it finished ('potpie status', or 'potpie daemon logs') "
        "before re-running — re-running a long operation can duplicate its work"
    )
    return exc


def _credential_refused(
    *,
    status_code: int,
    reason: str,
    label: str,
    endpoint: str | None,
) -> ContextEngineDisabled:
    """The error for an endpoint that answered and rejected the key.

    Reported as ``unavailable`` like every other member of this family — the
    exit codes and the error contract are not this function's to change — but
    the sentence has to say which of the two happened. "The managed host did not
    answer" for a host that answered in milliseconds sends the reader to check
    whether a service is running, restart it, and re-check the network, when the
    entire repair is one wrong token; the status code is the only thing that
    ever knew the difference, and it used to be dropped one frame above.

    ``status_code`` rides along on the exception so a caller that wraps this in
    its own wording can still tell a refusal from a silence without parsing the
    message.
    """
    where = f" at {endpoint}" if endpoint else ""
    tail = f": {reason}" if reason else ""
    exc = ContextEngineDisabled(
        f"{label}{where} refused the credential (HTTP {status_code}){tail}"
    )
    exc.status_code = status_code
    exc.recommended_next_action = (
        "for a managed host, re-run 'potpie host set <url> --token <key>' with "
        "the key that service expects; for the local daemon, run "
        "'potpie daemon restart' to pick up the current token"
    )
    return exc


def _raise_remote_error(
    data: dict[str, Any],
    *,
    status_code: int | None = None,
    endpoint: str | None = None,
    label: str = "Potpie daemon",
) -> None:
    """Re-raise a remote failure as the domain error it started out as.

    ``status_code`` is carried in because the transport knows something no
    envelope can: a ``401``/``403`` means this endpoint answered and refused the
    key. Every response without a recognised error envelope — which is what
    FastAPI's ``{"detail": ...}`` is — fell through to
    ``ContextEngineDisabled("Potpie daemon request failed.")``, so a rejected
    managed-host token was reported as the local daemon failing a request:
    wrong machine, wrong problem, and nothing left in the exception for a caller
    to notice the difference with.
    """
    if status_code is not None and status_code in _CREDENTIAL_REFUSED:
        raise _credential_refused(
            status_code=status_code,
            reason=_envelope_reason(data),
            label=label,
            endpoint=endpoint,
        )
    error = data.get("error") or {}
    code = str(error.get("code") or "daemon_error")
    message = str(error.get("message") or f"{label} request failed.")
    detail = error.get("detail")
    next_action = error.get("recommended_next_action")
    if code == "not_implemented":
        raise CapabilityNotImplemented(
            str(error.get("capability") or message),
            detail=detail,
            recommended_next_action=next_action,
        )
    if code == "pot_not_found":
        raise PotNotFound(message)
    if code == "unavailable":
        exc = ContextEngineDisabled(message)
        # The specific repair is the entire value of the domain error: a failed
        # pot teardown knows exactly what to restart, and dropping the hint here
        # leaves the CLI printing the generic "check readiness with 'potpie
        # doctor'" for a failure that never needed diagnosing.
        if next_action is not None:
            setattr(exc, "recommended_next_action", next_action)
        raise exc
    if code in {"validation_error", "value_error"} and message.startswith(
        _LEGACY_SURFACE_REFUSAL
    ):
        # Compatibility, not policy: this host has not adopted the
        # ``not_implemented`` answer that ``_validate_rpc_target`` now sends, so
        # the only place the truth can be recovered is here, in the one table
        # that decodes the wire envelope into domain errors. Splitting this row
        # out into a wrapper would leave the next reader with two tables to find.
        # Delete the branch once every deployment answers for itself.
        surface = message[len(_LEGACY_SURFACE_REFUSAL) :].strip()
        raise CapabilityNotImplemented(
            surface,
            detail=f"{label} does not implement the '{surface}' surface",
            recommended_next_action=(
                "run this command against a host that does, e.g. "
                "'potpie --host local ...'"
            ),
        )
    if code in {"validation_error", "value_error"}:
        exc = ValueError(message)
        # Re-attach structured guidance so the CLI error boundary can surface
        # detail/recommended_next_action exactly as with an in-process service.
        # ``error_code`` carries the domain's own stable code (a resource store
        # reports ``resource_chunk_too_large``, not just "validation_error"),
        # which the command's error envelope reports verbatim.
        if detail is not None:
            setattr(exc, "detail", detail)
        if next_action is not None:
            setattr(exc, "recommended_next_action", next_action)
        if error.get("error_code"):
            setattr(exc, "code", str(error["error_code"]))
        raise exc
    exc = ContextEngineDisabled(message)
    # Anything the envelope did not classify still keeps the status it arrived
    # with, so "the endpoint answered" survives the trip even here.
    if status_code is not None:
        exc.status_code = status_code
    raise exc


__all__ = ["DaemonRpcClient", "RemoteHostShell", "RemoteSurface"]
