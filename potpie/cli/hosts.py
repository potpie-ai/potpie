"""Host registry — which context-engine a command talks to.

Until now the CLI had exactly one host per process: ``CONTEXT_ENGINE_HOME``
picked a ``discovery.json``, and that was the daemon you got. Pots on a managed
service were invisible unless you pointed the whole CLI at it, one home at a
time. This module makes the host a *choice* the CLI can make per command, so
``pot list`` can show both and ``graph commit`` can route to whichever host owns
the pot.

Two origins, matching the vocabulary ``pot list`` already used for its
unimplemented flags:

``local``
    The daemon (or in-process host) at ``default_home()``. Always available;
    needs no configuration.
``managed``
    A hosted context-graph service, configured with ``potpie host set <url>``
    (``--token`` unless the service runs with auth disabled). This is what the
    old "managed pots require 'potpie login' — HU3" hint was pointing at, but
    it is stored under its own name rather than in the Potpie account fields:
    the account API is a different service at a different address, and sharing
    one ``api_base_url`` between them means configuring either silently
    repoints the other.

Three rules decide the awkward cases:

**Each host keeps its own active pot; the CLI keeps the active origin.**
A pot pointer is per host and stays there — the daemon has one, the service has
one per actor, and neither knows about the other. What the CLI adds is which
origin is *current*, so "the active pot" means "the active pot of the active
origin". ``potpie use managed:foo`` moves both. This is also why the pointer
cannot live in ``config.json``: that file is a host's own config service, and
asking a host which host is active is circular. It lives in
``<home>/cli_hosts.json``, client side, readable with no host at all.

**A bare ref resolves in the active origin first.**
Only if it is not there does the CLI look at the other origin, and finding it in
both is an ``ambiguous_pot`` error naming the qualified refs rather than a
guess. Pot names are per-host labels — ``default`` almost certainly exists in
both — so silently preferring one would eventually write to the wrong graph.
With no managed host configured this collapses to exactly the old behaviour.

**Enumeration degrades; targeting fails loud.**
``pot list`` with an unreachable service shows the local section and marks the
managed one unavailable, because a listing that dies when a remote is down is
useless offline. Anything aimed at a *specific* pot propagates the failure
instead: falling back to the other host would run the command against the wrong
graph, which is worse than not running it.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final
from urllib.parse import urlsplit

from potpie_context_core.errors import ContextEngineDisabled

LOCAL: Final = "local"
MANAGED: Final = "managed"
ORIGINS: Final[tuple[str, ...]] = (LOCAL, MANAGED)

_STATE_FILE: Final = "cli_hosts.json"

#: What a managed host with auth disabled sends instead of a key. The RPC client
#: refuses a discovery record with an empty token, so "no token" needs something
#: to be; see :meth:`StaticDiscovery.discovery` for why it is substituted there
#: and nowhere else.
NO_AUTH_TOKEN: Final = "no-auth"  # noqa: S105 - a placeholder for no credential

#: Process-local override for the current origin: a qualified ``--pot`` ref or
#: the global ``--host`` flag sets this for one invocation without persisting.
_current: dict[str, str | None] = {"origin": None}

#: Built hosts, keyed by what actually determines them — the home directory for
#: local, the endpoint *and credential* for managed — not merely by origin.
#: ``CONTEXT_ENGINE_HOME`` and a login can both change inside one process, and a
#: host cached under the bare origin would outlive the home it was built against
#: and keep talking to the old one.
_built: dict[tuple[str, ...], Any] = {}


def home_dir() -> Path:
    from potpie_context_engine.adapters.outbound.pots.local_pot_store import (
        default_home,
    )

    return Path(default_home())


# --- origin state ---------------------------------------------------------


class HostRegistryUnreadable(ContextEngineDisabled):
    """``cli_hosts.json`` is there, but nothing can be made of it.

    Reading it as ``{}`` — which is what swallowing the decode error did — makes
    a corrupt registry indistinguishable from a first run: every command reports
    "no managed host configured" and exits 0, and then the next command that
    persists *any* host state rewrites the file from that empty reading and
    takes the still-present url and token down with it. The recovery window was
    one command wide.

    So it fails, and it names the file: the token inside may exist nowhere else,
    and the only person who can decide between repairing and forgetting it is
    the one who can read the file.
    """

    def __init__(self, path: Path, reason: str) -> None:
        super().__init__(f"Host registry {path} is unreadable: {reason}")
        self.recommended_next_action = (
            f"repair {path} by hand, or run 'potpie host clear' to forget the "
            "managed host and re-add it with 'potpie host set <url> --token <token>'"
        )


class HostRegistryUnwritable(ContextEngineDisabled):
    """``cli_hosts.json`` could not be replaced.

    A half-written registry is worse than an unwritten one: it reads back as
    corrupt, which takes every later command down with it, and the token inside
    exists nowhere else. So the new contents land through a temp file and
    ``os.replace``, and the registry is either the old bytes or the new ones.

    This exists for the failure that survives even that. Without it an ENOSPC or
    a read-only directory surfaced as ``unexpected_cli_error`` / "Unexpected
    internal error." at exit 1, naming neither the file nor the repair — and the
    *next* command was the one that finally mentioned ``cli_hosts.json``.
    """

    def __init__(self, path: Path, reason: str) -> None:
        super().__init__(f"Host registry {path} could not be written: {reason}")
        self.recommended_next_action = (
            f"check permissions and free space on {path.parent}, then run "
            "'potpie host list' to confirm the stored host is still there"
        )


class HostRegistryNotSalvageable(ContextEngineDisabled):
    """The unreadable registry could not be moved aside, so it was left alone.

    ``host clear`` is the only writer allowed past the corrupt-file guard, and
    the only reason it is allowed is that the bytes it cannot parse are kept.
    Swallowing a failed move made the two outcomes indistinguishable — exit 0,
    ``salvaged_registry: null``, the same line either way — while one of them had
    destroyed what may be the only copy of a token. Nothing is overwritten until
    the copy exists.
    """

    def __init__(self, path: Path, kept: Path, reason: str) -> None:
        super().__init__(
            f"Host registry {path} is unreadable and could not be moved to "
            f"{kept}: {reason}. It has been left exactly as it was."
        )
        self.recommended_next_action = (
            f"move {path} aside by hand (it may hold the only copy of a managed "
            "token), then run 'potpie host clear' again"
        )


def _state_path() -> Path:
    return home_dir() / _STATE_FILE


def _read_state() -> dict[str, Any]:
    """The registry file's contents, or ``{}`` when there is no file yet.

    Only *absent* is silent. A file that exists and cannot be parsed is a
    different situation with a different repair, and conflating the two is what
    :class:`HostRegistryUnreadable` exists to stop.
    """
    path = _state_path()
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except OSError as exc:
        raise HostRegistryUnreadable(path, str(exc)) from exc
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HostRegistryUnreadable(path, f"invalid JSON ({exc})") from exc
    if not isinstance(parsed, dict):
        raise HostRegistryUnreadable(
            path, f"expected a JSON object, found {type(parsed).__name__}"
        )
    return parsed


def persisted_origin() -> str:
    """The origin ``potpie use`` last selected, defaulting to ``local``."""
    value = _read_state().get("active_origin")
    return value if value in ORIGINS else LOCAL


def _write_state(state: dict[str, Any], *, force: bool = False) -> None:
    """Replace the registry atomically, or fail naming the file.

    ``force`` skips the read-back guard and belongs to exactly one caller:
    :func:`clear_managed_endpoint`, whose whole job is to survive the corrupt
    file the guard exists to protect.
    """
    # Read first and throw the result away: every caller builds `state` by
    # editing a reading of this file, and one that skipped that step would
    # overwrite a registry that is corrupt but still holds a recoverable token.
    # The guard belongs here rather than in the callers because the thing being
    # protected is the file, not any one command.
    if not force:
        _read_state()
    path = _state_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
        )
        temp = Path(temp_name)
        try:
            # fdopen takes ownership of the descriptor immediately, so nothing
            # between here and the close can leak it.
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                # Credential permissions before the token is in the file and
                # before the file is reachable under its real name. The old
                # order — write, then chmod — left a freshly created registry
                # world-readable for the width of that gap under a lax umask.
                os.chmod(temp, 0o600)
                handle.write(json.dumps(state, indent=2) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            # The mode rides along with the inode, so this also re-applies 0600
            # to an overwrite of an existing file.
            os.replace(temp, path)
        except BaseException:
            temp.unlink(missing_ok=True)
            raise
    except OSError as exc:
        raise HostRegistryUnwritable(path, str(exc)) from exc


def set_persisted_origin(origin: str) -> None:
    require_origin(origin)
    state = _read_state()
    state["active_origin"] = origin
    _write_state(state)


def selected_origin() -> str:
    """The origin this invocation *asked* for, before any degrade.

    A per-invocation override (``--host``, or a qualified ref) wins over the
    persisted selection, so one command can reach across without moving the
    pointer under the next one.

    Split out from :func:`current_origin` because "what was selected" and "what
    is being talked to" are different answers exactly when something is wrong,
    and every diagnostic about the difference needs both.
    """
    override = _current["origin"]
    if override in ORIGINS:
        return str(override)
    return persisted_origin()


def current_origin() -> str:
    """The origin this invocation targets, after the unconfigured degrade.

    Only the *persisted* pointer degrades. A per-invocation override — ``--host``,
    ``--managed``, a ``managed:`` qualifier — is a target, and targeting fails
    loud: degrading one here is what let ``potpie --host managed pot list`` print
    the *local* pots at exit 0 under ``"active_origin": "managed"``, with no
    ``unavailable`` key to say the managed host was never asked. Left alone, the
    override reaches :func:`build_host`, which refuses with "No managed host
    configured" — and ``--host`` never gets that far, because
    :func:`potpie.cli.main._apply_host_override` refuses it at the door, where
    the error contract is what renders the refusal.
    """
    override = _current["origin"]
    if override in ORIGINS:
        return str(override)
    persisted = persisted_origin()
    # A persisted `managed` that is no longer configured would otherwise make
    # every command fail on a login that has since been cleared. Degrade to
    # local: the pointer is a convenience, not an authorization.
    if persisted == MANAGED and managed_endpoint() is None:
        return LOCAL
    return persisted


def origin_degraded() -> bool:
    """True when the origin in effect is not the origin that was asked for.

    The degrade in :func:`current_origin` is deliberate, but it is also
    invisible: commands keep exiting 0 while running against the *local* graph,
    and a pot name that exists on both hosts makes that look like nothing
    happened at all. Nothing here changes the routing — this only makes the
    state something ``potpie host list`` can say out loud.

    Stated as the difference between :func:`selected_origin` and
    :func:`current_origin` rather than re-deriving "managed and unconfigured"
    from :func:`persisted_origin`, because that re-derivation could disagree
    with the routing it claims to describe. It did, in both directions: it was
    blind to an origin that came from an override rather than the pointer, and
    it would now call an override that is *not* degraded — one heading for a
    loud refusal in :func:`build_host` — degraded, so ``host list`` reported
    ``configured: false``, ``active: true`` and ``degraded: false`` about the
    same host in one payload.
    """
    return selected_origin() != current_origin()


def origin_overridden() -> bool:
    """True when *this invocation* named its origin, rather than inheriting it.

    :func:`selected_origin` deliberately cannot answer this: it folds the
    override and the persisted pointer into one "what was asked for". That is
    right for reporting and wrong for the commands that are only ever local —
    ``setup`` provisions *this* machine, so a persisted ``managed`` pointer must
    keep working (it always has), while an explicit ``--host managed setup``
    has to fail loud rather than quietly provision somewhere the caller did not
    name. Telling those two apart is the only thing this exists for.
    """
    return _current["origin"] in ORIGINS


def set_current_origin(origin: str | None) -> None:
    """Override the origin for this process only (no persistence)."""
    if origin is not None:
        require_origin(origin)
    _current["origin"] = origin


def require_origin(origin: str) -> None:
    if origin not in ORIGINS:
        raise ValueError(
            f"unknown host origin: {origin!r} (expected one of {', '.join(ORIGINS)})"
        )


# --- refs -----------------------------------------------------------------


def split_ref(ref: str | None) -> tuple[str | None, str | None]:
    """Split ``managed:foo`` into ``("managed", "foo")``.

    Only a known origin counts as a prefix: pot *names* are user-chosen and may
    contain a colon, and stealing one would make a legitimate name unaddressable.
    """
    if not ref:
        return None, ref
    prefix, sep, rest = ref.partition(":")
    if sep and prefix in ORIGINS and rest:
        return prefix, rest
    return None, ref


def qualify(origin: str, ref: str) -> str:
    return f"{origin}:{ref}"


# --- hosts ----------------------------------------------------------------


def _resolve_managed() -> tuple[tuple[str, str] | None, str | None]:
    """``(endpoint, problem)`` — the managed host, and why it is unusable.

    Two answers rather than one because "no managed host" and "a managed host
    that can never be reached" are different states with different repairs.
    Collapsing them is how ``POTPIE_MANAGED_URL=http://127.0.0.1:8090x`` came to
    be reported as ``configured: true`` at exit 0: an address every later command
    fails on, named nowhere. The environment goes through the same normalizer as
    ``host set`` so a typo cannot enter by the one door that skips the CLI.
    """
    env_url = managed_env_override()
    if env_url:
        source, base_url = "POTPIE_MANAGED_URL", env_url
        token = normalize_managed_token(os.getenv("POTPIE_MANAGED_TOKEN"))
    else:
        state = _read_state()
        source = str(_state_path())
        base_url = str(state.get("managed_url") or "").strip()
        token = normalize_managed_token(str(state.get("managed_token") or ""))
    if not base_url:
        return None, None
    try:
        return (normalize_managed_url(base_url), token), None
    except ValueError as exc:
        return None, f"{source} holds an unusable managed host: {exc}"


def managed_endpoint() -> tuple[str, str] | None:
    """``(base_url, token)`` for the managed host, or ``None`` if unset.

    Deliberately *not* the account fields ``api_base_url`` / potpie API key.
    Those belong to the Potpie account API — a different service, quite
    possibly at a different address — so sharing them means configuring either
    one silently repoints the other, and an account login with ``--url`` would
    overwrite a self-hosted context-graph address.

    It is not a provider-credentials entry either: that door is GitHub-shaped
    (it demands an ``access_token`` and writes the GitHub keychain item), so a
    context-graph endpoint does not fit through it. This is host-registry
    state, and it lives with the rest of the host-registry state — in a file
    written 0600, because it may carry a token.

    The environment overrides the file, for a one-off run against a scratch
    service without persisting anything.

    The token comes back exactly as stored, empty string included: a service
    with auth disabled *has* no token, and the placeholder that lets the RPC
    client accept that belongs at the transport seam
    (:meth:`StaticDiscovery.discovery`), not here. Substituting it here put it
    above the point where ``build_host`` delegates to
    :func:`build_managed_host`, so the two construction paths disagreed about
    what an empty token means and ``host set <url>`` with no ``--token`` — the
    documented auth-disabled setup — was refused before a socket was opened.
    """
    return _resolve_managed()[0]


def managed_env_override() -> str | None:
    """The ``POTPIE_MANAGED_URL`` address that outranks the stored one, if any.

    Exposed rather than read at each call site so the one door that skips
    ``host set`` has one reader, and so a command that *writes* the registry can
    say that its write is not what the next command will use. ``host set``
    reported ``managed host → <url>`` at exit 0 while every later command kept
    talking to the environment's host, which is the same silence as writing to
    the wrong file.
    """
    url = os.getenv("POTPIE_MANAGED_URL", "").strip()
    return url or None


def managed_endpoint_problem() -> str | None:
    """Why the configured managed endpoint cannot be used, if it cannot.

    ``None`` covers both "usable" and "not configured"; only a configured-but-
    unusable endpoint has something to say. :func:`managed_endpoint` reports it
    as absent so nothing routes at it — this is what stops that from being
    silent.
    """
    return _resolve_managed()[1]


def stored_managed_token() -> str:
    """The token in the registry *file*, ignoring the environment override.

    ``host set`` writes the file, so the file is the only thing it can destroy.
    Reading the override here would make ``POTPIE_MANAGED_TOKEN`` in a shell
    block a first-time write over a token that write cannot touch.
    """
    return str(_read_state().get("managed_token") or "").strip()


def normalize_managed_url(url: str) -> str:
    """Validate a managed base URL and return it without its trailing slash.

    Syntax is checked before anything is stored, ``--no-check`` included: a URL
    that cannot be parsed is not "unreachable, try later", it is unusable, and
    storing one means every later command fails on an endpoint the CLI reported
    as saved. The typo that motivated this — a stray character in the port —
    parses fine as a string and only blows up at connect time, which is a
    command or two after the evidence.

    Raises ``ValueError``, which the CLI error boundary renders as
    ``validation_error`` / exit 1.
    """
    raw = (url or "").strip()
    if not raw:
        raise ValueError("Managed host URL is empty.")
    try:
        parts = urlsplit(raw)
    except ValueError as exc:
        raise ValueError(f"Managed host URL {raw!r} is not a URL: {exc}.") from exc
    if parts.scheme not in ("http", "https"):
        raise ValueError(
            f"Managed host URL {raw!r} needs an http:// or https:// scheme."
        )
    if not parts.hostname:
        raise ValueError(f"Managed host URL {raw!r} names no host.")
    try:
        # Reading the property *is* the check: urlsplit keeps the netloc
        # verbatim and only parses the port when asked for it.
        parts.port
    except ValueError as exc:
        raise ValueError(
            f"Managed host URL {raw!r} has an invalid port: {exc}."
        ) from exc
    return raw.rstrip("/")


def normalize_managed_token(token: str | None) -> str:
    """The token in the form it will be *sent* — which is the only form to check.

    Lives beside :func:`normalize_managed_url` because it is the same rule: the
    pair that is validated has to be the pair that is stored. ``_resolve_managed``
    has always stripped what it read, and nothing on the way in did, so a key
    pasted with a trailing space was probed as ``Bearer k3y `` by
    ``host set --check`` and sent as ``Bearer k3y`` by every command after it —
    a credential refused at the door in the one form that would have worked. A
    whitespace-only token was worse: probed as ``Bearer   `` and then sent as
    the auth-disabled placeholder.

    Empty is a real answer (a service with auth disabled has no token), so this
    normalizes and never refuses; :meth:`StaticDiscovery.discovery` is where
    "no token" becomes transport credentials.
    """
    return (token or "").strip()


def set_managed_endpoint(base_url: str, token: str | None) -> None:
    state = _read_state()
    # Validated here as well as at the CLI door, so no code path can put an
    # unusable endpoint — or a credential in a form nothing will send — in the
    # file.
    state["managed_url"] = normalize_managed_url(base_url)
    state["managed_token"] = normalize_managed_token(token)
    _write_state(state)
    _built.clear()


def clear_managed_endpoint() -> Path | None:
    """Forget the managed host; returns where an unreadable registry was kept.

    The one writer that survives a corrupt registry. Every other command fails
    loud on one, because the token inside may exist nowhere else and only the
    user can choose between repairing and forgetting it — but this *is* the
    forgetting, and refusing here left a hand edit as the only way out of the
    state whose documented remedy is "forget the managed host".

    The unreadable bytes are moved aside rather than dropped: the user asked to
    forget the host, not to lose a token they may never have been able to read.
    That move is a *precondition* of the overwrite below, not a courtesy beside
    it — see :func:`_quarantine_state`.
    """
    try:
        state = _read_state()
    except HostRegistryUnreadable:
        salvaged = _quarantine_state()
        _write_state({}, force=True)
        _built.clear()
        return salvaged
    state.pop("managed_url", None)
    state.pop("managed_token", None)
    _write_state(state)
    _built.clear()
    return None


def _quarantine_state() -> Path:
    """Move the unreadable registry aside, or refuse naming both paths.

    Raises :class:`HostRegistryNotSalvageable` rather than returning ``None``:
    the caller overwrites the file immediately afterwards, so "could not save
    the bytes" and "saved the bytes" cannot be allowed to reach it as the same
    answer.
    """
    path = _state_path()
    kept = path.parent / f"{path.name}.corrupt"
    try:
        os.replace(path, kept)
    except OSError as exc:
        raise HostRegistryNotSalvageable(path, kept, str(exc)) from exc
    return kept


@dataclass(frozen=True, slots=True)
class StaticDiscovery:
    """Stands in for ``Daemon`` where the endpoint is already known.

    ``DaemonRpcClient`` asks its daemon for ``{base_url, token}`` and cares
    about nothing else, so a managed host needs no process lifecycle — which is
    the point: there is no local process to start, stop, or recover.
    """

    base_url: str
    token: str

    def discovery(self) -> dict[str, str]:
        # The single seam where a managed token becomes transport credentials,
        # which is why the auth-disabled placeholder is applied here and nowhere
        # above it: every managed host — stored or probe — is built through
        # build_managed_host, so both paths cannot disagree about what an empty
        # token means. They did, and `host set <url>` with no --token failed the
        # RPC client's empty-token guard before it ever reached the network.
        return {"base_url": self.base_url, "token": self.token or NO_AUTH_TOKEN}

    def __getattr__(self, name: str) -> Any:
        from potpie_context_core.errors import CapabilityNotImplemented

        raise CapabilityNotImplemented(
            f"daemon.{name}",
            detail="the managed host is a remote service, not a local daemon process",
            recommended_next_action="run daemon commands against the local host ('potpie --host local ...')",
        )


def build_managed_host(base_url: str, token: str) -> Any:
    """Build a managed host for an explicit endpoint, reading no stored state.

    ``host set`` has to prove an endpoint answers *before* it replaces the
    stored one. Writing first and clearing on failure is not a rollback: it
    drops the previous url and token together, and a token that was pasted in
    once is recoverable from nowhere. Probing first is only possible if a host
    can be built from a pair nobody has saved yet — which is what this takes.

    Deliberately not cached: ``_built`` is keyed by base_url, so a probe host
    carrying a token that was *rejected* for that address would be handed
    straight back to the next ``build_host`` for the same address. Caching stays
    with the stored endpoint, in :func:`build_host`.
    """
    from potpie.daemon.client import DaemonRpcClient, RemoteHostShell

    return RemoteHostShell(
        rpc=DaemonRpcClient(
            daemon=StaticDiscovery(base_url=base_url, token=token),
            label=managed_label(base_url),
        ),
        profile=MANAGED,
    )


def build_host(origin: str) -> Any:
    """Build the host for ``origin``. Raises if managed is not configured."""
    require_origin(origin)

    if origin == LOCAL:
        key = (LOCAL, str(home_dir()))
        cached = _built.get(key)
        if cached is not None:
            return cached
        host = _build_local_host()
    else:
        endpoint = managed_endpoint()
        if endpoint is None:
            raise ContextEngineDisabled(
                "No managed host configured. Run 'potpie host set <base-url>'."
            )
        base_url, token = endpoint
        # The credential is part of the host's identity, not a detail of it.
        # `_built` is cleared only in the process that writes the registry, so a
        # host keyed by the address alone outlives the token it was built with:
        # the daemon serving `/ui?host=managed` kept sending a rotated-away key
        # until it was restarted, and reported the resulting 401 as the managed
        # service being down.
        key = (MANAGED, base_url, token)
        cached = _built.get(key)
        if cached is not None:
            return cached
        # Drop the pre-rotation host instead of accumulating one entry per
        # rotation for the life of a long-running process.
        for stale in [k for k in _built if k[:2] == (MANAGED, base_url) and k != key]:
            _built.pop(stale, None)
        host = build_managed_host(base_url, token)

    _built[key] = host
    return host


def _build_local_host() -> Any:
    mode = os.getenv("CONTEXT_ENGINE_HOST_MODE", "").strip().lower()
    if mode != "in_process":
        try:
            from potpie.daemon.client import RemoteHostShell
        except ModuleNotFoundError as exc:
            if exc.name not in {"potpie.daemon", "potpie.daemon.client"}:
                raise
        else:
            return RemoteHostShell()

    from potpie_context_engine.bootstrap.host_wiring import build_host_shell

    return build_host_shell()


def configured_origins() -> tuple[str, ...]:
    """Origins worth talking to: local always, managed once logged in."""
    return ORIGINS if managed_endpoint() is not None else (LOCAL,)


#: What each host answered at ``/surfaces``, keyed by the endpoint *and* the
#: credential — a 401 is one of the outcomes recorded as "does not say", and a
#: rotated key inside one process must not keep reading the old refusal. A
#: cached ``None`` is a real answer, not a miss; see :func:`advertised_surfaces`.
_surfaces: dict[tuple[str, str], frozenset[str] | None] = {}


def advertised_surfaces(host: Any) -> frozenset[str] | None:
    """The RPC surfaces ``host`` says it serves, or ``None`` if it does not say.

    **``None`` means "this host does not say", never "this host has nothing".**
    That distinction is the whole safety property of this function. The managed
    service predates ``GET /surfaces`` and answers 404; a client that read
    silence as an empty set would conclude that a host implementing everything
    implements nothing, and refuse commands that work perfectly well today.
    Nothing here gates a call — the refusal, when it comes, comes from the host
    — this only lets ``doctor`` say out loud which surfaces a host is missing
    relative to what this CLI knows how to call.

    Every failure is silence: a 404, a rejected credential, a proxy's HTML, an
    unreachable address, a host object with no transport at all. A diagnostic
    that raised while collecting diagnostics would be the bug it is looking for.
    """
    try:
        rpc = getattr(host, "rpc", None)
        discovery = rpc.daemon.discovery() if rpc is not None else None
    except Exception:  # noqa: BLE001 - see the docstring: silence, never a raise
        return None
    # An in-process host, or a test double, has no endpoint to ask. `isinstance`
    # rather than duck-typing on purpose: a MagicMock answers `__getitem__` with
    # another MagicMock, and formatting that into a URL would send a diagnostic
    # off to open a socket against nonsense.
    if not isinstance(discovery, dict):
        return None
    base_url = str(discovery.get("base_url") or "").rstrip("/")
    token = str(discovery.get("token") or "")
    if not base_url.startswith(("http://", "https://")):
        return None
    key = (base_url, token)
    if key in _surfaces:
        return _surfaces[key]
    answer = _fetch_surfaces(base_url, token)
    _surfaces[key] = answer
    return answer


def _fetch_surfaces(base_url: str, token: str) -> frozenset[str] | None:
    """One authenticated GET, with every failure collapsed to "does not say".

    The timeout is short and fixed: this is a diagnostic on a path nothing hot
    takes, and a host that is slow to answer a question about *itself* is one
    more thing not worth waiting on inside ``doctor``.
    """
    import httpx

    try:
        response = httpx.get(
            f"{base_url}/surfaces",
            headers={"Authorization": f"Bearer {token}"},
            timeout=5.0,
        )
        if response.status_code != 200:
            return None
        body = response.json()
    except Exception:  # noqa: BLE001 - transport, JSON, anything: it did not say
        return None
    if not isinstance(body, dict) or not isinstance(body.get("surfaces"), list):
        return None
    return frozenset(str(name) for name in body["surfaces"])


def managed_label(base_url: str) -> str:
    """``Managed (host:port)`` for an explicit endpoint.

    Takes the address rather than reading the stored one so a host built for an
    endpoint that is not stored — the ``host set`` probe — labels its own
    errors with the address it is actually calling, not the one it is about to
    replace.
    """
    trimmed = base_url.rstrip("/")
    for prefix in ("https://", "http://"):
        trimmed = trimmed[len(prefix) :] if trimmed.startswith(prefix) else trimmed
    return f"Managed ({trimmed})"


def origin_label(origin: str) -> str:
    """Human label for a section header, naming the endpoint for managed."""
    if origin == LOCAL:
        return "Local (daemon)"
    endpoint = managed_endpoint()
    if endpoint is None:
        return "Managed"
    return managed_label(endpoint[0])


def reset_for_tests() -> None:
    _built.clear()
    _surfaces.clear()
    _current["origin"] = None


__all__ = [
    "LOCAL",
    "MANAGED",
    "ORIGINS",
    "NO_AUTH_TOKEN",
    "HostRegistryNotSalvageable",
    "HostRegistryUnreadable",
    "HostRegistryUnwritable",
    "StaticDiscovery",
    "advertised_surfaces",
    "build_host",
    "build_managed_host",
    "clear_managed_endpoint",
    "set_managed_endpoint",
    "configured_origins",
    "current_origin",
    "home_dir",
    "managed_endpoint",
    "managed_endpoint_problem",
    "managed_env_override",
    "managed_label",
    "normalize_managed_token",
    "normalize_managed_url",
    "origin_degraded",
    "origin_label",
    "origin_overridden",
    "persisted_origin",
    "qualify",
    "require_origin",
    "reset_for_tests",
    "selected_origin",
    "set_current_origin",
    "set_persisted_origin",
    "split_ref",
    "stored_managed_token",
]
