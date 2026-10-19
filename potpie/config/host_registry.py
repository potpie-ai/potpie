"""Host registry: which context host a command talks to.

Two origins:

``local``
    The daemon (or in-process host) at ``default_home()``. Always available;
    needs no configuration.
``managed``
    A hosted context-graph service, configured with ``potpie host set <url>``.
    It is stored under its own name rather than in the Potpie account fields:
    the account API is a different service at a different address, and sharing
    one address between them means configuring either silently repoints the
    other.

The registry lives in ``<home>/cli_hosts.json``, client side, readable with no
host at all. It lives in ``potpie.config`` rather than the CLI because the local
daemon's explorer reads it too. It may hold a token, so it is written 0600 and
only ever replaced atomically.

Rules that decide the awkward cases:

**Each host keeps its own active pot; the CLI keeps the active origin.**
"The active pot" means "the active pot of the active origin". The pointer
cannot live in a host's own config, because asking a host which host is active
is circular.

**Enumeration degrades; targeting fails loud.**
A persisted ``managed`` pointer whose host is no longer configured degrades to
``local`` (the pointer is a convenience, not an authorization). An explicit
target (``--host managed``, a ``managed:`` ref) never degrades: running against
the other host's graph is worse than not running.

**Transport security.** A managed URL must be ``https://``. Plain ``http://`` is
accepted only for a loopback host, or for exactly one opted-in ``host[:port]``
(``host set <url> --allow-insecure`` for the stored host,
``POTPIE_ALLOW_INSECURE_HOST`` for ``POTPIE_MANAGED_URL``). The rule is checked
when ``host set`` runs and again whenever the registry is read, so a
hand-edited registry fails closed. An empty token is accepted only for a
loopback host.
"""

from __future__ import annotations

import ipaddress
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final
from urllib.parse import SplitResult, urlsplit

from potpie_context_engine.core.errors import ContextEngineDisabled

from potpie.config.local_paths import default_home

LOCAL: Final = "local"
MANAGED: Final = "managed"
ORIGINS: Final[tuple[str, ...]] = (LOCAL, MANAGED)

REGISTRY_FILE: Final = "cli_hosts.json"

MANAGED_URL_ENV: Final = "POTPIE_MANAGED_URL"
MANAGED_TOKEN_ENV: Final = "POTPIE_MANAGED_TOKEN"  # noqa: S105 - a variable name
ALLOW_INSECURE_HOST_ENV: Final = "POTPIE_ALLOW_INSECURE_HOST"

#: Process-local override for the current origin: the global ``--host`` flag
#: (and, later, a qualified pot ref) sets it for one invocation without
#: persisting anything.
_current: dict[str, str | None] = {"origin": None}

#: Base URLs this process has already warned about sending a key over plain
#: http, so the warning is one line per command rather than one per request.
_warned_insecure: set[str] = set()


def home_dir() -> Path:
    return default_home()


def registry_path() -> Path:
    return home_dir() / REGISTRY_FILE


# --- failures -------------------------------------------------------------


class HostRegistryUnreadable(ContextEngineDisabled):
    """``cli_hosts.json`` is there, but nothing can be made of it.

    Reading it as ``{}`` would make a corrupt registry indistinguishable from a
    first run, and the next command that persists any host state would rewrite
    the file from that empty reading, taking the url and token down with it. So
    it fails and names the file: the token inside may exist nowhere else.
    """

    def __init__(self, path: Path, reason: str) -> None:
        super().__init__(f"Host registry {path} is unreadable: {reason}")
        self.recommended_next_action = (
            f"repair {path} by hand, or run 'potpie host clear' to forget the "
            "managed host and re-add it with 'potpie host set <url> --token -' "
            "(key on stdin)"
        )


class HostRegistryUnwritable(ContextEngineDisabled):
    """``cli_hosts.json`` could not be replaced.

    New contents land through a temp file and ``os.replace``, so the registry
    is either the old bytes or the new ones. This names the file and the repair
    for the failure that survives even that (ENOSPC, a read-only directory).
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
    only because the bytes it cannot parse are kept. Nothing is overwritten
    until the copy exists.
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


# --- the file ---------------------------------------------------------------


def _read_state() -> dict[str, Any]:
    """The registry's contents, or ``{}`` when there is no file yet.

    Only *absent* is silent. A file that exists and cannot be parsed raises
    :class:`HostRegistryUnreadable`.
    """
    path = registry_path()
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


def _write_state(state: dict[str, Any], *, force: bool = False) -> None:
    """Replace the registry atomically at 0600, or fail naming the file.

    ``force`` skips the read-back guard and belongs to exactly one caller,
    :func:`clear_managed_endpoint`, whose job is to survive the corrupt file
    the guard protects.
    """
    # Read first and discard: a caller that skipped building `state` from a
    # reading would otherwise overwrite a corrupt registry that still holds a
    # recoverable token. The guard is on the file, not on any one command.
    if not force:
        _read_state()
    path = registry_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
        )
        temp = Path(temp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                # Credential permissions before the token is in the file and
                # before the file is reachable under its real name.
                os.chmod(temp, 0o600)
                handle.write(json.dumps(state, indent=2) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            # The mode rides along with the inode, so an overwrite of an
            # existing file is 0600 too.
            os.replace(temp, path)
        except BaseException:
            temp.unlink(missing_ok=True)
            raise
    except OSError as exc:
        raise HostRegistryUnwritable(path, str(exc)) from exc


def _quarantine_state() -> Path:
    """Move the unreadable registry aside, or refuse naming both paths."""
    path = registry_path()
    kept = path.parent / f"{path.name}.corrupt"
    try:
        os.replace(path, kept)
    except OSError as exc:
        raise HostRegistryNotSalvageable(path, kept, str(exc)) from exc
    return kept


# --- origins ----------------------------------------------------------------


def require_origin(origin: str) -> None:
    if origin not in ORIGINS:
        raise ValueError(
            f"unknown host origin: {origin!r} (expected one of {', '.join(ORIGINS)})"
        )


def persisted_origin() -> str:
    """The origin last selected with ``host use``, defaulting to ``local``."""
    value = _read_state().get("active_origin")
    return value if value in ORIGINS else LOCAL


def set_persisted_origin(origin: str) -> None:
    require_origin(origin)
    state = _read_state()
    state["active_origin"] = origin
    _write_state(state)


def set_current_origin(origin: str | None) -> None:
    """Override the origin for this process only (no persistence)."""
    if origin is not None:
        require_origin(origin)
    _current["origin"] = origin


def selected_origin() -> str:
    """The origin this invocation asked for, before any degrade."""
    override = _current["origin"]
    if override in ORIGINS:
        return str(override)
    return persisted_origin()


def current_origin() -> str:
    """The origin this invocation targets, after the unconfigured degrade.

    Only the persisted pointer degrades. An override is a target, and targeting
    fails loud: ``--host managed`` is refused at the CLI door when nothing is
    configured, rather than quietly running against the local graph.
    """
    override = _current["origin"]
    if override in ORIGINS:
        return str(override)
    persisted = persisted_origin()
    if persisted == MANAGED and managed_endpoint() is None:
        return LOCAL
    return persisted


def origin_degraded() -> bool:
    """True when the origin in effect is not the origin that was asked for."""
    return selected_origin() != current_origin()


def origin_overridden() -> bool:
    """True when this invocation named its origin rather than inheriting it."""
    return _current["origin"] in ORIGINS


# --- refs -----------------------------------------------------------------


def split_ref(ref: str | None) -> tuple[str | None, str | None]:
    """Split ``managed:foo`` into ``("managed", "foo")``.

    Only a known origin counts as a prefix: pot names are user-chosen and may
    contain a colon.
    """
    if not ref:
        return None, ref
    prefix, sep, rest = ref.partition(":")
    if sep and prefix in ORIGINS and rest:
        return prefix, rest
    return None, ref


def qualify(origin: str, ref: str) -> str:
    return f"{origin}:{ref}"


# --- address and credential rules -------------------------------------------


def is_loopback_host(hostname: str | None) -> bool:
    """``localhost``, ``127.0.0.0/8`` or ``::1``."""
    host = (hostname or "").strip().lower().strip("[]")
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _host_port_key(parts: SplitResult) -> str:
    host = (parts.hostname or "").lower()
    if ":" in host:
        host = f"[{host}]"
    port = parts.port
    return f"{host}:{port}" if port is not None else host


def normalize_insecure_host(value: str) -> str:
    """Validate an insecure opt-in and return it as ``host[:port]``.

    Raises ``ValueError`` for anything that is not exactly a host with an
    optional port: a URL, a path or credentials would make "exactly this host"
    ambiguous.
    """
    raw = (value or "").strip()
    if not raw:
        raise ValueError("The insecure-host opt-in is empty.")
    if any(mark in raw for mark in ("/", "@", "?", "#")):
        raise ValueError(
            f"The insecure-host opt-in {raw!r} is not a <host>[:<port>] "
            "(no scheme, path or credentials)."
        )
    parts = urlsplit(f"//{raw}")
    if not parts.hostname:
        raise ValueError(f"The insecure-host opt-in {raw!r} names no host.")
    try:
        parts.port
    except ValueError as exc:
        raise ValueError(
            f"The insecure-host opt-in {raw!r} has an invalid port: {exc}."
        ) from exc
    return _host_port_key(parts)


def insecure_host_key(url: str) -> str | None:
    """The ``host[:port]`` an opt-in must name for ``url`` to be accepted.

    ``None`` when ``url`` needs no opt-in (https, or a loopback host) or does
    not parse; :func:`normalize_managed_url` reports the latter.
    """
    try:
        parts = urlsplit((url or "").strip())
        if parts.scheme != "http" or not parts.hostname:
            return None
        if is_loopback_host(parts.hostname):
            return None
        return _host_port_key(parts)
    except ValueError:
        return None


def normalize_managed_url(url: str, *, allow_insecure_host: str | None = None) -> str:
    """Validate a managed base URL and return it without its trailing slash.

    Syntax and transport security are checked before anything is stored,
    ``--no-check`` included: an address that cannot be used is not
    "unreachable, try later". ``allow_insecure_host`` is the one opted-in
    ``host[:port]`` for which plain http is accepted; it must equal the URL's
    own ``host[:port]`` exactly.

    Raises ``ValueError``, which the CLI renders as ``validation_error``.
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
        # Reading the property is the check: urlsplit keeps the netloc verbatim
        # and only parses the port when asked for it.
        parts.port
    except ValueError as exc:
        raise ValueError(
            f"Managed host URL {raw!r} has an invalid port: {exc}."
        ) from exc
    if parts.scheme == "http" and not is_loopback_host(parts.hostname):
        key = _host_port_key(parts)
        if not (allow_insecure_host or "").strip():
            raise ValueError(
                f"Managed host URL {raw!r} uses plain http:// for a non-loopback "
                "host; use https://, or opt in this one host with "
                f"'potpie host set {raw} --allow-insecure' "
                f"({ALLOW_INSECURE_HOST_ENV}={key} for {MANAGED_URL_ENV})."
            )
        allowed = normalize_insecure_host(str(allow_insecure_host))
        if allowed != key:
            raise ValueError(
                f"Managed host URL {raw!r} uses plain http://, and the insecure "
                f"opt-in names {allowed!r}, not {key!r}."
            )
    return raw.rstrip("/")


def normalize_managed_token(token: str | None) -> str:
    """The token in the form it will be sent, which is the only form to check.

    Empty is a real answer for a loopback service with auth disabled, so this
    normalizes and never refuses; :func:`managed_token_problem` decides where
    empty is acceptable.
    """
    return (token or "").strip()


def managed_token_problem(base_url: str, token: str) -> str | None:
    """Why ``token`` cannot be used with ``base_url``, if it cannot."""
    if token:
        return None
    if is_loopback_host(urlsplit(base_url).hostname):
        return None
    return (
        f"the managed host {base_url} has no token, and an empty token is "
        "accepted only for a loopback host"
    )


# --- the managed host -------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ManagedHost:
    """The usable managed host, as resolved from the environment or the file."""

    base_url: str
    token: str
    #: The ``host[:port]`` whose plain-http opt-in is in force, when the URL is
    #: http to a non-loopback host; ``None`` otherwise.
    allow_insecure_host: str | None
    #: ``POTPIE_MANAGED_URL`` or the registry path: where the address came from.
    source: str

    @property
    def insecure(self) -> bool:
        return self.allow_insecure_host is not None


def _resolve_managed() -> tuple[ManagedHost | None, str | None]:
    """``(host, problem)``: the managed host, and why it is unusable.

    Two answers because "no managed host" and "a managed host that can never be
    used" are different states with different repairs. The environment goes
    through the same rules as ``host set``, so a typo cannot enter by the one
    door that skips the CLI. The address is checked before the credential, so
    an override that is both unusable and uncredentialed is reported as the
    syntax error the user can fix by reading the message.
    """
    state: dict[str, Any] = {}
    env_url = managed_env_override()
    if env_url:
        source, base_url = MANAGED_URL_ENV, env_url
        opt_in = os.getenv(ALLOW_INSECURE_HOST_ENV, "").strip() or None
    else:
        state = _read_state()
        source = str(registry_path())
        base_url = str(state.get("managed_url") or "").strip()
        opt_in = str(state.get("managed_allow_insecure_host") or "").strip() or None
    if not base_url:
        return None, None
    try:
        normalized = normalize_managed_url(base_url, allow_insecure_host=opt_in)
    except ValueError as exc:
        return None, f"{source} holds an unusable managed host: {exc}"
    if env_url:
        token, problem = _env_managed_token(normalized)
        if problem is not None:
            return None, problem
    else:
        token = normalize_managed_token(str(state.get("managed_token") or ""))
    token_problem = managed_token_problem(normalized, token or "")
    if token_problem is not None:
        repair = (
            f"set {MANAGED_TOKEN_ENV} to the key that host expects"
            if env_url
            else "store one with 'potpie host set <url> --token -' (key on stdin)"
        )
        return None, f"{source}: {token_problem}; {repair}"
    return (
        ManagedHost(
            base_url=normalized,
            token=token or "",
            allow_insecure_host=insecure_host_key(normalized),
            source=source,
        ),
        None,
    )


def _env_managed_token(base_url: str) -> tuple[str | None, str | None]:
    """The credential to pair with ``POTPIE_MANAGED_URL``, or why there is none.

    ``POTPIE_MANAGED_TOKEN`` set (the empty string included) is the caller's
    whole statement about the credential. Unset falls back to the stored token
    only when the address is the one it was stored for: a key pasted in for one
    service is not a key handed to whatever else the environment names.
    """
    raw = os.getenv(MANAGED_TOKEN_ENV)
    if raw is not None:
        return normalize_managed_token(raw), None
    stored = _read_state()
    token = normalize_managed_token(str(stored.get("managed_token") or ""))
    if not token:
        return "", None
    stored_url = str(stored.get("managed_url") or "").strip()
    if _address_key(stored_url) == _address_key(base_url):
        return token, None
    return None, (
        f"{MANAGED_URL_ENV}={base_url} is set with no {MANAGED_TOKEN_ENV}, and "
        f"the stored token belongs to {stored_url or 'no stored address'} rather "
        f"than to that host. Set {MANAGED_TOKEN_ENV} to the key that host expects"
    )


def _address_key(value: str) -> str:
    return value.strip().rstrip("/")


def managed_host() -> ManagedHost | None:
    """The usable managed host, or ``None`` when unset or unusable.

    The environment overrides the file, for a one-off run against another
    service without persisting anything.
    """
    return _resolve_managed()[0]


def managed_endpoint() -> tuple[str, str] | None:
    """``(base_url, token)`` for the managed host, or ``None`` if unusable."""
    host = managed_host()
    return (host.base_url, host.token) if host is not None else None


def managed_env_override() -> str | None:
    """The ``POTPIE_MANAGED_URL`` address that outranks the stored one, if any."""
    url = os.getenv(MANAGED_URL_ENV, "").strip()
    return url or None


def managed_endpoint_problem() -> str | None:
    """Why the configured managed endpoint cannot be used, if it cannot.

    ``None`` covers both "usable" and "not configured".
    """
    return _resolve_managed()[1]


def managed_unconfigured_refusal() -> tuple[str, str]:
    """``(message, next_action)`` for "managed was targeted, and there is none".

    Two ways to have no usable managed host, and only one is fixed by
    ``host set``: an unusable ``POTPIE_MANAGED_URL`` outranks the file, so
    telling its owner to write the file would not help.
    """
    problem = managed_endpoint_problem()
    if problem is None:
        return ("No managed host is configured.", "run 'potpie host set <url>'")
    return (
        f"The managed host is unusable: {problem.rstrip('.')}.",
        "run 'potpie host list' to see the managed host as it is currently resolved",
    )


def managed_unconfigured_error() -> ContextEngineDisabled:
    """:func:`managed_unconfigured_refusal` as a raisable error."""
    message, next_action = managed_unconfigured_refusal()
    error = ContextEngineDisabled(message)
    error.recommended_next_action = next_action  # type: ignore[attr-defined]
    return error


def stored_managed_token() -> str:
    """The token in the registry file, ignoring the environment override."""
    return str(_read_state().get("managed_token") or "").strip()


def set_managed_endpoint(
    base_url: str, token: str | None, *, allow_insecure_host: str | None = None
) -> None:
    """Store the managed host; url, token and insecure opt-in move together.

    Validated here as well as at the CLI door, so no code path can put an
    unusable endpoint, or a credential in a form nothing will send, in the
    file. The opt-in is stored only when the URL actually needs it.
    """
    normalized = normalize_managed_url(
        base_url, allow_insecure_host=allow_insecure_host
    )
    clean_token = normalize_managed_token(token)
    problem = managed_token_problem(normalized, clean_token)
    if problem is not None:
        raise ValueError(f"{problem[0].upper()}{problem[1:]}.")
    state = _read_state()
    state["managed_url"] = normalized
    state["managed_token"] = clean_token
    opt_in = insecure_host_key(normalized)
    if opt_in is not None:
        state["managed_allow_insecure_host"] = opt_in
    else:
        state.pop("managed_allow_insecure_host", None)
    _write_state(state)


def clear_managed_endpoint() -> Path | None:
    """Forget the managed host; returns where an unreadable registry was kept.

    The one writer that survives a corrupt registry. The unreadable bytes are
    moved aside first (a precondition of the overwrite, not a courtesy): the
    user asked to forget the host, not to lose a token they could not read.
    """
    try:
        state = _read_state()
    except HostRegistryUnreadable:
        salvaged = _quarantine_state()
        _write_state({}, force=True)
        return salvaged
    for key in ("managed_url", "managed_token", "managed_allow_insecure_host"):
        state.pop(key, None)
    _write_state(state)
    return None


def insecure_transport_warning(base_url: str) -> str:
    return (
        f"warning: {base_url} is reached over plain http (insecure opt-in); "
        "its API key is sent unencrypted"
    )


def warn_insecure_transport(base_url: str) -> None:
    """One stderr line, once per process, before a key goes out over plain http.

    Never on stdout: a ``--json`` caller parses stdout. A no-op for https and
    loopback URLs. Transports call this before sending a key; repeated calls
    for the same URL print nothing more.
    """
    if insecure_host_key(base_url) is None or base_url in _warned_insecure:
        return
    _warned_insecure.add(base_url)
    print(insecure_transport_warning(base_url), file=sys.stderr, flush=True)


# --- candidate sets and labels ----------------------------------------------


def configured_origins() -> tuple[str, ...]:
    """Origins worth talking to: local always, managed once usable."""
    return ORIGINS if managed_endpoint() is not None else (LOCAL,)


def targeting_origins() -> tuple[str, ...]:
    """Origins a bare targeting ref has to be accounted for against.

    Unlike :func:`configured_origins`, a configured-but-unusable managed host
    stays in: dropping it would assert that it holds no pot by that name, which
    a host nobody can enumerate cannot be shown to say.
    """
    if managed_endpoint() is not None or managed_endpoint_problem() is not None:
        return ORIGINS
    return (LOCAL,)


def managed_label(base_url: str) -> str:
    """``Managed (host:port)`` for an explicit endpoint."""
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
    """Drop process-local state: the origin override and warning memo."""
    _current["origin"] = None
    _warned_insecure.clear()


__all__ = [
    "ALLOW_INSECURE_HOST_ENV",
    "LOCAL",
    "MANAGED",
    "MANAGED_TOKEN_ENV",
    "MANAGED_URL_ENV",
    "ORIGINS",
    "REGISTRY_FILE",
    "HostRegistryNotSalvageable",
    "HostRegistryUnreadable",
    "HostRegistryUnwritable",
    "ManagedHost",
    "clear_managed_endpoint",
    "configured_origins",
    "current_origin",
    "home_dir",
    "insecure_host_key",
    "insecure_transport_warning",
    "is_loopback_host",
    "managed_endpoint",
    "managed_endpoint_problem",
    "managed_env_override",
    "managed_host",
    "managed_label",
    "managed_token_problem",
    "managed_unconfigured_error",
    "managed_unconfigured_refusal",
    "normalize_insecure_host",
    "normalize_managed_token",
    "normalize_managed_url",
    "origin_degraded",
    "origin_label",
    "origin_overridden",
    "persisted_origin",
    "qualify",
    "registry_path",
    "require_origin",
    "reset_for_tests",
    "selected_origin",
    "set_current_origin",
    "set_managed_endpoint",
    "set_persisted_origin",
    "split_ref",
    "stored_managed_token",
    "targeting_origins",
    "warn_insecure_transport",
]
