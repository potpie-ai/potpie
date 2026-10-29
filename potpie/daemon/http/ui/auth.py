"""Credentials and origin checks for the daemon's browser-facing ``/ui`` surface.

The graph-explorer server binds loopback, but loopback is not a credential:
every other program on this machine can reach it, and so can any web page the
user visits (directly, or through a DNS name re-pointed at ``127.0.0.1``).
Without a gate, any of them could read the whole project-memory graph and
``POST /ui/api/pots/use`` to move the active pot the CLI works against.

Two credentials are accepted, because a browser *navigation* cannot carry a
header:

* ``Authorization: Bearer <daemon token>`` -- the per-boot secret the daemon
  already writes to its owner-only credential file, for ``potpie ui`` and
  anything else that can read that file.
* a session cookie the browser obtained through the handoff below.

The handoff keeps the daemon token out of the browser entirely: ``potpie ui``
spends the bearer on a single-use, short-lived code, opens ``/ui?k=<code>``,
and the shell handler trades the code for an ``HttpOnly`` cookie and redirects
to the clean URL -- so the code is spent and gone from the address bar by the
time the page renders. A token in a URL would outlive the session in history.

Two request checks sit in front of the credential:

* ``LoopbackHostGuard`` refuses any request whose ``Host`` header is not a
  loopback name. A page that reached this port through DNS rebinding sends its
  own name there, so it is turned away before the shell or the API answers.
* ``require_same_origin`` refuses an ``Origin`` (or ``Referer``) that is not
  this daemon's own loopback origin, so another site's page cannot drive the
  API even if a browser were to attach the cookie.
"""

from __future__ import annotations

import secrets
import threading
import time
from typing import Any
from urllib.parse import SplitResult, urlsplit

from fastapi import HTTPException, Request
from starlette.datastructures import Headers
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp, Receive, Scope, Send

#: Prefix of the per-daemon browser session cookie. Path-scoped to ``/ui`` so
#: it is never attached to anything else served from this host.
SESSION_COOKIE = "potpie_ui_session"
COOKIE_PATH = "/ui"

#: A handoff code only has to survive the trip from ``potpie ui`` to the browser
#: it just opened. Two minutes leaves room for a copy-paste out of ``--no-open``
#: without leaving a usable credential lying around afterwards.
HANDOFF_TTL_SECONDS = 120.0

#: How long a browser session stays good. A daemon restart mints a new token and
#: forgets every session anyway, so this is only the ceiling on an idle tab.
SESSION_TTL_SECONDS = 12 * 60 * 60.0

#: The only names a browser can honestly be showing this daemon's page under.
#: The daemon binds loopback, so the set is closed and short -- which is what
#: lets both checks state their expectation instead of asking the request for it.
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})
_DEFAULT_PORTS = {"http": 80, "https": 443}

_CHALLENGE = {"WWW-Authenticate": "Bearer"}
_DENIED = (
    "unauthorized: the graph explorer needs a session -- run 'potpie ui' to open "
    "it again, or send the daemon token as a bearer token"
)


class UiAuth:
    """The credential the ``/ui`` surface accepts, plus the browser handoff.

    Lives on ``app.state`` rather than inside the router's closure because the
    two halves of the handoff sit in different modules: the router mints codes,
    the static shell is the only handler that can set a cookie on a navigation.

    Codes and sessions are held in this process' memory on purpose -- a file
    would outlive the daemon that issued them and hand a restarted daemon a
    credential nobody alive still holds.
    """

    def __init__(self, *, token: str) -> None:
        if not token:
            raise ValueError("the UI credential needs the daemon token")
        self._token = token
        # Cookies are scoped by host and path, not port. Several local homes can
        # run daemons on different ports in the same browser; a shared cookie
        # name would let opening one explorer log the other out. Keep the name
        # local to this daemon, just like the sessions it holds.
        self._cookie_name = f"{SESSION_COOKIE}_{secrets.token_hex(8)}"
        self._lock = threading.Lock()
        self._sessions: dict[str, float] = {}
        self._codes: dict[str, float] = {}

    @property
    def cookie_name(self) -> str:
        return self._cookie_name

    # -- credentials ---------------------------------------------------------

    def has_bearer(self, request: Request) -> bool:
        header = request.headers.get("authorization")
        if header is None:
            return False
        return _same(header, f"Bearer {self._token}")

    def has_session(self, request: Request) -> bool:
        value = request.cookies.get(self._cookie_name)
        if not value:
            return False
        now = time.monotonic()
        with self._lock:
            self._prune(self._sessions, now)
            # Compared one by one rather than by dict lookup so a wrong cookie
            # cannot be narrowed down by how long the answer took.
            return any(_same(known, value) for known in self._sessions)

    # -- handoff -------------------------------------------------------------

    def mint_code(self) -> tuple[str, int]:
        """A single-use code and its lifetime, for handing to a browser."""
        code = secrets.token_urlsafe(24)
        ttl = float(HANDOFF_TTL_SECONDS)
        now = time.monotonic()
        with self._lock:
            self._prune(self._codes, now)
            self._codes[code] = now + ttl
        return code, int(ttl)

    def redeem_code(self, code: str) -> str | None:
        """Spend a handoff code for a session token, or ``None`` if it is dead.

        Removing the code before answering is what makes it single use: a code
        that stayed valid would be a bearer token pinned in the browser history
        and terminal scrollback of whoever was handed the link.
        """
        now = time.monotonic()
        with self._lock:
            self._prune(self._codes, now)
            expiry = self._codes.pop(code, None)
            if expiry is None or expiry <= now:
                return None
            session = secrets.token_urlsafe(32)
            self._prune(self._sessions, now)
            self._sessions[session] = now + SESSION_TTL_SECONDS
        return session

    def attach_session(self, response: Response, session: str) -> None:
        """Set the session cookie in the shape a local, http-only daemon needs.

        ``HttpOnly`` keeps page scripts from reading it back out, ``SameSite``
        strict keeps another site's page from spending it, and the ``/ui`` path
        keeps it off every other route on this host.
        """
        response.set_cookie(
            self._cookie_name,
            session,
            max_age=int(SESSION_TTL_SECONDS),
            httponly=True,
            samesite="strict",
            path=COOKIE_PATH,
        )

    @staticmethod
    def _prune(table: dict[str, float], now: float) -> None:
        for key in [key for key, expiry in table.items() if expiry <= now]:
            del table[key]


def _same(known: str, given: str) -> bool:
    """Constant-time equality, over bytes.

    ``compare_digest`` refuses non-ASCII *strings*, and a header or cookie is
    whatever the caller put on the wire -- comparing the encoded bytes turns a
    hostile value into a plain mismatch instead of a 500.
    """
    return secrets.compare_digest(known.encode("utf-8"), given.encode("utf-8"))


def ui_auth(request: Request) -> UiAuth:
    """The app's configured credential, or a refusal.

    Fail closed: an app assembled without one must serve nothing rather than
    serve the graph to anybody who asks.
    """
    auth = getattr(request.app.state, "ui_auth", None)
    if not isinstance(auth, UiAuth):
        raise HTTPException(status_code=401, detail=_DENIED, headers=_CHALLENGE)
    return auth


def require_ui_credential(request: Request) -> None:
    """Bearer token or browser session -- the gate on every ``/ui/api`` route."""
    auth = ui_auth(request)
    if auth.has_bearer(request) or auth.has_session(request):
        return
    raise HTTPException(status_code=401, detail=_DENIED, headers=_CHALLENGE)


def require_bearer(request: Request) -> None:
    """Bearer token only -- for minting new browser sessions.

    A page holding the (unreadable) session cookie must not be able to turn it
    into a code it *can* read and pass somewhere else; issuing sessions stays
    with whoever can already read the daemon token off disk.
    """
    if not ui_auth(request).has_bearer(request):
        raise HTTPException(
            status_code=401,
            detail="unauthorized: minting a browser session needs the daemon token",
            headers=_CHALLENGE,
        )


def require_same_origin(request: Request) -> None:
    """Refuse a stated origin that is not one this daemon can honestly be at.

    ``SameSite`` already keeps another site's page from sending the cookie;
    this catches what it does not cover (a stale browser, a same-site but
    cross-origin page such as another local daemon's port).

    The expectation is the fixed loopback allowlist and the port the request
    arrived on -- never the request's own ``Host`` header. Deriving it from the
    header would make the check circular: under DNS rebinding the page at
    ``evil.example`` sends ``Origin`` and ``Host`` that agree with each other.

    No stated origin at all means no browser page made this request -- ``potpie
    ui`` and curl send the bearer token -- so the credential stays the lock.
    """
    stated = _stated_origin(request)
    if stated is None:
        return
    parts = urlsplit(stated)
    port = _stated_port(parts)
    # ``port is not None`` on its own: an origin with an unparseable port must
    # not match a daemon whose port is equally unknown.
    if (
        (parts.hostname or "").lower() in LOOPBACK_HOSTS
        and port is not None
        and port == _bound_port(request)
    ):
        return
    raise HTTPException(
        status_code=403,
        detail=f"cross-origin request refused (origin {stated})",
    )


class LoopbackHostGuard:
    """Refuse every request whose ``Host`` header is not a loopback name.

    The DNS-rebinding defence. A page at ``evil.example`` whose name has been
    re-pointed at ``127.0.0.1`` reaches this port, but its browser still sends
    ``Host: evil.example:<port>``. Turning that away before routing covers the
    API, the SPA shell and its assets alike, so a rebound page cannot load the
    explorer under its own origin or redeem a handoff code there.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] in {"http", "websocket"} and not is_loopback_host(
            Headers(scope=scope).get("host")
        ):
            response = JSONResponse(
                {
                    "detail": "request refused: the graph explorer answers on loopback only"
                },
                status_code=403,
            )
            await response(scope, receive, send)
            return
        await self.app(scope, receive, send)


def is_loopback_host(host: str | None) -> bool:
    """Whether a ``Host`` header value names this machine's loopback interface."""
    if not host:
        return False
    try:
        parts = urlsplit(f"//{host}")
        _ = parts.port  # raises ValueError on a malformed port
    except ValueError:
        return False
    return (parts.hostname or "").lower() in LOOPBACK_HOSTS


def _stated_origin(request: Request) -> str | None:
    origin = request.headers.get("origin")
    if not origin:
        referer = request.headers.get("referer")
        parts = urlsplit(referer) if referer else None
        if parts is None or not parts.netloc:
            return None
        origin = f"{parts.scheme}://{parts.netloc}"
    return origin.rstrip("/").lower()


def _stated_port(parts: SplitResult) -> int | None:
    try:
        port = parts.port
    except ValueError:  # a port that is not a number is not a port
        return None
    return port if port is not None else _DEFAULT_PORTS.get(parts.scheme)


def _bound_port(request: Request) -> int | None:
    """The port this request actually arrived on, from the ASGI server address.

    The socket, not the ``Host`` header: the header is the half of the
    comparison an attacker gets to choose.
    """
    server: Any = request.scope.get("server")
    if isinstance(server, (tuple, list)) and len(server) == 2 and server[1]:
        return int(server[1])
    return _DEFAULT_PORTS.get(request.url.scheme)


__all__ = [
    "COOKIE_PATH",
    "HANDOFF_TTL_SECONDS",
    "LOOPBACK_HOSTS",
    "SESSION_COOKIE",
    "SESSION_TTL_SECONDS",
    "LoopbackHostGuard",
    "UiAuth",
    "is_loopback_host",
    "require_bearer",
    "require_same_origin",
    "require_ui_credential",
    "ui_auth",
]
