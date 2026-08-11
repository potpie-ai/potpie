"""Credentials for the daemon's browser-facing ``/ui`` surface.

``/rpc`` has always required the bearer token from ``discovery.json``. The UI's
JSON API sat on the same loopback port with nothing in front of it, so any
process on the machine could read the whole project-memory graph — summaries,
claim text, provenance, actor identities — and ``POST /ui/api/pots/use`` let it
move the active pot *and* the active host, with the daemon attaching its stored
managed token on the anonymous caller's behalf. Loopback is not a credential:
every other local program is on it too.

Two credentials are accepted, because a browser *navigation* cannot carry a
header:

* ``Authorization: Bearer <daemon token>`` — the same token ``/rpc`` checks, for
  ``potpie ui`` and anything else that can already read ``discovery.json``.
* a session cookie the browser obtained through the handoff below.

The handoff keeps the daemon token out of the browser entirely: ``potpie ui``
spends the bearer on a single-use, short-lived code, opens ``/ui/?k=<code>``,
and the shell handler trades the code for an ``HttpOnly`` cookie and redirects
to the clean URL — so the code is spent and gone from the address bar (and from
history) by the time the page renders. A token in a URL would outlive the
session in both.
"""

from __future__ import annotations

import secrets
import threading
import time
from urllib.parse import SplitResult, urlsplit

from fastapi import HTTPException, Request
from starlette.responses import Response

#: Name of the browser session cookie. Path-scoped to ``/ui`` so it is never
#: attached to ``/rpc``, which takes the bearer token and nothing else.
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
#: The daemon binds loopback, so the set is closed and short — which is what
#: lets the same-origin check state its expectation instead of asking the
#: request for it.
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})
_DEFAULT_PORTS = {"http": 80, "https": 443}

_CHALLENGE = {"WWW-Authenticate": "Bearer"}
_DENIED = (
    "unauthorized: the graph explorer needs a session — run 'potpie ui' to open "
    "it, or send the daemon token from ~/.potpie/discovery.json as a bearer token"
)


class UiAuth:
    """The credential the ``/ui`` surface accepts, plus the browser handoff.

    Lives on ``app.state`` rather than inside the router's closure because the
    two halves of the handoff sit in different modules: the router mints codes,
    the static shell is the only handler that can set a cookie on a navigation.

    Codes and sessions are held in this process' memory on purpose — a file
    would outlive the daemon that issued them and hand a restarted daemon a
    credential nobody alive still holds.
    """

    def __init__(self, *, token: str) -> None:
        self._token = token
        self._lock = threading.Lock()
        self._sessions: dict[str, float] = {}
        self._codes: dict[str, float] = {}

    # -- credentials ---------------------------------------------------------

    def has_bearer(self, request: Request) -> bool:
        header = request.headers.get("authorization")
        if header is None:
            return False
        return _same(header, f"Bearer {self._token}")

    def has_session(self, request: Request) -> bool:
        value = request.cookies.get(SESSION_COOKIE)
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
        of whoever was handed the link.
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
        keeps it off every other route this daemon serves.
        """
        response.set_cookie(
            SESSION_COOKIE,
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
    whatever the caller put on the wire — comparing the encoded bytes turns a
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
    """Bearer token or browser session — the gate on every ``/ui/api`` route."""
    auth = ui_auth(request)
    if auth.has_bearer(request) or auth.has_session(request):
        return
    raise HTTPException(status_code=401, detail=_DENIED, headers=_CHALLENGE)


def require_bearer(request: Request) -> None:
    """Bearer token only — for minting new browser sessions.

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

    Defence in depth on the one route that writes: ``SameSite`` already keeps
    another site's page from sending the cookie, so this catches the cases it
    does not cover (a stale browser, a same-site-but-not-same-origin page) and
    turns a silent host switch into a refusal.

    The expectation is the fixed loopback allowlist and the port the request
    arrived on — never the request's own ``Host`` header. Deriving it from the
    header made the check circular: under DNS rebinding the page at
    ``evil.example`` (freshly re-pointed at 127.0.0.1) sends ``Origin`` and
    ``Host`` that agree with each other, so a Host-derived expectation always
    matched and the check passed trivially.

    No stated origin at all means no browser page made this request — ``potpie
    ui`` and curl carry the bearer token — so the credential stays the lock.
    """
    stated = _stated_origin(request)
    if stated is None:
        return
    parts = urlsplit(stated)
    port = _stated_port(parts)
    # ``port is not None`` on its own line of defence: an origin with an
    # unparseable port must not match a daemon whose port is equally unknown.
    if (
        (parts.hostname or "").lower() in _LOOPBACK_HOSTS
        and port is not None
        and port == _bound_port(request)
    ):
        return
    raise HTTPException(
        status_code=403,
        detail=f"cross-origin request refused (origin {stated})",
    )


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
    server = request.scope.get("server")
    if isinstance(server, (tuple, list)) and len(server) == 2 and server[1]:
        return int(server[1])
    return _DEFAULT_PORTS.get(request.url.scheme)


__all__ = [
    "COOKIE_PATH",
    "HANDOFF_TTL_SECONDS",
    "SESSION_COOKIE",
    "SESSION_TTL_SECONDS",
    "UiAuth",
    "require_bearer",
    "require_same_origin",
    "require_ui_credential",
    "ui_auth",
]
