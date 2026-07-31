"""HTTP security response headers (pentest remediations FW002 / FW004)."""

import os

from starlette.types import ASGIApp, Message, Receive, Scope, Send

HSTS_HEADER_NAME = b"strict-transport-security"
HSTS_HEADER_VALUE = b"max-age=31536000; includeSubDomains; preload"

X_FRAME_OPTIONS_HEADER_NAME = b"x-frame-options"
X_FRAME_OPTIONS_HEADER_VALUE = b"DENY"

CSP_HEADER_NAME = b"content-security-policy"
CSP_FRAME_ANCESTORS_VALUE = b"frame-ancestors 'self'"

# Public string forms for tests / docs (FW004 remediation values).
X_FRAME_OPTIONS_VALUE = "DENY"
CSP_FRAME_ANCESTORS = "frame-ancestors 'self'"

# Same env name as uvicorn/gunicorn ``--forwarded-allow-ips``.
_FORWARDED_ALLOW_IPS_ENV = "FORWARDED_ALLOW_IPS"


def trusted_proxy_hosts() -> frozenset[str]:
    """Hosts allowed to supply ``X-Forwarded-Proto`` / related proxy headers.

    Defaults to loopback only. Set ``FORWARDED_ALLOW_IPS`` to a comma-separated
    list of reverse-proxy addresses (or ``*`` only when the edge already
    overwrites client-supplied forwarded headers — see
    ``deployment/proxy/overwrite-forwarded-proto.conf``).
    """
    raw = os.getenv(_FORWARDED_ALLOW_IPS_ENV, "127.0.0.1").strip()
    if not raw:
        return frozenset()
    return frozenset(part.strip() for part in raw.split(",") if part.strip())


def _client_host(scope: Scope) -> str | None:
    client = scope.get("client")
    if not client:
        return None
    return client[0]


def _client_is_trusted_proxy(scope: Scope) -> bool:
    hosts = trusted_proxy_hosts()
    if "*" in hosts:
        return True
    host = _client_host(scope)
    return bool(host and host in hosts)


def _forwarded_proto(scope: Scope) -> str | None:
    for name, value in scope.get("headers", []):
        if name.lower() == b"x-forwarded-proto":
            # Leftmost value is the original client scheme when proxies append.
            token = value.decode("latin-1").split(",", 1)[0].strip().lower()
            return token or None
    return None


def request_is_https(scope: Scope) -> bool:
    """True when the request should be treated as HTTPS for HSTS.

    Uses ``scope["scheme"]`` first (including values rewritten by uvicorn
    ``ProxyHeadersMiddleware`` / ``--forwarded-allow-ips``).

    ``X-Forwarded-Proto`` is honored only when the connecting peer is in
    ``FORWARDED_ALLOW_IPS``. Arbitrary clients cannot enable HSTS by spoofing
    the header.
    """
    if scope.get("scheme") == "https":
        return True
    if _client_is_trusted_proxy(scope) and _forwarded_proto(scope) == "https":
        return True
    return False


def _header_present(headers: list[tuple[bytes, bytes]], name: bytes) -> bool:
    return any(key.lower() == name for key, _ in headers)


class SecurityHeadersMiddleware:
    """Attach anti-framing headers on all responses; HSTS on HTTPS only."""

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        add_hsts = request_is_https(scope)

        async def send_with_security_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                if not _header_present(headers, X_FRAME_OPTIONS_HEADER_NAME):
                    headers.append(
                        (X_FRAME_OPTIONS_HEADER_NAME, X_FRAME_OPTIONS_HEADER_VALUE)
                    )
                if not _header_present(headers, CSP_HEADER_NAME):
                    headers.append((CSP_HEADER_NAME, CSP_FRAME_ANCESTORS_VALUE))
                if add_hsts and not _header_present(headers, HSTS_HEADER_NAME):
                    headers.append((HSTS_HEADER_NAME, HSTS_HEADER_VALUE))
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, send_with_security_headers)


# Backward-compatible alias used by older imports/tests.
StrictTransportSecurityMiddleware = SecurityHeadersMiddleware
