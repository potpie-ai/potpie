"""HTTP security response headers (pentest remediations FW002 / FW004)."""

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


def request_is_https(scope: Scope) -> bool:
    """True when the client connection is HTTPS (direct or via proxy)."""
    for key, value in scope.get("headers", []):
        if key.lower() == b"x-forwarded-proto":
            # Take the left-most value (original client protocol).
            proto = value.split(b",", 1)[0].strip().lower()
            return proto == b"https"
    return scope.get("scheme") == "https"


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
