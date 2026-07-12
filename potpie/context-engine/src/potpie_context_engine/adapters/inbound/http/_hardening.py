"""Transport hardening for the standalone HTTP app (security review M-5).

Deny-by-default CORS, a request body-size cap, baseline security headers,
and a lightweight in-process per-principal rate limit on the expensive
ingest / webhook / query paths. All limits are env-tunable; the rate
limiter is best-effort (per-process) — a real multi-instance deployment
should also throttle at the edge, but this removes the "no throttle at
all" amplification surface.
"""

from __future__ import annotations

import os
import time
from collections import deque

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.cors import CORSMiddleware

_DEFAULT_MAX_BODY = 5 * 1024 * 1024  # 5 MB
_RATE_LIMITED_PREFIXES = ("/webhooks", "/api/v1/context")
_SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Cache-Control": "no-store",
}


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


class BodySizeLimitMiddleware(BaseHTTPMiddleware):
    """413 when a declared Content-Length exceeds the cap."""

    def __init__(self, app, max_bytes: int) -> None:
        super().__init__(app)
        self._max = max_bytes

    async def dispatch(self, request, call_next):
        cl = request.headers.get("content-length")
        if cl is not None:
            try:
                if int(cl) > self._max:
                    return JSONResponse(
                        status_code=413,
                        content={
                            "error": {
                                "code": "payload_too_large",
                                "message": f"body exceeds {self._max} bytes",
                            }
                        },
                    )
            except ValueError:
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": {
                            "code": "bad_content_length",
                            "message": "invalid Content-Length",
                        }
                    },
                )
        return await call_next(request)


class TracingMiddleware(BaseHTTPMiddleware):
    """Open a SERVER span per request and bind the trace to the
    correlation context so every downstream log line / span carries it.

    Outermost middleware (added last) so the span covers rate-limit and
    body-size rejections too. Best-effort: any observability failure must
    not affect the response.
    """

    async def dispatch(self, request, call_next):
        from potpie_context_engine.bootstrap.observability_context import (
            bind_correlation,
        )
        from potpie_context_engine.bootstrap.observability_runtime import (
            get_observability,
        )
        from potpie_context_engine.domain.ports.observability import SPAN_KIND_SERVER

        obs = get_observability()
        route = request.url.path
        try:
            cm = obs.span(
                f"HTTP {request.method} {route}",
                kind=SPAN_KIND_SERVER,
                attributes={
                    "http.method": request.method,
                    "http.route": route,
                },
            )
        except Exception:  # noqa: BLE001 — never break the request
            return await call_next(request)
        with cm as span:
            tp = obs.current_traceparent()
            if tp:
                # W3C: 00-<32 hex trace_id>-<16 hex span_id>-<flags>
                parts = tp.split("-")
                if len(parts) >= 2:
                    bind_correlation(trace_id=parts[1])
            try:
                response = await call_next(request)
            except Exception as exc:  # noqa: BLE001 — annotate + re-raise
                span.record_exception(exc)
                span.set_error(repr(exc))
                raise
            span.set_attribute("http.status_code", response.status_code)
            if response.status_code >= 500:
                span.set_error(f"status {response.status_code}")
            return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        resp = await call_next(request)
        for k, v in _SECURITY_HEADERS.items():
            resp.headers.setdefault(k, v)
        return resp


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Per-principal sliding-window limit on expensive path prefixes.

    Principal = the API key (``X-API-Key``) when present, else client IP.
    Off when ``limit`` <= 0.
    """

    def __init__(self, app, *, limit: int, window_s: float) -> None:
        super().__init__(app)
        self._limit = limit
        self._window = window_s
        self._hits: dict[str, deque[float]] = {}

    def _principal(self, request) -> str:
        key = request.headers.get("x-api-key")
        if key:
            return f"k:{hash(key)}"
        client = request.client
        return f"ip:{client.host}" if client else "ip:?"

    async def dispatch(self, request, call_next):
        if self._limit <= 0 or not request.url.path.startswith(_RATE_LIMITED_PREFIXES):
            return await call_next(request)
        now = time.monotonic()
        pid = self._principal(request)
        dq = self._hits.setdefault(pid, deque())
        cutoff = now - self._window
        while dq and dq[0] < cutoff:
            dq.popleft()
        if len(dq) >= self._limit:
            return JSONResponse(
                status_code=429,
                content={
                    "error": {
                        "code": "rate_limited",
                        "message": "too many requests; slow down",
                    }
                },
                headers={"Retry-After": str(int(self._window))},
            )
        dq.append(now)
        return await call_next(request)


def install_hardening(app: FastAPI) -> None:
    """Attach the hardening middleware stack (deny-by-default)."""
    raw_origins = os.getenv("CONTEXT_ENGINE_CORS_ORIGINS", "").strip()
    origins = [o.strip() for o in raw_origins.split(",") if o.strip()]
    # Deny-by-default: with no configured origins CORS grants nothing.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=bool(origins),
        allow_methods=["*"] if origins else [],
        allow_headers=["*"] if origins else [],
    )
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(
        RateLimitMiddleware,
        limit=_int_env("CONTEXT_ENGINE_RATE_LIMIT_PER_MIN", 120),
        window_s=60.0,
    )
    app.add_middleware(
        BodySizeLimitMiddleware,
        max_bytes=_int_env("CONTEXT_ENGINE_MAX_BODY_BYTES", _DEFAULT_MAX_BODY),
    )
    # Added last → outermost: the request span wraps rate-limit / body-size
    # rejections too. NoOp by default, so this is free until enabled.
    app.add_middleware(TracingMiddleware)
