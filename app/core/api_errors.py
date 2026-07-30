import json
import re
import uuid
from typing import Any

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.modules.utils.logger import setup_logger


logger = setup_logger(__name__)

_STATUS_MESSAGES = {
    400: ("Invalid request.", "invalid_request"),
    401: ("Authentication required.", "authentication_required"),
    403: ("Access denied.", "access_denied"),
    404: ("Resource not found.", "not_found"),
    405: ("Method not allowed.", "method_not_allowed"),
    409: ("The request conflicts with the current state.", "conflict"),
    422: ("Invalid request.", "validation_error"),
    429: ("Too many requests. Please try again later.", "rate_limited"),
}

_INTERNAL_ERROR = (
    "An internal error occurred. Please try again later.",
    "internal_error",
)
_SERVICE_UNAVAILABLE = (
    "Service temporarily unavailable. Please try again later.",
    "service_unavailable",
)
_OPERATION_FAILED = "The operation could not be completed."
_PRIVATE_ERROR_KEYS = {
    "exception",
    "traceback",
    "stack",
    "stack_trace",
}
_ERROR_VALUE_KEYS = {
    "_error",
    "error",
    "event_bus_error",
    "last_error",
}
_ERROR_CONTEXT_KEYS = {
    "content",
    "message",
    "detail",
    "details",
    "response",
    "stream_part",
    "tool_response",
}
_ERROR_MARKER = "[[SUBAGENT_ERROR]]"
_PUBLIC_ERROR_HEADER = "X-Potpie-Public-Error"
_PUBLIC_ERROR_HEADER_BYTES = _PUBLIC_ERROR_HEADER.lower().encode("ascii")
_REQUEST_ID_MAX_LENGTH = 128
_REQUEST_ID_PATTERN = re.compile(r"[A-Za-z0-9._-]+", re.ASCII)
_SUCCESS_JSON_INSPECTION_LIMIT_BYTES = 64 * 1024


class PublicErrorDetail(str):
    """Reviewed, static 4xx copy that is intentionally safe for clients."""


def public_error_detail(message: str) -> PublicErrorDetail:
    """Mark static 4xx copy as public; never wrap exception or user-derived text."""
    if not isinstance(message, str) or not message.strip():
        raise ValueError("Public error detail must be a non-empty string")
    return PublicErrorDetail(message)


def public_error_response(
    message: str,
    *,
    status_code: int,
    field: str = "error",
) -> JSONResponse:
    """Build an explicitly public 4xx JSON response for reviewed static copy."""
    if not 400 <= status_code < 500:
        raise ValueError("Public error responses are restricted to 4xx status codes")
    detail = public_error_detail(message)
    return JSONResponse(
        status_code=status_code,
        content={field: detail},
        headers={_PUBLIC_ERROR_HEADER: "1"},
    )


def _request_id_from_scope(scope: Scope) -> str:
    state = scope.setdefault("state", {})
    existing = state.get("request_id")
    if (
        isinstance(existing, str)
        and len(existing) <= _REQUEST_ID_MAX_LENGTH
        and _REQUEST_ID_PATTERN.fullmatch(existing)
    ):
        return existing

    request_id = None
    for key, value in scope.get("headers", []):
        if key.lower() == b"x-request-id":
            candidate = value.decode("latin-1")
            if (
                len(candidate) <= _REQUEST_ID_MAX_LENGTH
                and _REQUEST_ID_PATTERN.fullmatch(candidate)
            ):
                request_id = candidate
            break

    request_id = request_id or str(uuid.uuid4())
    state["request_id"] = request_id
    return request_id


def _public_status_message(status_code: int) -> tuple[str, str]:
    if status_code in (502, 503, 504):
        return _SERVICE_UNAVAILABLE
    if status_code >= 500:
        return _INTERNAL_ERROR
    return _STATUS_MESSAGES.get(
        status_code,
        ("The request could not be completed.", "request_failed"),
    )


def client_safe_error_message(_error: Any = None) -> str:
    """Return stable copy for non-HTTP error channels such as SSE and tool events."""
    return _OPERATION_FAILED


def _sanitize_error_value(value: Any) -> Any:
    if isinstance(value, str):
        prefix = f"{_ERROR_MARKER}\n" if _ERROR_MARKER in value else ""
        return f"{prefix}{client_safe_error_message(value)}"
    if isinstance(value, (dict, list, tuple)):
        return sanitize_client_payload(value, error_context=True)
    if value is None:
        return None
    return client_safe_error_message(value)


def _is_error_envelope(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False

    status = str(payload.get("status", "")).lower()
    event_type = str(payload.get("type", "")).lower()
    has_error_value = any(
        payload.get(key) not in (None, "", False) for key in _ERROR_VALUE_KEYS
    )
    has_error_marker = any(
        isinstance(payload.get(key), str) and _ERROR_MARKER in payload[key]
        for key in _ERROR_CONTEXT_KEYS
    )
    return (
        payload.get("success") is False
        or status in {"error", "failed", "failure"}
        or "error" in event_type
        or has_error_value
        or has_error_marker
    )


def sanitize_client_payload(payload: Any, error_context: bool = False) -> Any:
    """Recursively remove private details from error-shaped client payloads."""
    if isinstance(payload, dict):
        current_error_context = error_context or _is_error_envelope(payload)
        sanitized = {}
        for key, value in payload.items():
            key_lower = str(key).lower()
            if key_lower in _PRIVATE_ERROR_KEYS:
                sanitized[key] = client_safe_error_message(value)
            elif key_lower in _ERROR_VALUE_KEYS:
                sanitized[key] = _sanitize_error_value(value)
            elif current_error_context and key_lower in _ERROR_CONTEXT_KEYS:
                sanitized[key] = _sanitize_error_value(value)
            else:
                sanitized[key] = sanitize_client_payload(
                    value,
                    error_context=current_error_context,
                )
        return sanitized
    if isinstance(payload, list):
        return [
            sanitize_client_payload(item, error_context=error_context)
            for item in payload
        ]
    if isinstance(payload, tuple):
        return tuple(
            sanitize_client_payload(item, error_context=error_context)
            for item in payload
        )
    return payload


def sanitize_and_log_client_payload(payload: Any, channel: str) -> Any:
    """Sanitize an event payload and retain its private form in backend diagnostics."""
    sanitized = sanitize_client_payload(payload)
    if sanitized != payload:
        logger.bind(
            channel=channel,
            private_payload=payload,
        ).error("Sanitized client error payload")
    return sanitized


def _error_payload(
    status_code: int,
    request_id: str,
    detail: Any = None,
) -> dict[str, str]:
    default_message, code = _public_status_message(status_code)
    message = (
        str(detail)
        if status_code < 500 and isinstance(detail, PublicErrorDetail)
        else default_message
    )
    return {"detail": message, "code": code, "request_id": request_id}


def _safe_json_response(
    status_code: int,
    request_id: str,
    detail: Any = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    response_headers = dict(headers or {})
    response_headers["X-Request-ID"] = request_id
    return JSONResponse(
        status_code=status_code,
        content=_error_payload(status_code, request_id, detail),
        headers=response_headers,
    )


def _log_private_exception(
    request_id: str,
    scope: Scope,
    exc: BaseException,
    status_code: int,
) -> None:
    logger.bind(
        request_id=request_id,
        path=scope.get("path"),
        method=scope.get("method"),
        status_code=status_code,
        private_detail=str(exc),
    ).opt(exception=(type(exc), exc, exc.__traceback__)).error("API request failed")


async def _http_exception_handler(
    request: Request,
    exc: StarletteHTTPException,
) -> JSONResponse:
    request_id = _request_id_from_scope(request.scope)
    detail_is_private = exc.status_code >= 500 or not isinstance(
        exc.detail,
        PublicErrorDetail,
    )
    if detail_is_private and exc.detail not in (None, ""):
        _log_private_exception(request_id, request.scope, exc, exc.status_code)

    request.state.error_sanitized = True
    return _safe_json_response(
        exc.status_code,
        request_id,
        detail=exc.detail,
        headers=exc.headers,
    )


async def _validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    request_id = _request_id_from_scope(request.scope)
    errors = [
        {
            "loc": list(error.get("loc", [])),
            "msg": error.get("msg", "Invalid value"),
            "type": error.get("type", "validation_error"),
        }
        for error in exc.errors()
    ]
    request.state.error_sanitized = True
    response = JSONResponse(
        status_code=422,
        content={
            "detail": errors,
            "code": "validation_error",
            "request_id": request_id,
        },
        headers={"X-Request-ID": request_id},
    )
    return response


async def _unhandled_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    request_id = _request_id_from_scope(request.scope)
    _log_private_exception(request_id, request.scope, exc, 500)
    request.state.error_sanitized = True
    return _safe_json_response(500, request_id)


def register_api_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(StarletteHTTPException, _http_exception_handler)
    app.add_exception_handler(RequestValidationError, _validation_exception_handler)
    app.add_exception_handler(Exception, _unhandled_exception_handler)


class ServerErrorSanitizationMiddleware:
    """Sanitize error responses, including manually built and failure-envelope bodies."""

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = _request_id_from_scope(scope)
        start_message: Message | None = None
        explicitly_public_error = False
        inspect_success_json = False
        passthrough_response_body = False
        response_started = False
        response_body = bytearray()

        def header_value(headers: list[tuple[bytes, bytes]], name: bytes) -> str:
            for key, value in headers:
                if key.lower() == name:
                    return value.decode("latin-1")
            return ""

        def replace_headers(
            headers: list[tuple[bytes, bytes]],
            body: bytes,
            content_type: str | None = None,
        ) -> list[tuple[bytes, bytes]]:
            replaced = [
                (key, value)
                for key, value in headers
                if key.lower()
                not in (
                    b"content-length",
                    b"content-type",
                    b"content-md5",
                    b"etag",
                    b"x-request-id",
                    _PUBLIC_ERROR_HEADER_BYTES,
                )
            ]
            if content_type:
                replaced.append((b"content-type", content_type.encode("latin-1")))
            else:
                existing_content_type = header_value(headers, b"content-type")
                if existing_content_type:
                    replaced.append(
                        (b"content-type", existing_content_type.encode("latin-1"))
                    )
            replaced.extend(
                [
                    (b"content-length", str(len(body)).encode("ascii")),
                    (b"x-request-id", request_id.encode("latin-1")),
                ]
            )
            return replaced

        def parse_json_body(body: bytes) -> Any:
            try:
                return json.loads(body)
            except (json.JSONDecodeError, UnicodeDecodeError):
                return None

        def log_private_response(status_code: int, body: bytes) -> None:
            logger.bind(
                request_id=request_id,
                path=scope.get("path"),
                method=scope.get("method"),
                status_code=status_code,
                private_response=body.decode("utf-8", errors="replace"),
            ).error("Sanitized direct API error response")

        async def send_tracked(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        async def send_sanitized(message: Message) -> None:
            nonlocal explicitly_public_error, inspect_success_json
            nonlocal passthrough_response_body, response_body, start_message

            if message["type"] == "http.response.start":
                status_code = message["status"]
                already_sanitized = scope.get("state", {}).get("error_sanitized")
                content_type = header_value(message.get("headers", []), b"content-type")
                explicitly_public_error = (
                    400 <= status_code < 500
                    and header_value(
                        message.get("headers", []),
                        _PUBLIC_ERROR_HEADER_BYTES,
                    )
                    == "1"
                )
                content_length = header_value(
                    message.get("headers", []),
                    b"content-length",
                )
                try:
                    declared_content_length = int(content_length)
                except (TypeError, ValueError):
                    declared_content_length = None
                inspect_success_json = (
                    200 <= status_code < 300
                    and "json" in content_type.lower()
                    and declared_content_length is not None
                    and 0
                    <= declared_content_length
                    <= _SUCCESS_JSON_INSPECTION_LIMIT_BYTES
                )
                should_inspect = status_code >= 400 or inspect_success_json
                if should_inspect and not already_sanitized:
                    start_message = message
                    return
                await send_tracked(message)
                return

            if passthrough_response_body:
                await send_tracked(message)
                return

            if start_message is None:
                await send_tracked(message)
                return

            if message["type"] != "http.response.body":
                await send_tracked(message)
                return

            message_body = message.get("body", b"")
            if (
                inspect_success_json
                and len(response_body) + len(message_body)
                > _SUCCESS_JSON_INSPECTION_LIMIT_BYTES
            ):
                buffered_start = start_message
                buffered_body = bytes(response_body) + message_body
                start_message = None
                response_body.clear()
                passthrough_response_body = True

                flushed_body_message = dict(message)
                flushed_body_message["body"] = buffered_body
                await send_tracked(buffered_start)
                await send_tracked(flushed_body_message)
                return

            response_body.extend(message_body)
            if message.get("more_body", False):
                return

            status_code = start_message["status"]
            original_body = bytes(response_body)
            body = original_body
            content_type: str | None = None
            changed = False

            if explicitly_public_error:
                body = original_body
            elif status_code >= 500:
                log_private_response(status_code, original_body)
                body = json.dumps(
                    _error_payload(status_code, request_id),
                    separators=(",", ":"),
                ).encode("utf-8")
                content_type = "application/json"
                changed = True
            else:
                payload = parse_json_body(original_body)
                if payload is not None:
                    sanitized_payload = sanitize_client_payload(
                        payload,
                        error_context=status_code >= 400,
                    )
                    if sanitized_payload != payload:
                        log_private_response(status_code, original_body)
                        body = json.dumps(
                            sanitized_payload,
                            separators=(",", ":"),
                            default=str,
                        ).encode("utf-8")
                        content_type = "application/json"
                        changed = True
                elif status_code >= 400:
                    log_private_response(status_code, original_body)
                    body = json.dumps(
                        _error_payload(status_code, request_id),
                        separators=(",", ":"),
                    ).encode("utf-8")
                    content_type = "application/json"
                    changed = True

            start_message["headers"] = replace_headers(
                start_message.get("headers", []),
                body,
                content_type=content_type if changed else None,
            )
            await send_tracked(start_message)
            await send_tracked({"type": "http.response.body", "body": body})

        try:
            await self.app(scope, receive, send_sanitized)
        except Exception as exc:
            if response_started:
                raise
            _log_private_exception(request_id, scope, exc, 500)
            response = _safe_json_response(500, request_id)
            await response(scope, receive, send)
