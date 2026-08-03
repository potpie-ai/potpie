import asyncio
import json
from uuid import UUID

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel
from starlette.responses import JSONResponse, Response, StreamingResponse

from app.core.api_errors import (
    ServerErrorSanitizationMiddleware,
    _request_id_from_scope,
    client_safe_error_message,
    logger as api_error_logger,
    public_error_detail,
    public_error_response,
    register_api_error_handlers,
    sanitize_and_log_client_payload,
    sanitize_client_payload,
)
from app.modules.intelligence.agents.chat_agent import ChatAgentResponse, ChatContext
from app.modules.intelligence.agents.chat_agents.pydantic_agent import PydanticRagAgent


SENSITIVE_ERROR = (
    "connection to server at pgbouncer-rw.internal.svc.cluster.local "
    "(34.118.226.109), port 5432 failed; "
    "postgresql://db-user:db-password@pgbouncer-rw.internal/app"
)

REVIEW_SENSITIVE_4XX_DETAILS = [
    "Account alice@example.com cannot access this resource.",
    "Unable to connect to [2001:db8:85a3::8a2e:370:7334].",
    r"Failed to read C:\Users\Alice\secrets\config.json.",
    "AWS credential AKIAIOSFODNN7EXAMPLE is invalid.",
    (
        "Token eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
        "eyJzdWIiOiIxMjM0NTY3ODkwIn0."
        "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c is invalid."
    ),
    "Database db.internal.example.com is unavailable.",
]


def _build_app() -> FastAPI:
    app = FastAPI()
    register_api_error_handlers(app)
    app.add_middleware(ServerErrorSanitizationMiddleware)

    @app.get("/unhandled")
    async def unhandled():
        raise RuntimeError(SENSITIVE_ERROR)

    @app.get("/http-500")
    async def http_500():
        raise HTTPException(status_code=500, detail=SENSITIVE_ERROR)

    @app.get("/http-503")
    async def http_503():
        raise HTTPException(status_code=503, detail=SENSITIVE_ERROR)

    @app.get("/direct-500")
    async def direct_500():
        return JSONResponse(
            status_code=500,
            content={"error": SENSITIVE_ERROR},
            headers={
                "ETag": '"stale-error-body"',
                "Content-MD5": "stale-content-digest",
            },
        )

    @app.get("/unsafe-409")
    async def unsafe_409():
        return Response(
            status_code=409,
            content=json.dumps(
                {
                    "error": "GitHub account is already linked.",
                    "details": SENSITIVE_ERROR,
                }
            ),
        )

    @app.get("/failure-envelope")
    async def failure_envelope():
        return {
            "success": False,
            "error": SENSITIVE_ERROR,
            "details": {"last_error": SENSITIVE_ERROR},
        }

    @app.get("/success-envelope")
    async def success_envelope():
        return {
            "success": True,
            "message": "Repository contains postgresql:// examples.",
        }

    @app.get("/success-with-validators")
    async def success_with_validators():
        return JSONResponse(
            status_code=200,
            content={"success": True, "message": "ok"},
            headers={
                "ETag": '"stable-success-body"',
                "Content-MD5": "stable-content-digest",
            },
        )

    @app.get("/nested-failure-envelope")
    async def nested_failure_envelope():
        return {
            "column_oriented": {
                "_error": SENSITIVE_ERROR,
            }
        }

    @app.get("/unsafe-400")
    async def unsafe_400():
        raise HTTPException(status_code=400, detail=SENSITIVE_ERROR)

    @app.get("/safe-400")
    async def safe_400():
        raise HTTPException(
            status_code=400,
            detail="GitHub account not linked. Please link your GitHub account.",
        )

    @app.get("/review-unsafe-400/{case_index}")
    async def review_unsafe_400(case_index: int):
        raise HTTPException(
            status_code=400,
            detail=REVIEW_SENSITIVE_4XX_DETAILS[case_index],
        )

    @app.get("/review-direct-400/{case_index}")
    async def review_direct_400(case_index: int):
        return JSONResponse(
            status_code=400,
            content={"detail": REVIEW_SENSITIVE_4XX_DETAILS[case_index]},
        )

    @app.get("/public-400")
    async def public_400():
        raise HTTPException(
            status_code=400,
            detail=public_error_detail("Message content cannot be empty"),
        )

    @app.get("/public-direct-400")
    async def public_direct_400():
        return public_error_response("Missing uid", status_code=400)

    class LoginPayload(BaseModel):
        attempts: int

    @app.post("/validation")
    async def validation(payload: LoginPayload):
        return payload

    @app.get("/streaming-json")
    async def streaming_json():
        async def chunks():
            yield b'{"chunk":"first"}\n'
            await asyncio.sleep(0.05)
            yield b'{"chunk":"second"}\n'
            await asyncio.sleep(0.05)

        return StreamingResponse(chunks(), media_type="application/json")

    return app


def _assert_private_details_absent(response) -> None:
    body = response.text.lower()
    assert "pgbouncer" not in body
    assert "34.118.226.109" not in body
    assert "5432" not in body
    assert "postgresql://" not in body
    assert "db-password" not in body


def test_unhandled_exception_is_sanitized():
    response = TestClient(_build_app(), raise_server_exceptions=False).get(
        "/unhandled",
        headers={"X-Request-ID": "request-unhandled"},
    )

    assert response.status_code == 500
    assert response.json() == {
        "detail": "An internal error occurred. Please try again later.",
        "code": "internal_error",
        "request_id": "request-unhandled",
    }
    _assert_private_details_absent(response)


def test_http_500_and_503_are_sanitized():
    client = TestClient(_build_app(), raise_server_exceptions=False)

    internal_response = client.get("/http-500", headers={"X-Request-ID": "request-500"})
    unavailable_response = client.get(
        "/http-503", headers={"X-Request-ID": "request-503"}
    )

    assert internal_response.status_code == 500
    assert internal_response.json()["code"] == "internal_error"
    assert unavailable_response.status_code == 503
    assert unavailable_response.json() == {
        "detail": "Service temporarily unavailable. Please try again later.",
        "code": "service_unavailable",
        "request_id": "request-503",
    }
    _assert_private_details_absent(internal_response)
    _assert_private_details_absent(unavailable_response)


def test_direct_server_error_response_is_sanitized():
    response = TestClient(_build_app(), raise_server_exceptions=False).get(
        "/direct-500",
        headers={"X-Request-ID": "request-direct"},
    )

    assert response.status_code == 500
    assert response.json()["code"] == "internal_error"
    assert "etag" not in response.headers
    assert "content-md5" not in response.headers
    _assert_private_details_absent(response)


@pytest.mark.parametrize(
    "invalid_request_id",
    [
        "request id with spaces",
        "../../private/config",
        "request@example.com",
        "x" * 129,
    ],
)
def test_invalid_request_ids_are_replaced_before_storage(invalid_request_id):
    scope = {
        "type": "http",
        "headers": [(b"x-request-id", invalid_request_id.encode("latin-1"))],
        "state": {"request_id": invalid_request_id},
    }

    request_id = _request_id_from_scope(scope)

    UUID(request_id)
    assert request_id != invalid_request_id
    assert scope["state"]["request_id"] == request_id


def test_valid_request_id_from_state_is_reused():
    scope = {
        "type": "http",
        "headers": [(b"x-request-id", b"different-valid-id")],
        "state": {"request_id": "request_123.valid-id"},
    }

    request_id = _request_id_from_scope(scope)

    assert request_id == "request_123.valid-id"
    assert scope["state"]["request_id"] == request_id


def test_direct_server_error_response_is_logged_with_request_context():
    messages = []
    sink_id = api_error_logger.add(
        lambda message: messages.append(message.record),
        level="ERROR",
    )
    try:
        response = TestClient(_build_app(), raise_server_exceptions=False).get(
            "/direct-500",
            headers={"X-Request-ID": "request-direct-log"},
        )
    finally:
        api_error_logger.remove(sink_id)

    assert response.status_code == 500
    matching_records = [
        record
        for record in messages
        if record["extra"].get("request_id") == "request-direct-log"
    ]
    assert matching_records
    assert any(
        "pgbouncer-rw.internal.svc.cluster.local"
        in str(record["extra"].get("private_response", ""))
        for record in matching_records
    )


def test_direct_4xx_error_details_are_fail_closed():
    response = TestClient(_build_app(), raise_server_exceptions=False).get(
        "/unsafe-409",
        headers={"X-Request-ID": "request-409"},
    )

    assert response.status_code == 409
    assert response.json()["error"] == "The operation could not be completed."
    assert response.json()["details"] == "The operation could not be completed."
    assert response.headers["x-request-id"] == "request-409"
    _assert_private_details_absent(response)


def test_success_false_envelope_is_sanitized_but_success_payload_is_unchanged():
    client = TestClient(_build_app(), raise_server_exceptions=False)

    failure_response = client.get("/failure-envelope")
    nested_failure_response = client.get("/nested-failure-envelope")
    success_response = client.get("/success-envelope")

    assert failure_response.status_code == 200
    assert failure_response.json()["error"] == "The operation could not be completed."
    assert (
        failure_response.json()["details"]["last_error"]
        == "The operation could not be completed."
    )
    _assert_private_details_absent(failure_response)
    assert (
        nested_failure_response.json()["column_oriented"]["_error"]
        == "The operation could not be completed."
    )
    _assert_private_details_absent(nested_failure_response)
    assert success_response.json() == {
        "success": True,
        "message": "Repository contains postgresql:// examples.",
    }


def test_unchanged_success_json_preserves_etag_and_content_md5():
    response = TestClient(_build_app(), raise_server_exceptions=False).get(
        "/success-with-validators",
        headers={"X-Request-ID": "request-success-validators"},
    )

    assert response.status_code == 200
    assert response.json() == {"success": True, "message": "ok"}
    assert response.headers["etag"] == '"stable-success-body"'
    assert response.headers["content-md5"] == "stable-content-digest"
    assert response.headers["x-request-id"] == "request-success-validators"


def test_all_developer_supplied_4xx_details_are_sanitized():
    client = TestClient(_build_app(), raise_server_exceptions=False)

    unsafe_response = client.get("/unsafe-400")
    safe_response = client.get("/safe-400")

    assert unsafe_response.status_code == 400
    assert unsafe_response.json()["detail"] == "Invalid request."
    _assert_private_details_absent(unsafe_response)
    assert safe_response.status_code == 400
    assert safe_response.json()["detail"] == "Invalid request."


@pytest.mark.parametrize(
    "case_index",
    range(len(REVIEW_SENSITIVE_4XX_DETAILS)),
)
def test_review_sensitive_4xx_details_are_fail_closed(case_index):
    client = TestClient(_build_app(), raise_server_exceptions=False)

    exception_response = client.get(f"/review-unsafe-400/{case_index}")
    direct_response = client.get(f"/review-direct-400/{case_index}")

    assert exception_response.status_code == 400
    assert exception_response.json()["detail"] == "Invalid request."
    assert direct_response.status_code == 400
    assert direct_response.json()["detail"] == "Invalid request."
    assert direct_response.json()["code"] == "invalid_request"
    assert "request_id" in direct_response.json()
    assert REVIEW_SENSITIVE_4XX_DETAILS[case_index] not in exception_response.text
    assert REVIEW_SENSITIVE_4XX_DETAILS[case_index] not in direct_response.text


def test_explicitly_public_4xx_messages_are_preserved_without_internal_marker():
    client = TestClient(_build_app(), raise_server_exceptions=False)

    exception_response = client.get("/public-400")
    direct_response = client.get("/public-direct-400")

    assert exception_response.json()["detail"] == "Message content cannot be empty"
    assert direct_response.json()["error"] == "Missing uid"
    assert "x-potpie-public-error" not in direct_response.headers


def test_public_error_response_rejects_non_4xx_status():
    with pytest.raises(ValueError, match="restricted to 4xx"):
        public_error_response("Do not expose this", status_code=500)


@pytest.mark.asyncio
async def test_successful_json_stream_is_forwarded_incrementally():
    app = _build_app()
    first_chunk_received = asyncio.Event()
    second_chunk_received = asyncio.Event()
    body_chunks = []
    request_received = False
    disconnect = asyncio.Event()

    async def receive():
        nonlocal request_received
        if not request_received:
            request_received = True
            return {"type": "http.request", "body": b"", "more_body": False}
        await disconnect.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        if message["type"] != "http.response.body":
            return
        body = message.get("body", b"")
        if not body:
            return
        body_chunks.append(body)
        if body == b'{"chunk":"first"}\n':
            first_chunk_received.set()
        if body == b'{"chunk":"second"}\n':
            second_chunk_received.set()

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": "/streaming-json",
        "raw_path": b"/streaming-json",
        "query_string": b"",
        "root_path": "",
        "headers": [],
        "client": ("test-client", 50000),
        "server": ("test-server", 80),
        "state": {},
    }

    request_task = asyncio.create_task(app(scope, receive, send))

    await asyncio.wait_for(first_chunk_received.wait(), timeout=0.5)
    assert not request_task.done()
    await asyncio.wait_for(second_chunk_received.wait(), timeout=0.5)
    assert not request_task.done()
    await request_task

    assert body_chunks == [
        b'{"chunk":"first"}\n',
        b'{"chunk":"second"}\n',
    ]


@pytest.mark.asyncio
async def test_declared_bounded_success_json_stops_buffering_after_limit():
    sent_messages = []
    prefix = b'{"data":"'
    oversized_chunk = b"x" * (64 * 1024 + 1)
    suffix = b'"}'

    async def app(scope, receive, send):
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", b"1024"),
                ],
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": prefix,
                "more_body": True,
            }
        )
        assert sent_messages == []

        await send(
            {
                "type": "http.response.body",
                "body": oversized_chunk,
                "more_body": True,
            }
        )
        assert [message["type"] for message in sent_messages] == [
            "http.response.start",
            "http.response.body",
        ]

        await send(
            {
                "type": "http.response.body",
                "body": suffix,
                "more_body": False,
            }
        )

    async def receive():
        return {"type": "http.disconnect"}

    async def send(message):
        sent_messages.append(message)

    middleware = ServerErrorSanitizationMiddleware(app)
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/large-success-json",
        "headers": [],
        "state": {},
    }

    await middleware(scope, receive, send)

    assert sent_messages[0]["status"] == 200
    forwarded_headers = {
        key.lower(): value for key, value in sent_messages[0].get("headers", [])
    }
    assert b"content-length" not in forwarded_headers
    assert forwarded_headers.get(b"content-type") == b"application/json"
    assert b"".join(
        message.get("body", b"")
        for message in sent_messages
        if message["type"] == "http.response.body"
    ) == prefix + oversized_chunk + suffix


@pytest.mark.asyncio
async def test_oversized_error_json_is_fail_closed_not_flushed():
    sent_messages = []
    prefix = b'{"detail":"'
    oversized_chunk = b"x" * (64 * 1024 + 1)
    suffix = b'"}'

    async def app(scope, receive, send):
        await send(
            {
                "type": "http.response.start",
                "status": 400,
                "headers": [
                    (b"content-type", b"application/json"),
                    (
                        b"content-length",
                        str(
                            len(prefix) + len(oversized_chunk) + len(suffix)
                        ).encode("ascii"),
                    ),
                ],
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": prefix,
                "more_body": True,
            }
        )
        assert sent_messages == []

        await send(
            {
                "type": "http.response.body",
                "body": oversized_chunk,
                "more_body": True,
            }
        )
        assert [message["type"] for message in sent_messages] == [
            "http.response.start",
            "http.response.body",
        ]

        await send(
            {
                "type": "http.response.body",
                "body": suffix,
                "more_body": False,
            }
        )

    async def receive():
        return {"type": "http.disconnect"}

    async def send(message):
        sent_messages.append(message)

    middleware = ServerErrorSanitizationMiddleware(app)
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/large-error-json",
        "headers": [],
        "state": {},
    }

    await middleware(scope, receive, send)

    assert sent_messages[0]["status"] == 400
    body = b"".join(
        message.get("body", b"")
        for message in sent_messages
        if message["type"] == "http.response.body"
    )
    assert oversized_chunk not in body
    payload = json.loads(body)
    assert payload["detail"] == "Invalid request."
    assert payload["code"] == "invalid_request"
    assert "request_id" in payload
    assert len(sent_messages) == 2
    assert len(body) < 512


@pytest.mark.asyncio
async def test_oversized_explicitly_public_error_is_flushed_unchanged():
    """Approved public 4xx bodies must not be replaced when over the buffer limit."""
    from app.core.api_errors import _PUBLIC_ERROR_HEADER_BYTES

    sent_messages = []
    prefix = b'{"detail":"'
    oversized_chunk = b"y" * (64 * 1024 + 1)
    suffix = b'"}'
    public_body = prefix + oversized_chunk + suffix

    async def app(scope, receive, send):
        await send(
            {
                "type": "http.response.start",
                "status": 400,
                "headers": [
                    (b"content-type", b"application/json"),
                    (_PUBLIC_ERROR_HEADER_BYTES, b"1"),
                    (b"content-length", b"1024"),
                ],
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": prefix,
                "more_body": True,
            }
        )
        assert sent_messages == []

        await send(
            {
                "type": "http.response.body",
                "body": oversized_chunk,
                "more_body": True,
            }
        )
        assert [message["type"] for message in sent_messages] == [
            "http.response.start",
            "http.response.body",
        ]

        await send(
            {
                "type": "http.response.body",
                "body": suffix,
                "more_body": False,
            }
        )

    async def receive():
        return {"type": "http.disconnect"}

    async def send(message):
        sent_messages.append(message)

    middleware = ServerErrorSanitizationMiddleware(app)
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/large-public-error-json",
        "headers": [],
        "state": {},
    }

    await middleware(scope, receive, send)

    assert sent_messages[0]["status"] == 400
    forwarded_headers = {
        key.lower(): value for key, value in sent_messages[0].get("headers", [])
    }
    assert b"content-length" not in forwarded_headers
    assert _PUBLIC_ERROR_HEADER_BYTES not in forwarded_headers
    assert b"x-request-id" in forwarded_headers
    assert b"".join(
        message.get("body", b"")
        for message in sent_messages
        if message["type"] == "http.response.body"
    ) == public_body


@pytest.mark.asyncio
async def test_exception_after_response_start_is_reraised():
    sent_messages = []

    async def app(scope, receive, send):
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"text/plain")],
            }
        )
        raise RuntimeError("stream failed after response start")

    async def receive():
        return {"type": "http.disconnect"}

    async def send(message):
        sent_messages.append(message)

    middleware = ServerErrorSanitizationMiddleware(app)
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/started-response-failure",
        "headers": [],
        "state": {},
    }

    with pytest.raises(RuntimeError, match="stream failed after response start"):
        await middleware(scope, receive, send)

    assert [message["type"] for message in sent_messages] == ["http.response.start"]


def test_validation_response_does_not_echo_submitted_value():
    response = TestClient(_build_app(), raise_server_exceptions=False).post(
        "/validation",
        json={"attempts": "super-secret-user-input"},
    )

    assert response.status_code == 422
    assert "super-secret-user-input" not in response.text
    assert response.json()["code"] == "validation_error"


def test_success_shaped_context_fields_are_preserved_without_error_envelope():
    payload = {
        "success": True,
        "content": "repository README with postgresql:// examples",
        "message": "clone finished",
        "detail": "branch main",
        "response": {"ok": True},
        "tool_response": "file contents look fine",
    }

    assert sanitize_client_payload(payload) == payload


def test_stream_and_tool_error_payloads_are_sanitized():
    payload = {
        "type": "tool_call_stream_end",
        "status": "error",
        "message": SENSITIVE_ERROR,
        "result": {
            "success": False,
            "error": SENSITIVE_ERROR,
            "last_error": SENSITIVE_ERROR,
            "_error": SENSITIVE_ERROR,
        },
    }

    sanitized = sanitize_client_payload(payload)

    assert sanitized["message"] == "The operation could not be completed."
    assert sanitized["result"]["error"] == "The operation could not be completed."
    assert sanitized["result"]["last_error"] == "The operation could not be completed."
    assert sanitized["result"]["_error"] == "The operation could not be completed."
    assert "pgbouncer" not in str(sanitized).lower()


def test_safe_stream_error_helper_never_returns_exception_text():
    assert (
        client_safe_error_message(RuntimeError(SENSITIVE_ERROR))
        == "The operation could not be completed."
    )


def test_subagent_error_stream_parts_are_sanitized():
    payload = {
        "type": "tool_call_stream_part",
        "stream_part": f"[[SUBAGENT_ERROR]]\n{SENSITIVE_ERROR}",
        "tool_response": f"[[SUBAGENT_ERROR]]\n{SENSITIVE_ERROR}",
    }

    sanitized = sanitize_client_payload(payload)

    assert sanitized["stream_part"] == (
        "[[SUBAGENT_ERROR]]\nThe operation could not be completed."
    )
    assert sanitized["tool_response"] == (
        "[[SUBAGENT_ERROR]]\nThe operation could not be completed."
    )
    assert "pgbouncer" not in str(sanitized).lower()


def test_stream_sanitization_logs_private_payload():
    payload = {
        "status": "error",
        "message": SENSITIVE_ERROR,
    }
    records = []
    sink_id = api_error_logger.add(
        lambda message: records.append(message.record),
        level="ERROR",
    )
    try:
        sanitized = sanitize_and_log_client_payload(
            payload,
            channel="unit-test-stream",
        )
    finally:
        api_error_logger.remove(sink_id)

    assert sanitized["message"] == "The operation could not be completed."
    matching_records = [
        record
        for record in records
        if record["extra"].get("channel") == "unit-test-stream"
    ]
    assert matching_records
    assert any(
        "pgbouncer-rw.internal.svc.cluster.local"
        in str(record["extra"].get("private_payload", ""))
        for record in matching_records
    )


@pytest.mark.asyncio
async def test_chat_agent_error_response_does_not_expose_exception_text():
    class RaisingAsyncContext:
        async def __aenter__(self):
            raise RuntimeError(SENSITIVE_ERROR)

        async def __aexit__(self, exc_type, exc, traceback):
            return False

    class RaisingAgent:
        def run_mcp_servers(self):
            return RaisingAsyncContext()

        def iter(self, **kwargs):
            return RaisingAsyncContext()

    agent = PydanticRagAgent.__new__(PydanticRagAgent)
    agent._create_agent = lambda _ctx: RaisingAgent()
    context = ChatContext(
        project_id="project-1",
        project_name="Project",
        curr_agent_id="agent-1",
        history=[],
        query="Explain the codebase",
    )

    responses = [
        response async for response in agent._run_standard_stream(context)
    ]
    response_text = "".join(response.response for response in responses)

    assert responses
    assert all(isinstance(response, ChatAgentResponse) for response in responses)
    assert response_text == "\n\n*The request could not be completed.*\n\n"
    assert SENSITIVE_ERROR not in response_text
