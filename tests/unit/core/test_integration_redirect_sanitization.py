from urllib.parse import parse_qs, urlparse

import pytest
from starlette.requests import Request

from app.src.integrations.integrations.adapters.inbound.http import (
    integrations_router,
)

SENSITIVE_DATABASE_ERROR = (
    "(psycopg2.OperationalError) connection to server at "
    "pgbouncer-rw.internal.svc.cluster.local, port 5432 failed"
)


def _request(query: str) -> Request:
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/api/v1/integrations/linear/callback",
            "query_string": query.encode(),
            "headers": [],
            "scheme": "http",
            "server": ("backend.test", 80),
            "client": ("127.0.0.1", 1234),
        }
    )


@pytest.mark.asyncio
async def test_linear_callback_redirect_does_not_expose_internal_exception(
    monkeypatch,
):
    def fail_state_verification(state):
        raise RuntimeError(SENSITIVE_DATABASE_ERROR)

    monkeypatch.setenv("FRONTEND_URL", "http://frontend.test")
    monkeypatch.setattr(
        integrations_router,
        "_verify_oauth_state",
        fail_state_verification,
    )

    response = await integrations_router.linear_oauth_callback(
        _request("state=signed-state"),
        linear_oauth=object(),
    )

    location = response.headers["location"]
    error = parse_qs(urlparse(location).query)["error"][0]

    assert error == "OAuth authorization could not be completed"
    assert "psycopg2" not in location.lower()
    assert "pgbouncer" not in location.lower()
