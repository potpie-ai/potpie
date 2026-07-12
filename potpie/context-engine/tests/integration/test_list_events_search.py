"""GET /pots/{pot_id}/events?q=... passes the needle through to the filter."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from potpie_context_engine.adapters.inbound.http.api.v1.context.router import (
    create_context_router,
)


def _build_container() -> MagicMock:
    container = MagicMock()
    decision = SimpleNamespace(allowed=True, status_code=200, reason=None, detail=None)
    container.policy.return_value.authorize.return_value = decision

    page = SimpleNamespace(items=(), next_cursor=None)
    container.event_query_service.return_value.list_events.return_value = page
    return container


def _client(container: MagicMock) -> TestClient:
    app = FastAPI()
    app.include_router(
        create_context_router(
            require_auth=lambda: None,
            get_container=lambda: container,
            get_db=lambda: MagicMock(),
            get_db_optional=lambda: MagicMock(),
        ),
        prefix="/api/v1/context",
    )
    return TestClient(app)


class TestQParamPassthrough:
    def test_q_makes_it_into_the_filter_dataclass(self) -> None:
        container = _build_container()
        client = _client(container)
        r = client.get(
            "/api/v1/context/pots/p1/events",
            params={"q": "auth flow"},
        )
        assert r.status_code == 200
        list_events = container.event_query_service.return_value.list_events
        list_events.assert_called_once()
        _args, kwargs = list_events.call_args
        # Either positional or keyword — pull from the call.
        filters_obj = list_events.call_args[0][1]
        assert filters_obj.q == "auth flow"

    def test_blank_q_is_normalized_to_none(self) -> None:
        container = _build_container()
        client = _client(container)
        # "  " (whitespace-only) shouldn't trigger an ILIKE — that would
        # match everything in the DB.
        r = client.get(
            "/api/v1/context/pots/p1/events",
            params={"q": "   "},
        )
        assert r.status_code == 200
        list_events = container.event_query_service.return_value.list_events
        filters_obj = list_events.call_args[0][1]
        assert filters_obj.q is None

    def test_omitted_q_is_none(self) -> None:
        container = _build_container()
        client = _client(container)
        r = client.get("/api/v1/context/pots/p1/events")
        assert r.status_code == 200
        list_events = container.event_query_service.return_value.list_events
        filters_obj = list_events.call_args[0][1]
        assert filters_obj.q is None

    def test_overlong_q_is_422(self) -> None:
        container = _build_container()
        client = _client(container)
        long_q = "x" * 201
        r = client.get(
            "/api/v1/context/pots/p1/events",
            params={"q": long_q},
        )
        assert r.status_code == 422
