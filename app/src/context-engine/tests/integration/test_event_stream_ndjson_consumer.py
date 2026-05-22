"""CGT-10: NDJSON consumer contract for context-graph activity/status streams."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from adapters.inbound.http.api.v1.context.router import create_context_router

pytestmark = pytest.mark.unit


def _parse_ndjson(raw: bytes) -> list[dict]:
    lines = [ln for ln in raw.decode("utf-8").splitlines() if ln.strip()]
    return [json.loads(ln) for ln in lines]


def _container() -> SimpleNamespace:
    policy = MagicMock()
    policy.authorize.return_value = SimpleNamespace(
        allowed=True, status_code=200, reason=None, detail=None
    )
    return SimpleNamespace(
        policy=lambda: policy,
        agent_execution_log=lambda _db: MagicMock(),
    )


def _wire_event_lookup(
    container: SimpleNamespace,
    *,
    ev: SimpleNamespace,
    batch_id: str | None,
) -> None:
    events_store = MagicMock()
    events_store.get_event.return_value = ev
    container.event_query_service = lambda _db: events_store
    batches = MagicMock()
    batches.get_latest_batch_id_for_event.return_value = batch_id
    container.batch_repository = lambda _db: batches


def _client(container: SimpleNamespace) -> TestClient:
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


class TestEventActivityStream:
    def test_transient_end_when_event_not_yet_batched(self) -> None:
        container = _container()
        _wire_event_lookup(
            container,
            ev=SimpleNamespace(event_id="e1", pot_id="pot-1"),
            batch_id=None,
        )

        r = _client(container).get("/api/v1/context/events/e1/stream")
        assert r.status_code == 200
        assert "application/x-ndjson" in r.headers.get("content-type", "")
        events = _parse_ndjson(r.content)
        assert len(events) == 1
        assert events[0]["type"] == "end"
        assert events[0]["status"] == "queued"

    def test_replay_and_tail_from_execution_log(self) -> None:
        container = _container()
        _wire_event_lookup(
            container,
            ev=SimpleNamespace(event_id="e1", pot_id="pot-1"),
            batch_id="batch-1",
        )

        def _replay(**_kwargs):
            yield {"type": "status", "stream_id": "1", "status": "processing"}
            yield {"type": "run_finished", "stream_id": "2", "status": "done"}

        exec_log = MagicMock()
        exec_log.replay_and_tail.side_effect = _replay
        container.agent_execution_log = lambda _db: exec_log

        events = _parse_ndjson(
            _client(container).get("/api/v1/context/events/e1/stream").content
        )
        assert events[0]["type"] == "status"
        assert events[-1]["type"] == "run_finished"

    def test_iterator_error_yields_terminal_end(self) -> None:
        container = _container()
        _wire_event_lookup(
            container,
            ev=SimpleNamespace(event_id="e1", pot_id="pot-1"),
            batch_id="batch-1",
        )
        exec_log = MagicMock()
        exec_log.replay_and_tail.side_effect = RuntimeError("redis down")
        container.agent_execution_log = lambda _db: exec_log

        events = _parse_ndjson(
            _client(container).get("/api/v1/context/events/e1/stream").content
        )
        assert events[-1]["type"] == "end"
        assert events[-1]["status"] == "error"


class TestPotStatusStream:
    def test_pot_stream_uses_publisher_replay_and_tail(self) -> None:
        container = _container()

        def _tail(**_kwargs):
            yield {
                "type": "status",
                "stream_id": "10-0",
                "event_id": "e1",
                "status": "processing",
            }
            yield {"type": "end", "stream_id": "10-1", "status": "idle_timeout"}

        publisher = MagicMock()
        publisher.replay_and_tail_pot_status.side_effect = _tail
        container.event_stream_publisher = publisher

        with _client(container).stream(
            "GET",
            "/api/v1/context/pots/pot-1/events/stream",
            params={"idle_timeout_seconds": 1.0},
        ) as response:
            assert response.status_code == 200
            raw = b"".join(response.iter_bytes())
        events = _parse_ndjson(raw)
        assert any(e.get("type") == "status" for e in events)
        assert events[-1]["type"] == "end"
        assert events[-1]["status"] == "idle_timeout"
        publisher.replay_and_tail_pot_status.assert_called_once()

    def test_pot_stream_error_yields_terminal_end(self) -> None:
        container = _container()
        publisher = MagicMock()
        publisher.replay_and_tail_pot_status.side_effect = RuntimeError("stream broke")
        container.event_stream_publisher = publisher

        events = _parse_ndjson(
            _client(container)
            .get("/api/v1/context/pots/pot-1/events/stream")
            .content
        )
        assert events[-1]["type"] == "end"
        assert events[-1]["status"] == "error"
