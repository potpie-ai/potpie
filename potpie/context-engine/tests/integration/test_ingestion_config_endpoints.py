"""Behavior tests for the ingestion-config + force-flush HTTP endpoints."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from adapters.inbound.http.api.v1.context.router import create_context_router
from domain.ports.ingestion_config import IngestionConfig


def _allow_decision() -> SimpleNamespace:
    return SimpleNamespace(allowed=True, status_code=200, reason=None, detail=None)


def _build_container() -> MagicMock:
    container = MagicMock()
    container.policy.return_value.authorize.return_value = _allow_decision()
    cfg_port = MagicMock()
    cfg_port.get.return_value = IngestionConfig(
        pot_id="p1",
        mode="windowed",
        window_minutes=5,
        min_batch_size=None,
    )
    cfg_port.set.return_value = IngestionConfig(
        pot_id="p1",
        mode="immediate",
        window_minutes=3,
        min_batch_size=None,
    )
    container.ingestion_config.return_value = cfg_port

    batches = MagicMock()
    batches.get_open_batch_id_for_pot.return_value = "batch-1"
    container.batch_repository.return_value = batches

    container.jobs = MagicMock()
    return container


def _client(container: MagicMock) -> TestClient:
    app = FastAPI()
    app.include_router(
        create_context_router(
            require_auth=lambda: SimpleNamespace(id="user-1"),
            get_container=lambda: container,
            get_db=lambda: MagicMock(),
            get_db_optional=lambda: MagicMock(),
        ),
        prefix="/api/v1/context",
    )
    return TestClient(app)


class TestGetIngestionConfig:
    def test_returns_current_config(self) -> None:
        container = _build_container()
        client = _client(container)
        r = client.get("/api/v1/context/pots/p1/ingestion-config")
        assert r.status_code == 200
        body = r.json()
        assert body["pot_id"] == "p1"
        assert body["mode"] == "windowed"
        assert body["window_minutes"] == 5
        assert body["min_batch_size"] is None


class TestPutIngestionConfig:
    def test_happy_path_updates_and_returns_new_config(self) -> None:
        container = _build_container()
        client = _client(container)
        r = client.put(
            "/api/v1/context/pots/p1/ingestion-config",
            json={"mode": "immediate", "window_minutes": 3},
        )
        assert r.status_code == 200
        assert r.json()["mode"] == "immediate"
        # Adapter received the actor_user_id.
        port = container.ingestion_config.return_value
        port.set.assert_called_once()
        _args, kwargs = port.set.call_args
        assert kwargs["mode"] == "immediate"
        assert kwargs["window_minutes"] == 3
        assert kwargs["actor_user_id"] == "user-1"

    def test_rejects_unknown_mode(self) -> None:
        container = _build_container()
        client = _client(container)
        r = client.put(
            "/api/v1/context/pots/p1/ingestion-config",
            json={"mode": "weird", "window_minutes": 5},
        )
        # FastAPI validation would catch a literal mismatch, but our
        # adapter also validates. Either way the user gets 4xx.
        assert r.status_code in (400, 422)

    def test_rejects_out_of_range_window(self) -> None:
        container = _build_container()
        client = _client(container)
        r = client.put(
            "/api/v1/context/pots/p1/ingestion-config",
            json={"mode": "windowed", "window_minutes": 0},
        )
        assert r.status_code in (400, 422)


class TestForceFlush:
    def test_returns_batch_id_when_open(self) -> None:
        container = _build_container()
        client = _client(container)
        r = client.post("/api/v1/context/pots/p1/ingest/flush")
        assert r.status_code == 200
        body = r.json()
        assert body["batch_id"] == "batch-1"
        assert body["status"] == "queued"
        container.jobs.enqueue_batch.assert_called_once_with("batch-1")

    def test_returns_none_status_when_nothing_pending(self) -> None:
        container = _build_container()
        container.batch_repository.return_value.get_open_batch_id_for_pot.return_value = None
        client = _client(container)
        r = client.post("/api/v1/context/pots/p1/ingest/flush")
        assert r.status_code == 200
        body = r.json()
        assert body["batch_id"] is None
        assert body["status"] == "no_pending_batch"
        container.jobs.enqueue_batch.assert_not_called()

    def test_returns_none_status_when_nothing_pending_without_queue(self) -> None:
        container = _build_container()
        container.batch_repository.return_value.get_open_batch_id_for_pot.return_value = None
        container.jobs = None
        client = _client(container)
        r = client.post("/api/v1/context/pots/p1/ingest/flush")
        assert r.status_code == 200
        body = r.json()
        assert body["batch_id"] is None
        assert body["status"] == "no_pending_batch"

    def test_returns_503_when_open_batch_has_no_queue(self) -> None:
        container = _build_container()
        container.jobs = None
        client = _client(container)
        r = client.post("/api/v1/context/pots/p1/ingest/flush")
        assert r.status_code == 503
        assert r.json()["detail"] == "Context graph job queue is not configured."
