"""CGT-9: HTTP contract for ``POST /record`` and ingestion-config routes."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from adapters.inbound.http.api.v1.context.router import create_context_router
from domain.ingestion_event_models import EventReceipt
from domain.ports.ingestion_config import IngestionConfig, InMemoryIngestionConfig

pytestmark = pytest.mark.unit


def _allow_policy() -> SimpleNamespace:
    return SimpleNamespace(allowed=True, status_code=200, reason=None, detail=None)


def _container(*, ingestion_cfg: InMemoryIngestionConfig | None = None) -> MagicMock:
    c = MagicMock()
    c.policy.return_value.authorize.return_value = _allow_policy()
    submission = MagicMock()
    c.ingestion_submission.return_value = submission
    if ingestion_cfg is not None:
        c.ingestion_config.return_value = ingestion_cfg
    batches = MagicMock()
    c.batch_repository.return_value = batches
    c.jobs = MagicMock()
    return c


def _client(container: MagicMock) -> TestClient:
    app = FastAPI()
    app.include_router(
        create_context_router(
            require_auth=lambda: SimpleNamespace(id="user-1", surface="http"),
            get_container=lambda: container,
            get_db=lambda: MagicMock(),
            get_db_optional=lambda: MagicMock(),
        ),
        prefix="/api/v1/context",
    )
    return TestClient(app)


class TestPostRecord:
    def test_record_plumbs_scope_idempotency_and_occurred_at(self) -> None:
        container = _container()
        submission = container.ingestion_submission.return_value
        occurred = datetime(2026, 5, 22, 10, 0, tzinfo=timezone.utc)
        submission.submit.return_value = EventReceipt(
            event_id="evt-rec-1",
            status="queued",
            duplicate=False,
            job_id="batch-1",
        )

        client = _client(container)
        r = client.post(
            "/api/v1/context/record",
            json={
                "pot_id": "pot-1",
                "record": {
                    "type": "decision",
                    "summary": "Use windowed ingestion",
                    "details": {"area": "ingestion"},
                    "source_refs": ["pr:42"],
                },
                "scope": {
                    "repo_name": "acme/widgets",
                    "file_path": "app/main.py",
                    "source_refs": ["issue:7"],
                },
                "idempotency_key": "idem-abc",
                "occurred_at": occurred.isoformat(),
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["ok"] is True
        assert body["status"] == "queued"
        assert body["event_id"] == "evt-rec-1"
        assert body["record_type"] == "decision"
        assert body["source_id"]
        assert len(body["fallbacks"]) == 1
        assert body["fallbacks"][0]["code"] == "record_queued"

        req = submission.submit.call_args[0][0]
        assert req.pot_id == "pot-1"
        assert req.idempotency_key == "idem-abc"
        assert req.occurred_at == occurred
        assert req.repo_name == "acme/widgets"
        assert req.source_channel == "http"
        assert req.payload["scope"]["file_path"] == "app/main.py"

    def test_duplicate_record_surfaces_duplicate_status(self) -> None:
        container = _container()
        container.ingestion_submission.return_value.submit.return_value = (
            EventReceipt(
                event_id="evt-dup",
                status="queued",
                duplicate=True,
            )
        )
        r = _client(container).post(
            "/api/v1/context/record",
            json={
                "pot_id": "pot-1",
                "record": {"type": "decision", "summary": "same again"},
            },
        )
        assert r.status_code == 200
        assert r.json()["status"] == "duplicate"
        assert r.json()["fallbacks"] == []


class TestIngestionConfigRoutes:
    def test_get_returns_effective_config(self) -> None:
        cfg = InMemoryIngestionConfig(
            {
                "pot-1": IngestionConfig(
                    pot_id="pot-1",
                    mode="windowed",
                    window_minutes=10,
                    min_batch_size=3,
                )
            }
        )
        r = _client(_container(ingestion_cfg=cfg)).get(
            "/api/v1/context/pots/pot-1/ingestion-config"
        )
        assert r.status_code == 200
        body = r.json()
        assert body == {
            "pot_id": "pot-1",
            "mode": "windowed",
            "window_minutes": 10,
            "min_batch_size": 3,
        }

    def test_put_validates_mode_at_schema_level(self) -> None:
        r = _client(_container(ingestion_cfg=InMemoryIngestionConfig())).put(
            "/api/v1/context/pots/pot-1/ingestion-config",
            json={"mode": "burst", "window_minutes": 5},
        )
        assert r.status_code == 422

    def test_put_updates_and_returns_config(self) -> None:
        cfg = InMemoryIngestionConfig()
        r = _client(_container(ingestion_cfg=cfg)).put(
            "/api/v1/context/pots/pot-1/ingestion-config",
            json={"mode": "immediate", "window_minutes": 5, "min_batch_size": None},
        )
        assert r.status_code == 200
        assert r.json()["mode"] == "immediate"
        assert cfg.get("pot-1").mode == "immediate"

    def test_flush_without_pending_batch(self) -> None:
        cfg = InMemoryIngestionConfig()
        container = _container(ingestion_cfg=cfg)
        container.batch_repository.return_value.get_open_batch_id_for_pot.return_value = (
            None
        )
        r = _client(container).post("/api/v1/context/pots/pot-1/ingest/flush")
        assert r.status_code == 200
        body = r.json()
        assert body["batch_id"] is None
        assert body["status"] == "no_pending_batch"
        container.jobs.enqueue_batch.assert_not_called()

    def test_flush_enqueues_open_batch(self) -> None:
        cfg = InMemoryIngestionConfig()
        container = _container(ingestion_cfg=cfg)
        batches = container.batch_repository.return_value
        batches.get_open_batch_id_for_pot.return_value = "batch-open"
        r = _client(container).post("/api/v1/context/pots/pot-1/ingest/flush")
        assert r.status_code == 200
        assert r.json()["batch_id"] == "batch-open"
        assert r.json()["status"] == "queued"
        container.jobs.enqueue_batch.assert_called_once_with("batch-open")

    def test_flush_tolerates_enqueue_failure(self) -> None:
        cfg = InMemoryIngestionConfig()
        container = _container(ingestion_cfg=cfg)
        container.batch_repository.return_value.get_open_batch_id_for_pot.return_value = (
            "batch-open"
        )
        container.jobs.enqueue_batch.side_effect = RuntimeError("broker down")
        r = _client(container).post("/api/v1/context/pots/pot-1/ingest/flush")
        assert r.status_code == 200
        assert r.json()["batch_id"] == "batch-open"
