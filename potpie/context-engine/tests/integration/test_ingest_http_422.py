"""POST /ingest returns 422 with structured body on reconciliation_rejected."""

from __future__ import annotations

from unittest.mock import MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient


def test_ingest_422_json_body(monkeypatch) -> None:
    from application.use_cases.run_raw_episode_ingestion import (
        RunRawEpisodeIngestionResult,
    )
    from adapters.inbound.http.api.v1.context.router import create_context_router

    def fake_run(**_kwargs):
        return RunRawEpisodeIngestionResult(
            ok=False,
            status="reconciliation_rejected",
            event_id="3d6ab2c2-4f43-4a5f-8b3c-12bee7386613",
            episode_uuid=None,
            job_id="j1",
            error="ontology validation failed: sample",
            reconciliation_errors=[
                {"entity": "adr:0042", "issue": "unknown canonical labels: ADR"},
            ],
            downgrades=[],
        )

    monkeypatch.setattr(
        "adapters.inbound.http.api.v1.context.router.run_raw_episode_ingestion",
        fake_run,
    )
    monkeypatch.setenv("CONTEXT_ENGINE_INGEST_422", "1")

    app = FastAPI()
    app.include_router(
        create_context_router(
            require_auth=lambda: None,
            get_container=lambda: MagicMock(),
            get_db=lambda: MagicMock(),
            get_db_optional=lambda: MagicMock(),
        ),
        prefix="/api/v2/context",
    )
    client = TestClient(app)
    r = client.post(
        "/api/v2/context/ingest",
        json={
            "pot_id": "p1",
            "name": "n",
            "episode_body": "b",
            "source_description": "s",
        },
        params={"sync": "true"},
    )
    assert r.status_code == 422
    body = r.json()
    assert body["status"] == "reconciliation_rejected"
    assert body["event_id"] == "3d6ab2c2-4f43-4a5f-8b3c-12bee7386613"
    assert body["episode_uuid"] is None
    assert body["errors"] == [
        {"entity": "adr:0042", "issue": "unknown canonical labels: ADR"}
    ]
    assert body["downgrades"] == []


def test_ingest_legacy_503_when_422_disabled(monkeypatch) -> None:
    from application.use_cases.run_raw_episode_ingestion import (
        RunRawEpisodeIngestionResult,
    )
    from adapters.inbound.http.api.v1.context.router import create_context_router

    def fake_run(**_kwargs):
        return RunRawEpisodeIngestionResult(
            ok=False,
            status="reconciliation_rejected",
            event_id="e1",
            reconciliation_errors=[{"entity": "a", "issue": "b"}],
            error="ontology validation failed",
        )

    monkeypatch.setattr(
        "adapters.inbound.http.api.v1.context.router.run_raw_episode_ingestion",
        fake_run,
    )
    monkeypatch.setenv("CONTEXT_ENGINE_INGEST_422", "0")

    app = FastAPI()
    app.include_router(
        create_context_router(
            require_auth=lambda: None,
            get_container=lambda: MagicMock(),
            get_db=lambda: MagicMock(),
            get_db_optional=lambda: MagicMock(),
        ),
        prefix="/api/v2/context",
    )
    client = TestClient(app)
    r = client.post(
        "/api/v2/context/ingest",
        json={
            "pot_id": "p1",
            "name": "n",
            "episode_body": "b",
            "source_description": "s",
        },
        params={"sync": "true"},
    )
    assert r.status_code == 503
    assert r.json()["detail"] == "ontology validation failed"
