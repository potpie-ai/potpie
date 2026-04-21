"""Unified raw episode ingest orchestration."""

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from application.use_cases.run_raw_episode_ingestion import run_raw_episode_ingestion
from bootstrap.container import ContextEngineContainer
from domain.ports.pot_resolution import ResolvedPot, ResolvedPotRepo


def _container(
    episodic: MagicMock, *, jobs: MagicMock | None = None
) -> ContextEngineContainer:
    settings = MagicMock()
    settings.is_enabled.return_value = True
    pots = MagicMock()
    repo = ResolvedPotRepo(
        pot_id="p1",
        repo_id="r1",
        provider="github",
        provider_host="github.com",
        repo_name="o/r",
    )
    pots.resolve_pot.return_value = ResolvedPot(pot_id="p1", name="n", repos=[repo])
    structural = MagicMock()
    j = jobs if jobs is not None else MagicMock()
    return ContextEngineContainer(
        settings=settings,
        episodic=episodic,
        structural=structural,
        pots=pots,
        source_for_repo=lambda _r: MagicMock(),
        jobs=j,
    )


def test_async_without_database_errors():
    c = _container(MagicMock())
    out = run_raw_episode_ingestion(
        container=c,
        db=None,
        pot_id="p1",
        name="n",
        episode_body="b",
        source_description="src",
        reference_time=datetime.now(timezone.utc),
        idempotency_key=None,
        sync=False,
    )
    assert not out.ok
    assert out.error == "async_requires_database"


def test_legacy_direct_without_database():
    episodic = MagicMock()
    episodic.enabled = True
    episodic.add_episode.return_value = "uuid-1"
    c = _container(episodic)
    t = datetime(2025, 1, 2, tzinfo=timezone.utc)
    out = run_raw_episode_ingestion(
        container=c,
        db=None,
        pot_id="p1",
        name="n",
        episode_body="b",
        source_description="src",
        reference_time=t,
        idempotency_key=None,
        sync=True,
    )
    assert out.ok
    assert out.status == "legacy_direct"
    assert out.episode_uuid == "uuid-1"


def test_legacy_direct_standalone_pot_no_repo():
    """Raw ingest allows pots with no linked GitHub repo (context_graph_pots)."""
    episodic = MagicMock()
    episodic.enabled = True
    episodic.add_episode.return_value = "uuid-solo"
    settings = MagicMock()
    settings.is_enabled.return_value = True
    pots = MagicMock()
    pots.resolve_pot.return_value = ResolvedPot(pot_id="solo", name="solo", repos=[])
    c = ContextEngineContainer(
        settings=settings,
        episodic=episodic,
        structural=MagicMock(),
        pots=pots,
        source_for_repo=lambda _r: MagicMock(),
        jobs=MagicMock(),
    )
    t = datetime(2025, 1, 2, tzinfo=timezone.utc)
    out = run_raw_episode_ingestion(
        container=c,
        db=None,
        pot_id="solo",
        name="n",
        episode_body="b",
        source_description="src",
        reference_time=t,
        idempotency_key=None,
        sync=True,
    )
    assert out.ok
    assert out.status == "legacy_direct"
    episodic.add_episode.assert_called_once()


def test_unknown_pot():
    c = _container(MagicMock())
    c.pots.resolve_pot.return_value = None
    out = run_raw_episode_ingestion(
        container=c,
        db=None,
        pot_id="bad",
        name="n",
        episode_body="b",
        source_description="src",
        reference_time=datetime.now(timezone.utc),
        idempotency_key=None,
        sync=True,
    )
    assert not out.ok
    assert out.error == "unknown_pot_id"


@pytest.mark.parametrize("sync", [True, False])
def test_with_database_delegates_to_submission(monkeypatch, sync: bool):
    """Patch IngestionSubmissionService.submit to avoid DB setup."""
    from application.services.ingestion_submission_service import (
        DefaultIngestionSubmissionService,
    )
    from domain.ingestion_event_models import EventReceipt

    want_sync = sync
    queued = not sync
    episodic = MagicMock()
    episodic.enabled = True
    jobs = MagicMock()
    c = _container(episodic, jobs=jobs)

    def fake_submit(self, request, *, sync=False, wait=False, timeout_seconds=None):
        assert sync == want_sync
        assert request.pot_id == "p1"
        if queued:
            return EventReceipt(event_id="e1", status="queued", job_id="j1")
        return EventReceipt(
            event_id="e1", status="done", episode_uuid="u1", job_id="j1"
        )

    monkeypatch.setattr(DefaultIngestionSubmissionService, "submit", fake_submit)
    db = MagicMock()
    out = run_raw_episode_ingestion(
        container=c,
        db=db,
        pot_id="p1",
        name="n",
        episode_body="b",
        source_description="src",
        reference_time=datetime.now(timezone.utc),
        idempotency_key=None,
        sync=sync,
    )
    assert out.ok
    assert out.event_id == "e1"
    if queued:
        assert out.status == "queued"
    else:
        assert out.status == "applied"
        assert out.episode_uuid == "u1"


def test_raw_submission_service_routes_raw_episode_through_agent(monkeypatch):
    from application.services.ingestion_submission_service import (
        DefaultIngestionSubmissionService,
    )
    from domain.ingestion_event_models import EventReceipt, IngestionSubmissionRequest
    from domain.ingestion_kinds import INGESTION_KIND_RAW_EPISODE

    captured = {}
    svc = object.__new__(DefaultIngestionSubmissionService)

    def fake_agent_submit(self, request, *, sync, wait, timeout_seconds):
        captured["request"] = request
        captured["sync"] = sync
        captured["wait"] = wait
        captured["timeout_seconds"] = timeout_seconds
        return EventReceipt(event_id="e-agent", status="queued", job_id="j-agent")

    monkeypatch.setattr(
        DefaultIngestionSubmissionService,
        "_submit_agent_reconciliation",
        fake_agent_submit,
    )

    receipt = svc._submit_raw_episode(
        IngestionSubmissionRequest(
            pot_id="p1",
            ingestion_kind=INGESTION_KIND_RAW_EPISODE,
            source_channel="ui_raw_ingest",
            source_system="manual",
            event_type="raw_episode",
            action="submit",
            payload={
                "name": "n",
                "episode_body": "b",
                "source_description": "src",
                "url": "https://example.com",
            },
        ),
        sync=False,
        wait=True,
        timeout_seconds=45.0,
    )

    assert receipt.event_id == "e-agent"
    forwarded = captured["request"]
    assert forwarded.ingestion_kind == INGESTION_KIND_RAW_EPISODE
    assert forwarded.source_id.startswith("raw_episode_")
    assert forwarded.payload["name"] == "n"
    assert forwarded.payload["episode_body"] == "b"
    assert forwarded.payload["url"] == "https://example.com"
    assert captured["sync"] is False
    assert captured["wait"] is True
    assert captured["timeout_seconds"] == 45.0


def test_agent_reconciliation_submission_requires_agent():
    from application.services.ingestion_submission_service import (
        DefaultIngestionSubmissionService,
    )
    from domain.ingestion_event_models import IngestionSubmissionRequest
    from domain.ingestion_kinds import INGESTION_KIND_RAW_EPISODE

    svc = object.__new__(DefaultIngestionSubmissionService)
    svc._c = MagicMock()
    svc._c.reconciliation_agent = None

    with pytest.raises(ValueError, match="no_reconciliation_agent"):
        svc._submit_agent_reconciliation(
            IngestionSubmissionRequest(
                pot_id="p1",
                ingestion_kind=INGESTION_KIND_RAW_EPISODE,
                source_channel="ui_raw_ingest",
                source_system="manual",
                event_type="raw_episode",
                action="submit",
                source_id="manual_1",
                payload={"name": "n", "episode_body": "b"},
            ),
            sync=False,
            wait=False,
            timeout_seconds=None,
        )
