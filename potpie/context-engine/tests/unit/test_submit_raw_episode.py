"""Unified raw episode ingest orchestration."""

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from potpie.context_engine.application.use_cases.submit_raw_episode import submit_raw_episode
from potpie.context_engine.bootstrap.ingestion_server import IngestionServerContainer
from potpie.context_engine.domain.ports.pot_resolution import ResolvedPot, ResolvedPotRepo


def _container(
    episodic: MagicMock,
    *,
    jobs: MagicMock | None = None,
    context_graph: MagicMock | None = None,
) -> IngestionServerContainer:
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
    j = jobs if jobs is not None else MagicMock()
    return IngestionServerContainer(
        settings=settings,
        graph_writer=episodic,
        pots=pots,
        jobs=j,
        context_graph=context_graph,
    )


def test_async_without_database_errors():
    c = _container(MagicMock())
    out = submit_raw_episode(
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


def test_sync_without_database_requires_database():
    """Synchronous narrative ingest now requires the async pipeline."""
    c = _container(MagicMock())
    out = submit_raw_episode(
        container=c,
        db=None,
        pot_id="p1",
        name="n",
        episode_body="b",
        source_description="src",
        reference_time=datetime.now(timezone.utc),
        idempotency_key=None,
        sync=True,
    )
    assert not out.ok
    assert out.error == "requires_database"


@pytest.mark.parametrize("sync", [True, False])
def test_with_database_delegates_to_submission(monkeypatch, sync: bool):
    """Patch IngestionSubmissionService.submit to avoid DB setup."""
    from potpie.context_engine.application.services.ingestion_submission_service import (
        DefaultIngestionSubmissionService,
    )
    from potpie.context_engine.domain.ingestion_event_models import EventReceipt

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
        return EventReceipt(event_id="e1", status="done", mutation_id="u1", job_id="j1")

    monkeypatch.setattr(DefaultIngestionSubmissionService, "submit", fake_submit)
    db = MagicMock()
    out = submit_raw_episode(
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
        assert out.mutation_id == "u1"
