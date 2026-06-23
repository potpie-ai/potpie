from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from adapters.outbound.graph.backends.in_memory_backend import InMemoryGraphBackend
from application.services.graph_service import DefaultGraphService
from application.services.ingestion_submission_service import (
    DefaultIngestionSubmissionService,
)
from domain.ingestion_event_models import IngestionSubmissionRequest
from domain.ingestion_kinds import INGESTION_KIND_AGENT_RECONCILIATION
from domain.ports.claim_query import ClaimQueryFilter
from domain.ports.pot_resolution import ResolvedPot, ResolvedPotRepo


def _service(*, reconciliation_agent=None):
    settings = MagicMock()
    settings.is_enabled.return_value = True
    pots = MagicMock()
    pots.resolve_pot.return_value = ResolvedPot(
        pot_id="pot-1",
        name="test",
        repos=[
            ResolvedPotRepo(
                pot_id="pot-1",
                repo_id="repo-1",
                provider="github",
                provider_host="github.com",
                repo_name="o/r",
            )
        ],
    )
    graph = DefaultGraphService(backend=InMemoryGraphBackend())
    return graph, DefaultIngestionSubmissionService(
        settings=settings,
        pots=pots,
        graph=graph,
        reconciliation_agent=reconciliation_agent,
        reco_ledger=MagicMock(),
        events=MagicMock(),
        batches=MagicMock(),
        jobs=MagicMock(),
    )


def test_context_record_submits_without_reconciliation_agent() -> None:
    graph, service = _service(reconciliation_agent=None)

    receipt = service.submit(
        IngestionSubmissionRequest(
            pot_id="pot-1",
            ingestion_kind=INGESTION_KIND_AGENT_RECONCILIATION,
            source_channel="http",
            source_system="agent",
            event_type="context_record",
            action="preference",
            source_id="context-record:pref-1",
            payload={
                "record": {
                    "type": "preference",
                    "summary": "Prefer ruff for linting",
                    "details": {
                        "policy_kind": "style",
                        "prescription": "Prefer ruff for linting",
                    },
                    "source_refs": ["test:pref-1"],
                },
                "scope": {"service": "context-engine"},
            },
        ),
        sync=True,
    )

    assert receipt.status == "done"
    assert receipt.error is None
    rows = graph.backend.claim_query.find_claims(
        ClaimQueryFilter(pot_id="pot-1", predicate_in=("POLICY_APPLIES_TO",))
    )
    assert len(rows) == 1
    assert "ruff" in rows[0].fact.lower()


def test_non_record_submission_still_requires_reconciliation_agent() -> None:
    _, service = _service(reconciliation_agent=None)

    with pytest.raises(ValueError, match="no_reconciliation_agent"):
        service.submit(
            IngestionSubmissionRequest(
                pot_id="pot-1",
                ingestion_kind=INGESTION_KIND_AGENT_RECONCILIATION,
                source_channel="http",
                source_system="agent",
                event_type="pull_request",
                action="merged",
                source_id="pr-1",
                payload={"title": "merge"},
            )
        )
