"""Live end-to-end pipeline tests using REAL Postgres + REAL LLM.

These exercise the parts the deterministic suite (``test_e2e_topology.py``)
deliberately skipped:

- **Postgres** — a throwaway database is created inside the configured
  Postgres instance (see ``conftest.pg_test_db``), the context-engine schema
  is built, and event submission / batching round-trips through it. Dropped on
  teardown.
- **LLM** — the real reconciliation agent reconciles a submitted event into the
  graph (ingest → batch → ``process_batch`` → ``apply_plan``), and the real
  answer synthesizer answers a query over ingested topology.

Skips automatically when Neo4j / Postgres / an LLM key is unavailable.
"""

from __future__ import annotations

import asyncio

import pytest

from application.use_cases.process_batch import process_batch
from domain.context_events import EventRef
from domain.graph_mutations import EdgeUpsert, EntityUpsert
from domain.graph_query import (
    ContextGraphGoal,
    ContextGraphQuery,
    ContextGraphScope,
)
from domain.ingestion_event_models import IngestionSubmissionRequest
from domain.ingestion_kinds import INGESTION_KIND_AGENT_RECONCILIATION
from domain.reconciliation import ReconciliationPlan

pytestmark = pytest.mark.integration


_DEPLOY_EVENT = IngestionSubmissionRequest(
    pot_id="",  # filled per test
    ingestion_kind=INGESTION_KIND_AGENT_RECONCILIATION,
    source_channel="webhook",
    source_system="deploy",
    event_type="deployment",
    action="succeeded",
    payload={
        "service": "auth-svc",
        "environment": "prod",
        "version": "v2.3.1",
        "status": "succeeded",
        "summary": "Deployed service auth-svc version v2.3.1 to the prod environment.",
    },
    source_id="deploy-evt-1",
    repo_name="acme/platform",
)


def _request_for(pot_id: str) -> IngestionSubmissionRequest:
    from dataclasses import replace

    return replace(_DEPLOY_EVENT, pot_id=pot_id)


def _submit(container, sessionmaker, request) -> str:
    """Submit an event (async/debounced) and commit; return the event id."""
    session = sessionmaker()
    try:
        receipt = container.ingestion_submission(session).submit(request, sync=False)
        session.commit()
        return receipt.event_id
    finally:
        session.close()


def _claim_and_process(container, sessionmaker, *, pot_id, event_id):
    session = sessionmaker()
    try:
        batch_repo = container.batch_repository(session)
        batch_id = batch_repo.get_latest_batch_id_for_event(
            event_id
        ) or batch_repo.get_open_batch_id_for_pot(pot_id)
        assert batch_id, "no batch was created for the submitted event"
        batch = batch_repo.claim_batch_by_id(batch_id)
        assert batch is not None, "batch could not be claimed"
        outcome = process_batch(
            batch=batch,
            agent=container.reconciliation_agent,
            batches=batch_repo,
            reco_ledger=container.reconciliation_ledger(session),
            checkpoints=container.agent_checkpoint_store(session),
            pots=container.pots,
            execution_log=container.agent_execution_log(session),
        )
        session.commit()
        return outcome
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Postgres round-trip (real DB, no LLM)
# ---------------------------------------------------------------------------


class TestPostgresPipeline:
    def test_event_submission_persists_and_batches(
        self, db_container, pg_test_db, pot_id
    ) -> None:
        event_id = _submit(db_container, pg_test_db.sessionmaker, _request_for(pot_id))
        assert event_id

        session = pg_test_db.sessionmaker()
        try:
            batch_repo = db_container.batch_repository(session)
            batch_id = batch_repo.get_latest_batch_id_for_event(event_id)
            assert batch_id, "submitted event was not added to a batch"
            refs = batch_repo.list_events_for_batch(batch_id)
            assert event_id in {r.event_id for r in refs}
        finally:
            session.close()

    def test_open_batch_tracked_per_pot(self, db_container, pg_test_db, pot_id) -> None:
        _submit(db_container, pg_test_db.sessionmaker, _request_for(pot_id))
        session = pg_test_db.sessionmaker()
        try:
            batch_repo = db_container.batch_repository(session)
            assert batch_repo.get_open_batch_id_for_pot(pot_id) is not None
        finally:
            session.close()


# ---------------------------------------------------------------------------
# Full ingestion pipeline (real Postgres + real LLM reconciliation agent)
# ---------------------------------------------------------------------------


class TestLLMReconciliationPipeline:
    def test_agent_reconciles_event_into_graph(
        self, pipeline_container, pg_test_db, pot_id, settings
    ) -> None:
        container = pipeline_container
        assert container.reconciliation_agent is not None

        event_id = _submit(container, pg_test_db.sessionmaker, _request_for(pot_id))
        outcome = _claim_and_process(
            container, pg_test_db.sessionmaker, pot_id=pot_id, event_id=event_id
        )

        assert outcome.ok is True, f"batch failed: {outcome.error}"
        assert event_id in outcome.completed_event_ids

        # A deploy event must reconcile into the canonical topology *shape* —
        # a Service deployed to an Environment — not merely "some entity".
        # Entity keys are derived code-side (slugify), so the Service/Environment
        # labels and the DEPLOYED_TO edge below are robust against LLM phrasing.
        from neo4j import GraphDatabase

        drv = GraphDatabase.driver(
            settings.neo4j_uri(), auth=(settings.neo4j_user(), settings.neo4j_password())
        )
        try:
            with drv.session() as session:
                deployed = [
                    dict(r)
                    for r in session.run(
                        "MATCH (s:Service {group_id:$g})"
                        "-[r:RELATES_TO {name:'DEPLOYED_TO'}]->"
                        "(e:Environment {group_id:$g}) "
                        "RETURN s.entity_key AS svc, e.entity_key AS env, "
                        "r.environment AS environment",
                        g=pot_id,
                    )
                ]
        finally:
            drv.close()

        assert deployed, "agent did not wire Service-[:DEPLOYED_TO]->Environment"
        # If the agent enriched the edge's environment property, it must match.
        for row in deployed:
            if row["environment"] is not None:
                assert row["environment"].lower() == "prod"


# ---------------------------------------------------------------------------
# Read-side LLM query (goal=ANSWER) over deterministically-ingested topology
# ---------------------------------------------------------------------------


def _seed_topology(container, pot_id: str) -> None:
    plan = ReconciliationPlan(
        event_ref=EventRef(event_id="seed", source_system="test", pot_id=pot_id),
        summary="seed topology",
        entity_upserts=[
            EntityUpsert("service:web", ("Entity", "Service"), {"name": "web"}),
            EntityUpsert("environment:prod", ("Entity", "Environment"), {"name": "prod"}),
        ],
        edge_upserts=[
            EdgeUpsert(
                "DEPLOYED_TO",
                "service:web",
                "environment:prod",
                {"environment": "prod", "evidence_strength": "deterministic", "source_ref": "seed:1"},
            )
        ],
        edge_deletes=[],
        invalidations=[],
    )
    asyncio.run(container.context_graph.apply_plan_async(plan, expected_pot_id=pot_id))


class TestEnvelopeQuery:
    def test_retrieve_returns_evidence_envelope_over_topology(
        self, pipeline_container, pot_id
    ) -> None:
        container = pipeline_container
        _seed_topology(container, pot_id)

        query = ContextGraphQuery(
            pot_id=pot_id,
            query="Which environment runs the web service?",
            goal=ContextGraphGoal.RETRIEVE,
            scope=ContextGraphScope(services=["web"]),
        )
        result = asyncio.run(container.context_graph.query_async(query))

        assert result.error is None, f"resolve query errored: {result.error}"
        # One read contract: the envelope carries ranked evidence (items +
        # coverage), never a server-side synthesized answer block.
        assert isinstance(result.result, dict)
        assert "items" in result.result
        assert "answer" not in result.result
