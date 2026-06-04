"""In-process engine driver — run the bench without the HTTP server on :8001.

The default bench client (:class:`PotpieContextApiClient`) drives the engine
over HTTP, which means a gunicorn server must own port 8001 — the same port
the app uses. This module provides an alternative client that drives the
engine **in-process**: it builds the real ``IngestionServerContainer`` against
the shared datastores (Postgres / Neo4j) and reconciles **synchronously**
in-process, so a bench run needs neither the API server nor a Celery worker.

What it reuses (unchanged): Postgres, Neo4j, the connector registry (GitHub /
Linear / Notion + bench stubs), the real ``pydantic-deep`` reconciliation
agent, and the read trunk. What it drops: the HTTP hop, gunicorn on :8001,
Redis-as-task-broker, and the Celery worker. Reconciliation runs through the
same ``handle_process_batch`` verb the worker calls — just inline.

Activate with ``POTPIE_BENCH_INPROCESS=1`` (or ``run --local``); the bench's
``make_client`` returns this instead of the HTTP client. It duck-types the
subset of the HTTP client surface the harness uses, so the runner, lifecycle,
query and graph-inspect layers work unchanged. Ingestion takes the
``replay._replay_all_inprocess`` fast path (submit-all → process-once →
read-back), which keeps reconciliation single-threaded and batched.
"""

from __future__ import annotations

import logging
import os
import sys
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _ensure_repo_root_on_path() -> None:
    """Put the repo root on ``sys.path`` so ``app.*`` imports resolve.

    The context-engine packages (``adapters`` / ``domain`` / ``bootstrap``)
    are importable via the editable ``.pth``; the app-side wiring
    (``app.modules.context_graph.wiring`` etc.) lives at the repo root, which
    is only on the path when the process was launched from there. This file
    is ``app/src/context-engine/benchmarks/core/local_engine.py`` → the repo
    root is ``parents[4]``.
    """
    repo_root = str(Path(__file__).resolve().parents[4])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


_MODELS_LOADED = False


def _load_all_models_once() -> None:
    """Import the full SQLAlchemy model registry.

    The app registers every model via ``from app.core.models import *`` at
    startup; without it, cross-model relationships (e.g.
    ``ContextGraphPotMember.user`` → ``User``) fail to map. The in-process
    harness isn't the app, so we load the same registry explicitly once.
    """
    global _MODELS_LOADED
    if _MODELS_LOADED:
        return
    import app.core.models  # noqa: F401 — registers all SQLAlchemy models

    _MODELS_LOADED = True


class _Resp:
    """Tiny stand-in for an ``httpx.Response`` (only ``.json()`` is used)."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def json(self) -> dict[str, Any]:
        return self._payload


class InProcessEngineClient:
    """Drives the engine in-process against the shared datastores.

    Implements the slice of the ``PotpieContextApiClient`` surface the bench
    calls: ``create_context_pot``, ``get_event``, ``context_graph_query``,
    ``reset`` plus the in-process-only ``submit_only`` / ``process_pending``
    used by the ingestion fast path. ``inprocess = True`` lets the harness
    pick the single-threaded, batched ingestion path.
    """

    inprocess = True
    _base = "inprocess://local"
    _api_key = "inprocess"

    def __init__(self) -> None:
        _ensure_repo_root_on_path()
        _load_all_models_once()
        # A synthetic owner for the ephemeral bench pots. The worker-scoped
        # pot resolver does not enforce per-actor access, so any stable id works.
        self._uid = os.environ.get("POTPIE_BENCH_INPROCESS_UID", "bench-local")

    # -- imports are lazy so importing this module never drags the engine in --
    @staticmethod
    def _session():
        from app.core.database import SessionLocal

        return SessionLocal()

    @staticmethod
    def _build_container(db):
        from domain.ports.context_graph_job_queue import NoOpContextGraphJobQueue
        from app.modules.context_graph.wiring import build_container_for_session

        container = build_container_for_session(db)
        # Process batches inline (below) instead of enqueueing to Celery.
        container.jobs = NoOpContextGraphJobQueue()
        return container

    # ------------------------------------------------------------------
    # Pot lifecycle
    # ------------------------------------------------------------------
    def create_context_pot(
        self, *, slug: str, display_name: str | None, primary_repo_name: str | None
    ) -> dict[str, Any]:
        from app.modules.context_graph.context_graph_pot_member_model import (
            ContextGraphPotMember,
        )
        from app.modules.context_graph.context_graph_pot_model import ContextGraphPot
        from app.modules.context_graph.context_graph_pot_repository_model import (
            ContextGraphPotRepository,
        )
        from app.modules.context_graph.pot_member_roles import POT_ROLE_OWNER
        from app.modules.users.user_model import User

        pot_id = str(uuid.uuid4())
        db = self._session()
        try:
            # The pot's user_id is an FK to users.uid. Reuse an existing user
            # (any one — the worker-scoped resolver doesn't enforce per-actor
            # access) or create a minimal bench user if the DB is empty.
            uid = os.environ.get("POTPIE_BENCH_INPROCESS_UID") or self._uid
            existing = db.query(User.uid).filter(User.uid == uid).first()
            if existing is None:
                any_user = db.query(User.uid).first()
                if any_user is not None:
                    uid = any_user[0]
                else:
                    db.add(
                        User(
                            uid=uid,
                            email=f"{uid}@bench.local",
                            display_name="bench local",
                            email_verified=True,
                        )
                    )
                    db.flush()
            self._uid = uid
            db.add(
                ContextGraphPot(
                    id=pot_id,
                    user_id=self._uid,
                    created_by_user_id=self._uid,
                    display_name=(display_name or "").strip() or None,
                    slug=slug,
                    primary_repo_name=(primary_repo_name or "").strip() or None,
                )
            )
            db.add(
                ContextGraphPotMember(pot_id=pot_id, user_id=self._uid, role=POT_ROLE_OWNER)
            )
            prn = (primary_repo_name or "").strip()
            if prn and "/" in prn:
                owner, repo = (p.strip() for p in prn.split("/", 1))
                if owner and repo:
                    db.add(
                        ContextGraphPotRepository(
                            id=str(uuid.uuid4()),
                            pot_id=pot_id,
                            provider="github",
                            provider_host="github.com",
                            owner=owner,
                            repo=repo,
                            added_by_user_id=self._uid,
                        )
                    )
            db.commit()
        finally:
            db.close()
        logger.info("created in-process pot %s slug=%s repo=%s", pot_id, slug, primary_repo_name)
        return {"id": pot_id}

    def reset(self, body: dict[str, Any]) -> _Resp:
        pot_id = body["pot_id"]
        db = self._session()
        try:
            out = self._build_container(db).context_graph.reset_pot(pot_id)
            db.commit()
        finally:
            db.close()
        return _Resp(out if isinstance(out, dict) else {"ok": True})

    # ------------------------------------------------------------------
    # Ingestion (batched, inline) — used by replay._replay_all_inprocess
    # ------------------------------------------------------------------
    def submit_only(self, pot_id: str, body: dict[str, Any]) -> str:
        """Admit one event into the open batch (no enqueue, no processing).

        Returns the event id. Reconciliation happens later, once, in
        ``process_pending``.
        """
        from datetime import datetime, timezone

        from domain.ingestion_event_models import IngestionSubmissionRequest

        occurred_raw = body.get("occurred_at")
        occurred_at = None
        if isinstance(occurred_raw, str) and occurred_raw:
            try:
                occurred_at = datetime.fromisoformat(occurred_raw.replace("Z", "+00:00"))
                if occurred_at.tzinfo is None:
                    occurred_at = occurred_at.replace(tzinfo=timezone.utc)
            except ValueError:
                occurred_at = None

        req = IngestionSubmissionRequest(
            pot_id=pot_id,
            ingestion_kind=body.get("ingestion_kind") or "agent_reconciliation",
            source_channel=body.get("source_channel") or "benchmark",
            source_system=body["source_system"],
            event_type=body["event_type"],
            action=body["action"],
            payload=dict(body.get("payload") or {}),
            source_id=body.get("source_id"),
            repo_name=(body.get("repo_name") or None),
            occurred_at=occurred_at,
        )
        db = self._session()
        try:
            receipt = self._build_container(db).ingestion_submission(db).submit(req)
            db.commit()
        finally:
            db.close()
        return receipt.event_id

    def process_pending(self, pot_id: str, *, max_batches: int = 200) -> int:
        """Reconcile every pending batch for the pot, inline. Returns count."""
        from application.use_cases.context_graph_jobs import handle_process_batch
        from app.modules.context_graph.wiring import build_container_for_session

        processed = 0
        for _ in range(max_batches):
            db = self._session()
            try:
                batch_id = self._build_container(db).batch_repository(db).get_open_batch_id_for_pot(
                    pot_id
                )
                db.commit()
            finally:
                db.close()
            if not batch_id:
                break
            logger.info("in-process: reconciling batch %s (#%d)...", batch_id, processed + 1)
            db2 = self._session()
            try:
                handle_process_batch(
                    db2, batch_id, build_ingestion_server=build_container_for_session
                )
                db2.commit()
            except Exception:  # noqa: BLE001 — surfaced per-event via get_event
                db2.rollback()
                logger.exception("in-process batch %s failed", batch_id)
            finally:
                db2.close()
            processed += 1
        return processed

    def get_event(self, event_id: str) -> dict[str, Any]:
        from adapters.inbound.http.api.v1.context.event_payload import (
            ingestion_event_to_payload,
        )
        from adapters.outbound.postgres.reconciliation_ledger import (
            SqlAlchemyReconciliationLedger,
        )

        db = self._session()
        try:
            container = self._build_container(db)
            ev = container.event_query_service(db).get_event(event_id)
            if ev is None:
                return {}
            reco = SqlAlchemyReconciliationLedger(db)
            runs = []
            for run in reco.list_runs_for_event(event_id):
                runs.append(
                    {
                        "id": run.id,
                        "attempt_number": run.attempt_number,
                        "status": run.status,
                        "agent_name": run.agent_name,
                        "error": run.error,
                        "entity_mutation_count": run.entity_mutation_count,
                        "edge_mutation_count": run.edge_mutation_count,
                    }
                )
            out = ingestion_event_to_payload(ev)
            out["reconciliation_runs"] = runs
            return out
        finally:
            db.close()

    # ------------------------------------------------------------------
    # Reads (snapshot + resolve)
    # ------------------------------------------------------------------
    def context_graph_query(self, body: dict[str, Any]) -> dict[str, Any]:
        from domain.graph_query import ContextGraphQuery

        query = ContextGraphQuery(**body)
        db = self._session()
        try:
            result = self._build_container(db).context_graph.query(query)
        finally:
            db.close()
        return result.model_dump(mode="json")
