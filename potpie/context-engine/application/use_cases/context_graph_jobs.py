"""Shared handlers for context-graph background jobs (Celery, Hatchet, etc.)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from sqlalchemy.orm import Session

from application.use_cases.apply_episode_step import apply_episode_step_for_event
from application.use_cases.backfill_pot import backfill_pot_context
from domain.ingestion_event_models import IngestionSubmissionRequest
from domain.ingestion_kinds import INGESTION_KIND_GITHUB_MERGED_PR
from application.use_cases.run_ingestion_agent_worker import run_ingestion_agent_for_event
from bootstrap.container import ContextEngineContainer


def handle_backfill_pot(
    db: Session,
    pot_id: str,
    *,
    target_repo_name: str | None = None,
    build_container: Callable[[Session], ContextEngineContainer],
) -> dict[str, Any]:
    container = build_container(db)
    if container.context_graph is None:
        return {"status": "error", "error": "context_graph_unavailable"}
    return backfill_pot_context(
        settings=container.settings,
        pots=container.pots,
        source_for_repo=container.source_for_repo,
        ledger=container.ledger(db),
        context_graph=container.context_graph,
        pot_id=pot_id,
        target_repo_name=target_repo_name,
    )


def handle_ingest_pr(
    db: Session,
    pot_id: str,
    pr_number: int,
    *,
    is_live_bridge: bool = True,
    repo_name: str | None = None,
    build_container: Callable[[Session], ContextEngineContainer],
) -> dict[str, Any]:
    container = build_container(db)
    svc = container.ingestion_submission(db)
    req = IngestionSubmissionRequest(
        pot_id=pot_id,
        ingestion_kind=INGESTION_KIND_GITHUB_MERGED_PR,
        source_channel="async_job",
        source_system="github",
        event_type="pull_request",
        action="merged",
        payload={
            "pr_number": pr_number,
            "is_live_bridge": is_live_bridge,
            "repo_name": repo_name,
        },
        repo_name=repo_name,
    )
    receipt = svc.submit(req, sync=True)
    if receipt.duplicate:
        return {
            "status": "duplicate",
            "pot_id": pot_id,
            "pr_number": pr_number,
            "event_id": receipt.event_id,
        }
    if receipt.status == "error":
        return {
            "status": "error",
            "pot_id": pot_id,
            "pr_number": pr_number,
            "event_id": receipt.event_id,
            "error": receipt.error,
        }
    out = dict(receipt.extras or {})
    out.setdefault("status", "success")
    out["event_id"] = receipt.event_id
    out["pot_id"] = pot_id
    out["pr_number"] = pr_number
    return out


def handle_ingestion_agent_run(
    db: Session,
    event_id: str,
    *,
    build_container: Callable[[Session], ContextEngineContainer],
) -> dict[str, Any]:
    container = build_container(db)
    agent = container.reconciliation_agent
    if agent is None:
        return {"ok": False, "error": "no_reconciliation_agent"}
    return run_ingestion_agent_for_event(
        agent,
        container.reconciliation_ledger(db),
        event_id,
        container.jobs,
    )


def handle_apply_episode(
    db: Session,
    pot_id: str,
    event_id: str,
    sequence: int,
    *,
    build_container: Callable[[Session], ContextEngineContainer],
) -> dict[str, Any]:
    container = build_container(db)
    if container.context_graph is None:
        return {"ok": False, "error": "context_graph_unavailable"}
    r = apply_episode_step_for_event(
        container.context_graph,
        container.reconciliation_ledger(db),
        event_id,
        sequence,
    )
    return {
        "ok": r.ok,
        "event_id": event_id,
        "sequence": sequence,
        "error": r.error,
        "episode_uuids": r.episode_uuids,
    }
