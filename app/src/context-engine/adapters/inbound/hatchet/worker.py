"""
Hatchet worker entrypoint for context-graph jobs.

The host application must supply a SQLAlchemy session factory and
``build_container`` (Potpie wires ``SessionLocal`` + ``build_container_for_session``).

Event keys are defined in ``domain.hatchet_events`` and must match
``adapters.outbound.hatchet.hatchet_job_queue.HatchetContextGraphJobQueue``.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import timedelta

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from application.use_cases.context_graph_jobs import (
    handle_apply_episode,
    handle_backfill_pot,
    handle_ingest_pr,
    handle_ingestion_agent_run,
)
from adapters.outbound.hatchet.env_bootstrap import prepare_hatchet_client_env
from bootstrap.container import ContextEngineContainer
from domain.hatchet_events import (
    EVENT_APPLY_EPISODE,
    EVENT_BACKFILL,
    EVENT_INGEST_PR,
    EVENT_INGESTION_AGENT,
)


def run_hatchet_context_graph_worker(
    *,
    session_factory: Callable[[], Session],
    build_container: Callable[[Session], ContextEngineContainer],
) -> None:
    """Register Hatchet tasks and block until the worker exits."""
    from hatchet_sdk import Context, Hatchet

    prepare_hatchet_client_env()
    hatchet = Hatchet()

    class BackfillIn(BaseModel):
        model_config = ConfigDict(extra="allow")
        pot_id: str
        target_repo_name: str | None = None

    class IngestPrIn(BaseModel):
        model_config = ConfigDict(extra="allow")
        pot_id: str
        pr_number: int
        is_live_bridge: bool = True
        repo_name: str | None = None

    class IngestionAgentIn(BaseModel):
        model_config = ConfigDict(extra="allow")
        event_id: str
        pot_id: str = ""
        kind: str = ""

    class ApplyEpisodeIn(BaseModel):
        model_config = ConfigDict(extra="allow")
        pot_id: str
        event_id: str
        sequence: int = Field(..., ge=1)

    @hatchet.task(
        on_events=[EVENT_BACKFILL],
        input_validator=BackfillIn,
        execution_timeout=timedelta(hours=2),
    )
    def backfill_task(input: BackfillIn, ctx: Context) -> dict:
        del ctx
        db = session_factory()
        try:
            return handle_backfill_pot(
                db,
                input.pot_id,
                target_repo_name=input.target_repo_name,
                build_container=build_container,
            )
        finally:
            db.close()

    @hatchet.task(
        on_events=[EVENT_INGEST_PR],
        input_validator=IngestPrIn,
        execution_timeout=timedelta(hours=1),
    )
    def ingest_pr_task(input: IngestPrIn, ctx: Context) -> dict:
        del ctx
        db = session_factory()
        try:
            return handle_ingest_pr(
                db,
                input.pot_id,
                int(input.pr_number),
                is_live_bridge=bool(input.is_live_bridge),
                repo_name=input.repo_name,
                build_container=build_container,
            )
        finally:
            db.close()

    @hatchet.task(
        on_events=[EVENT_INGESTION_AGENT],
        input_validator=IngestionAgentIn,
        execution_timeout=timedelta(hours=1),
    )
    def ingestion_agent_task(input: IngestionAgentIn, ctx: Context) -> dict:
        del ctx
        db = session_factory()
        try:
            return handle_ingestion_agent_run(
                db, input.event_id, build_container=build_container
            )
        finally:
            db.close()

    @hatchet.task(
        on_events=[EVENT_APPLY_EPISODE],
        input_validator=ApplyEpisodeIn,
        execution_timeout=timedelta(minutes=30),
    )
    def apply_episode_task(input: ApplyEpisodeIn, ctx: Context) -> dict:
        del ctx
        db = session_factory()
        try:
            return handle_apply_episode(
                db,
                input.pot_id,
                input.event_id,
                int(input.sequence),
                build_container=build_container,
            )
        finally:
            db.close()

    worker = hatchet.worker(
        "potpie-context-graph",
        slots=4,
        workflows=[
            backfill_task,
            ingest_pr_task,
            ingestion_agent_task,
            apply_episode_task,
        ],
    )
    worker.start()
