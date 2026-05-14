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

from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session

from application.use_cases.context_graph_jobs import handle_backfill_pot
from adapters.outbound.hatchet.env_bootstrap import prepare_hatchet_client_env
from bootstrap.container import ContextEngineContainer
from domain.hatchet_events import EVENT_BACKFILL


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

    worker = hatchet.worker(
        "potpie-context-graph",
        slots=4,
        workflows=[backfill_task],
    )
    worker.start()
