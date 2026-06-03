"""
Potpie Hatchet worker for context-graph jobs.

Run (after self-hosting Hatchet per https://docs.hatchet.run/self-hosting and setting
``HATCHET_CLIENT_TOKEN``):

    uv run python -m app.modules.context_graph.hatchet_worker

Implementation lives in context-engine (``adapters.inbound.hatchet.worker``).
"""

from __future__ import annotations

from app.core.database import SessionLocal
from app.modules.context_graph.wiring import build_container_for_session
from adapters.inbound.hatchet.worker import run_hatchet_context_graph_worker


def _main() -> None:
    run_hatchet_context_graph_worker(
        session_factory=SessionLocal,
        build_container=build_container_for_session,
    )


if __name__ == "__main__":
    _main()
