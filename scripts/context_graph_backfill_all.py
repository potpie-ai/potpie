#!/usr/bin/env python3
"""One-time script to enqueue context graph backfill for all eligible projects.

Run from repo root:
  python -m scripts.context_graph_backfill_all

Requires CONTEXT_GRAPH_ENABLED=true and Celery worker consuming context-graph-etl queue.
"""

import os
import sys

# Add app to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

from app.core.database import SessionLocal
from app.modules.context_graph.tasks import context_graph_backfill_project
from app.modules.projects.projects_model import Project


def main() -> None:
    if os.getenv("CONTEXT_GRAPH_ENABLED", "false").lower() != "true":
        print("CONTEXT_GRAPH_ENABLED is not true. Set it in .env to run backfill.")
        sys.exit(1)
    db = SessionLocal()
    try:
        projects = (
            db.query(Project)
            .filter(
                Project.repo_name.isnot(None),
                Project.is_deleted == False,  # noqa: E712
                Project.status == "ready",
            )
            .all()
        )
        for p in projects:
            context_graph_backfill_project.delay(p.id)
            print(f"Enqueued backfill for project {p.id} ({p.repo_name})")
        print(f"Enqueued {len(projects)} backfill tasks.")
    finally:
        db.close()


if __name__ == "__main__":
    main()
