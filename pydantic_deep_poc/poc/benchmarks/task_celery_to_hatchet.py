"""
Task: Replace Celery with Hatchet across the Potpie project.

Run:
    cd pydantic_deep_poc
    uv run python -m poc.benchmarks.task_celery_to_hatchet

First run clones https://github.com/potpie-ai/potpie.git into .repos/potpie.git
and creates a worktree at .repos/potpie/poc-celery-to-hatchet.
View traces at: https://logfire.pydantic.dev/
"""

from __future__ import annotations

import asyncio
import os
import sys

from poc.agents.factory import assemble_agent, default_deps
from poc.benchmarks._common import format_result, run_benchmark_traced
from poc.benchmarks.tracing_setup import init_benchmark_tracing, shutdown_benchmark_tracing
from poc.managers.run_context import init_run_context

TASK = (
    "Replace Celery with Hatchet across the entire Potpie codebase. "
    "Use the orchestrator workflow: bounded discovery, sequential implementation slices, "
    "then verification and a final staged diff review. "
    "Hatchet docs: https://docs.hatchet.run"
)


async def main() -> None:
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("Set OPENROUTER_API_KEY in .env", file=sys.stderr)
        sys.exit(1)

    print("Setting up worktree (clones potpie on first run — may take a minute)…")
    run = init_run_context(branch="poc/celery-to-hatchet")
    print(f"Worktree: {run.worktree_path}")
    print(f"Branch:   {run.branch_name}")
    print(f"Session:  {run.session_id}\n")

    agent = assemble_agent(
        context_block=f"project_root={run.project_root}"
    )
    deps = default_deps(run)

    r = await run_benchmark_traced(
        agent,
        deps,
        TASK,
        scenario="celery_to_hatchet",
        impl="pydantic-deep",
    )
    print(format_result(r))
    if r.error:
        sys.exit(1)


if __name__ == "__main__":
    init_benchmark_tracing()
    try:
        asyncio.run(main())
    finally:
        shutdown_benchmark_tracing()
