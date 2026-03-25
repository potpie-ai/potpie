"""Scenario 5: document async delegation (check_task) — optional; same as multi by default."""

from __future__ import annotations

import asyncio
import os
import sys

from poc.agents.factory import assemble_agent, default_deps
from poc.benchmarks._common import format_result, run_benchmark_traced
from poc.benchmarks.tracing_setup import init_benchmark_tracing, shutdown_benchmark_tracing
from poc.managers.run_context import init_run_context


async def main() -> None:
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("Set OPENROUTER_API_KEY", file=sys.stderr)
        sys.exit(1)
    run = init_run_context()
    agent = assemble_agent(
        context_block=(
            f"project_root={run.project_root}\n"
            "You may use task(..., mode='async') then check_task when appropriate."
        )
    )
    deps = default_deps(run)
    task = os.environ.get(
        "POC_TASK",
        (
            "Decide whether this task can be split into parallel slices. "
            "If yes, use async worker delegation and then wait for completion."
        ),
    )
    r = await run_benchmark_traced(
        agent, deps, task, scenario="scenario_5_async", impl="pydantic-deep"
    )
    print(format_result(r))


if __name__ == "__main__":
    init_benchmark_tracing()
    try:
        asyncio.run(main())
    finally:
        shutdown_benchmark_tracing()
