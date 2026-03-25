"""Scenario 2: full 5 subagents, non-streaming."""

from __future__ import annotations

import asyncio
import os
import sys

from poc.agents.factory import assemble_agent, default_deps
from poc.benchmarks._common import BenchmarkResult, format_result, run_benchmark_traced
from poc.benchmarks.tracing_setup import init_benchmark_tracing, shutdown_benchmark_tracing
from poc.managers.run_context import init_run_context


async def main() -> BenchmarkResult:
    run = init_run_context()
    agent = assemble_agent(context_block=f"project_root={run.project_root}")
    deps = default_deps(run)
    task = os.environ.get(
        "POC_TASK",
        (
            "Use the orchestrator workflow to make a minimal edit: add a short comment "
            "# poc-test at the top of README.md in the worktree. "
            "Use bounded discovery, then delegate implementation, then verification."
        ),
    )
    return await run_benchmark_traced(
        agent, deps, task, scenario="scenario_2_multi", impl="pydantic-deep"
    )


if __name__ == "__main__":
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("Set OPENROUTER_API_KEY", file=sys.stderr)
        sys.exit(1)
    init_benchmark_tracing()
    try:
        r = asyncio.run(main())
        print(format_result(r))
    finally:
        shutdown_benchmark_tracing()
