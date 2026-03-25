"""Scenario 4: code change quality task (validation + test touch)."""

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
    agent = assemble_agent(context_block=f"project_root={run.project_root}")
    deps = default_deps(run)
    task = os.environ.get(
        "POC_TASK",
        "In the potpie worktree, choose a small module under app/ and add minimal input validation "
        "to one public function, plus a matching unit test under tests/. "
        "Use Code Changes Manager tools, then show_diff and apply_changes so changes land on disk.",
    )
    r = await run_benchmark_traced(
        agent, deps, task, scenario="scenario_4_quality", impl="pydantic-deep"
    )
    print(format_result(r))


if __name__ == "__main__":
    init_benchmark_tracing()
    try:
        asyncio.run(main())
    finally:
        shutdown_benchmark_tracing()
