"""Scenario 1: single deep agent, THINK_EXECUTE tools only, no subagents."""

from __future__ import annotations

import asyncio
import os
import sys

from poc.agents.factory import assemble_single_execute_only, default_deps
from poc.benchmarks._common import BenchmarkResult, format_result, run_benchmark_traced
from poc.benchmarks.tracing_setup import init_benchmark_tracing, shutdown_benchmark_tracing
from poc.managers.run_context import RunContext
from poc.repo_setup import base_worktree, setup


async def main() -> BenchmarkResult:
    setup()
    root = str(base_worktree().resolve())
    run = RunContext(project_root=root, worktree_path=root)
    agent = assemble_single_execute_only()
    deps = default_deps(run)
    task = os.environ.get(
        "POC_TASK",
        "List the top-level Python files in this repo using bash_command (ls).",
    )
    return await run_benchmark_traced(
        agent, deps, task, scenario="scenario_1_single", impl="pydantic-deep"
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
