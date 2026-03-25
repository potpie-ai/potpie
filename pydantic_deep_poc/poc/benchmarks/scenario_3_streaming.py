"""Scenario 3: same workload as scenario 2; streaming can be measured via pydantic-ai run_stream separately."""

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
        "Give a one-paragraph summary of what tools you have access to (names only).",
    )
    r = await run_benchmark_traced(
        agent, deps, task, scenario="scenario_3_streaming", impl="pydantic-deep"
    )
    print(format_result(r))
    print(
        "\nNote: For TTFT, use Agent.run_stream() and pydantic-ai StreamedRunResult; "
        "see https://ai.pydantic.dev/"
    )


if __name__ == "__main__":
    init_benchmark_tracing()
    try:
        asyncio.run(main())
    finally:
        shutdown_benchmark_tracing()
