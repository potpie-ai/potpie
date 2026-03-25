"""Compare pydantic-deep PoC vs Potpie CodeGenAgent (optional baseline).

Potpie baseline must run in the main repo venv with PYTHONPATH set; this harness
always runs the deep-agent path from the PoC venv.
"""

from __future__ import annotations

import asyncio
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from poc.agents.factory import assemble_agent, default_deps
from poc.benchmarks._common import BenchmarkResult, run_benchmark_traced
from poc.benchmarks.tracing_setup import init_benchmark_tracing, shutdown_benchmark_tracing
from poc.managers.run_context import init_run_context


def _poc_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _results_path() -> Path:
    d = _poc_root() / "poc" / "comparison" / "results"
    d.mkdir(parents=True, exist_ok=True)
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return d / f"{day}.md"


async def run_deep_sample(task: str) -> BenchmarkResult:
    run = init_run_context()
    agent = assemble_agent(context_block=f"project_root={run.project_root}")
    deps = default_deps(run)
    return await run_benchmark_traced(
        agent, deps, task, scenario="comparison_sample", impl="pydantic-deep"
    )


def write_report(deep: BenchmarkResult, extra: str = "") -> Path:
    path = _results_path()
    lines = [
        f"# PoC comparison {datetime.now(timezone.utc).isoformat()}",
        "",
        "## pydantic-deep (PoC venv)",
        f"- wall_s: {deep.wall_time_s:.3f}",
        f"- error: {deep.error or 'none'}",
        "",
        "```",
        deep.output_excerpt[:3000],
        "```",
        "",
        "## Potpie CodeGenAgent (main venv)",
        "",
        "Run manually from repo root, with secrets configured:",
        "",
        "```bash",
        "cd /path/to/potpie",
        "source .venv/bin/activate",
        "export PYTHONPATH=. ",
        "# python -c \"from app...CodeGenAgent ...\"  # wire ChatContext + run",
        "```",
        "",
        "This harness does not instantiate Potpie DI; fill timings in this file after manual run.",
        "",
        extra,
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


async def main_async() -> None:
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("OPENROUTER_API_KEY required for deep run", file=sys.stderr)
        sys.exit(1)
    task = os.environ.get(
        "POC_COMPARE_TASK",
        "Reply with one sentence describing your role, then stop.",
    )
    deep = await run_deep_sample(task)
    path = write_report(deep)
    print(format_result_line(deep))
    print("Wrote", path)


def format_result_line(r: BenchmarkResult) -> str:
    return f"impl={r.impl} wall_s={r.wall_time_s:.3f} err={r.error}"


def main() -> None:
    init_benchmark_tracing()
    try:
        asyncio.run(main_async())
    finally:
        shutdown_benchmark_tracing()


if __name__ == "__main__":
    main()
