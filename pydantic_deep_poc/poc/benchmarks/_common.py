"""Shared benchmark helpers."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from pydantic_ai.usage import UsageLimits

from poc.config.provider import get_model_settings
from poc.config.settings import USAGE_REQUEST_LIMIT, USAGE_TOOL_CALLS_LIMIT


def _usage_summary(usage: Any) -> dict[str, Any]:
    if usage is None:
        return {}
    out: dict[str, Any] = {}
    for name in (
        "input_tokens",
        "output_tokens",
        "requests",
        "tool_calls",
        "total_tokens",
    ):
        if hasattr(usage, name):
            out[name] = getattr(usage, name)
    try:
        out["repr"] = repr(usage)
    except Exception:
        pass
    return out


@dataclass
class BenchmarkResult:
    scenario: str
    impl: str
    wall_time_s: float
    output_excerpt: str = ""
    usage: Any = None
    usage_summary: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


async def run_benchmark_traced(
    agent: Any,
    deps: Any,
    task: str,
    *,
    scenario: str,
    impl: str,
) -> BenchmarkResult:
    from poc.config.settings import MODEL_NAME
    from poc.tracing.logfire_tracer import logfire_trace_metadata

    run = getattr(deps, "poc_run", None)
    run_id = getattr(run, "session_id", "") if run is not None else ""
    with logfire_trace_metadata(
        scenario=scenario,
        run_id=run_id,
        model=MODEL_NAME,
        impl=impl,
    ):
        return await run_agent_timed(agent, deps, task, scenario=scenario, impl=impl)


async def run_agent_timed(
    agent: Any,
    deps: Any,
    task: str,
    *,
    scenario: str,
    impl: str,
) -> BenchmarkResult:
    t0 = time.perf_counter()
    try:
        result = await agent.run(
            task,
            deps=deps,
            model_settings=get_model_settings(),
            usage_limits=UsageLimits(
                request_limit=USAGE_REQUEST_LIMIT,
                tool_calls_limit=USAGE_TOOL_CALLS_LIMIT,
            ),
        )
        wall = time.perf_counter() - t0
        out = str(result.output) if result.output is not None else ""
        usage = getattr(result, "usage", None)
        return BenchmarkResult(
            scenario=scenario,
            impl=impl,
            wall_time_s=wall,
            output_excerpt=out[:4000],
            usage=usage,
            usage_summary=_usage_summary(usage),
        )
    except Exception as e:
        return BenchmarkResult(
            scenario=scenario,
            impl=impl,
            wall_time_s=time.perf_counter() - t0,
            error=str(e),
        )


def format_result(r: BenchmarkResult) -> str:
    lines = [
        f"scenario={r.scenario} impl={r.impl} wall_s={r.wall_time_s:.3f}",
    ]
    if r.error:
        lines.append(f"ERROR: {r.error}")
    if r.usage_summary:
        lines.append(f"usage_summary={r.usage_summary}")
    elif r.usage:
        lines.append(f"usage={r.usage}")
    lines.append("--- output (excerpt) ---")
    lines.append(r.output_excerpt[:2000])
    return "\n".join(lines)
