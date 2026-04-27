"""Benchmark orchestration."""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import Any

from benchmarks.dataset import build_pr_seed_episodes
from benchmarks.evaluator import evaluate_response, grade, latency_stats, summarize_response
from benchmarks.models import BenchmarkDataset, BenchmarkReport, ScenarioResult
from benchmarks.runners import BenchmarkRunner


async def run_benchmark(
    runner: BenchmarkRunner,
    dataset: BenchmarkDataset,
    *,
    mode: str,
    iterations: int,
    concurrency: int,
    scenario_concurrency: int,
    seed: bool,
    ingest_pr_live: bool,
    baseline: dict[str, Any] | None = None,
) -> BenchmarkReport:
    seed_result = await _seed_dataset(runner, dataset, seed=seed, ingest_pr_live=ingest_pr_live)
    scenario_semaphore = asyncio.Semaphore(max(1, scenario_concurrency))

    async def run_one(scenario: dict[str, Any]) -> ScenarioResult:
        async with scenario_semaphore:
            return await _run_scenario(runner, scenario, iterations=iterations, concurrency=concurrency)

    scenario_tasks = [asyncio.create_task(run_one(s)) for s in dataset.scenarios]
    scenario_results = await asyncio.gather(*scenario_tasks)

    total_score = sum(item.score for item in scenario_results)
    max_score = sum(item.max_score for item in scenario_results)
    pass_threshold = float(dataset.thresholds.get("pass_score", 0.75))
    ratio = total_score / max_score if max_score else 0.0
    regressions = _compare_baseline(scenario_results, baseline, dataset.thresholds)
    summary = {
        "ok": (
            ratio >= pass_threshold
            and not regressions
            and all(item.errors == 0 for item in scenario_results)
            and all(item.ok for item in scenario_results)
        ),
        "score": round(ratio, 4),
        "grade": grade(total_score, max_score),
        "scenario_count": len(scenario_results),
        "error_count": sum(item.errors for item in scenario_results),
        "pass_score": pass_threshold,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    return BenchmarkReport(
        dataset={"name": dataset.name, "version": dataset.version},
        target={
            "mode": mode,
            "pot_id": runner.pot_id,
            "repo_name": runner.repo_name,
            "iterations": iterations,
            "concurrency": concurrency,
            "scenario_concurrency": scenario_concurrency,
        },
        summary=summary,
        scenarios=scenario_results,
        seed=seed_result,
        regressions=regressions,
    )


async def _seed_dataset(
    runner: BenchmarkRunner,
    dataset: BenchmarkDataset,
    *,
    seed: bool,
    ingest_pr_live: bool,
) -> dict[str, Any]:
    result: dict[str, Any] = {"enabled": seed, "episodes": 0, "records": 0, "prs": 0, "errors": []}
    if not seed:
        return result
    episodes = list(dataset.seed_episodes)
    if not ingest_pr_live:
        episodes.extend(build_pr_seed_episodes(dataset))

    for episode in episodes:
        try:
            await runner.seed_episode(episode)
            result["episodes"] += 1
        except Exception as exc:
            result["errors"].append({"kind": "episode", "name": episode.get("name"), "error": str(exc)})

    for record in dataset.seed_records:
        try:
            await runner.seed_record(record)
            result["records"] += 1
        except Exception as exc:
            result["errors"].append({"kind": "record", "type": record.get("type"), "error": str(exc)})

    if ingest_pr_live:
        for bundle in dataset.pr_bundles:
            try:
                await runner.ingest_pr(
                    int(bundle["pr_data"]["number"]),
                    str(bundle.get("repo_name") or dataset.repo_name),
                )
                result["prs"] += 1
            except Exception as exc:
                result["errors"].append({"kind": "pr", "number": bundle.get("pr_data", {}).get("number"), "error": str(exc)})
    return result


async def _run_scenario(
    runner: BenchmarkRunner,
    scenario: dict[str, Any],
    *,
    iterations: int,
    concurrency: int,
) -> ScenarioResult:
    latencies: list[float] = []
    errors = 0
    best_score = 0.0
    best_max = 0.0
    best_assertions: list[dict[str, Any]] = []
    last_response: dict[str, Any] = {}
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def once() -> tuple[float, dict[str, Any] | None, str | None]:
        async with semaphore:
            start = time.perf_counter()
            try:
                response = await runner.query(_scenario_query(runner, scenario))
                error = None
            except Exception as exc:
                response = None
                error = str(exc)
            latency = (time.perf_counter() - start) * 1000.0
            return latency, response, error

    for task in asyncio.as_completed([asyncio.create_task(once()) for _ in range(iterations)]):
        latency, response, error = await task
        latencies.append(latency)
        if error or response is None:
            errors += 1
            best_assertions.append({"name": "request", "passed": False, "weight": 1.0, "detail": error})
            continue
        normalized = _normalize_response(response)
        last_response = normalized
        score, max_score, assertions = evaluate_response(normalized, scenario, latency_ms=latency)
        if max_score and score / max_score >= (best_score / best_max if best_max else -1):
            best_score = score
            best_max = max_score
            best_assertions = assertions

    scenario_grade = grade(best_score, best_max)
    threshold = float(scenario.get("pass_score", 0.75))
    ok = errors == 0 and (best_score / best_max if best_max else 0.0) >= threshold
    return ScenarioResult(
        id=str(scenario["id"]),
        feature=str(scenario.get("feature") or "context_graph"),
        intent=scenario.get("intent"),
        ok=ok,
        score=best_score,
        max_score=best_max,
        grade=scenario_grade,
        errors=errors,
        iterations=iterations,
        latency=latency_stats(latencies),
        assertions=best_assertions,
        response_summary=summarize_response(last_response),
    )


def _scenario_query(runner: BenchmarkRunner, scenario: dict[str, Any]) -> dict[str, Any]:
    body = dict(scenario["request"])
    body.setdefault("pot_id", runner.pot_id)
    scope = dict(body.get("scope") or {})
    if runner.repo_name and "repo_name" not in scope:
        scope["repo_name"] = runner.repo_name
    if scope:
        body["scope"] = scope
    return body


def _normalize_response(response: dict[str, Any]) -> dict[str, Any]:
    result = response.get("result")
    if isinstance(result, dict) and (
        "answer" in result or "coverage" in result or "source_refs" in result
    ):
        normalized = dict(result)
        normalized.setdefault("kind", response.get("kind"))
        normalized.setdefault("goal", response.get("goal"))
        normalized.setdefault("strategy", response.get("strategy"))
        return normalized
    return response


def _compare_baseline(
    results: list[ScenarioResult],
    baseline: dict[str, Any] | None,
    thresholds: dict[str, Any],
) -> list[dict[str, Any]]:
    if not baseline:
        return []
    max_score_drop = float(thresholds.get("max_score_drop", 0.05))
    max_p95_latency_ratio = float(thresholds.get("max_p95_latency_ratio", 1.25))
    previous = {item["id"]: item for item in baseline.get("scenarios", []) if isinstance(item, dict)}
    regressions: list[dict[str, Any]] = []
    for current in results:
        old = previous.get(current.id)
        if not old:
            continue
        old_score = float(old.get("score") or 0.0) / float(old.get("max_score") or 1.0)
        new_score = current.score / current.max_score if current.max_score else 0.0
        if old_score - new_score > max_score_drop:
            regressions.append(
                {"id": current.id, "kind": "score_drop", "previous": round(old_score, 4), "current": round(new_score, 4)}
            )
        old_p95 = ((old.get("latency_ms") or {}).get("p95") or 0) if isinstance(old.get("latency_ms"), dict) else 0
        new_p95 = current.latency.p95_ms or 0
        if old_p95 and new_p95 / old_p95 > max_p95_latency_ratio:
            regressions.append(
                {"id": current.id, "kind": "latency_p95", "previous": old_p95, "current": new_p95}
            )
    return regressions
