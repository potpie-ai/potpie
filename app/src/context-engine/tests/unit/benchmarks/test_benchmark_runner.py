import asyncio
from dataclasses import dataclass, field
from typing import Any

import pytest

from benchmarks.models import BenchmarkDataset
from benchmarks.runner import run_benchmark


@dataclass
class _SpyRunner:
    pot_id: str = "test-pot"
    repo_name: str | None = "test-repo"
    query_delay: float = 0.05
    query_calls: list[dict[str, Any]] = field(default_factory=list)
    seed_episode_calls: list[dict[str, Any]] = field(default_factory=list)
    seed_record_calls: list[dict[str, Any]] = field(default_factory=list)
    ingest_pr_calls: list[tuple[int, str | None]] = field(default_factory=list)

    async def seed_episode(self, episode: dict[str, Any]) -> dict[str, Any]:
        self.seed_episode_calls.append(episode)
        return {"status": "ok"}

    async def seed_record(self, record: dict[str, Any]) -> dict[str, Any]:
        self.seed_record_calls.append(record)
        return {"status": "ok"}

    async def ingest_pr(self, pr_number: int, repo_name: str | None = None) -> dict[str, Any]:
        self.ingest_pr_calls.append((pr_number, repo_name))
        return {"status": "ok"}

    async def query(self, body: dict[str, Any]) -> dict[str, Any]:
        self.query_calls.append(body)
        await asyncio.sleep(self.query_delay)
        return {
            "kind": "context_resolution",
            "goal": "answer",
            "strategy": "resolve",
            "result": {
                "answer": {"summary": "mock response"},
                "coverage": {"status": "complete"},
                "quality": {"status": "ok"},
                "source_refs": [{"ref": "mock"}],
                "fallbacks": [],
            },
        }

    async def status(self, intent: str | None = None) -> dict[str, Any]:
        return {"ok": True, "status": "ready"}


def _make_dataset(scenario_ids: list[str]) -> BenchmarkDataset:
    return BenchmarkDataset(
        name="test",
        version="1.0.0",
        pot_id="test-pot",
        repo_name="test-repo",
        seed_episodes=[],
        seed_records=[],
        pr_bundles=[],
        scenarios=[
            {
                "id": sid,
                "feature": "context_graph",
                "intent": "feature",
                "request": {"goal": "answer", "query": f"query {sid}"},
                "expected": {
                    "must_contain": ["mock"],
                    "min_source_refs": 1,
                    "max_fallbacks": 1,
                },
            }
            for sid in scenario_ids
        ],
        thresholds={"pass_score": 0.75},
    )


@pytest.mark.asyncio
async def test_scenario_concurrency_one_runs_sequentially() -> None:
    runner = _SpyRunner(query_delay=0.05)
    dataset = _make_dataset(["s1", "s2"])

    start = asyncio.get_event_loop().time()
    report = await run_benchmark(
        runner,
        dataset,
        mode="mock",
        iterations=1,
        concurrency=1,
        scenario_concurrency=1,
        seed=False,
        ingest_pr_live=False,
    )
    elapsed = asyncio.get_event_loop().time() - start

    assert len(report.scenarios) == 2
    # Sequential: at least 2 * delay
    assert elapsed >= 0.10
    assert report.target["scenario_concurrency"] == 1


@pytest.mark.asyncio
async def test_scenario_concurrency_two_runs_in_parallel() -> None:
    runner = _SpyRunner(query_delay=0.05)
    dataset = _make_dataset(["s1", "s2"])

    start = asyncio.get_event_loop().time()
    report = await run_benchmark(
        runner,
        dataset,
        mode="mock",
        iterations=1,
        concurrency=1,
        scenario_concurrency=2,
        seed=False,
        ingest_pr_live=False,
    )
    elapsed = asyncio.get_event_loop().time() - start

    assert len(report.scenarios) == 2
    # Parallel: should be close to 1 * delay (with some overhead)
    assert elapsed < 0.09
    assert report.target["scenario_concurrency"] == 2


@pytest.mark.asyncio
async def test_results_are_ordered_by_scenario_input() -> None:
    runner = _SpyRunner(query_delay=0.01)
    dataset = _make_dataset(["s1", "s2", "s3"])

    report = await run_benchmark(
        runner,
        dataset,
        mode="mock",
        iterations=1,
        concurrency=1,
        scenario_concurrency=3,
        seed=False,
        ingest_pr_live=False,
    )

    ids = [s.id for s in report.scenarios]
    assert ids == ["s1", "s2", "s3"]
