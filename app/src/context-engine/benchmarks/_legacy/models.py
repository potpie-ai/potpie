"""Benchmark data and result models."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_DATASET = Path(__file__).resolve().parent / "data" / "manifest.json"
DEFAULT_REPORT = Path(__file__).resolve().parents[1] / ".tmp" / "context-graph-benchmark-report.json"


@dataclass(frozen=True)
class BenchmarkDataset:
    name: str
    version: str
    pot_id: str
    repo_name: str
    seed_episodes: list[dict[str, Any]]
    seed_records: list[dict[str, Any]]
    pr_bundles: list[dict[str, Any]]
    scenarios: list[dict[str, Any]]
    thresholds: dict[str, Any]
    linear_issues: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BenchmarkDataset":
        return cls(
            name=str(data["name"]),
            version=str(data["version"]),
            pot_id=str(data["pot_id"]),
            repo_name=str(data["repo_name"]),
            seed_episodes=list(data.get("seed_episodes") or []),
            seed_records=list(data.get("seed_records") or []),
            pr_bundles=list(data.get("pr_bundles") or []),
            scenarios=list(data.get("scenarios") or []),
            thresholds=dict(data.get("thresholds") or {}),
            linear_issues=list(data.get("linear_issues") or []),
        )


@dataclass
class LatencyStats:
    min_ms: float | None = None
    p50_ms: float | None = None
    p95_ms: float | None = None
    p99_ms: float | None = None
    max_ms: float | None = None
    mean_ms: float | None = None


@dataclass
class ScenarioResult:
    id: str
    feature: str
    intent: str | None
    ok: bool
    score: float
    max_score: float
    grade: str
    errors: int
    iterations: int
    latency: LatencyStats
    assertions: list[dict[str, Any]] = field(default_factory=list)
    response_summary: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkReport:
    dataset: dict[str, str]
    target: dict[str, Any]
    summary: dict[str, Any]
    scenarios: list[ScenarioResult]
    seed: dict[str, Any]
    regressions: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "target": self.target,
            "summary": self.summary,
            "seed": self.seed,
            "regressions": self.regressions,
            "scenarios": [
                {
                    "id": item.id,
                    "feature": item.feature,
                    "intent": item.intent,
                    "ok": item.ok,
                    "score": round(item.score, 4),
                    "max_score": round(item.max_score, 4),
                    "grade": item.grade,
                    "errors": item.errors,
                    "iterations": item.iterations,
                    "latency_ms": {
                        "min": item.latency.min_ms,
                        "p50": item.latency.p50_ms,
                        "p95": item.latency.p95_ms,
                        "p99": item.latency.p99_ms,
                        "max": item.latency.max_ms,
                        "mean": item.latency.mean_ms,
                    },
                    "assertions": item.assertions,
                    "response_summary": item.response_summary,
                }
                for item in self.scenarios
            ],
        }
