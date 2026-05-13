"""ScenarioResult and BenchmarkReport dataclasses.

The result shape is JSON-serializable end-to-end so reports can be diffed
across runs. Per-axis scores (ingestion / retrieval / synthesis) are
tracked separately so a synthesis regression doesn't get hidden by a
stable retrieval score.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class AxisScore:
    score: float  # 0..100
    passed: bool
    details: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


@dataclass
class ScenarioResult:
    id: str
    use_case: str
    tier: str
    aggregate_score: float
    aggregate_passed: bool
    ingestion: AxisScore
    retrieval: AxisScore
    synthesis: AxisScore
    latency_ms: dict[str, float] = field(default_factory=dict)  # ingest_total, query, judge
    pot_id: str | None = None
    error: str | None = None  # Set when the scenario could not even start.

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkReport:
    schema_version: str
    started_at: str
    finished_at: str
    engine_url: str
    tier: str
    use_case_filter: str | None
    scenario_count: int
    aggregate_score: float
    pass_rate: float  # fraction of scenarios that passed
    by_use_case: dict[str, dict[str, float]] = field(default_factory=dict)
    scenarios: list[ScenarioResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "engine_url": self.engine_url,
            "tier": self.tier,
            "use_case_filter": self.use_case_filter,
            "scenario_count": self.scenario_count,
            "aggregate_score": self.aggregate_score,
            "pass_rate": self.pass_rate,
            "by_use_case": dict(self.by_use_case),
            "scenarios": [s.to_dict() for s in self.scenarios],
        }


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
