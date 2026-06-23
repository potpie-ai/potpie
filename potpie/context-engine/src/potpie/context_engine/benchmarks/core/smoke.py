"""In-process pipeline regression test — no engine required.

The bench's primary value comes from running against a real engine, but
that takes minutes and requires credentials. ``smoke`` exercises every
loadable scenario through the *bench-side* pipeline only:

1. Schema-validate every scenario (already covered by ``discover_scenarios``).
2. Resolve every fixture referenced (signal + seed + distractor) — catches
   broken references and malformed envelopes.
3. Assemble the full ingestion timeline through ``assemble_timeline``.
4. Run the ingestion evaluator against a synthetic empty graph snapshot.
5. Run the retrieval evaluator against a synthetic empty response.
6. Confirm the runner's per-dimension scoring + the reporting layer
   produce a well-formed ``BenchmarkReport``.

This is fast (sub-second) and runs without an engine, so it's the right
gate for every PR. If a refactor breaks the harness, smoke catches it
before CI tries the full bench.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from potpie.context_engine.benchmarks.core.graph_inspect import GraphSnapshot
from potpie.context_engine.benchmarks.core.ingestion import IngestionOutcome
from potpie.context_engine.benchmarks.core.replay import (
    FixtureNotFound,
    FixtureValidationError,
    assemble_timeline,
)
from potpie.context_engine.benchmarks.core.reporting import aggregate_report
from potpie.context_engine.benchmarks.core.result import AxisScore, DimensionScore, ScenarioResult, now_iso
from potpie.context_engine.benchmarks.core.scenario import Scenario, discover_scenarios
from potpie.context_engine.benchmarks.core.universe import resolve_seeds_for_scenario
from potpie.context_engine.benchmarks.evaluators.ingestion_quality import evaluate_ingestion_quality
from potpie.context_engine.benchmarks.evaluators.retrieval import evaluate_retrieval


@dataclass
class SmokeFinding:
    scenario_id: str
    ok: bool
    detail: str = ""


@dataclass
class SmokeReport:
    findings: list[SmokeFinding] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(f.ok for f in self.findings)


def _smoke_scenario(
    scenario: Scenario, benchmarks_root: Path, fixtures_root: Path
) -> SmokeFinding:
    anchor = datetime.now(timezone.utc)

    # Step 1: resolve every fixture (seeds + signals + distractors) without
    # hitting the engine. This is the single most expensive thing the bench
    # does on broken corpora — fail fast.
    try:
        seeds = resolve_seeds_for_scenario(
            benchmarks_root, scenario.universe, scenario.seed
        )
        events = assemble_timeline(
            seed_steps=seeds,
            ingest_steps=scenario.ingest,
            distractor_steps=scenario.distractor_events,
            fixtures_root=fixtures_root,
            anchor=anchor,
        )
    except (FixtureNotFound, FixtureValidationError) as exc:
        return SmokeFinding(
            scenario_id=scenario.id,
            ok=False,
            detail=f"fixture resolution failed: {exc}",
        )
    except FileNotFoundError as exc:
        return SmokeFinding(
            scenario_id=scenario.id, ok=False, detail=f"universe missing: {exc}"
        )
    except Exception as exc:  # noqa: BLE001
        return SmokeFinding(
            scenario_id=scenario.id,
            ok=False,
            detail=f"timeline assembly raised: {exc!r}",
        )

    # Step 2: fake an empty graph snapshot + outcomes and exercise the
    # ingestion evaluator. The evaluator should produce a low score with
    # error messages naming the missing assertions, but it must not raise.
    empty_outcomes = [
        IngestionOutcome(
            fixture_path=e.fixture_path,
            event_id="smoke",
            terminal_status="reconciled",
            duration_s=0.0,
        )
        for e in events
    ]
    try:
        ingestion_eval = evaluate_ingestion_quality(
            snapshot=GraphSnapshot(),
            outcomes=empty_outcomes,
            assertions=scenario.post_ingest_assertions,
        )
    except Exception as exc:  # noqa: BLE001
        return SmokeFinding(
            scenario_id=scenario.id,
            ok=False,
            detail=f"ingestion evaluator raised: {exc!r}",
        )

    # Step 3: fake an empty engine response and exercise the retrieval evaluator.
    try:
        retrieval_eval = evaluate_retrieval(
            response={"answer": {"summary": ""}, "source_refs": [], "facts": {}},
            assertions=scenario.retrieval_assertions,
            anchor=anchor,
        )
    except Exception as exc:  # noqa: BLE001
        return SmokeFinding(
            scenario_id=scenario.id,
            ok=False,
            detail=f"retrieval evaluator raised: {exc!r}",
        )

    detail_bits = [
        f"events={len(events)}",
        f"ingestion={ingestion_eval.score:.0f}",
        f"retrieval={retrieval_eval.score:.0f}",
    ]
    return SmokeFinding(scenario_id=scenario.id, ok=True, detail=" ".join(detail_bits))


def run_smoke(benchmarks_root: Path) -> SmokeReport:
    report = SmokeReport()
    scenarios = discover_scenarios(benchmarks_root)
    fixtures_root = benchmarks_root / "fixtures"
    if not scenarios:
        report.findings.append(
            SmokeFinding(
                scenario_id="<corpus>", ok=False, detail="no scenarios discovered"
            )
        )
        return report
    for s in scenarios:
        report.findings.append(_smoke_scenario(s, benchmarks_root, fixtures_root))

    # Final step: confirm the reporting layer can aggregate a synthetic
    # ScenarioResult from each smoke finding without raising. This catches
    # ``aggregate_report``/``render_markdown`` regressions early.
    try:
        synth = [
            ScenarioResult(
                id=f.scenario_id,
                use_case="PREF",
                tier="quick",
                aggregate_score=50.0,
                aggregate_passed=False,
                ingestion=AxisScore(
                    score=0.0, passed=False, coverage=0.0, precision=100.0
                ),
                retrieval=AxisScore(
                    score=0.0, passed=False, coverage=0.0, precision=100.0
                ),
                synthesis=AxisScore(score=0.0, passed=False),
                difficulty="easy",
                source_mix="single",
                dimensions=["PREF"],
                by_dimension=[
                    DimensionScore(
                        dimension="PREF",
                        ingestion=0,
                        retrieval=0,
                        synthesis=0,
                        aggregate=0,
                    )
                ],
            )
            for f in report.findings
            if f.ok
        ]
        aggregate_report(
            schema_version="smoke",
            started_at=now_iso(),
            finished_at=now_iso(),
            engine_url="(smoke)",
            tier="smoke",
            use_case_filter=None,
            results=synth,
        )
    except Exception as exc:  # noqa: BLE001
        report.findings.append(
            SmokeFinding(
                scenario_id="<reporting>",
                ok=False,
                detail=f"aggregate_report raised: {exc!r}",
            )
        )
    return report


def render_smoke(report: SmokeReport) -> str:
    lines = []
    width = max((len(f.scenario_id) for f in report.findings), default=0)
    for f in report.findings:
        mark = "OK" if f.ok else "FAIL"
        lines.append(f"  [{mark:<4}] {f.scenario_id.ljust(width)}  {f.detail}")
    lines.append("")
    lines.append("status: " + ("CLEAN" if report.passed else "BROKEN"))
    return "\n".join(lines)
