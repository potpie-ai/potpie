"""Report writers: JSON for diffing across runs, Markdown for humans."""

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

from benchmarks.core.result import BenchmarkReport, ScenarioResult


def write_json(report: BenchmarkReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
        f.write("\n")


def render_markdown(report: BenchmarkReport) -> str:
    out = StringIO()
    print(f"# Benchmark report — {report.tier} tier", file=out)
    print(f"", file=out)
    print(f"- Started: `{report.started_at}`", file=out)
    print(f"- Finished: `{report.finished_at}`", file=out)
    print(f"- Engine: `{report.engine_url}`", file=out)
    print(f"- Use-case filter: `{report.use_case_filter or 'all'}`", file=out)
    print(f"- Scenarios: **{report.scenario_count}**", file=out)
    print(f"- Aggregate score: **{report.aggregate_score:.1f}** / 100", file=out)
    print(f"- Pass rate: **{report.pass_rate * 100:.0f}%**", file=out)
    print(f"", file=out)

    if report.by_use_case:
        print("## By use case", file=out)
        print("", file=out)
        print("| Use case | Scenarios | Aggregate | Ingestion | Retrieval | Synthesis |", file=out)
        print("|---|---:|---:|---:|---:|---:|", file=out)
        for uc, stats in sorted(report.by_use_case.items()):
            print(
                f"| {uc} | {int(stats.get('count', 0))} "
                f"| {stats.get('aggregate', 0):.1f} "
                f"| {stats.get('ingestion', 0):.1f} "
                f"| {stats.get('retrieval', 0):.1f} "
                f"| {stats.get('synthesis', 0):.1f} |",
                file=out,
            )
        print("", file=out)

    print("## Scenarios", file=out)
    print("", file=out)
    print("| Scenario | Use case | Aggregate | Ingestion | Retrieval | Synthesis | Pass |", file=out)
    print("|---|---|---:|---:|---:|---:|:--:|", file=out)
    for s in report.scenarios:
        check = "✓" if s.aggregate_passed else "✗"
        print(
            f"| `{s.id}` | {s.use_case} "
            f"| {s.aggregate_score:.1f} "
            f"| {s.ingestion.score:.1f} "
            f"| {s.retrieval.score:.1f} "
            f"| {s.synthesis.score:.1f} | {check} |",
            file=out,
        )
    print("", file=out)

    failed = [s for s in report.scenarios if not s.aggregate_passed]
    if failed:
        print("## Failures", file=out)
        print("", file=out)
        for s in failed:
            print(f"### `{s.id}`", file=out)
            print("", file=out)
            for axis_name, axis in (
                ("ingestion", s.ingestion),
                ("retrieval", s.retrieval),
                ("synthesis", s.synthesis),
            ):
                if axis.errors:
                    print(f"- **{axis_name}**: {axis.score:.1f}", file=out)
                    for err in axis.errors:
                        print(f"  - {err}", file=out)
            if s.error:
                print(f"- **scenario error**: {s.error}", file=out)
            print("", file=out)
    return out.getvalue()


def aggregate_report(
    *,
    schema_version: str,
    started_at: str,
    finished_at: str,
    engine_url: str,
    tier: str,
    use_case_filter: str | None,
    results: list[ScenarioResult],
) -> BenchmarkReport:
    if not results:
        return BenchmarkReport(
            schema_version=schema_version,
            started_at=started_at,
            finished_at=finished_at,
            engine_url=engine_url,
            tier=tier,
            use_case_filter=use_case_filter,
            scenario_count=0,
            aggregate_score=0.0,
            pass_rate=0.0,
        )

    aggregate = sum(r.aggregate_score for r in results) / len(results)
    pass_rate = sum(1 for r in results if r.aggregate_passed) / len(results)

    by_use_case: dict[str, dict[str, float]] = {}
    for r in results:
        bucket = by_use_case.setdefault(
            r.use_case, {"count": 0.0, "aggregate": 0.0, "ingestion": 0.0, "retrieval": 0.0, "synthesis": 0.0}
        )
        bucket["count"] += 1
        bucket["aggregate"] += r.aggregate_score
        bucket["ingestion"] += r.ingestion.score
        bucket["retrieval"] += r.retrieval.score
        bucket["synthesis"] += r.synthesis.score
    for stats in by_use_case.values():
        n = stats["count"]
        if n > 0:
            stats["aggregate"] /= n
            stats["ingestion"] /= n
            stats["retrieval"] /= n
            stats["synthesis"] /= n

    return BenchmarkReport(
        schema_version=schema_version,
        started_at=started_at,
        finished_at=finished_at,
        engine_url=engine_url,
        tier=tier,
        use_case_filter=use_case_filter,
        scenario_count=len(results),
        aggregate_score=aggregate,
        pass_rate=pass_rate,
        by_use_case=by_use_case,
        scenarios=results,
    )
