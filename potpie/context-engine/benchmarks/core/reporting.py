"""Report writers: JSON for diffing across runs, Markdown for humans.

The bench plan asks for four panels in the human-readable report
(bench-plan §7):

- ``by_use_case``   — primary view; six numbers per use case
- ``by_dimension``  — composite scenarios decomposed by what they exercise
- ``by_difficulty`` — easy / medium / hard / adversarial curve
- ``by_source_mix`` — single / dual / full / adversarial curve

All four are computed from the same ``ScenarioResult`` list so a single
run produces every angle in one pass.
"""

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path
from typing import Any

from benchmarks.core.result import BenchmarkReport, ScenarioResult


def write_json(report: BenchmarkReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
        f.write("\n")


# ----------------------------------------------------------------------
# Baseline diff
# ----------------------------------------------------------------------


_DIFF_KEYS_USE_CASE = (
    "aggregate",
    "ingestion",
    "retrieval",
    "synthesis",
    "coverage",
    "precision",
)
_DIFF_KEYS_PANEL = ("aggregate", "pass_rate")
_DIFF_KEYS_DIM = ("aggregate", "ingestion", "retrieval", "synthesis")

# Per-axis "regression" thresholds — a delta worse than these counts as a
# regression worth surfacing. Generous on purpose: bench numbers are noisy
# and we want to flag drift, not seismograph noise.
_REGRESSION_DELTA = -2.0


def _diff_panel(
    current: dict[str, dict[str, float]],
    baseline: dict[str, dict[str, float]],
    keys: tuple[str, ...],
) -> list[dict[str, Any]]:
    """One row per bucket present in either side; missing-from-baseline is annotated."""
    rows: list[dict[str, Any]] = []
    for bucket in sorted(set(current) | set(baseline)):
        cur = current.get(bucket) or {}
        base = baseline.get(bucket) or {}
        deltas: dict[str, float | None] = {}
        for k in keys:
            if k in cur and k in base:
                deltas[k] = round(cur[k] - base[k], 1)
            else:
                deltas[k] = None  # New or removed bucket — no delta.
        rows.append(
            {
                "bucket": bucket,
                "current": {k: cur.get(k) for k in keys},
                "baseline": {k: base.get(k) for k in keys},
                "delta": deltas,
                "new": bucket not in baseline,
                "dropped": bucket not in current,
            }
        )
    return rows


def _diff_scenarios(
    current: list[ScenarioResult],
    baseline: list[ScenarioResult],
) -> list[dict[str, Any]]:
    by_cur = {s.id: s for s in current}
    by_base = {s.id: s for s in baseline}
    out: list[dict[str, Any]] = []
    for sid in sorted(set(by_cur) | set(by_base)):
        c = by_cur.get(sid)
        b = by_base.get(sid)
        row: dict[str, Any] = {"id": sid, "new": b is None, "dropped": c is None}
        if c is not None and b is not None:
            row["delta_aggregate"] = round(c.aggregate_score - b.aggregate_score, 1)
            row["delta_ingestion"] = round(c.ingestion.score - b.ingestion.score, 1)
            row["delta_retrieval"] = round(c.retrieval.score - b.retrieval.score, 1)
            row["delta_synthesis"] = round(c.synthesis.score - b.synthesis.score, 1)
            row["pass_change"] = (
                "regressed"
                if (b.aggregate_passed and not c.aggregate_passed)
                else "recovered"
                if (not b.aggregate_passed and c.aggregate_passed)
                else "stable"
            )
        out.append(row)
    return out


def diff_reports(current: BenchmarkReport, baseline: BenchmarkReport) -> dict[str, Any]:
    """Build a structured diff between two reports."""
    return {
        "current_started_at": current.started_at,
        "baseline_started_at": baseline.started_at,
        "headline": {
            "aggregate_delta": round(
                current.aggregate_score - baseline.aggregate_score, 1
            ),
            "pass_rate_delta": round((current.pass_rate - baseline.pass_rate) * 100, 1),
            "scenario_count_delta": current.scenario_count - baseline.scenario_count,
        },
        "by_use_case": _diff_panel(
            current.by_use_case, baseline.by_use_case, _DIFF_KEYS_USE_CASE
        ),
        "by_dimension": _diff_panel(
            current.by_dimension, baseline.by_dimension, _DIFF_KEYS_DIM
        ),
        "by_difficulty": _diff_panel(
            current.by_difficulty, baseline.by_difficulty, _DIFF_KEYS_PANEL
        ),
        "by_source_mix": _diff_panel(
            current.by_source_mix, baseline.by_source_mix, _DIFF_KEYS_PANEL
        ),
        "scenarios": _diff_scenarios(current.scenarios, baseline.scenarios),
    }


def _fmt_delta(v: float | None) -> str:
    if v is None:
        return "   —  "
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:5.1f}"


def _render_diff_panel(
    title: str,
    rows: list[dict[str, Any]],
    keys: tuple[str, ...],
) -> list[str]:
    if not rows:
        return []
    out = [f"## {title}", ""]
    header = (
        "| "
        + "Bucket".ljust(12)
        + " | "
        + " | ".join(f"Δ {k}" for k in keys)
        + " | flag |"
    )
    sep = "|" + "---|" * (len(keys) + 2)
    out.append(header)
    out.append(sep)
    for row in rows:
        flags = []
        if row["new"]:
            flags.append("new")
        if row["dropped"]:
            flags.append("dropped")
        for k in keys:
            d = row["delta"].get(k)
            if isinstance(d, (int, float)) and d <= _REGRESSION_DELTA:
                flags.append(f"regressed:{k}")
        delta_cells = " | ".join(_fmt_delta(row["delta"].get(k)) for k in keys)
        out.append(f"| `{row['bucket']}` | {delta_cells} | {','.join(flags) or '—'} |")
    out.append("")
    return out


def render_diff_markdown(diff: dict[str, Any]) -> str:
    """Human-readable diff. One section per panel + a scenario-level table."""
    lines = ["# Benchmark diff", ""]
    h = diff["headline"]
    lines.append(f"- Current: `{diff['current_started_at']}`")
    lines.append(f"- Baseline: `{diff['baseline_started_at']}`")
    lines.append(
        f"- Aggregate Δ: **{_fmt_delta(h['aggregate_delta']).strip()}**  "
        f"Pass-rate Δ: **{_fmt_delta(h['pass_rate_delta']).strip()}pp**  "
        f"Scenario count Δ: **{h['scenario_count_delta']:+d}**"
    )
    lines.append("")
    lines.extend(
        _render_diff_panel("By use case", diff["by_use_case"], _DIFF_KEYS_USE_CASE)
    )
    lines.extend(
        _render_diff_panel("By dimension", diff["by_dimension"], _DIFF_KEYS_DIM)
    )
    lines.extend(
        _render_diff_panel("By difficulty", diff["by_difficulty"], _DIFF_KEYS_PANEL)
    )
    lines.extend(
        _render_diff_panel("By source mix", diff["by_source_mix"], _DIFF_KEYS_PANEL)
    )

    rows = diff["scenarios"]
    if rows:
        lines.append("## Scenarios")
        lines.append("")
        lines.append("| Scenario | Δ agg | Δ ing | Δ ret | Δ syn | pass |")
        lines.append("|---|---:|---:|---:|---:|:--:|")
        for r in rows:
            if r.get("dropped"):
                lines.append(
                    f"| `{r['id']}` |   —   |   —   |   —   |   —   | dropped |"
                )
                continue
            if r.get("new"):
                lines.append(f"| `{r['id']}` |   —   |   —   |   —   |   —   | new |")
                continue
            lines.append(
                f"| `{r['id']}` "
                f"| {_fmt_delta(r.get('delta_aggregate'))} "
                f"| {_fmt_delta(r.get('delta_ingestion'))} "
                f"| {_fmt_delta(r.get('delta_retrieval'))} "
                f"| {_fmt_delta(r.get('delta_synthesis'))} "
                f"| {r.get('pass_change', '?')} |"
            )
        lines.append("")
    return "\n".join(lines)


def _format_pct(v: float | None) -> str:
    if v is None:
        return "  —"
    return f"{v:5.1f}"


def render_markdown(report: BenchmarkReport) -> str:
    out = StringIO()
    print(f"# Benchmark report — {report.tier} tier", file=out)
    print("", file=out)
    print(f"- Started: `{report.started_at}`", file=out)
    print(f"- Finished: `{report.finished_at}`", file=out)
    print(f"- Engine: `{report.engine_url}`", file=out)
    print(f"- Use-case filter: `{report.use_case_filter or 'all'}`", file=out)
    print(f"- Scenarios: **{report.scenario_count}**", file=out)
    print(f"- Aggregate score: **{report.aggregate_score:.1f}** / 100", file=out)
    print(f"- Pass rate: **{report.pass_rate * 100:.0f}%**", file=out)
    print("", file=out)

    if report.by_use_case:
        print("## By use case", file=out)
        print("", file=out)
        print(
            "| Use case | N | Aggregate | Ingestion | Retrieval | Synthesis | Coverage | Precision |",
            file=out,
        )
        print("|---|---:|---:|---:|---:|---:|---:|---:|", file=out)
        for uc, stats in sorted(report.by_use_case.items()):
            print(
                f"| {uc} | {int(stats.get('count', 0))} "
                f"| {stats.get('aggregate', 0):.1f} "
                f"| {stats.get('ingestion', 0):.1f} "
                f"| {stats.get('retrieval', 0):.1f} "
                f"| {stats.get('synthesis', 0):.1f} "
                f"| {_format_pct(stats.get('coverage'))} "
                f"| {_format_pct(stats.get('precision'))} |",
                file=out,
            )
        print("", file=out)

    if report.by_dimension:
        print("## By dimension", file=out)
        print("", file=out)
        print(
            "| Dimension | N (scenarios touching it) | Ingestion | Retrieval | Synthesis | Aggregate |",
            file=out,
        )
        print("|---|---:|---:|---:|---:|---:|", file=out)
        for dim, stats in sorted(report.by_dimension.items()):
            print(
                f"| {dim} | {int(stats.get('count', 0))} "
                f"| {stats.get('ingestion', 0):.1f} "
                f"| {stats.get('retrieval', 0):.1f} "
                f"| {stats.get('synthesis', 0):.1f} "
                f"| {stats.get('aggregate', 0):.1f} |",
                file=out,
            )
        print("", file=out)

    if report.by_difficulty:
        print("## By difficulty", file=out)
        print("", file=out)
        print("| Difficulty | N | Aggregate | Pass rate |", file=out)
        print("|---|---:|---:|---:|", file=out)
        # Preserve canonical order.
        order = {"easy": 0, "medium": 1, "hard": 2, "adversarial": 3}
        for diff in sorted(report.by_difficulty.keys(), key=lambda d: order.get(d, 99)):
            stats = report.by_difficulty[diff]
            print(
                f"| {diff} | {int(stats.get('count', 0))} "
                f"| {stats.get('aggregate', 0):.1f} "
                f"| {stats.get('pass_rate', 0) * 100:.0f}% |",
                file=out,
            )
        print("", file=out)

    if report.by_source_mix:
        print("## By source mix", file=out)
        print("", file=out)
        print("| Source mix | N | Aggregate | Pass rate |", file=out)
        print("|---|---:|---:|---:|", file=out)
        order_mix = {"single": 0, "dual": 1, "full": 2, "adversarial": 3}
        for mix in sorted(
            report.by_source_mix.keys(), key=lambda d: order_mix.get(d, 99)
        ):
            stats = report.by_source_mix[mix]
            print(
                f"| {mix} | {int(stats.get('count', 0))} "
                f"| {stats.get('aggregate', 0):.1f} "
                f"| {stats.get('pass_rate', 0) * 100:.0f}% |",
                file=out,
            )
        print("", file=out)

    print("## Scenarios", file=out)
    print("", file=out)
    print(
        "| Scenario | Use case | Diff | Mix | Aggregate | Ing | Ret | Syn | Cov | Prec | Pass |",
        file=out,
    )
    print("|---|---|---|---|---:|---:|---:|---:|---:|---:|:--:|", file=out)
    for s in report.scenarios:
        check = "✓" if s.aggregate_passed else "✗"
        # Coverage / precision: take the mean across the two primary axes that compute them.
        cov_parts = [
            v for v in (s.ingestion.coverage, s.retrieval.coverage) if v is not None
        ]
        prec_parts = [
            v for v in (s.ingestion.precision, s.retrieval.precision) if v is not None
        ]
        cov_mean = (sum(cov_parts) / len(cov_parts)) if cov_parts else None
        prec_mean = (sum(prec_parts) / len(prec_parts)) if prec_parts else None
        print(
            f"| `{s.id}` | {s.use_case} | {s.difficulty} | {s.source_mix} "
            f"| {s.aggregate_score:.1f} "
            f"| {s.ingestion.score:.1f} "
            f"| {s.retrieval.score:.1f} "
            f"| {s.synthesis.score:.1f} "
            f"| {_format_pct(cov_mean)} "
            f"| {_format_pct(prec_mean)} "
            f"| {check} |",
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


def _mean(numerator: float, count: float) -> float:
    return numerator / count if count > 0 else 0.0


def _by_use_case(results: list[ScenarioResult]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for r in results:
        bucket = out.setdefault(
            r.use_case,
            {
                "count": 0.0,
                "aggregate": 0.0,
                "ingestion": 0.0,
                "retrieval": 0.0,
                "synthesis": 0.0,
                "coverage": 0.0,
                "precision": 0.0,
                "coverage_n": 0.0,
                "precision_n": 0.0,
            },
        )
        bucket["count"] += 1
        bucket["aggregate"] += r.aggregate_score
        bucket["ingestion"] += r.ingestion.score
        bucket["retrieval"] += r.retrieval.score
        bucket["synthesis"] += r.synthesis.score
        for axis in (r.ingestion, r.retrieval):
            if axis.coverage is not None:
                bucket["coverage"] += axis.coverage
                bucket["coverage_n"] += 1
            if axis.precision is not None:
                bucket["precision"] += axis.precision
                bucket["precision_n"] += 1
    for stats in out.values():
        n = stats["count"]
        if n > 0:
            stats["aggregate"] /= n
            stats["ingestion"] /= n
            stats["retrieval"] /= n
            stats["synthesis"] /= n
        stats["coverage"] = _mean(stats["coverage"], stats.pop("coverage_n"))
        stats["precision"] = _mean(stats["precision"], stats.pop("precision_n"))
    return out


def _by_dimension(results: list[ScenarioResult]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for r in results:
        dims = r.dimensions or [r.use_case] if r.use_case != "COMBO" else r.dimensions
        if not dims:
            continue
        per_dim = {dscore.dimension: dscore for dscore in r.by_dimension}
        for d in dims:
            bucket = out.setdefault(
                d,
                {
                    "count": 0.0,
                    "ingestion": 0.0,
                    "retrieval": 0.0,
                    "synthesis": 0.0,
                    "aggregate": 0.0,
                },
            )
            bucket["count"] += 1
            ds = per_dim.get(d)
            if ds is not None:
                bucket["ingestion"] += ds.ingestion
                bucket["retrieval"] += ds.retrieval
                bucket["synthesis"] += ds.synthesis
                bucket["aggregate"] += ds.aggregate
            else:
                # Non-composite scenario: the whole scenario maps to its
                # single dimension.
                bucket["ingestion"] += r.ingestion.score
                bucket["retrieval"] += r.retrieval.score
                bucket["synthesis"] += r.synthesis.score
                bucket["aggregate"] += r.aggregate_score
    for stats in out.values():
        n = stats["count"]
        if n > 0:
            for k in ("ingestion", "retrieval", "synthesis", "aggregate"):
                stats[k] /= n
    return out


def _by_key(results: list[ScenarioResult], key: str) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for r in results:
        k = getattr(r, key) or "unknown"
        bucket = out.setdefault(k, {"count": 0.0, "aggregate": 0.0, "passed": 0.0})
        bucket["count"] += 1
        bucket["aggregate"] += r.aggregate_score
        bucket["passed"] += 1 if r.aggregate_passed else 0
    for stats in out.values():
        n = stats["count"]
        if n > 0:
            stats["aggregate"] /= n
            stats["pass_rate"] = stats.pop("passed") / n
    return out


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
        by_use_case=_by_use_case(results),
        by_dimension=_by_dimension(results),
        by_difficulty=_by_key(results, "difficulty"),
        by_source_mix=_by_key(results, "source_mix"),
        scenarios=results,
    )
