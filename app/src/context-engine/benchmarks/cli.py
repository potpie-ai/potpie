"""Benchmark CLI.

    python -m benchmarks run [--use-case X] [--scenario ID] [--tier T] [--baseline F] [--skip-judge]
    python -m benchmarks list
    python -m benchmarks fixture validate
    python -m benchmarks fixture redact <path>
    python -m benchmarks report <report.json> [--format markdown|json]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from benchmarks.core.lifecycle import make_client
from benchmarks.core.replay import validate_all_fixtures
from benchmarks.core.reporting import aggregate_report, render_markdown, write_json
from benchmarks.core.result import now_iso
from benchmarks.core.scenario import Scenario, ScenarioLoadError, discover_scenarios
from benchmarks.fixtures.redaction import redact_file
from benchmarks.runner import run_scenario

SCHEMA_VERSION = "2026-05-bench-v2"
BENCHMARKS_ROOT = Path(__file__).parent
FIXTURES_ROOT = BENCHMARKS_ROOT / "fixtures"
REPORTS_ROOT = BENCHMARKS_ROOT / "reports"


def _load_scenarios() -> list[Scenario]:
    try:
        return discover_scenarios(BENCHMARKS_ROOT)
    except ScenarioLoadError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


def _filter(
    scenarios: list[Scenario],
    *,
    use_case: str | None,
    scenario_id: str | None,
    tier: str,
) -> list[Scenario]:
    out = scenarios
    if scenario_id:
        out = [s for s in out if s.id == scenario_id]
        if not out:
            print(f"error: no scenario with id '{scenario_id}'", file=sys.stderr)
            raise SystemExit(2)
        return out  # When picking a specific scenario, ignore tier.
    if use_case:
        out = [s for s in out if s.use_case == use_case]
    out = [s for s in out if s.tier == tier]
    return out


def _cmd_list(args: argparse.Namespace) -> int:
    scenarios = _load_scenarios()
    if args.json:
        out = [
            {
                "id": s.id,
                "use_case": s.use_case,
                "tier": s.tier,
                "description": s.description,
                "source_path": str(s.source_path.relative_to(BENCHMARKS_ROOT)),
            }
            for s in scenarios
        ]
        print(json.dumps(out, indent=2))
        return 0
    if not scenarios:
        print("(no scenarios found)")
        return 0
    width = max(len(s.id) for s in scenarios)
    for s in scenarios:
        print(f"  {s.id.ljust(width)}  {s.use_case:<11}  {s.tier:<8}  {s.description.splitlines()[0] if s.description else ''}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    scenarios = _filter(
        _load_scenarios(),
        use_case=args.use_case,
        scenario_id=args.scenario,
        tier=args.tier,
    )
    if not scenarios:
        print(
            f"no scenarios match (use_case={args.use_case!r} tier={args.tier!r})",
            file=sys.stderr,
        )
        return 2

    client = make_client()
    started = now_iso()
    results = []
    for scenario in scenarios:
        result = run_scenario(
            scenario=scenario,
            client=client,
            fixtures_root=FIXTURES_ROOT,
            ingest_timeout_s=args.ingest_timeout,
            skip_judge=args.skip_judge,
        )
        results.append(result)
        check = "✓" if result.aggregate_passed else "✗"
        print(
            f"  {check} {result.id:<48} agg={result.aggregate_score:5.1f}  "
            f"ing={result.ingestion.score:5.1f}  ret={result.retrieval.score:5.1f}  "
            f"syn={result.synthesis.score:5.1f}"
        )

    finished = now_iso()
    report = aggregate_report(
        schema_version=SCHEMA_VERSION,
        started_at=started,
        finished_at=finished,
        engine_url=client._base,
        tier=args.tier if not args.scenario else "n/a",
        use_case_filter=args.use_case,
        results=results,
    )

    REPORTS_ROOT.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_ROOT / f"{started.replace(':', '-')}.json"
    write_json(report, out_path)
    print()
    print(f"report: {out_path.relative_to(BENCHMARKS_ROOT)}")
    print(
        f"aggregate: {report.aggregate_score:.1f}  "
        f"pass_rate: {report.pass_rate * 100:.0f}%  "
        f"scenarios: {report.scenario_count}"
    )

    failed = [r for r in results if not r.aggregate_passed]
    return 1 if failed else 0


def _cmd_fixture(args: argparse.Namespace) -> int:
    if args.fixture_cmd == "validate":
        errors = validate_all_fixtures(FIXTURES_ROOT)
        if errors:
            for e in errors:
                print(f"  error: {e}", file=sys.stderr)
            return 1
        print("all fixtures OK")
        return 0
    if args.fixture_cmd == "redact":
        redact_file(Path(args.path))
        print(f"redacted: {args.path}")
        return 0
    print("unknown fixture subcommand", file=sys.stderr)
    return 2


def _cmd_report(args: argparse.Namespace) -> int:
    path = Path(args.report)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if args.format == "json":
        print(json.dumps(data, indent=2))
        return 0

    # Reconstruct enough of the BenchmarkReport for rendering. We don't
    # need full type fidelity — render_markdown only reads named fields.
    from benchmarks.core.result import AxisScore, BenchmarkReport, ScenarioResult

    scenarios = []
    for s in data.get("scenarios", []):
        scenarios.append(
            ScenarioResult(
                id=s["id"],
                use_case=s["use_case"],
                tier=s["tier"],
                aggregate_score=s["aggregate_score"],
                aggregate_passed=s["aggregate_passed"],
                ingestion=AxisScore(**s["ingestion"]),
                retrieval=AxisScore(**s["retrieval"]),
                synthesis=AxisScore(**s["synthesis"]),
                latency_ms=s.get("latency_ms") or {},
                pot_id=s.get("pot_id"),
                error=s.get("error"),
            )
        )
    report = BenchmarkReport(
        schema_version=data["schema_version"],
        started_at=data["started_at"],
        finished_at=data["finished_at"],
        engine_url=data["engine_url"],
        tier=data["tier"],
        use_case_filter=data.get("use_case_filter"),
        scenario_count=data["scenario_count"],
        aggregate_score=data["aggregate_score"],
        pass_rate=data["pass_rate"],
        by_use_case=data.get("by_use_case") or {},
        scenarios=scenarios,
    )
    print(render_markdown(report))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser("benchmarks", description="Context engine benchmarks")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="Run scenarios against a real engine")
    p_run.add_argument("--use-case", choices=("feature", "debugging", "review", "operations", "onboarding"))
    p_run.add_argument("--scenario", help="Run a single scenario by id (overrides --tier and --use-case)")
    p_run.add_argument("--tier", default="quick", choices=("quick", "extended"))
    p_run.add_argument("--ingest-timeout", type=float, default=180.0, help="Per-event reconciliation timeout (s)")
    p_run.add_argument("--skip-judge", action="store_true", help="Skip the LLM-judge axis (cheap dry runs)")
    p_run.add_argument("--verbose", "-v", action="store_true")
    p_run.set_defaults(func=_cmd_run)

    p_list = sub.add_parser("list", help="List discovered scenarios")
    p_list.add_argument("--json", action="store_true")
    p_list.set_defaults(func=_cmd_list)

    p_fix = sub.add_parser("fixture", help="Fixture utilities")
    p_fix_sub = p_fix.add_subparsers(dest="fixture_cmd", required=True)
    p_fix_sub.add_parser("validate", help="Validate all fixture envelopes")
    p_fix_redact = p_fix_sub.add_parser("redact", help="Redact a single fixture in place")
    p_fix_redact.add_argument("path")
    p_fix.set_defaults(func=_cmd_fixture)

    p_report = sub.add_parser("report", help="Render an existing report")
    p_report.add_argument("report", help="Path to a benchmark report JSON")
    p_report.add_argument("--format", default="markdown", choices=("markdown", "json"))
    p_report.set_defaults(func=_cmd_report)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
