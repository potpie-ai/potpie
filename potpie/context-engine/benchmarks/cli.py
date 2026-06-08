"""Benchmark CLI.

    python -m benchmarks run [--use-case X] [--difficulty D] [--source-mix M]
                             [--dimension D] [--scenario ID]
                             [--tier T] [--baseline F] [--skip-judge]
    python -m benchmarks list [--grid] [--json]
    python -m benchmarks fixture validate
    python -m benchmarks fixture redact <path>
    python -m benchmarks report <report.json> [--format markdown|json]

``--difficulty`` / ``--source-mix`` / ``--dimension`` are filters added
in the v3 rewrite (bench-plan §6.10) so a regression on `hard`
scenarios can be drilled into without running the whole tier.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

from benchmarks.core.lifecycle import make_client
from benchmarks.core.probe import render_probe, run_probe
from benchmarks.core.replay import build_source_id_index, validate_all_fixtures
from benchmarks.core.reporting import diff_reports, render_diff_markdown
from benchmarks.core.smoke import render_smoke, run_smoke
from benchmarks.evaluators.retrieval import set_fixture_source_id_lookup
from benchmarks.core.reporting import aggregate_report, render_markdown, write_json
from benchmarks.core.result import now_iso
from benchmarks.core.scenario import (
    DIFFICULTIES,
    DIMENSIONS,
    SOURCE_MIXES,
    USE_CASES,
    Scenario,
    ScenarioLoadError,
    discover_scenarios,
)
from benchmarks.fixtures.redaction import redact_file
from benchmarks.runner import run_scenario

SCHEMA_VERSION = "2026-05-bench-v3"
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
    difficulty: str | None,
    source_mix: str | None,
    dimension: str | None,
    light_only: bool = False,
) -> list[Scenario]:
    out = scenarios
    if scenario_id:
        out = [s for s in out if s.id == scenario_id]
        if not out:
            print(f"error: no scenario with id '{scenario_id}'", file=sys.stderr)
            raise SystemExit(2)
        return out  # When picking a specific scenario, ignore other filters.
    if light_only:
        # ``run-light`` selects the curated 5-scenario subset by tag;
        # tier / use_case / dimension filters do not apply.
        return [s for s in out if s.light]
    if use_case:
        out = [s for s in out if s.use_case == use_case]
    if difficulty:
        out = [s for s in out if s.difficulty == difficulty]
    if source_mix:
        out = [s for s in out if s.source_mix == source_mix]
    if dimension:
        out = [s for s in out if dimension in s.effective_dimensions]
    out = [s for s in out if s.tier == tier]
    return out


def _cmd_list(args: argparse.Namespace) -> int:
    scenarios = _load_scenarios()
    if args.json:
        out = [
            {
                "id": s.id,
                "use_case": s.use_case,
                "difficulty": s.difficulty,
                "source_mix": s.source_mix,
                "tier": s.tier,
                "light": s.light,
                "dimensions": list(s.effective_dimensions),
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
    if args.grid:
        # use_case × difficulty matrix so coverage gaps are visible.
        rows = sorted(USE_CASES)
        cols = list(DIFFICULTIES)
        cells: dict[tuple[str, str], list[str]] = {}
        for s in scenarios:
            cells.setdefault((s.use_case, s.difficulty), []).append(s.id)
        col_w = max(12, max((len(c) for c in cols), default=0) + 2)
        first_w = max(8, max(len(r) for r in rows))
        header = " " * first_w + "  " + "".join(c.ljust(col_w) for c in cols)
        print(header)
        for r in rows:
            row_cells = [str(len(cells.get((r, c), []))) for c in cols]
            print(r.ljust(first_w) + "  " + "".join(v.ljust(col_w) for v in row_cells))
        return 0
    width = max(len(s.id) for s in scenarios)
    for s in scenarios:
        print(
            f"  {s.id.ljust(width)}  {s.use_case:<6}  {s.difficulty:<11}  "
            f"{s.source_mix:<11}  {s.tier:<8}  "
            f"{s.description.splitlines()[0] if s.description else ''}"
        )
    return 0


def _configure_logging(verbose: bool) -> None:
    """Show benchmark progress (``benchmarks.*`` at INFO) without third-party noise.

    The runner / ingestion / in-process engine emit phase breadcrumbs under the
    ``benchmarks`` logger; those always show. ``--verbose`` additionally
    surfaces the chatty engine + HTTP loggers (httpx, openai, the in-process
    reconciliation agent, etc.).
    """
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("benchmarks").setLevel(logging.INFO)
    if not verbose:
        for noisy in (
            "httpx",
            "httpcore",
            "openai",
            "urllib3",
            "neo4j",
            "sqlalchemy",
            "app",
            "adapters",
            "application",
            "bootstrap",
            "LiteLLM",
            "litellm",
        ):
            logging.getLogger(noisy).setLevel(logging.WARNING)


def _run_scenario_worker(payload):  # type: ignore[no-untyped-def]
    """ProcessPool entrypoint for the in-process engine (module-level → picklable).

    Each scenario runs in its own process so the reconciliation agent's
    per-batch ``asyncio.run`` and the SQLAlchemy session are fully isolated —
    the safe way to parallelize the in-process path (threads share a process
    and hit the agent's 'Event loop is closed' race).
    """
    scenario, ingest_timeout, skip_judge, invariant = payload
    os.environ["POTPIE_BENCH_INPROCESS"] = "1"
    _configure_logging(False)
    set_fixture_source_id_lookup(build_source_id_index(FIXTURES_ROOT))
    return run_scenario(
        scenario=scenario,
        client=make_client(),
        fixtures_root=FIXTURES_ROOT,
        benchmarks_root=BENCHMARKS_ROOT,
        ingest_timeout_s=ingest_timeout,
        skip_judge=skip_judge,
        invariant=invariant,
    )


def _cmd_run(args: argparse.Namespace) -> int:
    _configure_logging(args.verbose)
    if getattr(args, "local", False):
        # In-process engine: no HTTP server on :8001, no Celery worker.
        os.environ["POTPIE_BENCH_INPROCESS"] = "1"
    inprocess = os.environ.get("POTPIE_BENCH_INPROCESS", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    light_only = bool(getattr(args, "light_only", False))
    scenarios = _filter(
        _load_scenarios(),
        use_case=args.use_case,
        scenario_id=args.scenario,
        tier=args.tier,
        difficulty=args.difficulty,
        source_mix=args.source_mix,
        dimension=args.dimension,
        light_only=light_only,
    )
    if not scenarios:
        if light_only:
            print(
                "no scenarios tagged `light: true` — tag one easy scenario per "
                "dimension and try again",
                file=sys.stderr,
            )
        else:
            print(
                f"no scenarios match (use_case={args.use_case!r} tier={args.tier!r} "
                f"difficulty={args.difficulty!r} source_mix={args.source_mix!r} "
                f"dimension={args.dimension!r})",
                file=sys.stderr,
            )
        return 2

    # Make structured citation matching available to the retrieval evaluator
    # before we run any scenarios — keeps must_cite_event_id from depending
    # on substring matches against the response haystack (bench-plan v3 §6).
    set_fixture_source_id_lookup(build_source_id_index(FIXTURES_ROOT))

    client = make_client()
    started = now_iso()
    n = len(scenarios)
    n_workers = min(args.concurrency, n) if args.concurrency > 1 and n > 1 else 1
    mode = (
        "in-process (no :8001 server / no worker)"
        if inprocess
        else f"http {client._base}"
    )
    invariant = bool(getattr(args, "invariant", False))
    judge = (
        "off (--skip-judge)"
        if args.skip_judge
        else ("invariant (schema-independent)" if invariant else "rubric")
    )
    label = "benchmark run-light" if light_only else "benchmark run"
    print(
        f"▶ {label} | {n} scenario(s) | mode={mode} | "
        f"concurrency={n_workers}"
        + (
            " (processes)"
            if (n_workers > 1 and inprocess)
            else (" (threads)" if n_workers > 1 else "")
        )
        + f" | tier={('light' if light_only else args.tier)} | judge={judge}"
    )
    run_started = time.monotonic()

    results: list = []
    done = 0
    passed = 0

    def _log_done(r) -> None:  # type: ignore[no-untyped-def]
        nonlocal done, passed
        done += 1
        if r.aggregate_passed:
            passed += 1
        check = "✓" if r.aggregate_passed else "✗"
        el = (r.latency_ms or {}).get("total_ms", 0.0) / 1000.0
        tail = f"  ERROR: {r.error}" if getattr(r, "error", None) else ""
        print(
            f"[{done:>3}/{n}] {check} {r.id:<46} "
            f"agg={r.aggregate_score:5.1f} ing={r.ingestion.score:5.1f} "
            f"ret={r.retrieval.score:5.1f} syn={r.synthesis.score:5.1f} "
            f"{el:5.1f}s  (passed {passed}/{done}){tail}"
        )

    def _run_one(scenario):  # type: ignore[no-untyped-def]
        return run_scenario(
            scenario=scenario,
            client=client,
            fixtures_root=FIXTURES_ROOT,
            benchmarks_root=BENCHMARKS_ROOT,
            ingest_timeout_s=args.ingest_timeout,
            skip_judge=args.skip_judge,
            invariant=invariant,
        )

    if n_workers > 1 and inprocess:
        # In-process: parallelize across PROCESSES so each scenario gets its
        # own asyncio loop / agent / DB session (thread-safe alternative would
        # hit the agent's 'Event loop is closed' race).
        from concurrent.futures import ProcessPoolExecutor, as_completed

        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(
                    _run_scenario_worker,
                    (s, args.ingest_timeout, args.skip_judge, invariant),
                ): s
                for s in scenarios
            }
            for fut in as_completed(futures):
                s = futures[fut]
                try:
                    result = fut.result()
                except Exception as exc:  # noqa: BLE001 — a crashed subprocess
                    result = _failed_scenario_result(s, f"subprocess crashed: {exc}")
                results.append(result)
                _log_done(result)
    elif n_workers > 1:
        # HTTP: reconciliation runs in the engine worker; bench threads just
        # submit + poll over HTTP (thread-safe). Pots are isolated.
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(
            max_workers=n_workers, thread_name_prefix="bench-scenario"
        ) as pool:
            futures = {pool.submit(_run_one, s): s for s in scenarios}
            for fut in as_completed(futures):
                result = fut.result()
                results.append(result)
                _log_done(result)
    else:
        for i, scenario in enumerate(scenarios, 1):
            print(
                f"[{i:>3}/{n}] ▶ {scenario.id}  ({scenario.use_case}/{scenario.difficulty}/{scenario.source_mix})"
            )
            result = _run_one(scenario)
            results.append(result)
            _log_done(result)

    # Preserve declared scenario order in the report so baseline diffs don't shuffle.
    order = {s.id: i for i, s in enumerate(scenarios)}
    results.sort(key=lambda r: order.get(r.id, 1_000_000))

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

    total_s = time.monotonic() - run_started
    failed = [r for r in results if not r.aggregate_passed]
    print()
    print(f"report: {out_path.relative_to(BENCHMARKS_ROOT)}")
    print(
        f"done: {len(results)} scenario(s) in {total_s:.1f}s | "
        f"{len(results) - len(failed)} passed, {len(failed)} failed | "
        f"aggregate {report.aggregate_score:.1f}  pass_rate {report.pass_rate * 100:.0f}%"
    )
    if failed:
        print(f"failed ({len(failed)}):")
        for r in failed:
            reason = (
                f" — {r.error}"
                if getattr(r, "error", None)
                else (
                    f" — ing={r.ingestion.score:.0f} ret={r.retrieval.score:.0f} syn={r.synthesis.score:.0f}"
                )
            )
            print(f"  ✗ {r.id}{reason}")
    slow = sorted(
        results, key=lambda r: (r.latency_ms or {}).get("total_ms", 0.0), reverse=True
    )[:3]
    if slow and (slow[0].latency_ms or {}).get("total_ms"):
        print(
            "slowest: "
            + ", ".join(
                f"{r.id} {(r.latency_ms or {}).get('total_ms', 0.0) / 1000:.0f}s"
                for r in slow
            )
        )
    return 1 if failed else 0


def _failed_scenario_result(scenario, error: str):  # type: ignore[no-untyped-def]
    """Build a zero-score ScenarioResult for a scenario whose worker crashed."""
    from benchmarks.core.result import AxisScore, ScenarioResult

    def _z(reason: str) -> AxisScore:
        return AxisScore(score=0.0, passed=False, errors=[reason])

    return ScenarioResult(
        id=scenario.id,
        use_case=scenario.use_case,
        tier=scenario.tier,
        aggregate_score=0.0,
        aggregate_passed=False,
        ingestion=_z(error),
        retrieval=_z("skipped (worker crashed)"),
        synthesis=_z("skipped (worker crashed)"),
        latency_ms={},
        pot_id=None,
        error=error,
        difficulty=scenario.difficulty,
        source_mix=scenario.source_mix,
        dimensions=list(scenario.effective_dimensions),
    )


def _cmd_smoke(args: argparse.Namespace) -> int:
    """Run the in-process pipeline smoke test (no engine required).

    Catches harness regressions (broken fixtures, evaluator crashes,
    reporting layer breakage) in <1 s without standing up the engine.
    Returns 1 if any scenario fails the pipeline.
    """
    report = run_smoke(BENCHMARKS_ROOT)
    if args.json:
        print(
            json.dumps(
                {
                    "passed": report.passed,
                    "findings": [f.__dict__ for f in report.findings],
                },
                indent=2,
            )
        )
    else:
        print(render_smoke(report))
    return 0 if report.passed else 1


def _cmd_probe(args: argparse.Namespace) -> int:
    """Pre-flight check before a long run.

    Catches: engine unreachable, auth wrong, connector kind missing,
    reconciliation queue not draining. Targets the failure modes that
    today cost 10+ minutes of bench wall-clock to expose.
    """
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    client = make_client()
    expected = tuple(k.strip() for k in args.connectors.split(",") if k.strip())
    report = run_probe(
        client, expected_connectors=expected, terminal_timeout_s=args.terminal_timeout
    )
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(render_probe(report))
    return 0 if report.passed else 1


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


def _load_report_from_path(path: Path):  # type: ignore[no-untyped-def]
    """Rehydrate a ``BenchmarkReport`` from the JSON shape written by ``run``."""
    from benchmarks.core.result import (
        AxisScore,
        BenchmarkReport,
        DimensionScore,
        ScenarioResult,
    )

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
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
                difficulty=s.get("difficulty", "easy"),
                source_mix=s.get("source_mix", "single"),
                dimensions=s.get("dimensions") or [],
                by_dimension=[
                    DimensionScore(**d) for d in (s.get("by_dimension") or [])
                ],
            )
        )
    return BenchmarkReport(
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
        by_dimension=data.get("by_dimension") or {},
        by_difficulty=data.get("by_difficulty") or {},
        by_source_mix=data.get("by_source_mix") or {},
        scenarios=scenarios,
    )


def _cmd_report(args: argparse.Namespace) -> int:
    report = _load_report_from_path(Path(args.report))
    if args.baseline:
        baseline = _load_report_from_path(Path(args.baseline))
        diff = diff_reports(report, baseline)
        if args.format == "json":
            print(json.dumps(diff, indent=2))
        else:
            print(render_diff_markdown(diff))
        # Exit code semantics: 1 if any panel cell regressed > threshold.
        any_regression = any(
            isinstance(d, (int, float)) and d <= -2.0
            for panel_name in (
                "by_use_case",
                "by_dimension",
                "by_difficulty",
                "by_source_mix",
            )
            for row in diff[panel_name]
            for d in row["delta"].values()
        ) or any(
            isinstance(r.get("delta_aggregate"), (int, float))
            and r["delta_aggregate"] <= -2.0
            for r in diff["scenarios"]
        )
        return 1 if any_regression else 0

    if args.format == "json":
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(render_markdown(report))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        "benchmarks", description="Context engine benchmarks"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="Run scenarios against a real engine")
    p_run.add_argument("--use-case", choices=sorted(USE_CASES))
    p_run.add_argument("--difficulty", choices=list(DIFFICULTIES))
    p_run.add_argument("--source-mix", choices=list(SOURCE_MIXES))
    p_run.add_argument("--dimension", choices=sorted(DIMENSIONS))
    p_run.add_argument(
        "--scenario",
        help="Run a single scenario by id (overrides --tier and --use-case)",
    )
    p_run.add_argument("--tier", default="quick", choices=("quick", "extended"))
    p_run.add_argument(
        "--ingest-timeout",
        type=float,
        default=180.0,
        help="Per-event reconciliation timeout (s)",
    )
    p_run.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Run N scenarios in parallel (pots are isolated; safe up to engine capacity).",
    )
    p_run.add_argument(
        "--skip-judge",
        action="store_true",
        help="Skip the LLM-judge axis (cheap dry runs)",
    )
    # Invariant (schema-independent) grading is the default: the read path
    # returns a pure evidence envelope (no server-side answer.summary), so the
    # judge grades the agent's answer against the scenario's input events alone
    # (faithfulness/coverage/clarity/usefulness). Ingestion and retrieval axes
    # stay in the report as diagnostics but do not gate.
    p_run.add_argument(
        "--rubric",
        dest="invariant",
        action="store_false",
        help="Opt out of invariant grading and use the legacy per-rubric "
        "synthesis judge (deprecated; requires a server-synthesised answer).",
    )
    p_run.add_argument(
        "--local",
        action="store_true",
        help="Run the engine in-process (no HTTP server on :8001, no Celery worker; "
        "reuses Postgres/Neo4j and reconciles inline). Equivalent to POTPIE_BENCH_INPROCESS=1.",
    )
    p_run.add_argument("--verbose", "-v", action="store_true")
    p_run.set_defaults(func=_cmd_run, light_only=False, invariant=True)

    # --- run-light: curated 5-scenario subset, 5-way parallel, invariant judge.
    p_light = sub.add_parser(
        "run-light",
        help="Quick 5-scenario invariant run (one per dimension; concurrency 5).",
    )
    p_light.add_argument("--ingest-timeout", type=float, default=180.0)
    p_light.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Default 5 — one process per scenario.",
    )
    p_light.add_argument("--skip-judge", action="store_true")
    p_light.add_argument(
        "--local",
        action="store_true",
        help="Run the engine in-process (recommended for light runs; no :8001 / no worker).",
    )
    p_light.add_argument("--verbose", "-v", action="store_true")
    # Defaults that match the regular `run` namespace so _cmd_run can share code.
    p_light.set_defaults(
        func=_cmd_run,
        tier="quick",
        use_case=None,
        scenario=None,
        difficulty=None,
        source_mix=None,
        dimension=None,
        invariant=True,
        light_only=True,
    )

    p_list = sub.add_parser("list", help="List discovered scenarios")
    p_list.add_argument("--json", action="store_true")
    p_list.add_argument(
        "--grid",
        action="store_true",
        help="Show a use_case × difficulty coverage matrix instead of a flat list.",
    )
    p_list.set_defaults(func=_cmd_list)

    p_smoke = sub.add_parser(
        "smoke",
        help="In-process pipeline regression test (no engine required)",
    )
    p_smoke.add_argument("--json", action="store_true")
    p_smoke.set_defaults(func=_cmd_smoke)

    p_probe = sub.add_parser(
        "probe",
        help="Pre-flight: connector inventory + per-kind drain check",
    )
    p_probe.add_argument(
        "--connectors",
        default="github,linear,repo_docs,slack,alerting,deploy,notion",
        help="Comma-separated connector kinds the run will need (default: all seven).",
    )
    p_probe.add_argument(
        "--terminal-timeout",
        type=float,
        default=15.0,
        help="Per-connector seconds to wait for an event to leave 'queued'.",
    )
    p_probe.add_argument("--json", action="store_true")
    p_probe.add_argument("--verbose", "-v", action="store_true")
    p_probe.set_defaults(func=_cmd_probe)

    p_fix = sub.add_parser("fixture", help="Fixture utilities")
    p_fix_sub = p_fix.add_subparsers(dest="fixture_cmd", required=True)
    p_fix_sub.add_parser("validate", help="Validate all fixture envelopes")
    p_fix_redact = p_fix_sub.add_parser(
        "redact", help="Redact a single fixture in place"
    )
    p_fix_redact.add_argument("path")
    p_fix.set_defaults(func=_cmd_fixture)

    p_report = sub.add_parser(
        "report", help="Render an existing report (with optional diff)"
    )
    p_report.add_argument("report", help="Path to a benchmark report JSON")
    p_report.add_argument(
        "--baseline",
        help="Path to a prior report JSON. With this set, the command renders a diff and exits non-zero on regressions.",
    )
    p_report.add_argument("--format", default="markdown", choices=("markdown", "json"))
    p_report.set_defaults(func=_cmd_report)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
