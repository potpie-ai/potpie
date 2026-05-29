"""CLI for comprehensive context graph benchmarks."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from benchmarks.dataset import load_dataset
from benchmarks.models import DEFAULT_DATASET, DEFAULT_REPORT
from benchmarks.runner import run_benchmark
from benchmarks.runners import make_runner


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["api", "http-e2e", "mock"], help="Benchmark target.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--baseline", type=Path, default=None, help="Previous report for regression checks.")
    parser.add_argument("--pot-id", default=None)
    parser.add_argument("--repo-name", default=None)
    parser.add_argument("--iterations", "-i", type=int, default=3)
    parser.add_argument("--concurrency", "-c", type=int, default=4)
    parser.add_argument(
        "--scenario-concurrency",
        "-s",
        type=int,
        default=1,
        help="How many scenarios to run in parallel (default: 1).",
    )
    parser.add_argument("--no-seed", action="store_true", help="Skip fixture seeding.")
    parser.add_argument(
        "--ingest-pr-live",
        action="store_true",
        help="Call /ingest-pr for PR fixtures instead of seeding PR episodes. Use only when live source-control is configured.",
    )
    parser.add_argument("--tag-filter", default=None, help="Comma-separated tags; only run scenarios matching any tag.")
    parser.add_argument("--list-scenarios", action="store_true", help="List all scenarios and exit without running.")
    parser.add_argument("--print-json", action="store_true")
    args = parser.parse_args(argv)

    dataset = load_dataset(args.dataset)

    if args.list_scenarios:
        _list_scenarios(dataset)
        return 0

    scenarios = _filter_scenarios(dataset.scenarios, args.tag_filter)
    if not scenarios:
        print("ERROR: No scenarios match the given filter.", file=sys.stderr)
        return 1

    # Build a filtered dataset for this run
    filtered_dataset = type(dataset)(
        name=dataset.name,
        version=dataset.version,
        pot_id=dataset.pot_id,
        repo_name=dataset.repo_name,
        seed_episodes=dataset.seed_episodes,
        seed_records=dataset.seed_records,
        pr_bundles=dataset.pr_bundles,
        scenarios=scenarios,
        thresholds=dataset.thresholds,
    )

    baseline = _load_baseline(args.baseline)
    try:
        runner = make_runner(args.mode, filtered_dataset, pot_id=args.pot_id, repo_name=args.repo_name)
        report = asyncio.run(
            run_benchmark(
                runner,
                filtered_dataset,
                mode=args.mode,
                iterations=max(1, args.iterations),
                concurrency=max(1, args.concurrency),
                scenario_concurrency=max(1, args.scenario_concurrency),
                seed=not args.no_seed,
                ingest_pr_live=args.ingest_pr_live,
                baseline=baseline,
            )
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    payload = report.to_dict()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.print_json:
        print(json.dumps(payload, indent=2))
    else:
        _print_report(payload)
        print(f"\nreport written to {args.report}")
    return 0 if payload["summary"]["ok"] else 1


def _load_baseline(path: Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _list_scenarios(dataset: Any) -> None:
    print(f"Scenarios in {dataset.name}@{dataset.version}")
    print(f"{'ID':<35} {'Feature':<20} {'Intent':<12} {'Tags'}")
    print("-" * 90)
    for s in dataset.scenarios:
        tags = ", ".join(s.get("tags", []))
        print(f"{s['id']:<35} {s.get('feature', ''):<20} {str(s.get('intent')):<12} {tags}")


def _filter_scenarios(scenarios: list[dict[str, Any]], tag_filter: str | None) -> list[dict[str, Any]]:
    if not tag_filter:
        return list(scenarios)
    wanted = {t.strip() for t in tag_filter.split(",")}
    return [s for s in scenarios if wanted & set(s.get("tags", []))]


def _print_report(report: dict[str, Any]) -> None:
    target = report["target"]
    summary = report["summary"]
    print("\nContext Graph Benchmark")
    print(f"dataset={report['dataset']['name']}@{report['dataset']['version']} mode={target['mode']} pot={target['pot_id']}")
    print(f"score={summary['score']:.2%} grade={summary['grade']} ok={summary['ok']} errors={summary['error_count']}")
    print(f"iterations={target['iterations']} concurrency={target['concurrency']} scenario_concurrency={target.get('scenario_concurrency', 1)}")
    print("")
    print(f"{'Scenario':<35} {'Feature':<20} {'Grade':<10} {'Score':>8} {'p95 ms':>10} {'Err':>5}")
    print("-" * 92)
    for item in report["scenarios"]:
        latency = item.get("latency_ms") or {}
        ratio = item["score"] / item["max_score"] if item["max_score"] else 0.0
        print(
            f"{item['id']:<35} {item['feature']:<20} {item['grade']:<10} "
            f"{ratio:>7.1%} {float(latency.get('p95') or 0):>10.2f} {item['errors']:>5}"
        )
    if report.get("regressions"):
        print("\nRegressions:")
        for reg in report["regressions"]:
            print(f"- {reg['id']}: {reg['kind']} previous={reg['previous']} current={reg['current']}")


if __name__ == "__main__":
    raise SystemExit(main())
