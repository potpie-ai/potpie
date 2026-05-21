"""Position B proper POC — orchestration.

Runs end-to-end:
  1. Connect to Neo4j, clean test pot, ensure indexes
  2. Ingest 25 fixture events (universe + signal + distractor) via LLM extraction
     + identity resolution → claims
  3. For each of 4 scenarios (one per PREF/INFRA/TIME/BUG), call the right reader,
     build an envelope, score against ground truth
  4. Print per-scenario scorecard + aggregate

Run:
    cd /Users/nandan/Desktop/Dev/potpie
    .venv/bin/python -m pocs.position_b_proper.main
"""

from __future__ import annotations

import asyncio
import os
import sys
import time

from neo4j import AsyncGraphDatabase
from openai import AsyncOpenAI

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from .engine import (
    AgentEnvelope,
    ScenarioScore,
    ingest_all_events,
    resolve_scenario,
    score_scenario,
)
from .fixtures import SCENARIOS, all_events
from .store import cleanup, ensure_indexes

POT = "pot:position-b-proper-poc"
OK = "\033[32m✓\033[0m"
FAIL = "\033[31m✗\033[0m"
WARN = "\033[33m▲\033[0m"


def hdr(t: str) -> None:
    print(f"\n{'━' * 76}\n  {t}\n{'━' * 76}")


def fmt_pct(x: float) -> str:
    return f"{x * 100:>5.1f}%"


def print_envelope(envelope: AgentEnvelope) -> None:
    print(f"    confidence: {envelope.confidence}")
    print(f"    coverage:   {envelope.coverage['status']}")
    print(f"    freshness:  {envelope.freshness['status']}")
    print(f"    answer:     {envelope.answer_summary[:200]}{'…' if len(envelope.answer_summary) > 200 else ''}")
    print(f"    source_refs ({len(envelope.source_refs)}):")
    for r in envelope.source_refs[:8]:
        print(f"      • [{r['strength']:>13s}] {r['event_id']}: {r['fact'][:90]}{'…' if len(r['fact']) > 90 else ''}")


def print_score(scenario_id: str, dim: str, sc: ScenarioScore) -> None:
    cov_marker = OK if sc.coverage >= 0.75 else WARN if sc.coverage >= 0.5 else FAIL
    prec_marker = OK if sc.precision == 1.0 else WARN if sc.precision >= 0.75 else FAIL
    phrase_marker = OK if sc.phrase_recall >= 0.66 else WARN if sc.phrase_recall >= 0.33 else FAIL
    hall_marker = OK if not sc.hallucination else FAIL
    conf_marker = OK if sc.confidence_match else WARN

    print(f"\n  [{dim}] {scenario_id}")
    print(f"    {cov_marker} coverage  {fmt_pct(sc.coverage)}  ({len(sc.surfaced_events)} expected events surfaced)")
    print(f"    {prec_marker} precision {fmt_pct(sc.precision)}  ({len(sc.surfaced_forbidden)} forbidden events leaked)")
    print(f"    {phrase_marker} phrases   {fmt_pct(sc.phrase_recall)}")
    print(f"    {hall_marker} hallucinate?  {'no' if not sc.hallucination else 'YES — forbidden phrases appeared'}")
    print(f"    {conf_marker} confidence    {sc.confidence_match} (expected match)")
    if sc.surfaced_forbidden:
        print(f"      forbidden leaks: {sorted(sc.surfaced_forbidden)}")


def aggregate_dim_scores(scores: list[ScenarioScore]) -> dict[str, dict[str, float]]:
    by_dim: dict[str, list[ScenarioScore]] = {}
    for sc in scores:
        by_dim.setdefault(sc.dimension, []).append(sc)
    out = {}
    for dim, xs in by_dim.items():
        out[dim] = {
            "coverage": sum(s.coverage for s in xs) / len(xs),
            "precision": sum(s.precision for s in xs) / len(xs),
            "phrase_recall": sum(s.phrase_recall for s in xs) / len(xs),
            "confidence_match": sum(1.0 if s.confidence_match else 0.0 for s in xs) / len(xs),
            "n": len(xs),
        }
    return out


async def main() -> None:
    uri = os.environ.get("NEO4J_URI")
    user = os.environ.get("NEO4J_USERNAME")
    password = os.environ.get("NEO4J_PASSWORD")
    if not (uri and user and password):
        print(f"{FAIL} missing NEO4J_URI/USERNAME/PASSWORD")
        sys.exit(1)
    if not os.environ.get("OPENAI_API_KEY"):
        print(f"{FAIL} missing OPENAI_API_KEY")
        sys.exit(1)

    driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
    client = AsyncOpenAI()

    try:
        hdr("Position B proper POC")
        print(f"  pot:    {POT}")
        print(f"  neo4j:  {uri}")

        hdr("Setup")
        await cleanup(driver, POT)
        await ensure_indexes(driver)
        # Vector index population is async — give it a moment to settle
        await asyncio.sleep(1.0)
        print(f"  {OK} cleared prior data + indexes ensured")

        hdr("Ingest (LLM extraction + identity resolution)")
        events = all_events()
        print(f"  {len(events)} events to ingest "
              f"({sum(1 for e in events if e.role == 'universe')} universe, "
              f"{sum(1 for e in events if e.role == 'signal')} signal, "
              f"{sum(1 for e in events if e.role == 'distractor')} distractor)")
        t0 = time.perf_counter()
        stats = await ingest_all_events(driver, POT, client, events)
        elapsed = time.perf_counter() - t0
        print(f"\n  {OK} ingested in {elapsed:.1f}s")
        print(f"  events processed:  {stats.events_processed}")
        print(f"  claims extracted:  {stats.claims_extracted}")
        print(f"  claims written:    {stats.claims_written}")
        print(f"  entities created:  {stats.entities_created}")
        print(f"  extractor errors:  {stats.extractor_errors}")

        # Identity convergence audit — how many distinct surface names converged
        # on the auth-svc canonical entity?
        async with driver.session() as s:
            r = await s.run(
                """
                MATCH (a:Alias {group_id: $pot})-[:ALIAS_OF]->(c:Entity)
                WHERE c.entity_key STARTS WITH 'service:' AND
                      (c.entity_key CONTAINS 'auth' OR c.name CONTAINS 'auth' OR c.name CONTAINS 'Auth')
                RETURN c.entity_key AS key, c.name AS name,
                       collect(DISTINCT a.surface_name) AS surfaces,
                       count(DISTINCT a.surface_name) AS n_surfaces
                ORDER BY n_surfaces DESC
                """,
                pot=POT,
            )
            auth_entities = [dict(rec) async for rec in r]
        print(f"\n  Auth-related canonical entities and their alias surfaces:")
        for ent in auth_entities:
            print(f"    {ent['key']} (name={ent['name']!r}) ← {ent['n_surfaces']} surfaces: {ent['surfaces']}")

        hdr("Resolve scenarios + score")
        scores: list[ScenarioScore] = []
        for scenario in SCENARIOS:
            envelope = await resolve_scenario(driver, POT, client, scenario)
            sc = score_scenario(scenario, envelope)
            scores.append(sc)
            print(f"\n  --- {scenario.scenario_id} ({scenario.dimension}) ---")
            print(f"  query: {scenario.query_text[:140]}")
            print_envelope(envelope)
            print_score(scenario.scenario_id, scenario.dimension, sc)

        hdr("Aggregate by dimension")
        per_dim = aggregate_dim_scores(scores)
        print(f"  {'Dim':<6} {'N':>3}  {'Coverage':>10}  {'Precision':>10}  {'Phrases':>10}  {'Conf-match':>11}")
        for dim, agg in per_dim.items():
            print(f"  {dim:<6} {int(agg['n']):>3}  "
                  f"{fmt_pct(agg['coverage']):>10}  "
                  f"{fmt_pct(agg['precision']):>10}  "
                  f"{fmt_pct(agg['phrase_recall']):>10}  "
                  f"{fmt_pct(agg['confidence_match']):>11}")

        # Overall headline
        total = len(scores)
        avg_cov = sum(s.coverage for s in scores) / total
        avg_prec = sum(s.precision for s in scores) / total
        any_hall = any(s.hallucination for s in scores)
        print(f"\n  Overall: coverage {fmt_pct(avg_cov)}, precision {fmt_pct(avg_prec)}, "
              f"hallucinations: {'NONE' if not any_hall else 'YES'}")

    finally:
        await driver.close()


if __name__ == "__main__":
    asyncio.run(main())
