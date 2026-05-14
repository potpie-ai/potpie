"""Per-scenario orchestration: create pot → ingest → snapshot → query → judge → drop.

This is the only place the three axes are composed into a single
``ScenarioResult``. Keeping the per-scenario lifecycle in one function
makes it obvious what the bench is doing and where it can fail.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from adapters.outbound.http.potpie_context_api_client import PotpieContextApiClient

from benchmarks.core.graph_inspect import snapshot_graph
from benchmarks.core.ingestion import IngestionOutcome, replay_all
from benchmarks.core.lifecycle import EphemeralPot, create_ephemeral_pot, reset_pot
from benchmarks.core.query import resolve_context
from benchmarks.core.replay import to_replay_events
from benchmarks.core.result import AxisScore, ScenarioResult
from benchmarks.core.scenario import Scenario
from benchmarks.evaluators.ingestion_quality import evaluate_ingestion_quality
from benchmarks.evaluators.llm_judge import evaluate_synthesis
from benchmarks.evaluators.retrieval import evaluate_retrieval

logger = logging.getLogger(__name__)


def _aggregate(scenario: Scenario, ingestion_score: float, retrieval_score: float, synthesis_score: float) -> float:
    w = scenario.axis_weights
    return ingestion_score * w.ingestion + retrieval_score * w.retrieval + synthesis_score * w.synthesis


def _empty_axis(reason: str) -> AxisScore:
    return AxisScore(score=0.0, passed=False, errors=[reason])


def run_scenario(
    *,
    scenario: Scenario,
    client: PotpieContextApiClient,
    fixtures_root: Path,
    judge_client: Any | None = None,
    ingest_timeout_s: float = 180.0,
    skip_judge: bool = False,
) -> ScenarioResult:
    """Run a single scenario end-to-end against a real engine.

    Always returns a ``ScenarioResult``. Internal failures are captured
    in axis errors / scenario.error rather than raised, so a single bad
    scenario doesn't sink the whole run.
    """
    logger.info("scenario %s: starting", scenario.id)
    start_total = time.monotonic()
    pot: EphemeralPot | None = None
    latency: dict[str, float] = {}

    try:
        pot = create_ephemeral_pot(client, scenario_id=scenario.id)
    except Exception as exc:  # noqa: BLE001
        logger.exception("scenario %s: pot creation failed", scenario.id)
        return ScenarioResult(
            id=scenario.id,
            use_case=scenario.use_case,
            tier=scenario.tier,
            aggregate_score=0.0,
            aggregate_passed=False,
            ingestion=_empty_axis(f"pot creation failed: {exc}"),
            retrieval=_empty_axis("skipped (no pot)"),
            synthesis=_empty_axis("skipped (no pot)"),
            latency_ms={},
            pot_id=None,
            error=f"pot_creation_failed: {exc}",
        )

    assert pot is not None  # narrows for type-checker after the try/except above.
    try:
        anchor = datetime.now(timezone.utc)
        events = to_replay_events(scenario.ingest, fixtures_root, anchor=anchor)

        ingest_started = time.monotonic()
        outcomes: list[IngestionOutcome] = replay_all(
            client, pot.pot_id, events,
            timeout_s=ingest_timeout_s,
            fallback_repo_name=pot.repo_name,
        )
        latency["ingest_total_ms"] = (time.monotonic() - ingest_started) * 1000

        snapshot = snapshot_graph(client, pot.pot_id)
        ingestion_eval = evaluate_ingestion_quality(
            snapshot=snapshot,
            outcomes=outcomes,
            assertions=scenario.post_ingest_assertions,
        )
        ingestion_axis = AxisScore(
            score=ingestion_eval.score,
            passed=ingestion_eval.passed,
            details=ingestion_eval.details,
            errors=ingestion_eval.errors,
        )

        query_started = time.monotonic()
        try:
            response = resolve_context(client, pot.pot_id, scenario.query)
            latency["query_ms"] = (time.monotonic() - query_started) * 1000
            retrieval_eval = evaluate_retrieval(response, scenario.retrieval_assertions)
            retrieval_axis = AxisScore(
                score=retrieval_eval.score,
                passed=retrieval_eval.passed,
                details=retrieval_eval.details,
                errors=retrieval_eval.errors,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("scenario %s: query failed", scenario.id)
            response = {}
            retrieval_axis = _empty_axis(f"query failed: {exc}")

        if skip_judge:
            synthesis_axis = AxisScore(
                score=0.0,
                passed=False,
                details={"skipped": True},
                errors=["judge skipped via --skip-judge"],
            )
        elif response:
            judge_started = time.monotonic()
            try:
                synthesis_eval = evaluate_synthesis(
                    description=scenario.description,
                    query=scenario.query,
                    response=response,
                    rubric=scenario.judge,
                    client=judge_client,
                )
                latency["judge_ms"] = (time.monotonic() - judge_started) * 1000
                synthesis_axis = AxisScore(
                    score=synthesis_eval.score,
                    passed=synthesis_eval.passed,
                    details=synthesis_eval.details,
                    errors=synthesis_eval.errors,
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("scenario %s: judge failed", scenario.id)
                synthesis_axis = _empty_axis(f"judge failed: {exc}")
        else:
            synthesis_axis = _empty_axis("skipped (no query response)")

        aggregate = _aggregate(scenario, ingestion_axis.score, retrieval_axis.score, synthesis_axis.score)
        passed = ingestion_axis.passed and retrieval_axis.passed and synthesis_axis.passed
        latency["total_ms"] = (time.monotonic() - start_total) * 1000

        return ScenarioResult(
            id=scenario.id,
            use_case=scenario.use_case,
            tier=scenario.tier,
            aggregate_score=aggregate,
            aggregate_passed=passed,
            ingestion=ingestion_axis,
            retrieval=retrieval_axis,
            synthesis=synthesis_axis,
            latency_ms=latency,
            pot_id=pot.pot_id,
        )
    finally:
        if pot is not None:
            reset_pot(client, pot)
