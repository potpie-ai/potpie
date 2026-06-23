"""Per-scenario orchestration: create pot → seed → ingest+noise → snapshot → query → judge → drop.

This is the only place the three axes (and their coverage/precision
sub-axes) are composed into a single ``ScenarioResult``. Keeping the
per-scenario lifecycle in one function makes it obvious what the bench
is doing and where it can fail.

Bench-plan v3 additions in this rewrite:

- **Universe seeding.** Scenarios declaring ``universe: acme`` get the
  Acme seed bundle injected at ``-365d`` ahead of signal/distractor
  events. Seed events go through the real ``/events/reconcile`` path —
  no shortcut into the graph.
- **Distractor injection.** ``distractor_events`` expand via
  ``replay.assemble_timeline`` and arrive interleaved with signals
  according to their declared ``at:`` (or ``at`` range).
- **Per-dimension scoring.** Composite scenarios produce a
  ``by_dimension`` decomposition derived from judge criterion
  attribution; non-composite scenarios attribute their full score to
  their single dimension.
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from potpie.context_engine.adapters.outbound.http.potpie_context_api_client import PotpieContextApiClient

from potpie.context_engine.benchmarks.core.graph_inspect import snapshot_graph
from potpie.context_engine.benchmarks.core.ingestion import IngestionOutcome, replay_all
from potpie.context_engine.benchmarks.core.lifecycle import EphemeralPot, create_ephemeral_pot, reset_pot
from potpie.context_engine.benchmarks.core.query import resolve_context
from potpie.context_engine.benchmarks.core.replay import assemble_timeline
from potpie.context_engine.benchmarks.core.result import AxisScore, DimensionScore, ScenarioResult
from potpie.context_engine.benchmarks.core.scenario import Scenario
from potpie.context_engine.benchmarks.core.universe import resolve_seeds_for_scenario
from potpie.context_engine.benchmarks.evaluators.ingestion_quality import evaluate_ingestion_quality
from potpie.context_engine.benchmarks.evaluators.llm_judge import evaluate_synthesis, synthesis_by_dimension
from potpie.context_engine.benchmarks.evaluators.llm_judge_invariant import evaluate_synthesis_invariant
from potpie.context_engine.benchmarks.evaluators.retrieval import evaluate_retrieval

logger = logging.getLogger(__name__)


def _aggregate(
    scenario: Scenario,
    ingestion_score: float,
    retrieval_score: float,
    synthesis_score: float,
) -> float:
    w = scenario.axis_weights
    return (
        ingestion_score * w.ingestion
        + retrieval_score * w.retrieval
        + synthesis_score * w.synthesis
    )


def _aggregate_invariant(synthesis_score: float) -> float:
    """In invariant mode the headline is the invariant judge, period.

    Ingestion + retrieval are still computed and recorded as diagnostics
    (see ``run_scenario``), but they don't enter the aggregate — by
    design, since the goal of invariant mode is to score the agent's
    answer against the input events without leaning on engine-internal
    shape.
    """
    return synthesis_score


def _empty_axis(reason: str) -> AxisScore:
    return AxisScore(score=0.0, passed=False, errors=[reason])


def _build_by_dimension(
    scenario: Scenario,
    *,
    ingestion: AxisScore,
    retrieval: AxisScore,
    synthesis: AxisScore,
    synthesis_details: dict[str, Any] | None,
    invariant: bool = False,
) -> list[DimensionScore]:
    """Build per-dimension scores for the result.

    For PREF/INFRA/TIME/BUG (non-composite), one entry is returned with
    the dimension equal to the scenario's use_case.

    For COMBO, each declared dimension gets its own entry: ingestion and
    retrieval scores broadcast equally (the deterministic axes don't
    distinguish dimensions today), and synthesis is taken from the judge
    rubric's dimension attribution.
    """
    dims = scenario.effective_dimensions
    if not dims:
        return []

    synth_by_dim = synthesis_by_dimension(synthesis_details or {})
    default_synth = synth_by_dim.get("_default", synthesis.score)

    out: list[DimensionScore] = []
    weights_by_dim: dict[str, int] = {}
    for c in scenario.judge.criteria:
        for d in c.dimensions:
            weights_by_dim[d] = weights_by_dim.get(d, 0) + c.weight

    for dim in dims:
        synth_score = synth_by_dim.get(dim, default_synth)
        if invariant:
            # Invariant mode: ingestion/retrieval don't enter the aggregate;
            # the invariant judge returns a single answer-level score that
            # broadcasts across declared dimensions for COMBO scenarios.
            agg = _aggregate_invariant(synth_score)
        else:
            agg = _aggregate(scenario, ingestion.score, retrieval.score, synth_score)
        out.append(
            DimensionScore(
                dimension=dim,
                ingestion=ingestion.score,
                retrieval=retrieval.score,
                synthesis=synth_score,
                aggregate=agg,
                judge_weight=weights_by_dim.get(dim, 0),
            )
        )
    return out


def run_scenario(
    *,
    scenario: Scenario,
    client: PotpieContextApiClient,
    fixtures_root: Path,
    benchmarks_root: Path,
    judge_client: Any | None = None,
    ingest_timeout_s: float = 180.0,
    skip_judge: bool = False,
    invariant: bool = False,
) -> ScenarioResult:
    """Run a single scenario end-to-end against a real engine.

    Always returns a ``ScenarioResult``. Internal failures are captured
    in axis errors / scenario.error rather than raised, so a single bad
    scenario doesn't sink the whole run.
    """
    logger.info(
        "scenario %s: starting (diff=%s mix=%s)",
        scenario.id,
        scenario.difficulty,
        scenario.source_mix,
    )
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
            difficulty=scenario.difficulty,
            source_mix=scenario.source_mix,
            dimensions=list(scenario.effective_dimensions),
        )

    assert pot is not None
    try:
        anchor = datetime.now(timezone.utc)

        # Universe seeds + scenario signals + distractors → single ordered timeline.
        merged_seeds = resolve_seeds_for_scenario(
            benchmarks_root, scenario.universe, scenario.seed
        )
        events = assemble_timeline(
            seed_steps=merged_seeds,
            ingest_steps=scenario.ingest,
            distractor_steps=scenario.distractor_events,
            fixtures_root=fixtures_root,
            anchor=anchor,
        )
        roles = Counter(e.role for e in events)
        logger.info(
            "scenario %s: ingesting %d events (seed=%d signal=%d distractor=%d) into pot %s",
            scenario.id,
            len(events),
            roles.get("seed", 0),
            roles.get("signal", 0),
            roles.get("distractor", 0),
            pot.pot_id,
        )

        ingest_started = time.monotonic()
        outcomes: list[IngestionOutcome] = replay_all(
            client,
            pot.pot_id,
            events,
            timeout_s=ingest_timeout_s,
            fallback_repo_name=pot.repo_name,
        )
        latency["ingest_total_ms"] = (time.monotonic() - ingest_started) * 1000
        latency["events_replayed"] = float(len(events))
        _ingest_failed = sum(1 for o in outcomes if o.error is not None)
        logger.info(
            "scenario %s: ingested %d/%d ok, %d failed in %.1fs",
            scenario.id,
            len(outcomes) - _ingest_failed,
            len(outcomes),
            _ingest_failed,
            latency["ingest_total_ms"] / 1000,
        )

        snapshot = snapshot_graph(client, pot.pot_id)
        logger.info(
            "scenario %s: graph snapshot — %d entities, %d edges",
            scenario.id,
            len(snapshot.entities),
            len(snapshot.edges),
        )
        ingestion_eval = evaluate_ingestion_quality(
            snapshot=snapshot,
            outcomes=outcomes,
            assertions=scenario.post_ingest_assertions,
        )
        ingestion_axis = AxisScore(
            score=ingestion_eval.score,
            passed=ingestion_eval.passed,
            coverage=ingestion_eval.coverage,
            precision=ingestion_eval.precision,
            details=dict(ingestion_eval.details),
            errors=ingestion_eval.errors,
        )

        query_started = time.monotonic()
        logger.info(
            "scenario %s: resolving context (intent=%s, include=%s)",
            scenario.id,
            scenario.query.intent,
            ",".join(scenario.query.include) or "-",
        )
        try:
            response = resolve_context(client, pot.pot_id, scenario.query)
            latency["query_ms"] = (time.monotonic() - query_started) * 1000
            retrieval_eval = evaluate_retrieval(
                response, scenario.retrieval_assertions, anchor=anchor
            )
            retrieval_axis = AxisScore(
                score=retrieval_eval.score,
                passed=retrieval_eval.passed,
                coverage=retrieval_eval.coverage,
                precision=retrieval_eval.precision,
                details=dict(retrieval_eval.details),
                errors=retrieval_eval.errors,
            )
            _inc = retrieval_eval.details.get("includes_used") or []
            _inc_str = (
                ",".join(str(x) for x in _inc)
                if isinstance(_inc, (list, tuple))
                else "-"
            )
            logger.info(
                "scenario %s: retrieval %.0f (%d source_refs, includes=%s) in %.1fs",
                scenario.id,
                retrieval_axis.score,
                retrieval_eval.details.get("source_ref_count", 0),
                _inc_str or "-",
                latency["query_ms"] / 1000,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("scenario %s: query failed", scenario.id)
            response = {}
            retrieval_axis = _empty_axis(f"query failed: {exc}")

        synthesis_details: dict[str, Any] = {}
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
                if invariant:
                    # Schema-independent judge: grades the answer against
                    # the scenario's signal events only — the engine's
                    # graph shape, include vocabulary, and query params
                    # do not enter the score.
                    signal_events = [e for e in events if e.role == "signal"]
                    seed_count = sum(1 for e in events if e.role == "seed")
                    distractor_count = sum(1 for e in events if e.role == "distractor")
                    logger.info(
                        "scenario %s: invariant judge on %d signal event(s) "
                        "(+ %d seed, %d distractor in context)...",
                        scenario.id,
                        len(signal_events),
                        seed_count,
                        distractor_count,
                    )
                    synthesis_eval = evaluate_synthesis_invariant(
                        description=scenario.description,
                        query=scenario.query,
                        response=response,
                        signal_events=signal_events,
                        seed_count=seed_count,
                        distractor_count=distractor_count,
                        client=judge_client,
                    )
                else:
                    logger.info(
                        "scenario %s: judging synthesis (%d criteria)...",
                        scenario.id,
                        len(scenario.judge.criteria),
                    )
                    synthesis_eval = evaluate_synthesis(
                        description=scenario.description,
                        query=scenario.query,
                        response=response,
                        rubric=scenario.judge,
                        client=judge_client,
                    )
                latency["judge_ms"] = (time.monotonic() - judge_started) * 1000
                synthesis_details = dict(synthesis_eval.details)
                synthesis_axis = AxisScore(
                    score=synthesis_eval.score,
                    passed=synthesis_eval.passed,
                    details=synthesis_details,
                    errors=synthesis_eval.errors,
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("scenario %s: judge failed", scenario.id)
                synthesis_axis = _empty_axis(f"judge failed: {exc}")
        else:
            synthesis_axis = _empty_axis("skipped (no query response)")

        if invariant:
            # Headline = invariant judge. Ingestion/retrieval are
            # diagnostic-only in this mode (kept in the report so we can
            # spot regressions, but they do not gate pass/fail).
            aggregate = _aggregate_invariant(synthesis_axis.score)
            passed = synthesis_axis.passed
        else:
            aggregate = _aggregate(
                scenario,
                ingestion_axis.score,
                retrieval_axis.score,
                synthesis_axis.score,
            )
            passed = (
                ingestion_axis.passed
                and retrieval_axis.passed
                and synthesis_axis.passed
            )
        latency["total_ms"] = (time.monotonic() - start_total) * 1000
        if invariant:
            sub = (
                synthesis_axis.details.get("scores")
                if isinstance(synthesis_axis.details, dict)
                else None
            )
            sub_str = (
                f" [F={sub.get('faithfulness')} C={sub.get('coverage')} "
                f"Cl={sub.get('clarity')} U={sub.get('usefulness')}]"
                if isinstance(sub, dict)
                else ""
            )
            logger.info(
                "scenario %s: %s invariant=%.1f%s (det ing=%.1f ret=%.1f) in %.1fs",
                scenario.id,
                "PASS" if passed else "FAIL",
                aggregate,
                sub_str,
                ingestion_axis.score,
                retrieval_axis.score,
                latency["total_ms"] / 1000,
            )
        else:
            logger.info(
                "scenario %s: %s agg=%.1f (ing=%.1f ret=%.1f syn=%.1f) in %.1fs",
                scenario.id,
                "PASS" if passed else "FAIL",
                aggregate,
                ingestion_axis.score,
                retrieval_axis.score,
                synthesis_axis.score,
                latency["total_ms"] / 1000,
            )

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
            difficulty=scenario.difficulty,
            source_mix=scenario.source_mix,
            dimensions=list(scenario.effective_dimensions),
            by_dimension=_build_by_dimension(
                scenario,
                ingestion=ingestion_axis,
                retrieval=retrieval_axis,
                synthesis=synthesis_axis,
                synthesis_details=synthesis_details,
                invariant=invariant,
            ),
        )
    finally:
        if pot is not None:
            reset_pot(client, pot)
