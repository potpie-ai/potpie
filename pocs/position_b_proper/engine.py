"""Envelope builder + ingestion orchestration + scoring.

This is the engine layer: it ingests the fixture events through the LLM
extractor + identity layer, then exposes a `resolve()` function that returns
the agent envelope, and a `score_scenario()` that grades the envelope
against the scenario's expectations.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from neo4j import AsyncDriver
from openai import AsyncOpenAI

from .extractor import ExtractedClaim, extract_claims
from .fixtures import (
    DISTRACTOR_EVENTS,
    FixtureEvent,
    SCENARIOS,
    SIGNAL_EVENTS,
    Scenario,
    UNIVERSE_SEED,
    all_events,
)
from .identity import embed_text, resolve_or_create
from .readers import (
    ReaderHit,
    bug_reader,
    infra_reader,
    pref_reader,
    time_reader,
)
from .store import (
    Claim,
    Episode,
    cleanup,
    ensure_indexes,
    now_utc,
    write_claim,
    write_episode,
)


# ---------------------------------------------------------------------------
# Ingestion: event → extractor → identity → claim
# ---------------------------------------------------------------------------


@dataclass
class IngestionStats:
    events_processed: int = 0
    claims_extracted: int = 0
    claims_written: int = 0
    entities_created: int = 0
    aliases_written: int = 0
    extractor_errors: int = 0
    elapsed_seconds: float = 0.0
    # Track which surface_name resolutions happened
    resolutions: list[dict[str, Any]] = field(default_factory=list)


def _resolve_event_time(event: FixtureEvent) -> datetime:
    return now_utc() + timedelta(days=event.occurred_at_offset_days)


async def ingest_one_event(
    driver: AsyncDriver,
    pot: str,
    client: AsyncOpenAI,
    event: FixtureEvent,
    stats: IngestionStats,
) -> None:
    # Write the episode first (audit log + body embedding for episode search)
    valid_at = _resolve_event_time(event)
    body_embedding = await embed_text(client, event.body)
    episode_uuid = await write_episode(
        driver, pot,
        Episode(
            name=f"{event.source}/{event.kind}/{event.event_id}",
            body=event.body,
            source_description=f"{event.source}:{event.kind}",
            reference_time=valid_at,
            event_id=event.event_id,
        ),
        body_embedding=body_embedding,
    )

    # Extract claims via LLM
    try:
        claims = await extract_claims(
            client,
            event_body=event.body,
            event_kind=event.kind,
            event_source=event.source,
        )
    except Exception as exc:
        print(f"      ⚠ extractor failed on {event.event_id}: {exc}")
        stats.extractor_errors += 1
        return

    stats.claims_extracted += len(claims)

    # For each claim, resolve subject + object identity, then write claim edge
    for c in claims:
        subj = await resolve_or_create(
            driver, pot, client,
            surface_name=c.subject_name, entity_type=c.subject_type,
            source_event_id=event.event_id, source_event_body=event.body,
            source_system=event.source,
        )
        obj = await resolve_or_create(
            driver, pot, client,
            surface_name=c.object_name, entity_type=c.object_type,
            source_event_id=event.event_id, source_event_body=event.body,
            source_system=event.source,
        )
        stats.resolutions.append({
            "event_id": event.event_id,
            "subject": (c.subject_name, c.subject_type, subj.entity_key, subj.matched_via),
            "object": (c.object_name, c.object_type, obj.entity_key, obj.matched_via),
        })
        if subj.matched_via == "new":
            stats.entities_created += 1
        if obj.matched_via == "new":
            stats.entities_created += 1

        # Compute fact embedding so semantic search works on it
        fact_emb = await embed_text(client, c.fact)
        claim = Claim(
            subject_key=subj.entity_key,
            predicate=c.predicate,
            object_key=obj.entity_key,
            source_event_id=event.event_id,
            source_system=event.source,
            evidence_strength=c.evidence_strength,
            fact=c.fact,
            valid_at=valid_at,
            environment=c.environment,
            episode_uuid=episode_uuid,
            confidence=None,
            fact_embedding=fact_emb,
        )
        await write_claim(driver, pot, claim)
        stats.claims_written += 1

    stats.events_processed += 1


async def ingest_all_events(
    driver: AsyncDriver, pot: str, client: AsyncOpenAI, events: list[FixtureEvent]
) -> IngestionStats:
    stats = IngestionStats()
    t0 = time.perf_counter()
    for i, event in enumerate(events, start=1):
        print(f"    [{i:>2}/{len(events)}] ingesting {event.event_id} "
              f"({event.source}:{event.kind}, role={event.role})")
        await ingest_one_event(driver, pot, client, event, stats)
    stats.elapsed_seconds = time.perf_counter() - t0
    return stats


# ---------------------------------------------------------------------------
# Resolve: scenario → reader fan-out → envelope
# ---------------------------------------------------------------------------


@dataclass
class AgentEnvelope:
    """A small but realistic envelope mirroring `context_resolve`'s shape."""

    ok: bool
    answer_summary: str
    source_refs: list[dict[str, Any]]  # event_id + facts referenced
    evidence: list[ReaderHit]
    confidence: str  # high | medium | low | unknown
    coverage: dict[str, Any]  # status + available + missing
    freshness: dict[str, Any]
    quality_drift: dict[str, Any]
    fallbacks: list[dict[str, str]]
    recommended_next_actions: list[dict[str, str]]


async def resolve_service_key(driver: AsyncDriver, pot: str, surface_name: str) -> str | None:
    """Look up a canonical Service key from a query-time surface_name via the alias table."""
    async with driver.session() as s:
        r = await s.run(
            """
            MATCH (a:Alias {group_id: $pot, surface_name_lower: $name})
                  -[:ALIAS_OF]->(c:Entity)
            WHERE 'Service' IN labels(c)
            RETURN c.entity_key AS key
            LIMIT 1
            """,
            pot=pot, name=surface_name.strip().lower(),
        )
        record = await r.single()
        return record["key"] if record else None


def _build_envelope(
    hits: list[ReaderHit],
    *,
    requested_dim: str,
) -> AgentEnvelope:
    """Build a context_resolve-shaped envelope from reader hits.

    Confidence derivation:
      - high   = ≥2 hits, dominant strength = deterministic|attested, scores cohere
      - medium = ≥1 hit at attested/deterministic
      - low    = only inferred/hypothesized hits or few hits
      - unknown = no hits

    This isn't the full D3 derivation (no decay yet here — would come from a
    central belief service), but it's grounded in real signals (strength,
    count, score), not coverage-bucket constants.
    """
    if not hits:
        return AgentEnvelope(
            ok=True,
            answer_summary="No relevant project memory found for this scope.",
            source_refs=[],
            evidence=[],
            confidence="unknown",
            coverage={"status": "empty", "available": [], "missing": [requested_dim.lower()]},
            freshness={"status": "unknown", "stale_refs": []},
            quality_drift={"status": "unknown", "signals": {}},
            fallbacks=[{
                "code": "no_matching_claims",
                "message": "No claims matched the query scope and dimension.",
            }],
            recommended_next_actions=[{"action": "broaden_scope",
                                       "reason": "Try a wider time window or higher-level scope."}],
        )

    strengths = [h.evidence_strength for h in hits]
    has_deterministic = "deterministic" in strengths
    has_attested = "attested" in strengths
    top_score = hits[0].score
    n_strong = sum(1 for s in strengths if s in {"deterministic", "attested"})

    if n_strong >= 2 and top_score >= 0.5:
        confidence = "high"
    elif has_deterministic or has_attested:
        confidence = "medium"
    elif hits:
        confidence = "low"
    else:
        confidence = "unknown"

    source_refs = [
        {"event_id": h.source_event_id, "fact": h.fact, "strength": h.evidence_strength}
        for h in hits
    ]
    # Stale refs heuristic: any claim older than 180 days
    now = datetime.now(timezone.utc)
    stale_refs: list[str] = []
    for h in hits:
        if h.valid_at is None:
            continue
        # Coerce Neo4j DateTime → Python datetime
        va = h.valid_at
        to_native = getattr(va, "to_native", None)
        if callable(to_native):
            va = to_native()
        if va.tzinfo is None:
            va = va.replace(tzinfo=timezone.utc)
        if (now - va) > timedelta(days=180):
            stale_refs.append(h.source_event_id)

    freshness_status = "stale" if stale_refs else "fresh"
    summary = _summarize_hits(hits, requested_dim)
    return AgentEnvelope(
        ok=True,
        answer_summary=summary,
        source_refs=source_refs,
        evidence=hits,
        confidence=confidence,
        coverage={"status": "complete", "available": [requested_dim.lower()], "missing": []},
        freshness={"status": freshness_status, "stale_refs": stale_refs},
        quality_drift={
            "status": "good" if not stale_refs else "watch",
            "signals": {"stale_refs": len(stale_refs), "hit_count": len(hits)},
        },
        fallbacks=[],
        recommended_next_actions=(
            [{"action": "verify_stale", "reason": f"{len(stale_refs)} stale ref(s) may need re-verification."}]
            if stale_refs else []
        ),
    )


def _summarize_hits(hits: list[ReaderHit], dim: str) -> str:
    """Short deterministic summary from the top hits' facts.
    Not LLM-synthesized; a production engine would optionally pass through a
    synthesizer, but this is honest output that doesn't pretend to reason."""
    top = hits[:5]
    bullets = "; ".join(h.fact for h in top)
    return f"Top {len(top)} {dim}-relevant facts: {bullets}"


async def resolve_scenario(
    driver: AsyncDriver, pot: str, client: AsyncOpenAI, scenario: Scenario
) -> AgentEnvelope:
    """Run the appropriate reader for the scenario's dimension and build envelope."""
    if scenario.dimension == "PREF":
        hits = await pref_reader(
            driver, pot, client,
            query_text=scenario.query_text,
            file_path=str(scenario.query_scope.get("file_path")) if scenario.query_scope.get("file_path") else None,
            language=str(scenario.query_scope.get("language")) if scenario.query_scope.get("language") else None,
        )
    elif scenario.dimension == "INFRA":
        service_name = str(scenario.query_scope.get("service") or "")
        service_key = await resolve_service_key(driver, pot, service_name)
        if not service_key:
            return _build_envelope([], requested_dim=scenario.dimension)
        hits = await infra_reader(
            driver, pot,
            service_key=service_key,
            environment=str(scenario.query_scope.get("environment")) if scenario.query_scope.get("environment") else None,
        )
    elif scenario.dimension == "TIME":
        service_name = str(scenario.query_scope.get("service") or "")
        service_key = await resolve_service_key(driver, pot, service_name)
        if not service_key:
            return _build_envelope([], requested_dim=scenario.dimension)
        hits = await time_reader(driver, pot, service_key=service_key, window_days=7.0)
    elif scenario.dimension == "BUG":
        hits = await bug_reader(driver, pot, client, symptom_text=scenario.query_text)
    else:
        hits = []

    return _build_envelope(hits, requested_dim=scenario.dimension)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


@dataclass
class ScenarioScore:
    scenario_id: str
    dimension: str
    coverage: float  # recall: fraction of expected events surfaced
    precision: float  # 1 - (fraction of forbidden events surfaced)
    phrase_recall: float
    hallucination: bool  # any must_not_mention phrase found?
    confidence_match: bool  # derived confidence matches expected
    surfaced_events: set[str] = field(default_factory=set)
    surfaced_forbidden: set[str] = field(default_factory=set)
    notes: list[str] = field(default_factory=list)


def score_scenario(scenario: Scenario, envelope: AgentEnvelope) -> ScenarioScore:
    expected = scenario.expected
    surfaced_event_ids = {ref["event_id"] for ref in envelope.source_refs}

    # Coverage = recall of must_surface
    expected_set = expected.must_surface_event_ids
    surfaced_expected = expected_set & surfaced_event_ids
    coverage = (len(surfaced_expected) / len(expected_set)) if expected_set else 1.0

    # Precision = how cleanly we excluded must_not_surface
    forbidden_surfaced = expected.must_not_surface_event_ids & surfaced_event_ids
    precision = 1.0 - (len(forbidden_surfaced) / max(1, len(expected.must_not_surface_event_ids)))

    # Phrase recall — coarse synthesis check
    summary_lower = envelope.answer_summary.lower()
    refs_text = " ".join(r["fact"].lower() for r in envelope.source_refs)
    combined = f"{summary_lower} {refs_text}"
    phrase_hits = sum(1 for p in expected.must_mention_phrases if p.lower() in combined)
    phrase_recall = phrase_hits / max(1, len(expected.must_mention_phrases))

    # Hallucination check
    hallucination = any(p.lower() in combined for p in expected.must_not_mention_phrases)

    confidence_match = envelope.confidence == expected.expected_confidence

    return ScenarioScore(
        scenario_id=scenario.scenario_id,
        dimension=scenario.dimension,
        coverage=coverage,
        precision=precision,
        phrase_recall=phrase_recall,
        hallucination=hallucination,
        confidence_match=confidence_match,
        surfaced_events=surfaced_expected,
        surfaced_forbidden=forbidden_surfaced,
    )
