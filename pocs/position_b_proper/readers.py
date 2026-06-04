"""Per-dimension readers — minimal implementations sufficient for the POC.

Each reader takes a scenario query and returns a ranked list of candidate
claims + the source events they were derived from. The envelope builder then
combines reader outputs into the agent response shape.

These are NOT production readers — they exercise the substrate enough to score
each UC. A production `InfraTopologyReader` would do bounded blast-radius,
diff-between-envs, etc. The POC version proves the substrate supports those
queries cheaply.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from openai import AsyncOpenAI

from .store import (
    AsyncDriver,
    claims_for_subject,
    semantic_search_claims,
    semantic_search_episodes,
)
from .identity import embed_text


@dataclass
class ReaderHit:
    """One claim or episode surfaced to the agent, with derivation metadata."""

    source_event_id: str
    fact: str
    predicate: str
    subject: str
    object: str
    score: float
    evidence_strength: str
    valid_at: datetime | None
    source_system: str
    reader: str  # which reader produced this hit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_STRENGTH_RANK = {
    "deterministic": 4.0,
    "attested": 3.0,
    "inferred": 2.0,
    "hypothesized": 1.0,
}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _to_py_dt(value: Any) -> datetime | None:
    """Neo4j returns neo4j.time.DateTime; coerce to a tz-aware Python datetime."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    # Neo4j DateTime
    to_native = getattr(value, "to_native", None)
    if callable(to_native):
        dt = to_native()
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    return None


def _recency_weight(valid_at: Any, half_life_days: float = 90.0) -> float:
    """Linear-decay-ish weight: 1.0 at now, 0.0 by ~5*half_life past."""
    dt = _to_py_dt(valid_at)
    if dt is None:
        return 0.4
    age_days = (_now() - dt).total_seconds() / 86400.0
    if age_days < 0:
        return 1.0
    return max(0.0, 1.0 - (age_days / (5 * half_life_days)))


# ---------------------------------------------------------------------------
# PREF reader
# ---------------------------------------------------------------------------


async def pref_reader(
    driver: AsyncDriver,
    pot: str,
    client: AsyncOpenAI,
    *,
    query_text: str,
    file_path: str | None = None,
    language: str | None = None,
) -> list[ReaderHit]:
    """Surface preferences relevant to a coding task.

    Strategy:
      1. Semantic similarity over all GOVERNS / DESCRIBES claim facts.
      2. Boost by source authority (deterministic > attested > inferred).
      3. Filter out claims whose subject has been SUPERSEDED by a newer
         claim (so the old ADR-003 doesn't surface alongside ADR-007).
    """
    task_signature = (
        f"writing code, task: {query_text}. "
        f"file={file_path or 'unknown'}, language={language or 'python'}. "
        "Relevant preferences, conventions, style rules, library choices, "
        "error handling patterns, logging conventions, naming."
    )
    embedding = await embed_text(client, task_signature)
    raw_hits = await semantic_search_claims(
        driver, pot, embedding, top_k=15, min_score=0.40
    )

    # Filter to preference-shaped predicates and remove SUPERSEDED items
    superseded = await _superseded_subjects(driver, pot)
    out: list[ReaderHit] = []
    for h in raw_hits:
        if h["subject"] in superseded:
            continue
        if h["predicate"] not in {"GOVERNS", "DESCRIBES", "DOCUMENTS"}:
            continue
        strength = _STRENGTH_RANK.get(h["strength"], 2.0)
        recency = _recency_weight(h["valid_at"], half_life_days=180.0)
        ranked_score = h["score"] * (0.5 + 0.5 * strength / 4.0) * (0.6 + 0.4 * recency)
        out.append(ReaderHit(
            source_event_id=h["source_event_id"],
            fact=h["fact"],
            predicate=h["predicate"],
            subject=h["subject"],
            object=h["object"],
            score=ranked_score,
            evidence_strength=h["strength"],
            valid_at=h["valid_at"],
            source_system=h["source_system"],
            reader="pref",
        ))
    out.sort(key=lambda r: r.score, reverse=True)
    return out[:8]


async def _superseded_subjects(driver: AsyncDriver, pot: str) -> set[str]:
    """Return subjects of any claim that has been SUPERSEDES'd by another claim."""
    async with driver.session() as s:
        r = await s.run(
            """
            MATCH (a:Entity)-[r:RELATES_TO {group_id: $pot, name: 'SUPERSEDES'}]->(b:Entity)
            WHERE r.invalid_at IS NULL
            RETURN DISTINCT b.entity_key AS key
            """,
            pot=pot,
        )
        return {rec["key"] async for rec in r}


# ---------------------------------------------------------------------------
# INFRA reader
# ---------------------------------------------------------------------------


async def infra_reader(
    driver: AsyncDriver,
    pot: str,
    *,
    service_key: str,
    environment: str | None = None,
) -> list[ReaderHit]:
    """Surface infra topology claims for a service, env-filtered.

    Returns: all live claims where the subject is the service. If environment
    is given, filter to claims either env-tagged with the same env OR
    environment-agnostic (no env tag = applies across).
    """
    rows = await claims_for_subject(
        driver, pot, service_key, include_invalidated=False
    )
    out: list[ReaderHit] = []
    for row in rows:
        if row["predicate"] not in {
            "DEPENDS_ON", "STORED_IN", "USES", "EXPOSES",
            "DEPLOYED_TO", "OWNED_BY", "CONFIGURED_BY",
            "PRODUCES_TO", "CONSUMES_FROM",
        }:
            continue
        # Env filter
        if environment is not None and row["environment"] is not None:
            if row["environment"] != environment:
                continue
        strength = _STRENGTH_RANK.get(row["strength"], 2.0)
        recency = _recency_weight(row["valid_at"], half_life_days=30.0)
        ranked_score = (0.5 + 0.5 * strength / 4.0) * (0.6 + 0.4 * recency)
        out.append(ReaderHit(
            source_event_id=row["source_event_id"],
            fact=row["fact"],
            predicate=row["predicate"],
            subject=row["subject"],
            object=row["object"],
            score=ranked_score,
            evidence_strength=row["strength"],
            valid_at=row["valid_at"],
            source_system=row["source_system"],
            reader="infra",
        ))
    out.sort(key=lambda r: r.score, reverse=True)
    return out


# ---------------------------------------------------------------------------
# TIME reader
# ---------------------------------------------------------------------------


async def time_reader(
    driver: AsyncDriver,
    pot: str,
    *,
    service_key: str,
    window_days: float = 7.0,
) -> list[ReaderHit]:
    """Surface activity claims touching a service inside a time window."""
    window_start = _now() - timedelta(days=window_days)
    # Get claims where this service is involved as subject OR object
    async with driver.session() as s:
        r = await s.run(
            """
            MATCH (a:Entity {group_id: $pot})-[r:RELATES_TO {group_id: $pot}]->(b:Entity {group_id: $pot})
            WHERE (a.entity_key = $key OR b.entity_key = $key)
              AND r.invalid_at IS NULL
              AND r.valid_at >= $window_start
              AND r.name IN ['MERGED_BY', 'REVIEWED_BY', 'MODIFIED',
                             'CLOSES_ISSUE', 'ADDRESSES', 'DEPLOYED_TO',
                             'AFFECTS', 'RESOLVED_BY']
            RETURN a.entity_key AS subject, b.entity_key AS object,
                   r.name AS predicate, r.source_event_id AS source_event_id,
                   r.source_system AS source_system, r.fact AS fact,
                   r.evidence_strength AS strength, r.valid_at AS valid_at
            ORDER BY r.valid_at DESC
            """,
            pot=pot, key=service_key, window_start=window_start,
        )
        rows = [dict(rec) async for rec in r]
    out = [
        ReaderHit(
            source_event_id=row["source_event_id"],
            fact=row["fact"],
            predicate=row["predicate"],
            subject=row["subject"],
            object=row["object"],
            score=_recency_weight(row["valid_at"], half_life_days=window_days / 2.0),
            evidence_strength=row["strength"],
            valid_at=row["valid_at"],
            source_system=row["source_system"],
            reader="time",
        )
        for row in rows
    ]
    return out


# ---------------------------------------------------------------------------
# BUG reader — the killer query: symptom-similarity over fact + episode bodies
# ---------------------------------------------------------------------------


async def bug_reader(
    driver: AsyncDriver,
    pot: str,
    client: AsyncOpenAI,
    *,
    symptom_text: str,
) -> list[ReaderHit]:
    """Find prior bug-shape knowledge similar to a fresh symptom.

    Composes:
      1. Semantic similarity over claim facts (incidents, fixes, patterns).
      2. Semantic similarity over episode bodies (postmortems, discussions).
    """
    symptom_signature = f"Symptom: {symptom_text}. Looking for prior incident, root cause, fix, postmortem, recurring pattern."
    embedding = await embed_text(client, symptom_signature)

    claim_hits = await semantic_search_claims(driver, pot, embedding, top_k=10, min_score=0.40)
    episode_hits = await semantic_search_episodes(driver, pot, embedding, top_k=5, min_score=0.45)

    out: list[ReaderHit] = []
    # Filter claims to bug-shape predicates
    for h in claim_hits:
        if h["predicate"] not in {
            "HAS_ROOT_CAUSE", "RESOLVED_BY", "MITIGATES", "MATCHES_PATTERN",
            "AFFECTS", "ADDRESSES", "DESCRIBES",
        }:
            continue
        strength = _STRENGTH_RANK.get(h["strength"], 2.0)
        ranked = h["score"] * (0.6 + 0.4 * strength / 4.0)
        out.append(ReaderHit(
            source_event_id=h["source_event_id"],
            fact=h["fact"],
            predicate=h["predicate"],
            subject=h["subject"],
            object=h["object"],
            score=ranked,
            evidence_strength=h["strength"],
            valid_at=h["valid_at"],
            source_system=h["source_system"],
            reader="bug-claim",
        ))

    # Episode hits — surface as their own ReaderHit shape (episode is the source)
    for h in episode_hits:
        out.append(ReaderHit(
            source_event_id=h["event_id"],
            fact=h["body"][:200] + ("…" if len(h["body"]) > 200 else ""),
            predicate="EPISODE",
            subject="(episode)",
            object="(episode)",
            score=h["score"] * 0.95,  # slight de-weighting; claims usually preferred
            evidence_strength="attested",
            valid_at=None,
            source_system=h["source"],
            reader="bug-episode",
        ))

    out.sort(key=lambda r: r.score, reverse=True)
    # Dedupe by source_event_id, keeping highest-scoring entry
    seen: set[str] = set()
    dedup: list[ReaderHit] = []
    for hit in out:
        if hit.source_event_id in seen:
            continue
        seen.add(hit.source_event_id)
        dedup.append(hit)
    return dedup[:8]
