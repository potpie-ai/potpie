"""Identity / canonicalization layer — D2 made concrete.

Resolves a `surface_name` (whatever the source called the thing) to a
canonical `entity_key`. Three stages, in order:

  Stage 1 — Exact-match on existing canonical entity_keys (after slugifying).
  Stage 2 — Alias-table lookup (a prior source recorded this surface_name).
  Stage 3 — Embedding similarity over existing entity names within the same
            type; if top hit > THRESHOLD, propose match; LLM disambiguation
            confirms or rejects.
  Stage 4 — New canonical entity. Mint a slug; record the surface_name as the
            first alias.

Every accepted match writes an :Alias node + :ALIAS_OF edge so an agent (or a
human auditor) can introspect *why* two source names converged on one entity.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from openai import AsyncOpenAI

from .store import Alias, AsyncDriver, Entity, now_utc, write_alias, write_entity


SIMILARITY_THRESHOLD = 0.82  # above this, propose match for LLM confirmation
AUTO_ACCEPT_THRESHOLD = 0.95  # above this, accept without LLM call


@dataclass(frozen=True)
class ResolutionResult:
    entity_key: str
    matched_via: str  # "exact" | "alias" | "embedding-auto" | "embedding-llm" | "new"
    confidence: str   # "deterministic" | "attested" | "inferred"
    detail: str = ""


def slugify(name: str, entity_type: str) -> str:
    """Mint a deterministic slug for a new canonical entity_key."""
    type_prefix = {
        "Service": "service",
        "Person": "person",
        "Team": "team",
        "Repository": "repo",
        "Environment": "env",
        "DataStore": "datastore",
        "Dependency": "dep",
        "PullRequest": "pr",
        "Issue": "issue",
        "Incident": "incident",
        "Decision": "decision",
        "Policy": "policy",
        "Document": "doc",
        "Deployment": "deploy",
        "Topic": "topic",
        "BugPattern": "bug",
        "Fix": "fix",
    }.get(entity_type, entity_type.lower())
    body = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
    if not body:
        body = "unknown"
    return f"{type_prefix}:{body}"


async def embed_text(client: AsyncOpenAI, text: str) -> list[float]:
    r = await client.embeddings.create(model="text-embedding-3-small", input=text)
    return r.data[0].embedding


async def find_existing_by_alias(
    driver: AsyncDriver, pot: str, surface_name: str, entity_type: str
) -> str | None:
    """Stage 2 — alias-table exact lookup (case-insensitive)."""
    cypher = """
        MATCH (a:Alias {group_id: $pot, surface_name_lower: $surface_lower})
              -[:ALIAS_OF]->(c:Entity)
        WHERE $type IN labels(c)
        RETURN c.entity_key AS key
        LIMIT 1
    """
    async with driver.session() as s:
        r = await s.run(cypher, pot=pot, surface_lower=surface_name.strip().lower(), type=entity_type)
        record = await r.single()
        return record["key"] if record else None


async def find_existing_by_key(
    driver: AsyncDriver, pot: str, candidate_key: str, entity_type: str
) -> str | None:
    """Stage 1 — exact-key match (the candidate slug is already a known entity_key)."""
    cypher = """
        MATCH (c:Entity {group_id: $pot, entity_key: $key})
        WHERE $type IN labels(c)
        RETURN c.entity_key AS key
        LIMIT 1
    """
    async with driver.session() as s:
        r = await s.run(cypher, pot=pot, key=candidate_key, type=entity_type)
        record = await r.single()
        return record["key"] if record else None


async def find_similar_by_embedding(
    driver: AsyncDriver,
    pot: str,
    name_embedding: list[float],
    entity_type: str,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """Stage 3 — embedding similarity over existing entity names of the same type."""
    cypher = """
        CALL db.index.vector.queryNodes(
            'entity_name_embeddings', $top_k, $embedding
        ) YIELD node AS n, score
        WHERE n.group_id = $pot AND $type IN labels(n)
        RETURN n.entity_key AS key, n.name AS name, score
        ORDER BY score DESC
        LIMIT $top_k
    """
    async with driver.session() as s:
        r = await s.run(cypher, pot=pot, embedding=name_embedding, type=entity_type, top_k=top_k)
        return [dict(rec) async for rec in r]


async def confirm_match_via_llm(
    client: AsyncOpenAI,
    surface_name: str,
    entity_type: str,
    source_event_body: str,
    candidate_name: str,
    candidate_key: str,
) -> bool:
    """Ask an LLM whether surface_name in the given source context is THE SAME
    thing as the existing canonical candidate. Returns True iff confirmed."""
    prompt = (
        "You are a project-knowledge-graph identity resolver.\n\n"
        f"Type: {entity_type}\n\n"
        f"Source event excerpt:\n---\n{source_event_body[:600]}\n---\n\n"
        f"This event uses the name: \"{surface_name}\"\n\n"
        f"An existing canonical entity has name \"{candidate_name}\" "
        f"(key: {candidate_key}).\n\n"
        "Are these the SAME logical thing? Answer ONLY 'yes' or 'no'. If "
        "unsure (e.g. they could be sibling services with similar names), "
        "answer 'no'."
    )
    r = await client.chat.completions.create(
        model="gpt-5.4-mini",
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=4,
    )
    answer = (r.choices[0].message.content or "").strip().lower()
    return answer.startswith("y")


async def resolve_or_create(
    driver: AsyncDriver,
    pot: str,
    client: AsyncOpenAI,
    *,
    surface_name: str,
    entity_type: str,
    source_event_id: str,
    source_event_body: str,
    source_system: str,
) -> ResolutionResult:
    """Resolve a surface_name to a canonical entity_key, creating one if necessary.

    Always writes an Alias node recording the surface_name → entity_key mapping
    for audit, regardless of how the match was found.
    """
    # Stage 1 — exact-key match
    candidate_key = slugify(surface_name, entity_type)
    existing = await find_existing_by_key(driver, pot, candidate_key, entity_type)
    if existing:
        await write_alias(driver, pot, Alias(
            surface_name=surface_name,
            canonical_entity_key=existing,
            source_event_id=source_event_id,
            source_system=source_system,
            evidence_strength="deterministic",
            observed_at=now_utc(),
        ))
        return ResolutionResult(existing, "exact", "deterministic",
                                 detail=f"slugify→{existing}")

    # Stage 2 — alias lookup
    aliased = await find_existing_by_alias(driver, pot, surface_name, entity_type)
    if aliased:
        await write_alias(driver, pot, Alias(
            surface_name=surface_name,
            canonical_entity_key=aliased,
            source_event_id=source_event_id,
            source_system=source_system,
            evidence_strength="attested",
            observed_at=now_utc(),
        ))
        return ResolutionResult(aliased, "alias", "attested",
                                 detail=f"alias-table→{aliased}")

    # Stage 3 — embedding similarity within type
    name_embedding = await embed_text(client, surface_name)
    similar = await find_similar_by_embedding(driver, pot, name_embedding, entity_type)
    if similar:
        top = similar[0]
        if top["score"] >= AUTO_ACCEPT_THRESHOLD:
            # Very high similarity — auto-accept
            await write_alias(driver, pot, Alias(
                surface_name=surface_name,
                canonical_entity_key=top["key"],
                source_event_id=source_event_id,
                source_system=source_system,
                evidence_strength="attested",
                observed_at=now_utc(),
            ))
            return ResolutionResult(top["key"], "embedding-auto", "attested",
                                     detail=f"score={top['score']:.3f} ≥ {AUTO_ACCEPT_THRESHOLD}")
        if top["score"] >= SIMILARITY_THRESHOLD:
            # Borderline — ask LLM to confirm
            confirmed = await confirm_match_via_llm(
                client, surface_name, entity_type, source_event_body,
                top["name"], top["key"],
            )
            if confirmed:
                await write_alias(driver, pot, Alias(
                    surface_name=surface_name,
                    canonical_entity_key=top["key"],
                    source_event_id=source_event_id,
                    source_system=source_system,
                    evidence_strength="inferred",
                    observed_at=now_utc(),
                ))
                return ResolutionResult(top["key"], "embedding-llm", "inferred",
                                         detail=f"score={top['score']:.3f}, LLM confirmed")
            # LLM rejected — fall through to new entity

    # Stage 4 — new canonical entity
    new_key = candidate_key
    # If this slug already exists (collision with a different type or name),
    # disambiguate by appending a small suffix.
    suffix = 1
    while await find_existing_by_key(driver, pot, new_key, entity_type):
        new_key = f"{candidate_key}-{suffix}"
        suffix += 1
    await write_entity(driver, pot, Entity(
        entity_key=new_key,
        labels=(entity_type, "Entity"),
        name=surface_name,
        properties={},
    ), name_embedding=name_embedding)
    await write_alias(driver, pot, Alias(
        surface_name=surface_name,
        canonical_entity_key=new_key,
        source_event_id=source_event_id,
        source_system=source_system,
        evidence_strength="deterministic",  # this name *defines* the entity
        observed_at=now_utc(),
    ))
    return ResolutionResult(new_key, "new", "deterministic",
                             detail=f"created {new_key}")
