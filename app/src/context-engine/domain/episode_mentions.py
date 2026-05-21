"""MENTIONS-provenance helpers (rebuild plan P5 / F4).

Per the proper POC's F4 failure mode: TIME reader queries like "what
activities touched service:auth-svc in last 7d" missed PR events
because PR Activity entities link to ``person`` (via PERFORMED_BY) and
``pr:1042`` (via TOUCHED) but never to ``service:auth-svc`` — the
service is mentioned in the PR body, not in a structured field.

The fix per the Graphiti research: every entity mentioned in an
enriched episode body gets a ``:RELATES_TO {name: 'MENTIONS'}`` claim
back from the episode's Activity entity. The TIME reader then does
``MATCH (a:Activity)-[:MENTIONS]->(target {entity_key: $key})``.

This module gives connectors + the LLM-extraction agent a deterministic
helper that emits MENTIONS edge upserts for every entity referenced in
a body — without inventing scope, without LLM round-trips for the
linking step.
"""

from __future__ import annotations

from typing import Iterable

from domain.graph_mutations import EdgeUpsert


def build_mentions_edges(
    *,
    activity_entity_key: str,
    mentioned_entity_keys: Iterable[str],
    source_ref: str,
    source_system: str,
    fact_template: str | None = None,
) -> list[EdgeUpsert]:
    """Emit one ``MENTIONS`` :class:`EdgeUpsert` per mentioned entity.

    Parameters
    ----------
    activity_entity_key
        The source Activity entity (e.g. ``activity:github:pr:1042``).
    mentioned_entity_keys
        Iterable of canonical entity_keys mentioned in the episode body.
        Duplicates are deduped; the activity-key itself is filtered out
        (an episode mentioning its own activity is meaningless).
    source_ref
        Identifies the source the MENTIONS were extracted from (PR body
        URL, commit message, doc path). Becomes part of the MERGE key
        so re-extracting the same body updates in place.
    source_system
        Source-system identifier (e.g. ``github``, ``linear``) — written
        to the edge's ``source_system`` property for belief weighting.
    fact_template
        Optional human-readable fact text template ``"{activity} mentions
        {target}"``. Default produces a stable sentence per pair.
    """
    unique_targets: list[str] = []
    seen: set[str] = set()
    for key in mentioned_entity_keys:
        if not isinstance(key, str) or not key:
            continue
        if key == activity_entity_key:
            continue
        if key in seen:
            continue
        seen.add(key)
        unique_targets.append(key)

    edges: list[EdgeUpsert] = []
    template = fact_template or "{activity} mentions {target}"
    for target_key in unique_targets:
        fact = template.format(activity=activity_entity_key, target=target_key)
        edges.append(
            EdgeUpsert(
                edge_type="MENTIONS",
                from_entity_key=activity_entity_key,
                to_entity_key=target_key,
                properties={
                    "source_ref": source_ref,
                    "source_system": source_system,
                    # MENTIONS is always at least "attested" — the episode
                    # body literally contains the reference. Promoting to
                    # "deterministic" requires a deterministic parser
                    # (e.g. PR-body issue-ref extractor), so leave LLM-
                    # extracted mentions at attested.
                    "evidence_strength": "attested",
                    "fact": fact,
                },
            )
        )
    return edges


__all__ = ["build_mentions_edges"]
