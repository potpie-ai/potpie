"""LLM-based claim extractor.

Given a raw event body, emits (subject, predicate, object) claims with names
in the SOURCE's vocabulary (we DON'T tell the LLM to resolve identity — that's
the identity layer's job downstream). This is the *minimum viable* extractor
that mirrors what Position B asks an LLM to do per ingest event.

Why this is a meaningful POC piece: it tests whether GPT-class models can
reliably produce well-shaped claim triples from realistic mixed-source event
bodies, with reasonable predicate selection and evidence-strength tagging.
That's the bottleneck that determines whether the rest of Position B works.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from openai import AsyncOpenAI


# The ontology surface we allow the extractor to use. Kept small per E1's
# spirit — promote when a real use case demands an entity type.
ALLOWED_ENTITY_TYPES = [
    "Service", "Component", "DataStore", "Dependency", "Topic",
    "Environment", "Repository",
    "Person", "Team",
    "PullRequest", "Commit", "Issue", "Deployment", "Release",
    "Incident", "Alert", "BugPattern", "Fix",
    "Decision", "Policy",
    "Document", "Conversation",
]

ALLOWED_PREDICATES = [
    # Topology
    "DEPENDS_ON", "STORED_IN", "DEPLOYED_TO", "USES", "EXPOSES",
    "OWNED_BY", "MEMBER_OF",
    "CONFIGURED_BY", "PRODUCES_TO", "CONSUMES_FROM",
    # Change pipeline
    "MERGED_BY", "REVIEWED_BY", "MODIFIED",
    "CLOSES_ISSUE", "ADDRESSES",
    # Decisions / norms
    "GOVERNS", "SUPERSEDES",
    # Reliability
    "AFFECTS", "RESOLVED_BY", "MATCHES_PATTERN",
    "HAS_ROOT_CAUSE", "MITIGATES",
    # Provenance
    "DESCRIBES", "DOCUMENTS",
]


@dataclass(frozen=True)
class ExtractedClaim:
    subject_name: str
    subject_type: str
    predicate: str
    object_name: str
    object_type: str
    fact: str  # one short sentence grounded in source text
    environment: str | None  # "prod" | "staging" | "dev" | None
    evidence_strength: str   # deterministic | attested | inferred | hypothesized


_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "subject_name": {"type": "string"},
                    "subject_type": {"type": "string", "enum": ALLOWED_ENTITY_TYPES},
                    "predicate": {"type": "string", "enum": ALLOWED_PREDICATES},
                    "object_name": {"type": "string"},
                    "object_type": {"type": "string", "enum": ALLOWED_ENTITY_TYPES},
                    "fact": {"type": "string"},
                    "environment": {"type": ["string", "null"]},
                    "evidence_strength": {
                        "type": "string",
                        "enum": ["deterministic", "attested", "inferred", "hypothesized"],
                    },
                },
                "required": [
                    "subject_name", "subject_type", "predicate",
                    "object_name", "object_type", "fact",
                    "environment", "evidence_strength",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["claims"],
    "additionalProperties": False,
}


_SYSTEM_PROMPT = """\
You extract structured claims from project events (PRs, issues, k8s manifests,
ADRs, postmortems, Slack threads, CODEOWNERS).

Rules:
1. Each claim is a (subject, predicate, object) triple with names taken from
   the EVENT TEXT VERBATIM — do NOT normalize names, do NOT correct casing,
   do NOT collapse "auth-svc" and "auth-service" yourself. Identity resolution
   is handled downstream. Your job is to faithfully report what the source SAID.
2. Use only the predicates and entity types in the allowed schema.
3. evidence_strength rules:
   - "deterministic" — fact comes from a structured artifact (k8s YAML,
     CODEOWNERS file, manifest, deployment record).
   - "attested" — explicitly stated in a written doc (ADR, postmortem, README).
   - "inferred" — stated in a discussion (Slack, PR comment) where it's clear
     guidance or fact but not a structured source.
   - "hypothesized" — implied or speculative.
4. environment — set if the event mentions prod/staging/dev/local; otherwise null.
5. Emit only claims clearly supported by the text. Do not invent. If the event
   contains no extractable claims (e.g. a generic announcement), return claims=[].
6. fact: one short sentence (≤120 chars) restating the claim in the source's
   own terms.
7. Prefer multiple atomic claims over one compound claim. e.g.
   "auth-svc owned by alice and uses postgres" → two claims, not one.
"""


def _extraction_user_prompt(event_kind: str, event_body: str, event_source: str) -> str:
    return (
        f"Event source: {event_source}\n"
        f"Event kind: {event_kind}\n\n"
        f"--- BEGIN EVENT BODY ---\n{event_body}\n--- END EVENT BODY ---\n\n"
        "Extract claims as JSON per the schema."
    )


async def extract_claims(
    client: AsyncOpenAI,
    *,
    event_body: str,
    event_kind: str,
    event_source: str,
    model: str = "gpt-5.4-mini",
) -> list[ExtractedClaim]:
    """Run a single structured-output LLM call to extract claims.

    Returns [] for events with no extractable claims.
    """
    r = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _extraction_user_prompt(event_kind, event_body, event_source)},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "ClaimSet",
                "strict": True,
                "schema": _SCHEMA,
            },
        },
        max_completion_tokens=2000,
    )
    raw = r.choices[0].message.content or "{}"
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return []
    claims_data = parsed.get("claims", [])
    return [
        ExtractedClaim(
            subject_name=c["subject_name"],
            subject_type=c["subject_type"],
            predicate=c["predicate"],
            object_name=c["object_name"],
            object_type=c["object_type"],
            fact=c["fact"],
            environment=c.get("environment"),
            evidence_strength=c["evidence_strength"],
        )
        for c in claims_data
    ]
