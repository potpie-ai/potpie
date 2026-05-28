"""Shape an :class:`AgentEnvelope` into a compact synthesis prompt.

Keeps the LLM-facing payload deterministic and bounded so synthesis latency
stays predictable and so unit tests can pin the exact structure.
"""

from __future__ import annotations

import json
from typing import Any

from domain.agent_envelope import AgentEnvelope

_MAX_PER_FAMILY = 8
_MAX_FIELD_LEN = 600

SYNTHESIS_INSTRUCTIONS = """You synthesize a short answer for a coding agent from a pre-resolved, ranked evidence envelope.

Requirements:
- Answer the `query` using only the facts under `evidence` (grouped by include family). Do not invent facts.
- Keep the answer to 2-4 sentences. No markdown headers, no bullet lists.
- Cite a `source_ref` inline as (ref) when a fact you reference carries one.
- When `confidence != "high"`, when any `coverage` entry is not `complete`, or when `unsupported_includes` is non-empty, end with one short caveat sentence naming the gap.
- If `evidence` is empty, return exactly: "No project context found for this query."
"""


def _trim(value: Any) -> Any:
    if isinstance(value, str):
        return value.strip()[:_MAX_FIELD_LEN]
    return value


def build_synthesis_payload(envelope: AgentEnvelope) -> dict[str, Any]:
    """Produce the compact dict handed to the LLM."""
    evidence: dict[str, list[dict[str, Any]]] = {}
    for item in envelope.items:
        bucket = evidence.setdefault(item.include, [])
        if len(bucket) >= _MAX_PER_FAMILY:
            continue
        bucket.append({k: _trim(v) for k, v in dict(item.payload).items()})
    return {
        "query": str(envelope.metadata.get("query", "") or "").strip()[:_MAX_FIELD_LEN],
        "intent": envelope.intent,
        "confidence": envelope.overall_confidence,
        "coverage": [
            {"include": c.include, "status": c.status} for c in envelope.coverage
        ],
        "unsupported_includes": [u.name for u in envelope.unsupported_includes],
        "evidence": evidence,
    }


def build_synthesis_prompt(envelope: AgentEnvelope) -> str:
    payload = build_synthesis_payload(envelope)
    return json.dumps(payload, indent=2, default=str)
