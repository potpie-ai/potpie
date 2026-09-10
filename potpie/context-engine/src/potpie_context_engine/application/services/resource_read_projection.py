"""Keep passage evidence usable when crossing into the named-view contract."""

from typing import Any

from potpie_context_core.agent_envelope import EvidenceItem


def resource_read_item(item: EvidenceItem) -> dict[str, Any]:
    payload = item.payload
    source_ref = payload.get("source_ref")
    return {
        "entity_key": item.candidate_key,
        "entity_type": "ResourceChunk",
        "score": item.score,
        "summary": payload.get("snippet") or payload.get("label") or item.candidate_key,
        "chunk_ids": [payload["resource_id"]],
        "source_refs": [source_ref] if source_ref else [],
        "fetch": payload.get("fetch"),
        "retrieval": dict(payload.get("retrieval") or {}),
        "coverage_status": item.coverage_status,
        "breakdown": dict(item.breakdown),
        "relations": [],
    }
