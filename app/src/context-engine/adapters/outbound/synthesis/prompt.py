"""Shape an :class:`IntelligenceBundle` into a compact synthesis prompt.

Keeps the LLM-facing payload deterministic and bounded so synthesis latency
stays predictable and so unit tests can pin the exact structure.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from domain.intelligence_models import IntelligenceBundle

_MAX_PER_FAMILY = 8
_MAX_SOURCE_REFS = 12
_MAX_FIELD_LEN = 600

SYNTHESIS_INSTRUCTIONS = """You synthesize a short answer for a coding agent from a pre-resolved evidence bundle.

Requirements:
- Answer the `query` using only the fields under `evidence`. Do not invent facts.
- Keep the answer to 2-4 sentences. No markdown headers, no bullet lists.
- Cite at least one `source_refs` entry inline as (kind:ref) when you reference a fact that has one.
- When `coverage.status != "complete"` or `fallbacks` is non-empty, end with one short caveat sentence naming the gap (e.g. "Missing: change_history." or "Fallback: decisions via semantic seeds.").
- When `source_refs` include verification_state == "needs_verification", add "Needs verification." to the caveat.
- If the evidence is empty, return exactly: "No project context found for this query."
"""


def _trim(value: Any) -> Any:
    if isinstance(value, str):
        s = value.strip()
        return s[:_MAX_FIELD_LEN]
    return value


def _pick(d: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k in keys:
        if k in d and d[k] not in (None, "", []):
            out[k] = _trim(d[k])
    return out


def _records(items: list[Any], keys: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in items[:_MAX_PER_FAMILY]:
        d = asdict(item) if hasattr(item, "__dataclass_fields__") else dict(item)
        out.append(_pick(d, keys))
    return out


def build_synthesis_payload(bundle: IntelligenceBundle) -> dict[str, Any]:
    """Produce the compact dict handed to the LLM."""
    return {
        "query": (bundle.request.query or "").strip()[:_MAX_FIELD_LEN],
        "scope": _pick(
            asdict(bundle.request.scope) if bundle.request.scope else {},
            [
                "repo_name",
                "file_path",
                "function_name",
                "pr_number",
                "services",
                "features",
                "environment",
                "ticket_ids",
            ],
        ),
        "coverage": {
            "status": bundle.coverage.status,
            "missing": list(bundle.coverage.missing),
        },
        "evidence": {
            "decisions": _records(
                bundle.decisions, ["decision", "rationale", "pr_number", "source_ref"]
            ),
            "recent_changes": _records(
                bundle.changes, ["pr_number", "title", "summary", "artifact_ref"]
            ),
            "discussions": _records(
                bundle.discussions, ["source_ref", "summary"]
            ),
            "project_map": _records(
                bundle.project_map, ["family", "name", "summary", "source_uri"]
            ),
            "debugging_memory": _records(
                bundle.debugging_memory,
                ["title", "summary", "root_cause", "status", "source_ref"],
            ),
            "ownership": _records(
                bundle.ownership, ["file_path", "owner", "confidence_signal"]
            ),
            "artifacts": _records(
                bundle.artifacts, ["kind", "identifier", "title", "summary"]
            ),
        },
        "fallbacks": [
            _pick(asdict(fb), ["code", "message", "impact", "ref"])
            for fb in bundle.fallbacks
        ],
        "source_refs": [
            _pick(
                asdict(ref),
                ["source_type", "ref", "verification_state", "freshness"],
            )
            for ref in bundle.source_refs[:_MAX_SOURCE_REFS]
        ],
    }


def build_synthesis_prompt(bundle: IntelligenceBundle) -> str:
    payload = build_synthesis_payload(bundle)
    return json.dumps(payload, indent=2, default=str)
