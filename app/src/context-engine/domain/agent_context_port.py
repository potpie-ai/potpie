"""Stable agent-facing context port vocabulary and helpers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import datetime
from typing import Any

from domain.intelligence_models import IntelligenceBundle

CONTEXT_INTENTS: frozenset[str] = frozenset(
    {
        "feature",
        "debugging",
        "review",
        "operations",
        "planning",
        "docs",
        "onboarding",
        "refactor",
        "test",
        "security",
        "unknown",
    }
)

CONTEXT_INCLUDE_VALUES: frozenset[str] = frozenset(
    {
        "purpose",
        "feature_map",
        "service_map",
        "repo_map",
        "docs",
        "tickets",
        "decisions",
        "recent_changes",
        "owners",
        "prior_fixes",
        "diagnostic_signals",
        "incidents",
        "alerts",
        "deployments",
        "runbooks",
        "local_workflows",
        "scripts",
        "config",
        "preferences",
        "agent_instructions",
        "source_status",
        "operations",
        "discussions",
        "artifact",
        "semantic_search",
        "causal_chain",
    }
)

CONTEXT_RECORD_TYPES: frozenset[str] = frozenset(
    {
        "decision",
        "fix",
        "bug_pattern",
        "investigation",
        "diagnostic_signal",
        "preference",
        "workflow",
        "feature_note",
        "service_note",
        "runbook_note",
        "integration_note",
        "incident_summary",
        "doc_reference",
    }
)

DEBUGGING_MEMORY_INCLUDES: frozenset[str] = frozenset(
    {
        "prior_fixes",
        "diagnostic_signals",
        "incidents",
        "alerts",
    }
)

PROJECT_MAP_INCLUDES: frozenset[str] = frozenset(
    {
        "purpose",
        "feature_map",
        "service_map",
        "repo_map",
        "docs",
        "tickets",
        "deployments",
        "runbooks",
        "local_workflows",
        "scripts",
        "config",
        "preferences",
        "agent_instructions",
        "operations",
    }
)

DEFAULT_INTENT_INCLUDES: dict[str, tuple[str, ...]] = {
    "feature": (
        "purpose",
        "feature_map",
        "service_map",
        "docs",
        "tickets",
        "decisions",
        "recent_changes",
        "owners",
        "preferences",
        "source_status",
    ),
    "debugging": (
        "prior_fixes",
        "diagnostic_signals",
        "incidents",
        "alerts",
        "causal_chain",
        "recent_changes",
        "config",
        "deployments",
        "owners",
        "source_status",
    ),
    "review": (
        "artifact",
        "discussions",
        "owners",
        "recent_changes",
        "decisions",
        "preferences",
        "source_status",
    ),
    "operations": (
        "deployments",
        "runbooks",
        "alerts",
        "incidents",
        "scripts",
        "config",
        "owners",
        "source_status",
    ),
    "planning": (
        "purpose",
        "feature_map",
        "service_map",
        "docs",
        "tickets",
        "decisions",
        "recent_changes",
        "source_status",
    ),
    "docs": ("docs", "decisions", "source_status"),
    "onboarding": (
        "purpose",
        "repo_map",
        "service_map",
        "docs",
        "local_workflows",
        "agent_instructions",
        "source_status",
    ),
    "refactor": (
        "service_map",
        "repo_map",
        "recent_changes",
        "decisions",
        "owners",
        "source_status",
    ),
    "test": (
        "recent_changes",
        "decisions",
        "local_workflows",
        "scripts",
        "source_status",
    ),
    "security": (
        "service_map",
        "docs",
        "decisions",
        "incidents",
        "config",
        "owners",
        "source_status",
    ),
    "unknown": ("semantic_search", "recent_changes", "decisions", "source_status"),
}

CONTEXT_RESOLVE_RECIPES: dict[str, dict[str, Any]] = {
    "feature": {
        "intent": "feature",
        "include": list(DEFAULT_INTENT_INCLUDES["feature"]),
        "mode": "fast",
        "source_policy": "references_only",
        "when": "Before feature work, behavior changes, or cross-repo implementation.",
    },
    "debugging": {
        "intent": "debugging",
        "include": list(DEFAULT_INTENT_INCLUDES["debugging"]),
        "mode": "fast",
        "source_policy": "references_only",
        "when": "Before investigating a bug, incident, failing workflow, alert, or flaky behavior.",
    },
    "review": {
        "intent": "review",
        "include": list(DEFAULT_INTENT_INCLUDES["review"]),
        "mode": "balanced",
        "source_policy": "summary",
        "when": "Before reviewing a PR or checking risky changes against project memory.",
    },
    "operations": {
        "intent": "operations",
        "include": list(DEFAULT_INTENT_INCLUDES["operations"]),
        "mode": "balanced",
        "source_policy": "summary",
        "when": "Before deployment, environment, runbook, alert, or production-impacting work.",
    },
    "docs": {
        "intent": "docs",
        "include": list(DEFAULT_INTENT_INCLUDES["docs"]),
        "mode": "fast",
        "source_policy": "references_only",
        "when": "When locating or validating project documentation and decision context.",
    },
    "onboarding": {
        "intent": "onboarding",
        "include": list(DEFAULT_INTENT_INCLUDES["onboarding"]),
        "mode": "fast",
        "source_policy": "references_only",
        "when": "When entering an unfamiliar pot, repo, service, or local workflow.",
    },
    "planning": {
        "intent": "planning",
        "include": list(DEFAULT_INTENT_INCLUDES["planning"]),
        "mode": "balanced",
        "source_policy": "references_only",
        "when": "Before roadmap, sprint, architecture, or cross-team coordination work.",
    },
    "refactor": {
        "intent": "refactor",
        "include": list(DEFAULT_INTENT_INCLUDES["refactor"]),
        "mode": "balanced",
        "source_policy": "references_only",
        "when": "Before restructuring code, migrating services, or cleaning up technical debt.",
    },
    "test": {
        "intent": "test",
        "include": list(DEFAULT_INTENT_INCLUDES["test"]),
        "mode": "fast",
        "source_policy": "references_only",
        "when": "Before writing, modifying, or reviewing tests and test infrastructure.",
    },
    "security": {
        "intent": "security",
        "include": list(DEFAULT_INTENT_INCLUDES["security"]),
        "mode": "balanced",
        "source_policy": "verify",
        "when": "Before security review, audit, vulnerability assessment, or hardening work.",
    },
    "unknown": {
        "intent": "unknown",
        "include": list(DEFAULT_INTENT_INCLUDES["unknown"]),
        "mode": "fast",
        "source_policy": "references_only",
        "when": "When the task does not match a more specific recipe.",
    },
}

FALLBACK_ONLY_INCLUDES: frozenset[str] = frozenset()


def normalize_context_intent(intent: str | None) -> str:
    value = (intent or "unknown").strip().lower()
    return value if value in CONTEXT_INTENTS else "unknown"


def normalize_context_values(values: list[str] | tuple[str, ...] | None) -> list[str]:
    if not values:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for raw in values:
        value = raw.strip().lower()
        if not value or value in seen:
            continue
        out.append(value)
        seen.add(value)
    return out


def includes_for_request(
    intent: str | None, include: list[str], exclude: list[str]
) -> list[str]:
    requested = normalize_context_values(include)
    if not requested:
        requested = list(DEFAULT_INTENT_INCLUDES[normalize_context_intent(intent)])
    excluded = set(normalize_context_values(exclude))
    return [value for value in requested if value not in excluded]


def unsupported_include_values(include: list[str]) -> list[str]:
    return [
        value
        for value in normalize_context_values(include)
        if value not in CONTEXT_INCLUDE_VALUES
    ]


def context_recipe_for_intent(intent: str | None) -> dict[str, Any]:
    normalized = normalize_context_intent(intent)
    recipe = CONTEXT_RESOLVE_RECIPES.get(normalized)
    if recipe is None:
        recipe = {
            "intent": normalized,
            "include": list(DEFAULT_INTENT_INCLUDES[normalized]),
            "mode": "fast",
            "source_policy": "references_only",
            "when": "When the task does not match a more specific recipe.",
        }
    return {**recipe, "include": list(recipe["include"])}


def context_port_manifest() -> dict[str, Any]:
    """Return stable instructions for agent-facing context usage."""
    return {
        "tools": {
            "context_resolve": {
                "role": "primary",
                "description": "Return a bounded context wrap for a task, including facts, evidence, source refs, freshness, coverage, and fallbacks.",
            },
            "context_search": {
                "role": "secondary",
                "description": "Use for narrow follow-up lookup after context_resolve, not as the default entrypoint.",
            },
            "context_record": {
                "role": "write",
                "description": "Record durable project learnings such as decisions, fixes, preferences, workflows, feature notes, and incident summaries.",
            },
            "context_status": {
                "role": "cheap_status",
                "description": "Check pot readiness, source coverage, freshness gaps, and recommended next context action.",
            },
        },
        "recipes": CONTEXT_RESOLVE_RECIPES,
        "rules": [
            "Start non-trivial work with context_status or context_resolve for the active pot and task scope.",
            "Use intent/include/mode/source_policy presets instead of separate context tools per use case.",
            "Prefer mode=fast and source_policy=references_only first; escalate to summary, verify, snippets, or deep only when coverage or risk requires it.",
            "Inspect coverage, freshness, quality, fallbacks, open_conflicts, and source_refs before relying on graph memory.",
            "If quality.status is watch or degraded, verify relevant facts against source truth before high-impact work.",
            "Use context_search only for specific follow-up lookup when the needed entity or phrase is already known.",
            "Use context_record when a durable decision, fix, workflow, preference, feature note, document reference, or incident summary should become reusable project memory.",
        ],
    }


def normalize_record_type(record_type: str) -> str:
    value = record_type.strip().lower()
    if value not in CONTEXT_RECORD_TYPES:
        allowed = ", ".join(sorted(CONTEXT_RECORD_TYPES))
        raise ValueError(
            f"Unsupported context record type '{record_type}'. Use one of: {allowed}."
        )
    return value


def build_context_record_source_id(
    *,
    record_type: str,
    summary: str,
    scope: dict[str, Any],
    source_refs: list[str],
    idempotency_key: str | None,
) -> str:
    if idempotency_key and idempotency_key.strip():
        return idempotency_key.strip()
    payload = {
        "record_type": record_type,
        "summary": summary,
        "scope": scope,
        "source_refs": source_refs,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
    return f"context_record:{record_type}:{digest}"


def _confidence_for_coverage(status: str) -> float:
    if status == "complete":
        return 0.82
    if status == "partial":
        return 0.55
    return 0.2


def _verification_state(bundle: IntelligenceBundle) -> str:
    if not bundle.source_refs:
        return "unknown"
    states = {ref.verification_state for ref in bundle.source_refs}
    if states == {"verified"}:
        return "verified"
    if "verification_failed" in states:
        return "verification_failed"
    if "needs_verification" in states:
        return "needs_verification"
    return "unverified"


def _summary(bundle: IntelligenceBundle) -> str:
    parts: list[str] = []
    if bundle.artifacts:
        parts.append(f"{len(bundle.artifacts)} artifact")
    if bundle.changes:
        parts.append(f"{len(bundle.changes)} recent change")
    if bundle.decisions:
        parts.append(f"{len(bundle.decisions)} decision")
    if bundle.discussions:
        parts.append(f"{len(bundle.discussions)} discussion")
    if bundle.ownership:
        parts.append(f"{len(bundle.ownership)} owner signal")
    if bundle.project_map:
        parts.append(f"{len(bundle.project_map)} project-map fact")
    if bundle.debugging_memory:
        parts.append(f"{len(bundle.debugging_memory)} debugging memory item")
    if bundle.causal_chain:
        parts.append(f"{len(bundle.causal_chain)} causal chain step")
    if bundle.semantic_hits:
        parts.append(f"{len(bundle.semantic_hits)} memory hit")
    if not parts:
        return "No matching project context was found for this request."
    return "Resolved " + ", ".join(parts) + " for this request."


def bundle_to_agent_envelope(bundle: IntelligenceBundle) -> dict[str, Any]:
    """Return the stable minimal-port response envelope for agents."""
    bundle_dict = asdict(bundle)
    return {
        "ok": True,
        "answer": {
            "summary": _summary(bundle),
            "artifacts": bundle_dict["artifacts"],
            "recent_changes": bundle_dict["changes"],
            "decisions": bundle_dict["decisions"],
            "discussions": bundle_dict["discussions"],
            "owners": bundle_dict["ownership"],
            "project_map": bundle_dict["project_map"],
            "debugging_memory": bundle_dict["debugging_memory"],
        },
        "facts": {
            "changes": bundle_dict["changes"],
            "decisions": bundle_dict["decisions"],
            "ownership": bundle_dict["ownership"],
            "project_map": bundle_dict["project_map"],
            "debugging_memory": bundle_dict["debugging_memory"],
            "causal_chain": bundle_dict["causal_chain"],
        },
        "evidence": bundle_dict["semantic_hits"] + bundle_dict["discussions"],
        "source_refs": bundle_dict["source_refs"],
        "source_resolution": bundle_dict["source_resolution"],
        "confidence": _confidence_for_coverage(bundle.coverage.status),
        "as_of": _iso_or_none(bundle.request.as_of),
        "open_conflicts": bundle_dict["open_conflicts"],
        "coverage": bundle_dict["coverage"],
        "freshness": bundle_dict["freshness"],
        "quality": bundle_dict["quality"],
        "verification_state": _verification_state(bundle),
        "fallbacks": bundle_dict["fallbacks"],
        "recommended_next_actions": bundle_dict["recommended_next_actions"],
        "errors": bundle_dict["errors"],
        "meta": bundle_dict["meta"],
        "bundle": bundle_dict,
    }


def _iso_or_none(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None
