"""Stable agent-facing context port vocabulary and helpers.

Agent-surface adapter over the unified ontology catalog. The record-type
vocabulary, the advertised include families, and the planned-vs-backed
split are *derived* from :data:`domain.ontology.RECORD_TYPES` and
:data:`domain.ontology.STRUCTURAL_INCLUDES`; this module only owns the
intent set, the resolve recipes, and the per-intent default-include map.

The read orchestrator (``application/services/read_orchestrator``)
answers exactly :data:`READER_BACKED_INCLUDES`. :data:`PLANNED_INCLUDES`
holds advertised-but-not-yet-implemented keys; the orchestrator surfaces
them as ``unsupported_include`` (reason ``not_implemented``) rather than
silent zeros. Adding a backing reader moves the key over automatically.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from domain.ontology import (
    PUBLIC_RECORD_TYPES,
    advertised_include_families,
)

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

# --- Derived vocabularies ---------------------------------------------------
# Derived from RECORD_TYPES + STRUCTURAL_INCLUDES so the agent surface and
# the graph schema cannot drift apart. The coherence check enforces that
# READER_BACKED_INCLUDES matches the runtime reader registry.

CONTEXT_RECORD_TYPES: frozenset[str] = PUBLIC_RECORD_TYPES

# ``READER_BACKED_INCLUDES`` mirrors ``ReadOrchestrator.backed_includes`` —
# the coherence check asserts they match. Adding a reader to the
# orchestrator's ``_routing`` dict requires adding the key here too.
READER_BACKED_INCLUDES: frozenset[str] = frozenset(
    {
        "coding_preferences",
        "features",
        "infra_topology",
        "timeline",
        "prior_bugs",
        "decisions",
        "owners",
        "docs",
        # Visualization read: the full canonical subgraph (all RELATES_TO,
        # incl. generic RELATED_TO). Backed by RawGraphReader; used by the
        # graph explorer, not an agent use-case family.
        "raw_graph",
    }
)

# The full published agent include contract — derived from the ontology.
CONTEXT_INCLUDE_VALUES: frozenset[str] = advertised_include_families()

# Anything advertised but not yet reader-backed.
PLANNED_INCLUDES: frozenset[str] = CONTEXT_INCLUDE_VALUES - READER_BACKED_INCLUDES

# Keys the resolver honestly flags as not-yet-implemented (no backing reader).
# Retained as an alias for back-compat.
FALLBACK_ONLY_INCLUDES: frozenset[str] = PLANNED_INCLUDES

DEFAULT_INTENT_INCLUDES: dict[str, tuple[str, ...]] = {
    "feature": ("coding_preferences", "infra_topology", "decisions", "owners", "docs"),
    "debugging": ("prior_bugs", "infra_topology", "timeline"),
    "review": ("coding_preferences", "decisions", "timeline", "owners"),
    "operations": ("infra_topology", "timeline", "owners"),
    "planning": ("infra_topology", "decisions", "timeline", "docs"),
    "docs": ("docs", "decisions"),
    "onboarding": ("infra_topology", "coding_preferences", "docs", "owners"),
    "refactor": ("infra_topology", "coding_preferences", "timeline"),
    "test": ("coding_preferences", "timeline"),
    "security": ("infra_topology", "prior_bugs", "decisions"),
    "unknown": ("infra_topology", "timeline", "decisions"),
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


# --- Deterministic intent detection -----------------------------------------
# A dependency-free keyword/phrase matcher over the free-text task, mirroring
# the "no-guess" discipline of ``domain.ontology_classifier``: when nothing
# matches confidently it returns ``None`` (the caller surfaces that as the
# neutral ``unknown`` intent) rather than guessing an intent. The tables are
# seeded from the natural-language trigger phrases in each recipe's ``"when"``
# field (see :data:`CONTEXT_RESOLVE_RECIPES`). ``unknown`` is intentionally
# absent — it is the fallback, never a detected result.
#
# Detection runs *before* ``normalize_context_intent`` and does not modify the
# intent vocabulary, recipes, or default-include map.
_INTENT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "security": (
        "vulnerability",
        "vulnerabilities",
        "cve",
        "exploit",
        "security",
        "audit",
        "hardening",
        "injection",
        "auth bypass",
    ),
    "debugging": (
        "bug",
        "error",
        "errors",
        "crash",
        "crashing",
        "incident",
        "failing",
        "failure",
        "exception",
        "traceback",
        "stack trace",
        "flaky",
        "500",
        "throwing",
        "broken",
    ),
    "operations": (
        "deploy",
        "deployment",
        "runbook",
        "production",
        "rollback",
        "outage",
        "on-call",
        "oncall",
        "alert",
    ),
    "refactor": (
        "refactor",
        "technical debt",
        "tech debt",
        "migrate",
        "migration",
        "cleanup",
        "clean up",
        "restructure",
    ),
    "test": ("test", "tests", "pytest", "coverage", "test suite"),
    "review": (
        "review",
        "pull request",
        "what changed",
        "recently",
        "recent changes",
        "latest changes",
        "risky",
    ),
    "docs": ("documentation", "docs", "readme", "document"),
    "onboarding": (
        "onboard",
        "onboarding",
        "unfamiliar",
        "getting started",
        "new to",
        "up to speed",
    ),
    "planning": ("roadmap", "sprint", "architecture", "design doc", "milestone"),
    "feature": ("feature", "implement", "add support", "endpoint", "behavior change"),
}


_INTENT_DETECTION_PRIORITY: tuple[str, ...] = tuple(_INTENT_KEYWORDS)

_INTENT_MATCHERS: dict[str, re.Pattern[str]] = {
    intent: re.compile(
        r"|".join(r"\b" + re.escape(kw) + r"\b" for kw in keywords),
        re.IGNORECASE,
    )
    for intent, keywords in _INTENT_KEYWORDS.items()
}


def detect_context_intent(task: str | None) -> str | None:
    """Map a free-text task to a canonical intent, or ``None`` if unsure.

    Deterministic, case-insensitive, word-boundary keyword match. When the
    task triggers more than one intent, :data:`_INTENT_DETECTION_PRIORITY`
    breaks the tie (most specific / highest-risk first). Returns ``None`` when
    nothing matches — favouring "no guess" over a wrong guess — which the CLI
    surfaces as the neutral ``unknown`` intent with ``intent_source=default``.
    """
    if not task or not task.strip():
        return None
    for intent in _INTENT_DETECTION_PRIORITY:
        if _INTENT_MATCHERS[intent].search(task):
            return intent
    return None


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
        "include_families": {
            # Honest capability map: ``reader_backed`` keys are answered by a P9
            # reader today; ``planned`` keys are requestable but surface as
            # ``unsupported_include`` until a reader backs them.
            "reader_backed": sorted(READER_BACKED_INCLUDES),
            "planned": sorted(PLANNED_INCLUDES),
        },
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
