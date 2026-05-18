"""Episodic edge classification: collapse vague verbs into canonical edges.

This module collapses the long tail of LLM-emitted relation names into a small
set of canonical edges from :mod:`domain.ontology`. Lifecycle verbs
(``PLANNED``, ``DELIVERED``, ``DEPRECATED``, ``DECOMMISSIONED``,
``MIGRATED_TO``, ``ADDED_TO``, ``REMOVED_FROM``, ``GENERIC_ACTION``) collapse
into the single :data:`LIFECYCLE_TRANSITION` edge with a ``verb`` property and
a ``lifecycle_status`` carried separately.

The remap table is data, not control flow — adding a new verb-family means
adding rows to :data:`VAGUE_VERB_REMAP_RULES` here. Canonical edge types are
sourced from the ontology.
"""

from __future__ import annotations

import re
from typing import Iterable

from domain.ontology import (
    CODE_GRAPH_LABELS,
    EDGE_TYPES,
    LifecycleStatus,
    normalize_graphiti_edge_name,
)

# Stored on Graphiti entity edges under ``attributes["lifecycle_status"]``.
LIFECYCLE_STATUS_VALUES: tuple[str, ...] = tuple(s.value for s in LifecycleStatus)

# Canonical fallback edge for verbs that don't match anything specific. With
# the v2 ontology the lifecycle wildcard verbs collapse here.
LIFECYCLE_TRANSITION_EDGE_NAME = "LIFECYCLE_TRANSITION"
GENERIC_ACTION_EDGE_NAME = LIFECYCLE_TRANSITION_EDGE_NAME  # back-compat alias


def normalize_relation_name(name: str) -> str:
    return normalize_graphiti_edge_name(name or "")


def is_legitimate_pr_code_modified(
    source_labels: Iterable[str], target_labels: Iterable[str]
) -> bool:
    """True when MODIFIED is the PR→code-file edge (narrow Git sense)."""
    src = set(source_labels) | {"Entity"}
    tgt = set(target_labels) | {"Entity"}
    if "PullRequest" not in src:
        return False
    codeish = CODE_GRAPH_LABELS | {"CodeAsset"}
    return bool(tgt & codeish)


# Lifecycle-status heuristics: tense/modality cues over the edge fact.
_LIFECYCLE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "decommissioned",
        re.compile(
            r"\b(decommissioned|decommissioning|torn down|shut down|removed from production)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "deprecated",
        re.compile(
            r"\b(deprecated|deprecation|sunset|end[- ]of[- ]life|eol)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "planned",
        re.compile(
            r"\b(will be|shall be|going to|planned for|planning to|on the roadmap|"
            r"q[1-4]\s*20\d{2}|next quarter|upcoming)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "proposed",
        re.compile(
            r"\b(proposed|proposal|consider|might|may introduce|could add)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "in_progress",
        re.compile(
            r"\b(in progress|being migrated|is being|currently rolling out|underway|actively)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "completed",
        re.compile(
            r"\b(was migrated|has been|have been|completed|shipped|merged|rolled out|"
            r"deployed to production|now uses|switched to)\b",
            re.IGNORECASE,
        ),
    ),
)


def infer_lifecycle_status(fact: str) -> str:
    text = (fact or "").strip()
    if not text:
        return "unknown"
    for status, pattern in _LIFECYCLE_PATTERNS:
        if pattern.search(text):
            return status
    return "unknown"


def coalesce_lifecycle(existing: str | None, inferred: str) -> str:
    if existing in LIFECYCLE_STATUS_VALUES and existing != LifecycleStatus.unknown.value:
        return existing
    return inferred


# Verb remap rules: short keyword tests → canonical edge type.
# Rules earlier in the list take precedence. Adding a new family = adding one
# row. The canonical edge name on the right must exist in EDGE_TYPES.
VAGUE_VERB_REMAP_RULES: tuple[tuple[str, str], ...] = (
    # Specific edges (must stay distinct from lifecycle).
    ("replac", "REPLACES"),
    ("instead of", "REPLACES"),
    ("depend", "DEPENDS_ON"),
    ("relies on", "DEPENDS_ON"),
    ("stored in", "STORED_IN"),
    ("persisted in", "STORED_IN"),
    ("caused", "CAUSED"),
    ("led to", "CAUSED"),
    # Lifecycle transitions collapse to one edge with a verb property — the
    # lifecycle_status is also inferred separately.
    ("migrat", LIFECYCLE_TRANSITION_EDGE_NAME),
    ("decommission", LIFECYCLE_TRANSITION_EDGE_NAME),
    ("shut down", LIFECYCLE_TRANSITION_EDGE_NAME),
    ("torn down", LIFECYCLE_TRANSITION_EDGE_NAME),
    ("deprecat", LIFECYCLE_TRANSITION_EDGE_NAME),
    ("sunset", LIFECYCLE_TRANSITION_EDGE_NAME),
    ("end of life", LIFECYCLE_TRANSITION_EDGE_NAME),
    ("eol", LIFECYCLE_TRANSITION_EDGE_NAME),
    ("will be", LIFECYCLE_TRANSITION_EDGE_NAME),
    (" shall ", LIFECYCLE_TRANSITION_EDGE_NAME),
    ("going to", LIFECYCLE_TRANSITION_EDGE_NAME),
    ("planned", LIFECYCLE_TRANSITION_EDGE_NAME),
    ("roadmap", LIFECYCLE_TRANSITION_EDGE_NAME),
    ("was added", LIFECYCLE_TRANSITION_EDGE_NAME),
    ("were added", LIFECYCLE_TRANSITION_EDGE_NAME),
    ("introduced", LIFECYCLE_TRANSITION_EDGE_NAME),
    ("rolled out", LIFECYCLE_TRANSITION_EDGE_NAME),
    ("shipped", LIFECYCLE_TRANSITION_EDGE_NAME),
    ("completed", LIFECYCLE_TRANSITION_EDGE_NAME),
    ("merged", LIFECYCLE_TRANSITION_EDGE_NAME),
    # Telemetry / instrumentation
    ("telemetry", LIFECYCLE_TRANSITION_EDGE_NAME),
    ("instrument", LIFECYCLE_TRANSITION_EDGE_NAME),
    # Deployment / environment
    ("deploy", "DEPLOYED_TO"),
)


def remap_vague_modified(fact: str) -> str:
    """Map a vague MODIFIED fact to a specific canonical edge."""
    low = (fact or "").lower()
    for needle, canonical in VAGUE_VERB_REMAP_RULES:
        if needle in low:
            return canonical
    return LIFECYCLE_TRANSITION_EDGE_NAME


def classify_episodic_edge(
    relation_name: str,
    fact: str,
    source_labels: Iterable[str],
    target_labels: Iterable[str],
    *,
    allowed_normalized_names: frozenset[str],
    existing_lifecycle: str | None = None,
) -> tuple[str, str]:
    """Return ``(relation_name, lifecycle_status)`` after collapse / allow-list rules."""
    norm = normalize_relation_name(relation_name)
    inferred = infer_lifecycle_status(fact)
    src = tuple(source_labels)
    tgt = tuple(target_labels)

    if norm == "MODIFIED":
        if is_legitimate_pr_code_modified(src, tgt):
            return (
                "MODIFIED",
                coalesce_lifecycle(
                    existing_lifecycle if isinstance(existing_lifecycle, str) else None,
                    inferred,
                ),
            )
        replacement = remap_vague_modified(fact)
        norm = normalize_relation_name(replacement)

    if norm not in allowed_normalized_names:
        norm = LIFECYCLE_TRANSITION_EDGE_NAME

    return (
        norm,
        coalesce_lifecycle(
            existing_lifecycle if isinstance(existing_lifecycle, str) else None,
            inferred,
        ),
    )


def generic_modified_ratio_before_normalize(
    edges: list[object],
    uuid_to_labels: dict[str, tuple[str, ...]],
) -> float:
    """Share of extracted edges that are vague MODIFIED (not PR→code)."""
    if not edges:
        return 0.0
    vague = 0
    for edge in edges:
        name = normalize_relation_name(getattr(edge, "name", "") or "")
        if name != "MODIFIED":
            continue
        src = uuid_to_labels.get(getattr(edge, "source_node_uuid", ""), ())
        tgt = uuid_to_labels.get(getattr(edge, "target_node_uuid", ""), ())
        if is_legitimate_pr_code_modified(src, tgt):
            continue
        vague += 1
    return vague / float(len(edges))


# Self-check at import time: rules must reference real edge types.
def _validate_remap_rules() -> None:
    for _, edge_type in VAGUE_VERB_REMAP_RULES:
        if edge_type not in EDGE_TYPES:
            raise RuntimeError(
                f"VAGUE_VERB_REMAP_RULES references unknown canonical edge: {edge_type!r}"
            )


_validate_remap_rules()
