"""Heuristics for Graphiti episodic extraction (edge-type collapse / lifecycle).

See docs/context-graph-improvements/02-edge-type-collapse.md.
"""

from __future__ import annotations

import re
from typing import Iterable

from domain.ontology import (
    CODE_GRAPH_LABELS,
    LifecycleStatus,
    normalize_graphiti_edge_name,
)

# Stored on Graphiti entity edges under ``attributes["lifecycle_status"]`` (Neo4j rel props).
LIFECYCLE_STATUS_VALUES: tuple[str, ...] = tuple(s.value for s in LifecycleStatus)

# Canonical fallback when the LLM emits a vague MODIFIED or an unknown relation name.
GENERIC_ACTION_EDGE_NAME = "GENERIC_ACTION"


def normalize_relation_name(name: str) -> str:
    return normalize_graphiti_edge_name(name or "")


def is_legitimate_pr_code_modified(
    source_labels: Iterable[str], target_labels: Iterable[str]
) -> bool:
    """True when MODIFIED is the PR→code-file edge (narrow Git sense), not a vague catch-all."""
    src = set(source_labels) | {"Entity"}
    tgt = set(target_labels) | {"Entity"}
    if "PullRequest" not in src:
        return False
    codeish = CODE_GRAPH_LABELS | {"CodeAsset"}
    return bool(tgt & codeish)


def infer_lifecycle_status(fact: str) -> str:
    """Lightweight tense/modality heuristic over the edge fact sentence."""
    text = (fact or "").strip()
    if not text:
        return "unknown"
    low = text.lower()

    if re.search(
        r"\b(decommissioned|decommissioning|torn down|shut down|removed from production)\b",
        low,
    ):
        return "decommissioned"
    if re.search(r"\b(deprecated|deprecation|sunset|end[- ]of[- ]life|eol)\b", low):
        return "deprecated"

    if re.search(
        r"\b(will be|shall be|going to|planned for|planning to|on the roadmap|"
        r"q[1-4]\s*20\d{2}|next quarter|upcoming)\b",
        low,
    ):
        return "planned"
    if re.search(r"\b(proposed|proposal|consider|might|may introduce|could add)\b", low):
        return "proposed"
    if re.search(r"\b(in progress|being migrated|is being|currently rolling out|"
                 r"underway|actively)\b", low):
        return "in_progress"

    if re.search(
        r"\b(was migrated|has been|have been|completed|shipped|merged|rolled out|"
        r"deployed to production|now uses|switched to)\b",
        low,
    ):
        return "completed"

    return "unknown"


def coalesce_lifecycle(existing: str | None, inferred: str) -> str:
    if existing in LIFECYCLE_STATUS_VALUES and existing != LifecycleStatus.unknown.value:
        return existing
    return inferred


def remap_vague_modified(fact: str) -> str:
    """Map a vague MODIFIED fact to a specific verb; fall back to GENERIC_ACTION."""
    low = (fact or "").lower()
    if "migrat" in low:
        return "MIGRATED_TO"
    if "decommission" in low or "shut down" in low or "torn down" in low:
        return "DECOMMISSIONED"
    if "deprecat" in low or "sunset" in low or " end of life" in low or "eol" in low:
        return "DEPRECATED"
    if (
        "will be" in low
        or " shall " in low
        or "going to" in low
        or "planned" in low
        or "roadmap" in low
    ):
        return "PLANNED"
    if (
        "was added" in low
        or "were added" in low
        or "introduced" in low
        or "rolled out" in low
        or "shipped" in low
        or "completed" in low
        or "merged" in low
    ):
        return "DELIVERED"
    if "replac" in low or "instead of" in low:
        return "REPLACES"
    if "depend" in low or "relies on" in low:
        return "DEPENDS_ON"
    if "deploy" in low and ("prod" in low or "staging" in low or "environment" in low):
        return "DEPLOYED_TO"
    if "caused" in low or "led to" in low:
        return "CAUSED"
    if "added" in low and ("telemetry" in low or "span" in low or "instrument" in low):
        return "ADDED_TO"
    return GENERIC_ACTION_EDGE_NAME


def classify_episodic_edge(
    relation_name: str,
    fact: str,
    source_labels: Iterable[str],
    target_labels: Iterable[str],
    *,
    allowed_normalized_names: frozenset[str],
    existing_lifecycle: str | None = None,
) -> tuple[str, str]:
    """Return ``(relation_name, lifecycle_status)`` after collapse / allow-list rules.

    Used both at ingest (Graphiti extract) and by the ``classify_modified_edges`` job.
    """
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
        norm = GENERIC_ACTION_EDGE_NAME

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
    """Share of extracted edges that are vague MODIFIED (not PR→code), for regression metrics."""
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
