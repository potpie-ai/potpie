"""Deterministic ontology classifier: entity signals → canonical labels to add.

The classifier is the single rule engine behind two paths:

* Reconciliation plan enrichment (``domain.canonical_label_inference``) — runs
  before structural writes so the governed validator sees canonical labels.
* Post-Graphiti Neo4j pass (``adapters.outbound.graphiti.ontology_classifier_pass``) —
  runs after episodic extraction so flexible LLM output is pinned to the ontology.

Rules are drawn from three signal sources and only fire when the inference is
unambiguous. When the signals conflict or do not uniquely pick a label, the
classifier returns nothing rather than guessing — a missing label is cheaper
than a wrong one.

Signal sources:
    1. Edge-endpoint rules (``EDGE_ENDPOINT_INFERRED_LABELS``): the canonical
       shape of an edge fixes the label of its endpoints.
    2. Property signatures: Graphiti entity schemas carry discriminating
       properties (``pr_number``, ``sha``, ``fix_type``, ``signal_type``, …).
    3. Text cues on ``name`` / ``title`` / ``summary`` / ``statement``: high-
       precision regex patterns for Decision / Fix / Incident / Runbook /
       BugPattern / Constraint / Preference / AgentInstruction.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from domain.ontology import (
    ENTITY_TYPES,
    inferred_labels_for_episodic_edge_endpoint,
    is_canonical_entity_label,
    normalize_graphiti_edge_name,
)


@dataclass(frozen=True, slots=True)
class EntitySignals:
    """All deterministic signals available for classifying one entity."""

    labels: tuple[str, ...]
    properties: Mapping[str, Any]
    outgoing_edge_names: frozenset[str]
    incoming_edge_names: frozenset[str]


def build_signals(
    labels: Iterable[str],
    properties: Mapping[str, Any],
    outgoing_edge_names: Iterable[str] = (),
    incoming_edge_names: Iterable[str] = (),
) -> EntitySignals:
    return EntitySignals(
        labels=tuple(labels),
        properties=dict(properties),
        outgoing_edge_names=frozenset(
            normalize_graphiti_edge_name(n) for n in outgoing_edge_names if n
        ),
        incoming_edge_names=frozenset(
            normalize_graphiti_edge_name(n) for n in incoming_edge_names if n
        ),
    )


_TEXT_PROPERTY_KEYS: tuple[str, ...] = (
    "name",
    "title",
    "summary",
    "description",
    "statement",
    "fact",
    "rationale",
)


# High-precision text patterns. Recall is intentionally low — the classifier
# only fires when the cue is strong enough to pick one canonical label.
_TEXT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "Decision",
        re.compile(
            r"\b("
            r"(we|the\s+team|engineering|product)\s+(decided|chose|adopted|agreed)|"
            r"(adopted|selected|chose|going\s+with)\s+[\w\-./]+\s+(over|instead\s+of|as)|"
            r"architecture\s+decision|"
            r"design\s+decision|"
            r"decision\s+record|"
            r"\badr[- ]?\d*\b"
            r")",
            re.IGNORECASE,
        ),
    ),
    (
        "Incident",
        re.compile(
            r"\b("
            r"incident|outage|downtime|postmortem|post[- ]mortem|"
            r"p[0-4]\s+(incident|event|issue)|"
            r"sev[- ]?[0-4]\b"
            r")",
            re.IGNORECASE,
        ),
    ),
    (
        "Fix",
        re.compile(
            r"\b("
            r"hotfix|bug\s*fix|bugfix|"
            r"patch(ed|ing)?\s+(the\s+)?(bug|issue|regression|vulnerability)|"
            r"workaround\s+for|mitigation\s+for"
            r")",
            re.IGNORECASE,
        ),
    ),
    (
        "Alert",
        re.compile(
            r"\b("
            r"pagerduty|paged\s+on[- ]?call|pager\s+duty|"
            r"alert\s+fired|alerting\s+rule|threshold\s+alert"
            r")",
            re.IGNORECASE,
        ),
    ),
    (
        "Runbook",
        re.compile(
            r"\b(runbook|playbook|operational\s+procedure|recovery\s+steps|on[- ]?call\s+procedure)",
            re.IGNORECASE,
        ),
    ),
    (
        "BugPattern",
        re.compile(
            r"\b("
            r"flaky\s+(test|suite|spec)|bug\s+pattern|anti[- ]pattern|"
            r"recurring\s+(failure|bug|regression)|"
            r"known\s+(issue|failure\s+mode|bad\s+pattern)"
            r")",
            re.IGNORECASE,
        ),
    ),
    (
        "Preference",
        re.compile(
            r"\b("
            r"team\s+prefers|we\s+prefer|preferred\s+approach|style\s+preference"
            r")",
            re.IGNORECASE,
        ),
    ),
    (
        "Constraint",
        re.compile(
            r"\b("
            r"hard\s+constraint|architectural\s+constraint|compliance\s+requirement|"
            r"must\s+not\s+(be\s+)?(use|used|stored|exposed|committed|logged)|"
            r"never\s+(commit|log|store|expose|call)\s+\w+|"
            r"do\s+not\s+(call|use|import|modify)\s+\w+"
            r")",
            re.IGNORECASE,
        ),
    ),
    (
        "AgentInstruction",
        re.compile(
            r"\b("
            r"agents?\.md|claude\.md|cursor\.md|\.cursorrules|"
            r"agent\s+instruction|skill\s+definition|mcp\s+guidance"
            r")",
            re.IGNORECASE,
        ),
    ),
)


# Property signatures: presence of a property (with a plausible value) forces a label.
def _classify_from_properties(properties: Mapping[str, Any]) -> set[str]:
    out: set[str] = set()

    if properties.get("pr_number") is not None:
        out.add("PullRequest")

    sha = properties.get("sha")
    if isinstance(sha, str) and re.fullmatch(r"[0-9a-fA-F]{7,40}", sha):
        out.add("Commit")

    if properties.get("issue_number") is not None:
        out.add("Issue")

    if properties.get("github_login") or properties.get("display_name"):
        out.add("Person")

    _if_string(properties, "interface_type", out, "Interface")
    _if_string(properties, "store_type", out, "DataStore")
    _if_string(properties, "agent_type", out, "Agent")
    _if_string(properties, "fix_type", out, "Fix")
    _if_string(properties, "signal_type", out, "DiagnosticSignal")
    _if_string(properties, "integration_type", out, "Integration")
    _if_string(properties, "dependency_type", out, "Dependency")
    _if_string(properties, "component_type", out, "Component")
    _if_string(properties, "workflow_type", out, "LocalWorkflow")
    _if_string(properties, "constraint_type", out, "Constraint")
    _if_string(properties, "instruction_type", out, "AgentInstruction")
    _if_string(properties, "role_type", out, "Role")
    _if_string(properties, "strategy_type", out, "DeploymentStrategy")
    _if_string(properties, "environment_type", out, "Environment")
    _if_string(properties, "asset_type", out, "CodeAsset")
    _if_string(properties, "ref_type", out, "SourceReference")
    _if_string(properties, "job_type", out, "MaintenanceJob")
    _if_string(properties, "metric_type", out, "Metric")
    _if_string(properties, "pattern_type", out, "MaterializedAccessPath")

    if properties.get("preference_type") and properties.get("scope_kind"):
        out.add("Preference")

    return out


def _if_string(
    properties: Mapping[str, Any], key: str, out: set[str], label: str
) -> None:
    value = properties.get(key)
    if isinstance(value, str) and value.strip():
        out.add(label)


def _classify_from_text(text: str) -> set[str]:
    if not text:
        return set()
    out: set[str] = set()
    for label, pattern in _TEXT_PATTERNS:
        if pattern.search(text):
            out.add(label)
    return out


def _text_blob(properties: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for key in _TEXT_PROPERTY_KEYS:
        value = properties.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value)
    return " \n ".join(parts)


def _classify_from_edges(signals: EntitySignals) -> set[str]:
    out: set[str] = set()
    for name in signals.outgoing_edge_names:
        out.update(inferred_labels_for_episodic_edge_endpoint(name, "source"))
    for name in signals.incoming_edge_names:
        out.update(inferred_labels_for_episodic_edge_endpoint(name, "target"))
    return out


def _classify_from_canonical_type_hint(properties: Mapping[str, Any]) -> set[str]:
    """Entity extraction schemas can stamp ``canonical_type`` to pin an ontology label."""
    hint = properties.get("canonical_type")
    if not isinstance(hint, str):
        return set()
    value = hint.strip()
    if not value or not is_canonical_entity_label(value):
        return set()
    return {value}


def classify_entity(signals: EntitySignals) -> tuple[str, ...]:
    """Return canonical labels to ADD. Never returns labels the entity already has.

    Idempotent. Never returns non-canonical labels. Deterministic — the same
    signals always yield the same output — so the classifier pass can run on
    every episode ingest without churning the graph.
    """
    existing = frozenset(signals.labels)
    suggested: set[str] = set()
    suggested |= _classify_from_edges(signals)
    suggested |= _classify_from_properties(signals.properties)
    suggested |= _classify_from_text(_text_blob(signals.properties))
    suggested |= _classify_from_canonical_type_hint(signals.properties)

    canonical_new = {
        label
        for label in suggested
        if label not in existing and label in ENTITY_TYPES
    }
    return tuple(sorted(canonical_new))
