"""Deterministic ontology classifier: entity signals → canonical labels to add.

The classifier is fully spec-driven: it reads its rule tables from
:mod:`domain.ontology` rather than hardcoding label strings. To teach the
classifier about a new entity, add ``text_patterns`` / ``property_signatures``
to the entity spec — no changes needed here.

It runs on two paths:

* Reconciliation plan enrichment (``domain.canonical_label_inference``) — runs
  before structural writes so the validator sees canonical labels.
* Neo4j classifier pass over the canonical graph —
  runs after episodic extraction so flexible LLM output is pinned to the
  ontology.

Signal sources (all derived from spec metadata):

1. **Edge-endpoint inference** — :func:`inferred_labels_for_episodic_edge_endpoint`
   reads :data:`EDGE_ENDPOINT_INFERRED_LABELS` built from edge specs'
   ``source_inferred_labels`` / ``target_inferred_labels`` fields.
2. **Property signatures** — :data:`ENTITY_PROPERTY_SIGNATURES` maps property
   names declared on entity specs to the labels they imply.
3. **Text patterns** — :data:`ENTITY_TEXT_CLASSIFIERS` compiles regexes
   declared on entity specs.

When the signals do not uniquely pick a label, the classifier returns nothing
rather than guessing — a missing label is cheaper than a wrong one.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from context_engine.domain.ontology import (
    ENTITY_PROPERTY_SIGNATURES,
    ENTITY_TEXT_CLASSIFIERS,
    ENTITY_TYPES,
    inferred_labels_for_episodic_edge_endpoint,
    is_canonical_entity_label,
    normalize_edge_name,
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
            normalize_edge_name(n) for n in outgoing_edge_names if n
        ),
        incoming_edge_names=frozenset(
            normalize_edge_name(n) for n in incoming_edge_names if n
        ),
    )


_TEXT_PROPERTY_KEYS: tuple[str, ...] = (
    "name",
    "title",
    "summary",
    "description",
    "statement",
    "question",
    "fact",
    "rationale",
)

# Special cases where presence of a property is necessary but not sufficient
# for the label — these require additional checks beyond the spec's
# property_signatures list. Kept tiny on purpose.
_SHA_PATTERN = re.compile(r"^[0-9a-fA-F]{7,40}$")


def _classify_from_properties(properties: Mapping[str, Any]) -> set[str]:
    """Map property presence to canonical labels using spec metadata."""
    out: set[str] = set()
    for prop_name, labels in ENTITY_PROPERTY_SIGNATURES.items():
        value = properties.get(prop_name)
        if value is None:
            continue
        # Numeric IDs are valid signatures too (pr_number, issue_number).
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            for label in labels:
                out.add(label)
            continue
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                continue
            # ``sha`` only counts when it looks like a git SHA.
            if prop_name == "sha" and not _SHA_PATTERN.fullmatch(stripped):
                continue
            for label in labels:
                out.add(label)
    return out


def _classify_from_text(text: str) -> set[str]:
    if not text:
        return set()
    out: set[str] = set()
    for label, pattern in ENTITY_TEXT_CLASSIFIERS:
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
        label for label in suggested if label not in existing and label in ENTITY_TYPES
    }
    return tuple(sorted(canonical_new))
